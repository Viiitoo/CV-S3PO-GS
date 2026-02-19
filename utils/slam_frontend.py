import time
import numpy as np
import torch
import torch.multiprocessing as mp
import os

from utils.time_utils import attach_time_to_viewpoint

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.init_pose import get_pose, get_depth
from utils.depth_utils import process_depth
# 延迟导入光流相关模块，避免在未启用光流时也需要yacs依赖
# from utils.flow_utils import init_optical_flow_model, compute_optical_flow
# from utils.warp_utils import calculate_camera_flow
# from utils.flow_viz import save_flow_visualization

class FrontEnd(mp.Process):
    def __init__(self, config, model, save_dir=None):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.save_dir = save_dir

        self.initialized = False            
        self.kf_indices = []
        
        # [Modified] 1. 定义存储完整轨迹的列表
        self.full_trajectory_data = {
            "trj_id": [],
            "trj_est": []
        }
        
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
        self.model = model  # MASt3R Model
        self.theta = 0
        
        # 光流相关初始化
        self.flownet = None
        self.flow_2d_gt_list = []  # 缓存光流结果
        self.optical_flow_dict = {}  # {frame_idx: (optical_flow, target_kf_idx)} 存储原始光流和目标关键帧索引
        self.use_optical_flow = False
        self.use_camera_flow = False
        self.flow_visualization = False
        self.use_gt_pose = False
        
        # 计时相关属性初始化
        self._timing_get_pose = 0.0
        self._timing_get_depth = 0.0
        self._timing_optical_flow = 0.0
        self._timing_pose_opt = 0.0
        self._timing_pose_opt_iters = 0
        self._timing_process_depth = 0.0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.plot_dir = os.path.join(self.save_dir, "plot")
        
        if self.save_results:
            os.makedirs(self.plot_dir, exist_ok=True)
        
        # 读取光流相关配置
        self.use_optical_flow = self.config.get("Training", {}).get("use_optical_flow", False)
        self.use_camera_flow = self.config.get("Training", {}).get("use_camera_flow", False)
        self.flow_visualization = self.config.get("Training", {}).get("flow_visualization", False)
        self.use_gt_pose = self.config.get("Training", {}).get("use_gt_pose", False)
        
        # 初始化光流模型（延迟导入，避免未启用光流时也需要yacs依赖）
        if self.use_optical_flow or self.use_camera_flow:
            try:
                from utils.flow_utils import init_optical_flow_model
                flow_config = self.config.get("Flow", {})
                model_path = flow_config.get("model_path", "./gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth")
                self.flownet = init_optical_flow_model(model_path)
                Log("光流模型初始化成功", tag="Flow")
            except ImportError as e:
                Log(f"光流模块导入失败: {e}", tag="Flow")
                Log("提示: 如果不需要光流功能，请在配置文件中设置 use_optical_flow: False 和 use_camera_flow: False", tag="Flow")
                self.use_optical_flow = False
                self.use_camera_flow = False
                self.flownet = None
            except Exception as e:
                Log(f"光流模型初始化失败: {e}", tag="Flow")
                self.use_optical_flow = False
                self.use_camera_flow = False
                self.flownet = None
        
        # 创建光流可视化目录
        if self.flow_visualization:
            self.flow_viz_dir = os.path.join(self.save_dir, "flow_viz")
            os.makedirs(self.flow_viz_dir, exist_ok=True)       
    
    # Add a new keyframe. Create valid pixel mask using RGB boundary threshold from config, then generate initial depth map
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        # Compute angular difference with the previous frame (not used)
        R_now = viewpoint.R
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]      # Check if sum of RGB channels exceeds threshold; add a new dimension to match expected shape
        if self.monocular:
            if depth is None:
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)     # For the first frame, use MASt3R to estimate depth during map initialization
                initial_depth[~valid_rgb.cpu()] = 0
                return initial_depth[0].numpy()
            else:                                               # For non-initial keyframes, use rendered depth
                depth = depth.detach().clone()
                opacity = opacity.detach()
                
                initial_depth = depth
                
                # Compute scale factor and adjust rendered depth (Pointmap Replacement)
                render_depth = initial_depth.cpu().numpy()[0]
                torch.cuda.synchronize()
                t_process_depth_start = time.time()
                initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth(render_depth, viewpoint.mono_depth, last_depth = viewpoint_last.mono_depth, 
                                                                                             im1 = viewpoint_last.original_image, im2 = viewpoint.original_image, model = self.model,
                                                                                             patch_size = self.config["depth"]["patch_size"], 
                                                                                             mean_threshold = self.config["depth"]["mean_threshold"], std_threshold = self.config["depth"]["std_threshold"],
                                                                                             error_threshold = self.config["depth"]["error_threshold"], final_error_threshold = self.config["depth"]["final_error_threshold"],
                                                                                             min_accurate_pixels_ratio = self.config["depth"]["min_accurate_pixels_ratio"])
                torch.cuda.synchronize()
                t_process_depth_end = time.time()
                self._timing_process_depth = (t_process_depth_end - t_process_depth_start) * 1000  # ms

                # Correct MASt3R scale
                viewpoint.mono_depth = viewpoint.mono_depth * scale_factor

                pixel_num = viewpoint.image_height * viewpoint.image_width
                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0 
            return initial_depth
        # Keep ground truth depth usage
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)     
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # initial_depth is a 4D tensor (1, C, H, W); extract the first channel as (C, H, W)
    
    # Initialize the SLAM system: clear backend queue, reset state, set current frame to ground-truth pose, 
    # add a new keyframe, and push related info into the backend queue
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        # get mono_depth from MASt3R
        img = viewpoint.original_image
        mast_viz_cfg = self.config.get("mast3r_edge_viz", None)
        if isinstance(mast_viz_cfg, dict):
            _cfg = dict(mast_viz_cfg)
            if _cfg.get("enabled", False):
                viz_dir = os.path.join(self.save_dir, "viz_mast3r_pc_edge")
                _cfg["dir"] = viz_dir
                _cfg["frame_idx"] = int(cur_frame_idx)
                _cfg["tag"] = "f{:06d}".format(int(cur_frame_idx))
            mast_viz_cfg = _cfg
        torch.cuda.synchronize()
        t_init_depth_start = time.time()
        depth = get_depth(img, img, self.model, return_conf=False, mast3r_edge_viz=mast_viz_cfg)
        torch.cuda.synchronize()
        t_init_depth_end = time.time()
        viewpoint.mono_depth = depth
        
        self.kf_indices = []
        t_init_kf_start = time.time()
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        t_init_kf_end = time.time()
        self.request_init(cur_frame_idx, viewpoint, depth_map)      # Request initialization and push related info into the backend queue
        self.reset = False
        
        t_init_total = (t_init_kf_end - t_init_depth_start) * 1000
        Log(f"[Frame {cur_frame_idx}] 初始化总计: {t_init_total:.1f}ms | "
            f"MASt3R深度: {(t_init_depth_end - t_init_depth_start)*1000:.1f}ms | "
            f"添加关键帧: {(t_init_kf_end - t_init_kf_start)*1000:.1f}ms",
            tag="Timing")
   
    def tracking(self, cur_frame_idx, viewpoint):    
        ##=====================Pointmap Anchored Pose Estimation(PAPE)=====================
        # The previous frame
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        pose_prev = getWorld2View2(prev.R, prev.T)
        
        # adjacent keyframe
        last_keyframe_idx = self.current_window[0]
        last_kf = self.cameras[last_keyframe_idx]
        pose_last_kf = getWorld2View2(last_kf.R, last_kf.T)
        img1 = last_kf.original_image
        
        # Estimate the relative pose between the current frame and its adjacent keyframe
        img2 = viewpoint.original_image
        rgb_edge_pnp_cfg = self.config.get("rgb_edge_pnp", None)
        torch.cuda.synchronize()
        t_pose_start = time.time()
        rel_pose, render_depth = get_pose(img1=img1, img2=img2, model=self.model, dist_coeffs=self.dataset.dist_coeffs, 
                            viewpoint=last_kf, gaussians=self.gaussians, pipeline_params=self.pipeline_params, background=self.background,
                            rgb_edge_pnp=rgb_edge_pnp_cfg)
        torch.cuda.synchronize()
        t_pose_end = time.time()
        self._timing_get_pose = (t_pose_end - t_pose_start) * 1000  # ms
        
        # [加速] 延迟 depth 计算到关键帧处理时（非关键帧不需要 mono_depth）
        viewpoint.mono_depth = None
        self._timing_get_depth = 0.0
        
        # [加速] 光流仅在 request_keyframe() 的回溯机制中计算，非关键帧无需每帧计算
        self._timing_optical_flow = 0.0
        
        # Compute current frame's pose estimation
        identity_matrix = torch.eye(4, device=self.device)
        rel_pose = torch.from_numpy(rel_pose).to(self.device).float()
        # If the relative pose is identity (no motion), treat as a failure and use the previous pose
        if torch.allclose(rel_pose, identity_matrix, atol=1e-6):  
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(prev.R, prev.T)
        else:
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(pose_init[:3, :3], pose_init[:3, 3])

        # Use previous frame pose (for ablation)
        #viewpoint.update_RT(prev.R, prev.T)
        
        ## ===================================Pose Optimization=================================
        opt_params = []     # Exposure parameters a and b, used to adjust image brightness
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        torch.cuda.synchronize()
        t_opt_start = time.time()
        for tracking_itr in range(self.tracking_itr_num):
            attach_time_to_viewpoint(viewpoint, frame_idx=cur_frame_idx, num_frames=len(self.dataset))
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            # Check if rendering failed (e.g., no points initialized yet)
            if render_pkg is None:
                # Return None to indicate tracking failed
                return None
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 

            if tracking_itr % 10 == 0:              
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        # ====== 实验：强制使用GT位姿 ======
        force_gt_pose = self.config.get("Training", {}).get("force_gt_pose", False)
        if force_gt_pose and hasattr(viewpoint, 'R_gt') and hasattr(viewpoint, 'T_gt'):
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.cam_rot_delta.data.fill_(0)
            viewpoint.cam_trans_delta.data.fill_(0)

        torch.cuda.synchronize()
        t_opt_end = time.time()
        self._timing_pose_opt = (t_opt_end - t_opt_start) * 1000  # ms
        self._timing_pose_opt_iters = tracking_itr + 1  # 实际迭代次数
        
        self.median_depth = get_median_depth(depth, opacity)        # Median of rendered depth, used to determine whether the frame is a keyframe
        return render_pkg
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)  
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)          
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])        # Get transformation matrix from current frame to previous keyframe; extract translation and compute distance
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        # Avoid division-by-zero if both visibility masks are empty.
        point_ratio_2 = intersection / (union + 1e-6)

        rel = pose_CW @ torch.linalg.inv(last_kf_CW)

        R_rel = rel[:3, :3]
        t_rel = rel[:3, 3]
        dist_trans = torch.norm(t_rel)

        # 旋转角度
        angle = torch.acos( torch.clamp((torch.trace(R_rel) - 1) / 2, -1+1e-6, 1-1e-6) )

        rot_equiv = angle * self.median_depth
        alpha = 0.05
        # dist_total = torch.sqrt(dist_trans**2 + (alpha * rot_equiv)**2)
        
        # rotation_check = dist_total > kf_translation * self.median_depth
        # rotation_check2 = dist_total > kf_min_translation * self.median_depth
        # Keyframe if we have enough motion (translation or rotation), or low co-visibility with minimum motion.
        return (rot_equiv > alpha * self.median_depth) or dist_check or (
            (point_ratio_2 < kf_overlap) and dist_check2
        )
    
    # Add current frame to the window and remove the least important keyframe based on overlap ratio to keep window size within limit
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if (point_ratio_2 <= cut_off) and (len(window) > self.config["Training"]["window_size"]):        
            #if (point_ratio_2 <= cut_off):
                to_remove.append(kf_idx)
        # Remove the earliest keyframe among those with overlap below the threshold
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # If the window is still too large, remove the farthest keyframe
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        return window, removed_frame
    
    # Request to add a new keyframe and push related info into the backend queue
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        # 回溯计算: 当创建关键帧 K' 时，计算前一个关键帧 K → K' 的 optical flow
        # 因为创建 K' 时，K' 刚加入 kf_indices，不可能有比 K' 更大的关键帧
        # 所以我们为 K（前一个关键帧）计算 flow，而不是为 K' 计算
        prev_kf_optical_flow = None  # (prev_kf_idx, optical_flow, cur_frame_idx) or None

        if (self.use_optical_flow and
            self.flownet is not None and
            len(self.kf_indices) >= 2):
            # kf_indices[-1] == cur_frame_idx (刚被 add_new_keyframe 添加)
            # kf_indices[-2] == 前一个关键帧
            prev_kf_idx = self.kf_indices[-2]

            try:
                from utils.flow_utils import compute_optical_flow

                # 加载前一个关键帧和当前关键帧的图像
                prev_gt_color, _, _, _ = self.dataset[prev_kf_idx]
                prev_gt_image = prev_gt_color.cuda()
                cur_gt_image = viewpoint.original_image.cuda()

                # 计算 prev_kf → cur_kf 的 optical flow
                flow_2d = compute_optical_flow(prev_gt_image, cur_gt_image, self.flownet)

                if flow_2d is not None:
                    optical_flow_data = (flow_2d, cur_frame_idx)
                    self.optical_flow_dict[prev_kf_idx] = optical_flow_data
                    prev_kf_optical_flow = (prev_kf_idx, flow_2d, cur_frame_idx)
                    Log(f"回溯存储 optical flow: 帧{prev_kf_idx} -> 帧{cur_frame_idx}", tag="Flow")

            except Exception as e:
                import traceback
                Log(f"计算关键帧之间的optical flow失败: {e}", tag="Flow")
                Log(f"详细错误: {traceback.format_exc()}", tag="Flow")

        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.theta, prev_kf_optical_flow]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
    
    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
    
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True
    # Synchronize data from backend, including Gaussian scene, occlusion-aware visibility, and keyframe info; update keyframe
    def sync_backend(self, data):
        self.gaussians = data[1]
        self.occ_aware_visibility = data[2]
        keyframes = data[3]
        # self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
    # Clear current frame's camera data; clear CUDA cache every 10 frames
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    # Main execution loop: process messages in frontend and backend queues, perform tracking, keyframe management, 
    # synchronize data, clean up resources, and save results
    def run(self):
        # ... (前面的初始化代码保持不变) ...
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(    
            znear=0.01, zfar=100.0, fx=self.dataset.fx, fy=self.dataset.fy,
            cx=self.dataset.cx, cy=self.dataset.cy, W=self.dataset.width, H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)      
        toc = torch.cuda.Event(enable_timing=True)

        import json # 引入 json 库

        while True:
            # ... (队列处理代码保持不变) ...
            if self.q_vis2main.empty():      
                if self.pause: continue
            else:
                # ... (暂停逻辑保持不变) ...
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():    
                tic.record()
                
                # =========================================================
                # [Modified] 4. 程序结束时，保存我们自己收集的完整轨迹
                # =========================================================
                if cur_frame_idx >= len(self.dataset):  
                    if self.save_results:
                        # 1. 仍然调用原来的 eval_ate 保存关键帧评估结果 (可选)
                        eval_ate(
                            self.cameras, self.kf_indices, self.save_dir, 0,
                            final=True, monocular=self.monocular,
                        )
                        save_gaussians(self.gaussians, self.save_dir, "final", final=True)
                        
                        # 2. 保存所有帧的轨迹到 full_trajectory.json
                        full_trj_path = os.path.join(self.plot_dir, "full_trajectory.json")
                        # Log(f"[Traj] Saving full trajectory ({len(self.full_trajectory_data['trj_id'])} frames)")  # 减少冗余输出
                        with open(full_trj_path, "w") as f:
                            json.dump(self.full_trajectory_data, f, indent=4)
                            
                    break
                # =========================================================

                # ... (中间的 sleep 和 init 逻辑保持不变) ...
                if self.requested_init:
                    time.sleep(0.01)
                    continue
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                torch.cuda.synchronize()
                t_frame_start = time.time()
                
                t_cam_init_start = time.time()
                viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, projection_matrix)
                attach_time_to_viewpoint(viewpoint, frame_idx=cur_frame_idx, num_frames=len(self.dataset))
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint
                torch.cuda.synchronize()
                t_cam_init_end = time.time()

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                # Tracking
                torch.cuda.synchronize()
                t_tracking_start = time.time()
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                torch.cuda.synchronize()
                t_tracking_end = time.time()
                
                # Check if tracking failed (e.g., no points initialized yet)
                if render_pkg is None:
                    # Skip this frame if tracking failed
                    cur_frame_idx += 1
                    continue
                
                # =========================================================
                # [Modified] 3. 强制记录每一帧的位姿 (在 Tracking 之后)
                # =========================================================
                # 获取 World-to-Camera 矩阵
                w2c = getWorld2View2(viewpoint.R, viewpoint.T)
                # 计算 Camera-to-World (C2W) 矩阵，这通常是画轨迹需要的
                c2w = torch.linalg.inv(w2c)
                
                # 转为 list 存入字典
                self.full_trajectory_data["trj_id"].append(cur_frame_idx)
                self.full_trajectory_data["trj_est"].append(c2w.cpu().numpy().tolist())
                # =========================================================
                
                # ... (后续代码保持不变: cleanup, keyframe logic 等) ...
                
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                # ... (后面所有的 keyframe 判断逻辑和 eval_ate 调用保持原样即可) ...
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(cur_frame_idx, last_keyframe_idx, curr_visibility, self.occ_aware_visibility)
                
                if len(self.current_window) < self.window_size:    
                    union = torch.logical_or(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                    intersection = torch.logical_and(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (check_time and point_ratio < self.config["Training"]["kf_overlap"])
                
                if self.single_thread:      
                    create_kf = check_time and create_kf
                
                t_kf_start = time.time()
                if create_kf:
                    # [加速] 延迟计算：仅为关键帧计算 MASt3R 深度
                    if viewpoint.mono_depth is None:
                        mast_viz_cfg = self.config.get("mast3r_edge_viz", None)
                        if isinstance(mast_viz_cfg, dict):
                            _cfg = dict(mast_viz_cfg)
                            if _cfg.get("enabled", False):
                                viz_dir = os.path.join(self.save_dir, "viz_mast3r_pc_edge")
                                _cfg["dir"] = viz_dir
                                _cfg["frame_idx"] = int(cur_frame_idx)
                                _cfg["tag"] = "f{:06d}".format(int(cur_frame_idx))
                            mast_viz_cfg = _cfg
                        torch.cuda.synchronize()
                        t_depth_start = time.time()
                        viewpoint.mono_depth = get_depth(
                            viewpoint.original_image, viewpoint.original_image,
                            self.model, return_conf=False, mast3r_edge_viz=mast_viz_cfg)
                        torch.cuda.synchronize()
                        self._timing_get_depth = (time.time() - t_depth_start) * 1000
                    self.current_window, removed = self.add_to_window(cur_frame_idx, curr_visibility, self.occ_aware_visibility, self.current_window)       
                    depth_map = self.add_new_keyframe(cur_frame_idx, depth=render_pkg["depth"], opacity=render_pkg["opacity"], init=False)
                    self.request_keyframe(cur_frame_idx, viewpoint, self.current_window, depth_map)
                else:
                    self.cleanup(cur_frame_idx)
                torch.cuda.synchronize()
                t_kf_end = time.time()
                
                # ==================== 帧计时汇总 ====================
                t_frame_end = time.time()
                t_frame_total = (t_frame_end - t_frame_start) * 1000
                t_cam_init = (t_cam_init_end - t_cam_init_start) * 1000
                t_tracking = (t_tracking_end - t_tracking_start) * 1000
                t_kf_creation = (t_kf_end - t_kf_start) * 1000
                
                kf_tag = " [KF]" if create_kf else ""
                Log(f"[Frame {cur_frame_idx}]{kf_tag} 总计: {t_frame_total:.1f}ms | "
                    f"相机初始化: {t_cam_init:.1f}ms | "
                    f"Tracking: {t_tracking:.1f}ms ("
                    f"get_pose: {self._timing_get_pose:.1f}ms, "
                    f"get_depth: {self._timing_get_depth:.1f}ms, "
                    f"光流: {self._timing_optical_flow:.1f}ms, "
                    f"位姿优化: {self._timing_pose_opt:.1f}ms/{self._timing_pose_opt_iters}itr) | "
                    f"KF处理: {t_kf_creation:.1f}ms"
                    + (f" (process_depth: {getattr(self, '_timing_process_depth', 0.0):.1f}ms)" if create_kf else ""),
                    tag="Timing")
                # ====================================================
                
                cur_frame_idx += 1          

                if (self.save_results and self.save_trj and create_kf and len(self.kf_indices) % self.save_trj_kf_intv == 0):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(self.cameras, self.kf_indices, self.save_dir, cur_frame_idx, monocular=self.monocular)
                
                toc.record()
                torch.cuda.synchronize()       
                if create_kf:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:      
                # ... (backend sync 逻辑保持不变) ...
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break