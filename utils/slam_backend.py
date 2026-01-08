import random
import time

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import os

from utils.time_utils import attach_time_to_viewpoint

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.init_pose import save_depth_comparison


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir
        self.num_frames = None  # 将在 slam.py 中设置为 len(dataset)
        self.checkpoint_iterations = config.get("Results", {}).get("checkpoint_iterations", [])
        self.iteration_count_global = 0  # 全局迭代计数，用于checkpoint保存


        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.theta = 0

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.global_BA_itr_num = self.config["Training"]["global_BA_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    # Insert new Gaussians into the Gaussian scene based on the new keyframe's viewpoint and geometry
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    # Initialize the SLAM map by optimizing Gaussians through multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1

            attach_time_to_viewpoint(viewpoint, frame_idx=cur_frame_idx, num_frames=self.num_frames)
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(  
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(                 
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:  
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()                         
                self.gaussians.optimizer.zero_grad(set_to_none=True)    

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        
        # 输出初始化信息（包含形变状态）
        num_points = self.gaussians._xyz.shape[0]
        has_deformation = hasattr(self.gaussians, '_w_pos') and self.gaussians._w_pos.numel() > 0
        deform_info = ""
        if has_deformation:
            w_pos = self.gaussians._w_pos
            deform_info = f", Deform params: {w_pos.shape}, K_time={self.gaussians.K_time}"
        Log(f"[Init] Map initialized at frame {cur_frame_idx}, Points: {num_points}{deform_info}")
        return render_pkg
    # Optimize keyframe poses and Gaussians scene
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items():  # Add viewpoints outside the current window to the random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append((cam_idx, viewpoint))       
            
        for _ in range(iters):
            self.iteration_count += 1
            self.iteration_count_global += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []                 
            visibility_filter_acm = []                      
            radii_acm = []                                  
            n_touched_acm = []                            
            # Track which keyframes successfully rendered (kf_idx -> index in n_touched_acm)
            kf_to_n_touched_idx = {}

            keyframes_opt = []          

            for cam_idx in range(len(current_window)):      # For each keyframe in the current window, perform rendering and compute loss
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                kf_idx = current_window[cam_idx]
                attach_time_to_viewpoint(viewpoint, frame_idx=kf_idx, num_frames=self.num_frames)

                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                # Check if rendering failed (e.g., no points initialized yet)
                if render_pkg is None:
                    # Skip this keyframe if no points are available yet
                    continue
                (                                          
                    image,
                    viewspace_point_tensor,                 
                    visibility_filter,                     
                    radii,                                  
                    depth,                                 
                    opacity,                                
                    n_touched,                              
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                # Record the mapping from kf_idx to index in n_touched_acm
                kf_to_n_touched_idx[kf_idx] = len(n_touched_acm) - 1     
                
            # In each iteration, randomly select two non-window keyframes for optimization
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:     
                kf_idx, viewpoint = random_viewpoint_stack[cam_idx]
                attach_time_to_viewpoint(viewpoint, frame_idx=kf_idx, num_frames=self.num_frames)

                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                # Check if rendering failed (e.g., no points initialized yet)
                if render_pkg is None:
                    # Skip this keyframe if no points are available yet
                    continue
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            # isotropic regularization
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            
            # time deformation regularization - 防止位移权重过大导致飘移
            if hasattr(self.gaussians, '_w_pos') and self.gaussians._w_pos.numel() > 0:
                # L2 正则化：鼓励小位移
                w_pos_reg = self.gaussians._w_pos.pow(2).mean()
                loss_mapping += 0.1 * w_pos_reg
                
                # 时间平滑正则化：相邻时间的位移应该相似（可选）
                # sigma_reg = F.softplus(self.gaussians.t_sigma_raw).mean()
                # loss_mapping += 0.01 * (1.0 / (sigma_reg + 1e-6))  # 鼓励较宽的基函数
            
            loss_mapping.backward()
            gaussian_split = False
            
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                # Check if gaussians are initialized
                if hasattr(self.gaussians, '_xyz') and self.gaussians._xyz is not None and self.gaussians._xyz.shape[0] > 0:
                    # Only process keyframes that successfully rendered
                    for kf_idx in current_window:
                        if kf_idx in kf_to_n_touched_idx:
                            n_touched_idx = kf_to_n_touched_idx[kf_idx]
                            n_touched = n_touched_acm[n_touched_idx]
                            self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                        else:
                            # If rendering failed, set visibility to zero
                            self.occ_aware_visibility[kf_idx] = torch.zeros(
                                self.gaussians._xyz.shape[0], dtype=torch.long, device=self.device
                            )
                else:
                    # If no gaussians initialized yet, create empty visibility dict
                    for kf_idx in current_window:
                        self.occ_aware_visibility[kf_idx] = None

                # Only prune on the last iteration and when we have full window
                if prune:     
                    if len(current_window) == self.config["Training"]["window_size"]:
                        # Check if we have valid visibility data
                        has_valid_visibility = any(
                            v is not None for v in self.occ_aware_visibility.values()
                        )
                        if has_valid_visibility:
                            prune_mode = self.config["Training"]["prune_mode"]
                            prune_coviz = self.config["Training"]["prune_num"]  # prune parameter
                            self.gaussians.n_obs.fill_(0)
                            for window_idx, visibility in self.occ_aware_visibility.items():
                                if visibility is not None:
                                    self.gaussians.n_obs += visibility.cpu()
                            to_prune = None
                            if prune_mode == "odometry":
                                to_prune = self.gaussians.n_obs < 3
                                # make sure we don't split the gaussians, break here.
                            if prune_mode == "slam":
                                # only prune keyframes which are relatively new
                                sorted_window = sorted(current_window, reverse=True)
                                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                                if not self.initialized:
                                    mask = self.gaussians.unique_kfIDs >= 0
                                to_prune = torch.logical_and(
                                    self.gaussians.n_obs <= prune_coviz, mask
                                )
                            if to_prune is not None and self.monocular:       
                                self.gaussians.prune_points(to_prune.cuda())
                                for idx in range((len(current_window))):
                                    current_idx = current_window[idx]
                                    if self.occ_aware_visibility.get(current_idx) is not None:
                                        self.occ_aware_visibility[current_idx] = (                
                                            self.occ_aware_visibility[current_idx][~to_prune]
                                        )
                        if not self.initialized:
                            self.initialized = True
                            num_points = self.gaussians._xyz.shape[0]
                            Log(f"[Init] SLAM initialized, Total points: {num_points}")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    num_points = self.gaussians._xyz.shape[0]
                    Log(f"[Densify] Resetting opacity of non-visible Gaussians, Points: {num_points}")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                
                # 监控时间变形状态（每200次迭代输出一次）
                if self.iteration_count % 200 == 0 and hasattr(self.gaussians, '_w_pos'):
                    if self.gaussians._w_pos.numel() > 0:
                        w_pos = self.gaussians._w_pos
                        w_rot = self.gaussians._w_rot if self.gaussians._w_rot.numel() > 0 else None
                        w_scale = self.gaussians._w_scale if self.gaussians._w_scale.numel() > 0 else None
                        
                        # 位置形变统计
                        w_pos_norm = w_pos.norm().item()
                        w_pos_max = w_pos.abs().max().item()
                        w_pos_mean = w_pos.abs().mean().item()
                        
                        # 旋转形变统计
                        w_rot_info = ""
                        if w_rot is not None:
                            w_rot_norm = w_rot.norm().item()
                            w_rot_max = w_rot.abs().max().item()
                            w_rot_info = f", rot_norm={w_rot_norm:.6f}, rot_max={w_rot_max:.6f}"
                        
                        # 放缩形变统计
                        w_scale_info = ""
                        if w_scale is not None:
                            w_scale_norm = w_scale.norm().item()
                            w_scale_max = w_scale.abs().max().item()
                            w_scale_info = f", scale_norm={w_scale_norm:.6f}, scale_max={w_scale_max:.6f}"
                        
                        # 时间基函数信息
                        t_mu = self.gaussians.t_mu.detach().cpu().numpy()
                        t_sigma = torch.nn.functional.softplus(self.gaussians.t_sigma_raw).detach().cpu().numpy()
                        t_mu_range = f"[{t_mu.min():.3f}, {t_mu.max():.3f}]"
                        t_sigma_mean = t_sigma.mean()
                        
                        Log(f"[Deform] iter={self.iteration_count}, points={w_pos.shape[0]}, "
                            f"pos_norm={w_pos_norm:.6f}, pos_max={w_pos_max:.6f}, pos_mean={w_pos_mean:.6f}"
                            f"{w_rot_info}{w_scale_info}, "
                            f"t_mu={t_mu_range}, t_sigma_mean={t_sigma_mean:.4f}")
                
                # 保存checkpoint（参考EH-SurGS的方式）
                if self.save_dir and len(self.checkpoint_iterations) > 0:
                    if self.iteration_count_global in self.checkpoint_iterations:
                        checkpoint_path = os.path.join(self.save_dir, "checkpoints", f"chkpnt_{self.iteration_count_global}.pth")
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        checkpoint_data = (self.gaussians.capture(), self.iteration_count_global)
                        torch.save(checkpoint_data, checkpoint_path)
                        Log(f"[Checkpoint] Saved checkpoint at iteration {self.iteration_count_global} to {checkpoint_path}")
                
                # Pose update
                if up_pose:
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
        return gaussian_split
                
    # Run color refinement as a post-processing step after SLAM
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())      
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]      
            attach_time_to_viewpoint(viewpoint_cam, frame_idx=viewpoint_cam_idx, num_frames=self.num_frames)
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():       
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(26000)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
            
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    # Main execution loop: 
    # process backend messages, perform initialization, optimize keyframe map, color refinement,
    # synchronize data, and push updates to the frontend
    def run(self):
        # 在子进程中显式设置 CUDA 设备，确保使用正确的 GPU
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES 已经限制了可见设备，所以这里用 0
        torch.cuda.empty_cache()
        
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:       
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()

                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]

                    attach_time_to_viewpoint(viewpoint, frame_idx=cur_frame_idx, num_frames=self.num_frames)

                    Log(f"[Init] Resetting system at frame {cur_frame_idx}")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    self.theta = data[5]
                    
                    # 输出关键帧和形变状态信息
                    if hasattr(self.gaussians, '_w_pos') and self.gaussians._w_pos.numel() > 0:
                        num_points = self.gaussians._xyz.shape[0]
                        w_pos_active = (self.gaussians._w_pos.abs() > 1e-6).any(dim=-1).sum().item()
                        w_pos_ratio = w_pos_active / num_points if num_points > 0 else 0
                        Log(f"[Keyframe] Frame {cur_frame_idx}, Window: {current_window}, "
                            f"Points: {num_points}, Active deform: {w_pos_active} ({w_pos_ratio*100:.1f}%)")
                    else:
                        Log(f"[Keyframe] Frame {cur_frame_idx}, Window: {current_window}")

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)

                    attach_time_to_viewpoint(viewpoint, frame_idx=cur_frame_idx, num_frames=self.num_frames)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_nosingle = self.config["Training"]["mapping_itr_nosingle"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else iter_nosingle
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            num_points = self.gaussians._xyz.shape[0]
                            Log(f"[BA] Performing initial BA, Points: {num_points}, Iters: {iter_per_kf}")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):     
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:        
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf, up_pose=True)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
