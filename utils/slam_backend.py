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
        self.optical_flow_dict = {}  # {frame_idx: (optical_flow, target_kf_idx)} 原始光流 + 目标帧索引

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
            from utils.logging_utils import format_number
            deform_info = f" | 形变参数: [cyan]{w_pos.shape}[/cyan], K_time={self.gaussians.K_time}"
        Log(f"地图初始化完成 | 帧: {cur_frame_idx} | 点数: [green]{num_points:,}[/green]{deform_info}", tag="Init")
        return render_pkg
    # Optimize keyframe poses and Gaussians scene
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return

        # 首次调用 map() 时打印光流光栅化器配置摘要
        if not getattr(self, '_map_flow_config_logged', False):
            _fr_cfg = self.config["Training"].get("use_flow_rasterizer", False)
            _fr_fb = self.config["Training"].get("flow_rasterizer_fallback", True)
            _fl_w = self.config["Training"].get("flow_loss_weight", 0)
            _fl_wu = self.config["Training"].get("flow_warm_up", 1000)
            _fl_int = self.config["Training"].get("flow_loss_interval", 5)
            from gaussian_splatting.gaussian_renderer import FLOW_RASTERIZER_AVAILABLE
            Log(f"[FlowRasterizer 配置] use_flow_rasterizer={_fr_cfg} | "
                f"available={FLOW_RASTERIZER_AVAILABLE} | fallback={_fr_fb} | "
                f"flow_loss_weight={_fl_w} | warm_up={_fl_wu} | interval={_fl_int}",
                tag="Backend")
            self._map_flow_config_logged = True

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items():  # Add viewpoints outside the current window to the random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append((cam_idx, viewpoint))

        # ========== 解耦优化配置 ==========
        decouple_cfg = self.config.get("Training", {}).get("decouple_pose_deform", None)
        decouple_enabled = (decouple_cfg is not None and
                           decouple_cfg.get("enabled", False) and
                           hasattr(self.gaussians, '_coefs') and
                           self.gaussians._coefs.numel() > 0 and
                           not getattr(self.gaussians, 'freeze_coefs', False))
        if decouple_enabled:
            decouple_pose_steps = decouple_cfg.get("pose_steps", 3)
            decouple_deform_steps = decouple_cfg.get("deform_steps", 1)
            decouple_cycle = decouple_pose_steps + decouple_deform_steps

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

                # 读取光流光栅化器配置
                use_flow_raster_cfg = self.config["Training"].get("use_flow_rasterizer", False)
                flow_loss_weight = self.config["Training"].get("flow_loss_weight", 0)
                flow_loss_interval = self.config["Training"].get("flow_loss_interval", 5)
                flow_warm_up = self.config["Training"].get("flow_warm_up", 1000)
                flow_rasterizer_fallback = self.config["Training"].get("flow_rasterizer_fallback", True)

                # 优化：只在最近的N对关键帧上计算flow loss，避免重复计算所有frame对
                # flow_on_all_frames=True: 在所有frame对上计算
                # flow_on_all_frames=False (默认): 只在最近的1对上计算（cam_idx=1，即次新帧->最新帧）
                # 注意: cam_idx=0是最新帧，通常没有next frame，不会计算flow loss
                flow_on_all_frames = self.config["Training"].get("flow_on_all_frames", False)
                flow_max_pairs = self.config["Training"].get("flow_max_pairs", 1)  # 最多计算几对关键帧

                # 判断本次迭代是否需要使用光流光栅化器
                # 仅在以下条件全部满足时使用：配置启用、有光流数据、超过预热期、到达计算间隔
                use_flow_raster_now = (
                    use_flow_raster_cfg and
                    flow_loss_weight > 0 and
                    self.iteration_count >= flow_warm_up and
                    (self.iteration_count % flow_loss_interval) == 0 and
                    kf_idx in self.optical_flow_dict and
                    (flow_on_all_frames or cam_idx < flow_max_pairs + 1)  # 默认只在最近的N对上计算(cam_idx=1到N)
                )

                # 首次激活光流光栅化器时打印详细日志
                if use_flow_raster_now and not getattr(self, '_flow_raster_first_logged', False):
                    from gaussian_splatting.gaussian_renderer import FLOW_RASTERIZER_AVAILABLE
                    Log(f"[FlowRasterizer] 首次激活 | iter={self.iteration_count} kf={kf_idx} | "
                        f"cfg={use_flow_raster_cfg} available={FLOW_RASTERIZER_AVAILABLE} "
                        f"fallback={flow_rasterizer_fallback} | flow_max_pairs={flow_max_pairs}", tag="Backend")
                    self._flow_raster_first_logged = True

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background,
                    use_flow_rasterizer=use_flow_raster_now
                )
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
                
                # ============================================================
                # Flow Loss: 使用原始 optical_flow，后端重新计算 camera_flow
                # 修复：1) 帧对齐验证 2) 用当前优化位姿计算camera_flow 3) 梯度截断
                # 当 use_flow_rasterizer=True 时，render返回10个输出，
                # 包含 proj_2D, conic_2D 等完整模式所需的中间信息
                # ============================================================
                flow_loss_weight = self.config["Training"].get("flow_loss_weight", 0)
                flow_loss_interval = self.config["Training"].get("flow_loss_interval", 5)
                flow_warm_up = self.config["Training"].get("flow_warm_up", 1000)
                flow_on_all_frames = self.config["Training"].get("flow_on_all_frames", False)
                flow_max_pairs = self.config["Training"].get("flow_max_pairs", 1)

                if (flow_loss_weight > 0 and
                    self.iteration_count >= flow_warm_up and
                    (self.iteration_count % flow_loss_interval) == 0 and
                    kf_idx in self.optical_flow_dict and
                    (flow_on_all_frames or cam_idx < flow_max_pairs + 1)):  # 默认只在最近的N对上计算

                    # 取出前端存储的 (optical_flow, target_kf_idx)
                    optical_flow_data = self.optical_flow_dict[kf_idx]
                    if isinstance(optical_flow_data, tuple) and len(optical_flow_data) == 2:
                        optical_flow_gt, stored_target_kf = optical_flow_data
                    else:
                        # 兼容旧格式（直接存储 motion_flow tensor）
                        optical_flow_gt = optical_flow_data
                        stored_target_kf = None

                    # 找到实际的下一个关键帧
                    actual_next_kf = None
                    for candidate_idx in sorted(self.viewpoints.keys()):
                        if candidate_idx > kf_idx:
                            actual_next_kf = candidate_idx
                            break

                    # 帧对齐验证：存储的目标帧必须与实际下一关键帧一致
                    if (actual_next_kf is None or
                        (stored_target_kf is not None and stored_target_kf != actual_next_kf)):
                        _log_flow = (self.iteration_count % max(1, flow_loss_interval * 10)) == 0
                        if _log_flow and actual_next_kf is not None:
                            Log(f"Flow loss 跳过: 帧对齐不一致 stored={stored_target_kf} actual={actual_next_kf} | kf={kf_idx}", tag="Flow")
                    elif actual_next_kf in self.viewpoints:
                        try:
                            from utils.flow_utils import calculate_gs_flow
                            from utils.warp_utils import warping_gs_flow, calculate_camera_flow
                            from utils.slam_utils import flow_loss

                            viewpoint_next = self.viewpoints[actual_next_kf]
                            attach_time_to_viewpoint(viewpoint_next, frame_idx=actual_next_kf, num_frames=self.num_frames)

                            render_pkg_next = render(
                                viewpoint_next, self.gaussians, self.pipeline_params, self.background,
                                use_flow_rasterizer=use_flow_raster_now
                            )

                            if render_pkg_next is not None:
                                depth_for_flow = depth
                                if depth_for_flow.dim() == 2:
                                    depth_for_flow = depth_for_flow.unsqueeze(0)

                                # --- 用当前优化后的位姿重新计算 camera_flow ---
                                # 避免使用前端计算时的旧位姿，消除负反馈循环
                                depth_for_cam_flow = depth_for_flow
                                if depth_for_cam_flow.dim() == 3:
                                    depth_for_cam_flow = depth_for_cam_flow.unsqueeze(0)  # (1, 1, H, W)
                                camera_flow = calculate_camera_flow(depth_for_cam_flow, viewpoint, viewpoint_next)

                                # motion_flow = optical_flow - camera_flow（使用最新位姿）
                                motion_flow = optical_flow_gt.detach() - camera_flow.detach()

                                # --- 计算 GS Flow ---
                                proj_2D = render_pkg.get("proj_2D")
                                conic_2D = render_pkg.get("conic_2D")
                                conic_2D_inv = render_pkg.get("conic_2D_inv")
                                gs_per_pixel = render_pkg.get("gs_per_pixel")
                                weight_per_gs_pixel = render_pkg.get("weight_per_gs_pixel")
                                x_mu = render_pkg.get("x_mu")
                                next_proj_2D = render_pkg_next.get("proj_2D")
                                next_conic_2D = render_pkg_next.get("conic_2D")

                                full_mode_params = {
                                    "proj_2D": proj_2D is not None,
                                    "conic_2D": conic_2D is not None,
                                    "conic_2D_inv": conic_2D_inv is not None,
                                    "gs_per_pixel": gs_per_pixel is not None,
                                    "weight_per_gs_pixel": weight_per_gs_pixel is not None,
                                    "x_mu": x_mu is not None,
                                    "next_proj_2D": next_proj_2D is not None,
                                    "next_conic_2D": next_conic_2D is not None,
                                }
                                _log_flow = (self.iteration_count % max(1, flow_loss_interval * 10)) == 0

                                if all(full_mode_params.values()):
                                    # 完整模式：使用各向异性GS光流计算
                                    # 验证张量形状一致性
                                    H_render, W_render = render_pkg["render"].shape[1:]
                                    if gs_per_pixel.shape[-2:] != (H_render, W_render):
                                        if _log_flow:
                                            Log(f"GS Flow: [red]形状不匹配[/red] gs_per_pixel {gs_per_pixel.shape} vs render ({H_render}, {W_render})，降级到简化模式", tag="Flow")
                                        gs_flow = calculate_gs_flow(depth1=depth_for_flow, cam1=viewpoint, cam2=viewpoint_next)
                                        gs_flow_aligned = warping_gs_flow(depth_for_flow, gs_flow, viewpoint, viewpoint_next)
                                    else:
                                        if _log_flow:
                                            Log(f"GS Flow: [green]完整模式[/green] | {kf_idx} -> {actual_next_kf} | iter={self.iteration_count}", tag="Flow")
                                        gs_flow = calculate_gs_flow(
                                            gs_per_pixel=gs_per_pixel,
                                            weight_per_gs_pixel=weight_per_gs_pixel,
                                            next_conic_2D=next_conic_2D,
                                            conic_2D_inv=conic_2D_inv,
                                            proj_2D=proj_2D,
                                            next_proj_2D=next_proj_2D,
                                            x_mu=x_mu
                                        )
                                        gs_flow_aligned = gs_flow  # 完整模式内部已处理对齐
                                else:
                                    if _log_flow:
                                        missing = [k for k, v in full_mode_params.items() if not v]
                                        Log(f"GS Flow: [yellow]简化模式[/yellow] | {kf_idx} -> {actual_next_kf} | 缺少: {missing} | iter={self.iteration_count}", tag="Flow")
                                    gs_flow = calculate_gs_flow(depth1=depth_for_flow, cam1=viewpoint, cam2=viewpoint_next)
                                    gs_flow_aligned = warping_gs_flow(depth_for_flow, gs_flow, viewpoint, viewpoint_next)

                                H, W = image.shape[-2:]
                                flow_loss_value = flow_loss(gs_flow_aligned, motion_flow, H, W)

                                # 截断异常大的 flow loss，防止错误梯度主导优化
                                flow_loss_clamped = torch.clamp(flow_loss_value, max=0.5)
                                loss_mapping += flow_loss_weight * flow_loss_clamped

                                if _log_flow:
                                    Log(f"Flow loss: {flow_loss_value.item():.6f} (clamped: {flow_loss_clamped.item():.6f}) | weight: {flow_loss_weight} | iter={self.iteration_count}", tag="Flow")

                        except Exception as e:
                            import traceback
                            Log(f"Flow loss计算失败: {e}", tag="Flow")
                
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
            
            # time deformation regularization - 对 _coefs 添加 L2 正则化
            # 注：旧代码引用 _w_pos（已废弃字段，永远为空），现改为对 _coefs 正则化
            coefs_reg_weight = self.config.get("Training", {}).get("coefs_reg_weight", 0.0)
            if coefs_reg_weight > 0 and hasattr(self.gaussians, '_coefs') and self.gaussians._coefs.numel() > 0:
                coefs_reg = self.gaussians._coefs.pow(2).mean()
                loss_mapping += coefs_reg_weight * coefs_reg
            
            loss_mapping.backward()

            # ========== 解耦优化：选择性清零梯度 ==========
            if decouple_enabled:
                phase_idx = (self.iteration_count - 1) % decouple_cycle
                if phase_idx < decouple_pose_steps:
                    # Phase P：清零 _coefs 梯度，只更新位姿+高斯几何
                    if self.gaussians._coefs.grad is not None:
                        self.gaussians._coefs.grad.zero_()
                else:
                    # Phase D：清零位姿梯度，只更新 _coefs+高斯几何
                    if self.keyframe_optimizers is not None:
                        for group in self.keyframe_optimizers.param_groups:
                            for p in group["params"]:
                                if p.grad is not None:
                                    p.grad.zero_()

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
                            if n_touched is not None:
                                self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                            else:
                                # FlowRasterizer doesn't return n_touched, assume all points visible
                                self.occ_aware_visibility[kf_idx] = torch.ones(
                                    self.gaussians._xyz.shape[0], dtype=torch.long, device=self.device
                                )
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
                                # 在prune后更新deformation table，确保只有需要形变的点被标记
                                # 这样可以优化性能并改善建模效果
                                if hasattr(self.gaussians, 'update_deformation_table') and getattr(self.gaussians, 'use_deformation', True):
                                    self.gaussians.update_deformation_table()
                                for idx in range((len(current_window))):
                                    current_idx = current_window[idx]
                                    if self.occ_aware_visibility.get(current_idx) is not None:
                                        self.occ_aware_visibility[current_idx] = (                
                                            self.occ_aware_visibility[current_idx][~to_prune]
                                        )
                        if not self.initialized:
                            self.initialized = True
                            num_points = self.gaussians._xyz.shape[0]
                            Log(f"SLAM系统初始化完成 | 总点数: [green]{num_points:,}[/green]", tag="Init")
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
                    # 在densify_and_prune后更新deformation table
                    # 新点的deformation_table会在densify_and_prune中初始化，这里更新确保阈值正确
                    if hasattr(self.gaussians, 'update_deformation_table') and getattr(self.gaussians, 'use_deformation', True):
                        self.gaussians.update_deformation_table()
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    num_points = self.gaussians._xyz.shape[0]
                    # #region agent log
                    try:
                        log_dir = '/home/sjw/data0/lsx/S3PO_baseline/.cursor'
                        os.makedirs(log_dir, exist_ok=True)
                        with open('/home/sjw/data0/lsx/S3PO_baseline/.cursor/debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"slam_backend.py:422","message":"Before Log call","data":{"num_points":int(num_points),"Log_in_locals":"Log" in locals(),"Log_in_globals":"Log" in globals()},"timestamp":int(time.time()*1000)}) + '\n')
                    except Exception:
                        pass
                    # #endregion
                    Log(f"重置不可见高斯点不透明度 | 当前点数: [yellow]{num_points:,}[/yellow]", tag="Densify")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                
                # 监控时间变形状态（每200次迭代输出一次）
                if self.iteration_count % 200 == 0:
                    if hasattr(self.gaussians, '_coefs') and self.gaussians._coefs.numel() > 0:
                        coefs = self.gaussians._coefs
                        Log(f"[Deform Stats] coefs_norm={coefs.norm().item():.6f}, "
                            f"coefs_max={coefs.abs().max().item():.6f}, "
                            f"coefs_mean={coefs.abs().mean().item():.6f}", tag="BACKEND")
                    if hasattr(self.gaussians, '_deformation_table') and self.gaussians._deformation_table.numel() > 0:
                        table = self.gaussians._deformation_table
                        ratio = table.sum().item() / table.numel()
                        Log(f"[Deform Stats] dynamic_ratio={ratio:.4f} ({table.sum().item()}/{table.numel()})", tag="BACKEND")
                
                pcd_save_interval = self.config["Results"].get("pcd_save_interval", 1000) 
            
                if self.save_dir and self.iteration_count_global % pcd_save_interval == 0:
                    # 构造保存路径
                    pcd_path = os.path.join(self.save_dir, "point_clouds", f"iteration_{self.iteration_count_global}.ply")
                    os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
                    
                    # 调用 GaussianModel 的 save_ply 方法
                    if hasattr(self.gaussians, 'save_ply'):
                        self.gaussians.save_ply(pcd_path)
                        Log(f"已保存点云 | 迭代: [cyan]{self.iteration_count_global}[/cyan] | 路径: {pcd_path}", tag="IO")
                    else:
                        Log("错误: self.gaussians 对象没有 save_ply 方法", tag="Error")



                # 保存checkpoint（参考EH-SurGS的方式）
                if self.save_dir and len(self.checkpoint_iterations) > 0:
                    if self.iteration_count_global in self.checkpoint_iterations:
                        checkpoint_path = os.path.join(self.save_dir, "checkpoints", f"chkpnt_{self.iteration_count_global}.pth")
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        checkpoint_data = (self.gaussians.capture(), self.iteration_count_global)
                        torch.save(checkpoint_data, checkpoint_path)
                        Log(f"已保存检查点 | 迭代: [cyan]{self.iteration_count_global}[/cyan] | 路径: {checkpoint_path}", tag="INFO")
                
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
        Log("开始颜色精化", tag="INFO")

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
        Log("地图精化完成", tag="INFO")

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
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("[Backend] 收到中断信号，正常退出")
        except Exception as e:
            print(f"[Backend] 异常退出: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_loop(self):
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

                    Log(f"重置系统 | 帧: {cur_frame_idx}", tag="Init")
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
                    # 接收回溯计算的 optical_flow: (prev_kf_idx, optical_flow, target_kf_idx) 或 None
                    prev_kf_flow_data = data[6] if len(data) > 6 else None
                    if prev_kf_flow_data is not None:
                        if isinstance(prev_kf_flow_data, tuple) and len(prev_kf_flow_data) == 3:
                            # 新格式: (prev_kf_idx, optical_flow, target_kf_idx)
                            prev_kf_idx, optical_flow, target_kf_idx = prev_kf_flow_data
                            self.optical_flow_dict[prev_kf_idx] = (optical_flow, target_kf_idx)
                        elif isinstance(prev_kf_flow_data, tuple) and len(prev_kf_flow_data) == 2:
                            # 兼容旧格式: (optical_flow, target_kf_idx)
                            self.optical_flow_dict[cur_frame_idx] = prev_kf_flow_data
                    
                    # 输出关键帧信息（使用颜色标记）
                    num_points = self.gaussians._xyz.shape[0]
                    from utils.logging_utils import format_percentage
                    # 计算动态点比例（如果deformation_table存在）
                    if hasattr(self.gaussians, '_deformation_table') and self.gaussians._deformation_table.numel() > 0:
                        if self.gaussians._deformation_table.shape[0] == num_points:
                            dynamic_points = self.gaussians._deformation_table.sum().item()
                            dynamic_ratio = dynamic_points / num_points if num_points > 0 else 0
                            dynamic_pct = format_percentage(dynamic_ratio, 1)
                            Log(f"关键帧: [cyan]{cur_frame_idx}[/cyan] | "
                                f"窗口: {current_window} | "
                                f"点数: [green]{num_points:,}[/green] | "
                                f"动态点: {dynamic_pct}", 
                                tag="Keyframe")
                        else:
                            Log(f"关键帧: [cyan]{cur_frame_idx}[/cyan] | "
                                f"窗口: {current_window} | "
                                f"点数: [green]{num_points:,}[/green]", 
                                tag="Keyframe")
                    else:
                        Log(f"关键帧: [cyan]{cur_frame_idx}[/cyan] | "
                            f"窗口: {current_window} | "
                            f"点数: [green]{num_points:,}[/green]", 
                            tag="Keyframe")

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
                            Log(f"执行初始BA优化 | 点数: [green]{num_points:,}[/green] | 迭代: [cyan]{iter_per_kf}[/cyan]", tag="BA")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        force_gt_pose = self.config.get("Training", {}).get("force_gt_pose", False)
                        if cam_idx < frames_to_optimize and not force_gt_pose:
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
                        # force_gt_pose 模式下强制重置位姿到GT
                        if force_gt_pose and hasattr(viewpoint, 'R_gt') and hasattr(viewpoint, 'T_gt'):
                            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                            viewpoint.cam_rot_delta.data.fill_(0)
                            viewpoint.cam_trans_delta.data.fill_(0)
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
