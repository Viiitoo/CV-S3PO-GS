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
        self.motion_flow_dict = {}  # 存储每个关键帧的motion_flow {frame_idx: motion_flow}

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
                
                # 计算flow loss（如果启用且存在motion_flow）
                flow_loss_value = None
                # #region agent log
                try:
                    log_dir = '/home/sjw/data0/lsx/S3PO_baseline/.cursor'
                    os.makedirs(log_dir, exist_ok=True)
                    with open('/home/sjw/data0/lsx/S3PO_baseline/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"slam_backend.py:232","message":"Flow loss check entry","data":{"Log_in_locals":"Log" in locals(),"Log_in_globals":"Log" in globals(),"flow_loss_weight":self.config["Training"].get("flow_loss_weight", 0),"iteration_count":self.iteration_count,"flow_warm_up":self.config["Training"].get("flow_warm_up", 1000),"kf_idx_in_dict":kf_idx in self.motion_flow_dict},"timestamp":int(time.time()*1000)}) + '\n')
                except Exception:
                    pass
                # #endregion
                
                # ====================================================================
                # 【问题1：关键帧索引不匹配】详细分析 - 用最简单的话说明
                # ====================================================================
                # 
                # 【背景知识：什么是关键帧？】
                # 想象你在看一部电影，电影有1000帧画面
                # 但不需要保存所有1000帧，只需要保存其中重要的几帧（比如第0、5、10、15...帧）
                # 这些重要的帧就叫"关键帧"
                # 关键帧之间的间隔叫做"关键帧间隔"（kf_interval），比如间隔是5
                # 
                # 【背景知识：什么是motion flow？】
                # motion flow（运动光流）就像是在两张照片之间画箭头
                # 箭头表示：第一张照片的某个点，在第二张照片中移动到了哪里
                # 比如：第5帧照片中的一只鸟，在第6帧照片中移动到了右边
                # 
                # 【问题发生的全过程】
                # 
                # 步骤1：前端（frontend）计算motion flow并存储
                # - 当程序处理第5帧时，计算从第5帧到第6帧的motion flow
                # - 然后把这个motion flow存到一个"字典"（类似电话本）里
                # - 字典的"名字"是5（第5帧），"内容"是从第5帧到第6帧的motion flow
                # - 就像：电话本里写着"张三的电话是123456"
                # - 这里就是："第5帧的motion flow是从第5帧到第6帧的"
                # 
                # 步骤2：前端把motion flow传给后端（backend）
                # - 当第5帧被选为关键帧时，前端从字典里找到"第5帧的motion flow"
                # - 然后把这个motion flow发送给后端
                # - 后端收到后，也存到自己的字典里，名字还是"5"
                # 
                # 步骤3：后端使用motion flow计算loss（错误发生的地方）
                # - 后端现在要计算loss，需要知道"从当前关键帧到下一个关键帧"的flow
                # - 当前关键帧是第5帧，下一个关键帧应该是第10帧（因为间隔是5）
                # - 但是！代码里写的是：下一个关键帧 = 当前关键帧 + 1 = 5 + 1 = 6
                # - 这是错误的！因为下一个关键帧应该是10，不是6！
                # 
                # 【具体例子说明错误】
                # 
                # 假设：
                # - 关键帧间隔 = 5（每5帧选一个关键帧）
                # - 所有关键帧是：第0帧、第5帧、第10帧、第15帧、第20帧...
                # 
                # 当程序处理第5帧（这是一个关键帧）时：
                # 
                # 正确的情况应该是：
                # - 当前关键帧：第5帧
                # - 下一个关键帧：第10帧
                # - 应该使用：从第5帧到第10帧的motion flow
                # 
                # 但代码实际做的是：
                # - 当前关键帧：第5帧
                # - 代码认为下一个关键帧：第5帧 + 1 = 第6帧（错误！）
                # - 代码使用：从第5帧到第6帧的motion flow（错误！）
                # 
                # 为什么这是错误的？
                # - 因为第6帧不是关键帧！它只是普通帧
                # - 我们应该比较的是"关键帧到关键帧"的flow
                # - 而不是"关键帧到普通帧"的flow
                # 
                # 【这个错误会导致什么后果？】
                # 1. 使用了错误的motion flow（第5帧→第6帧，而不是第5帧→第10帧）
                # 2. 计算了错误的GS flow（也是第5帧→第6帧）
                # 3. 把两个错误的flow放在一起比较，得到错误的loss
                # 4. 这个错误的loss会误导程序优化，导致结果变差
                # 5. 最终导致轨迹误差（RMSE）增大，而不是减小
                # 
                # 【用生活比喻】
                # 就像你要从北京到上海，但代码却计算了从北京到天津的距离
                # 然后用这个错误的距离来指导你走，你当然走不到上海！
                # 
                # ====================================================================
                
                # 【第280-282行】检查是否要计算flow loss
                # 这行代码的意思是：如果满足三个条件，就计算flow loss
                # 条件1：flow_loss_weight > 0（flow loss的权重大于0，说明启用了flow loss）
                # 条件2：iteration_count >= flow_warm_up（迭代次数够了，可以开始用flow loss了）
                # 条件3：kf_idx in self.motion_flow_dict（字典里有这个关键帧的motion flow）
                if (self.config["Training"].get("flow_loss_weight", 0) > 0 and
                    self.iteration_count >= self.config["Training"].get("flow_warm_up", 1000) and
                    kf_idx in self.motion_flow_dict):  # 【问题1-1】在字典里查找当前关键帧的motion flow
                    
                    # 【问题1-1详细解释】
                    # kf_idx 是当前关键帧的编号，比如5（表示第5帧）
                    # self.motion_flow_dict 是一个字典，就像电话本
                    # "kf_idx in self.motion_flow_dict" 意思是：检查电话本里有没有"5"这个名字
                    # 
                    # 问题在于：
                    # - 字典里确实有"5"这个名字，对应的内容是从第5帧到第6帧的motion flow
                    # - 但我们需要的是从第5帧到第10帧（下一个关键帧）的motion flow
                    # - 所以虽然找到了，但找到的内容是错的！
                    
                    # 【第286行】从字典里取出motion flow
                    # 【修复说明】motion_flow_dict存储的是从kf_idx到kf_idx+1的flow
                    # 但我们需要的是从kf_idx到next_kf_idx（下一个关键帧）的flow
                    # 目前先检查是否有对应的motion flow，如果没有就跳过
                    # 
                    # 注意：理想情况下，应该在前端计算并存储关键帧之间的flow
                    # 或者在这里重新计算从kf_idx到next_kf_idx的flow
                    # 但为了保持代码简单，我们先检查motion_flow_dict中是否有next_kf_idx-1对应的flow
                    # 如果没有，就跳过flow loss计算
                    
                    # 尝试获取从当前关键帧到下一个关键帧的motion flow
                    # 由于motion_flow_dict存储的是从frame_idx到frame_idx+1的flow
                    # 我们需要检查是否有从kf_idx到next_kf_idx的flow
                    # 如果没有，就跳过flow loss计算
                    motion_flow = None
                    
                    # 方法1：检查是否有直接存储的关键帧之间的flow（未来可以改进）
                    # 方法2：暂时跳过，因为motion_flow_dict存储的是相邻帧的flow
                    # 为了正确计算flow loss，需要在前端计算关键帧之间的flow
                    # 或者在这里重新计算（需要optical flow模型）
                    
                    # 【修复】找到下一个关键帧的正确索引
                    # 方法：从所有关键帧（viewpoints的key）中找到大于当前kf_idx的最小值
                    # 这样就能正确找到下一个关键帧，而不是简单地+1
                    next_kf_idx = None
                    for candidate_idx in sorted(self.viewpoints.keys()):
                        if candidate_idx > kf_idx:
                            next_kf_idx = candidate_idx
                            break
                    
                    # 如果找不到下一个关键帧，说明当前是最后一个关键帧，跳过flow loss计算
                    if next_kf_idx is None:
                        # 当前是关键帧列表中的最后一个，没有下一个关键帧，跳过flow loss计算
                        continue
                    
                    # 【修复】获取从当前关键帧到下一个关键帧的motion flow
                    # 前端在request_keyframe时已经计算并传递了关键帧之间的flow
                    # motion_flow_dict中存储的flow是从kf_idx到下一个关键帧的flow
                    # 如果前端传递了flow，就使用它；否则跳过flow loss计算
                    motion_flow = self.motion_flow_dict.get(kf_idx, None)
                    
                    # 如果motion_flow不存在，说明前端没有计算或传递flow，跳过
                    if motion_flow is None:
                        continue
                    
                    # 【修复完成】现在可以使用前端传递的关键帧之间的flow来计算flow loss了
                    # 检查下一个关键帧是否存在
                    if next_kf_idx is not None and next_kf_idx in self.viewpoints:
                            
                            # 【修复完成】检查下一个关键帧是否存在
                            # 现在next_kf_idx是正确的下一个关键帧索引（已修复）
                            try:
                                # 【第305-307行】导入需要的函数
                                # 这些函数是用来计算flow的
                                from utils.flow_utils import calculate_gs_flow
                                from utils.warp_utils import warping_gs_flow
                                from utils.slam_utils import flow_loss
                                
                                # 【修复完成】获取下一个关键帧的相机视角信息
                                viewpoint_next = self.viewpoints[next_kf_idx]
                                # 现在next_kf_idx是正确的下一个关键帧索引（已修复）
                                
                                # 【第316行】给下一帧添加时间信息
                                attach_time_to_viewpoint(viewpoint_next, frame_idx=next_kf_idx, num_frames=self.num_frames)
                                
                                # 【第317行】渲染下一帧的图像
                                # 这行代码的意思是：用当前的3D模型（gaussians）渲染下一帧应该长什么样
                                render_pkg_next = render(viewpoint_next, self.gaussians, self.pipeline_params, self.background)
                                
                                # 【第319行】检查渲染是否成功
                                if render_pkg_next is not None:
                                    # 如果渲染成功了，就继续往下执行
                                    
                                    # 【第320行】从渲染结果中取出深度图
                                    depth_next = render_pkg_next["depth"]
                                    
                                    # 【第323-325行】确保深度图的格式正确
                                    # 深度图需要是3维的（1, 高度, 宽度），如果是2维的（高度, 宽度），就加一维
                                    depth_for_flow = depth
                                    if depth_for_flow.dim() == 2:
                                        depth_for_flow = depth_for_flow.unsqueeze(0)
                                    
                                    # 【修复完成】计算GS Flow（这是3D模型预测的光流）
                                    # 现在计算的是从当前关键帧到下一个关键帧的GS flow（已修复）
                                    gs_flow = calculate_gs_flow(depth1=depth_for_flow, cam1=viewpoint, cam2=viewpoint_next)
                                    
                                    # 【第337行】对齐GS Flow到Motion Flow的坐标系
                                    gs_flow_aligned = warping_gs_flow(depth_for_flow, gs_flow, viewpoint, viewpoint_next)
                                    
                                    # 【问题1-7详细解释】
                                    # 这行代码试图把gs_flow转换到motion_flow的坐标系
                                    # 
                                    # 问题在于：
                                    # - 即使gs_flow和motion_flow都是第5帧→第6帧的flow
                                    # - 这个对齐操作也可能引入额外的误差
                                    # - 因为坐标系转换本身就可能出错
                                    # 
                                    # 用生活比喻：
                                    # - 就像你要把一张地图从一种比例尺转换成另一种比例尺
                                    # - 转换过程中可能会引入误差
                                    
                                    # 【第343行】获取图像的高度和宽度
                                    H, W = image.shape[-2:]
                                    
                                    # 【第344行】计算flow loss（这是最关键的步骤）
                                    flow_loss_value = flow_loss(gs_flow_aligned, motion_flow.detach(), H, W)
                                    
                                    # 【问题1-8详细解释 - 这是最严重的问题！】
                                    # 
                                    # 这行代码比较两个flow：
                                    # 1. gs_flow_aligned：3D模型预测的光流（第5帧→第6帧）
                                    # 2. motion_flow：真实的光流（也是第5帧→第6帧）
                                    # 
                                    # 问题在于：
                                    # - 虽然两个flow都是第5帧→第6帧，看起来对应
                                    # - 但我们需要的是第5帧→第10帧的flow！
                                    # - 所以这个比较没有意义
                                    # 
                                    # 更严重的情况：
                                    # - 如果关键帧间隔是5，关键帧是0, 5, 10, 15...
                                    # - 当处理第5帧时，next_kf_idx = 5 + 1 = 6
                                    # - 但第6帧可能不存在（如果总共只有5帧）
                                    # - 或者第6帧存在，但它不是关键帧
                                    # - 无论哪种情况，这个loss都是错的
                                    # 
                                    # 用生活比喻：
                                    # - 就像你要从北京到上海
                                    # - 但代码却比较了"从北京到天津的路线"和"从北京到天津的真实路线"
                                    # - 虽然这两个路线都是对的，但这不是你要的！
                                    # - 你应该比较"从北京到上海的路线"和"从北京到上海的真实路线"
                                    
                                    # 【第347行】把flow loss加入到总loss中
                                    loss_mapping += self.config["Training"]["flow_loss_weight"] * flow_loss_value
                                    
                                    # 【问题1-9详细解释 - 这是导致RMSE增大的直接原因！】
                                    # 
                                    # 这行代码的意思是：
                                    # 总loss = 总loss + flow_loss_weight × flow_loss_value
                                    # 
                                    # 问题在于：
                                    # - flow_loss_value是错的（比较的是错误的flow）
                                    # - 这个错误的loss被加权后加入到总loss中
                                    # - 程序会根据总loss来优化模型
                                    # - 由于loss是错的，优化方向也是错的
                                    # 
                                    # 这会导致什么后果？
                                    # 1. 位姿优化方向错误：程序会往错误的方向调整相机位置
                                    # 2. 高斯点位置优化错误：程序会往错误的方向调整3D点的位置
                                    # 3. 最终导致轨迹误差（RMSE）增大，而不是减小
                                    # 
                                    # 用生活比喻：
                                    # - 就像你要去上海，但导航却告诉你"往北走"（因为它在计算去天津的路）
                                    # - 你往北走，当然离上海越来越远！
                                    # - 这就是为什么启用flow loss后，RMSE反而增大了
                            except Exception as e:
                                # 如果flow loss计算失败，记录但不中断训练
                                # #region agent log
                                try:
                                    log_dir = '/home/sjw/data0/lsx/S3PO_baseline/.cursor'
                                    os.makedirs(log_dir, exist_ok=True)
                                    with open('/home/sjw/data0/lsx/S3PO_baseline/.cursor/debug.log', 'a') as f:
                                        import json
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"slam_backend.py:270","message":"Flow loss exception caught","data":{"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
                                except Exception:
                                    pass
                                # #endregion
                                import traceback
                                # Log已经在文件顶部导入，不需要再次导入
                                Log(f"Flow loss计算失败: {e}", tag="Flow")
                                pass     
                
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
                                # 在prune后更新deformation table，确保只有需要形变的点被标记
                                # 这样可以优化性能并改善建模效果
                                if hasattr(self.gaussians, 'update_deformation_table'):
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
                    if hasattr(self.gaussians, 'update_deformation_table'):
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
                        
                        # 移除详细的形变参数数值输出，这些对用户没有直观价值
                        # 只保留关键信息，如动态点比例等（在update_deformation_table中输出）
                        pass
                
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
                    # 接收motion_flow（如果存在）
                    motion_flow = data[6] if len(data) > 6 else None
                    if motion_flow is not None:
                        self.motion_flow_dict[cur_frame_idx] = motion_flow
                    
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
