import json
import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
import cv2
import evo
import numpy as np
import torch
from PIL import Image
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
matplotlib.use('Agg')  
print("Current backend:", matplotlib.get_backend())
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.time_utils import attach_time_to_viewpoint
from utils.camera_utils import Camera
from copy import deepcopy

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from utils.logging_utils import Log

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    
    # 尝试对齐轨迹，如果失败则使用简单的位置对齐
    try:
        traj_est_aligned = trajectory.align_trajectory(
            traj_est, traj_ref, correct_scale=monocular
        )
    except Exception as e:
        Log(f"Warning: Trajectory alignment failed ({str(e)}), using simple translation alignment", tag="Eval")
        # 使用简单的位置对齐：计算质心偏移
        positions_ref = traj_ref.positions_xyz
        positions_est = traj_est.positions_xyz
        
        if len(positions_ref) < 3:
            Log(f"Warning: Too few poses ({len(positions_ref)}) for alignment, skipping ATE calculation", tag="Eval")
            # 确保目录存在
            mkdir_p(plot_dir)
            # 返回一个默认值
            ape_stat = 0.0
            ape_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "rmse": 0.0, "sse": 0.0, "std": 0.0}
            # 仍然保存统计信息
            with open(
                os.path.join(plot_dir, "stats_{}.json".format(str(label))),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(ape_stats, f, indent=4)
            return ape_stat
        
        # 计算质心
        centroid_ref = np.mean(positions_ref, axis=0)
        centroid_est = np.mean(positions_est, axis=0)
        translation = centroid_ref - centroid_est
        
        # 应用简单的平移对齐
        aligned_poses = []
        for pose in poses_est:
            aligned_pose = pose.copy()
            aligned_pose[:3, 3] += translation
            aligned_poses.append(aligned_pose)
        
        traj_est_aligned = PosePath3D(poses_se3=aligned_poses)
    
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))))
    plt.close(fig) 

    return ape_stat

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, BA=False):
    # 检查关键帧数量
    if len(kf_ids) < 2:
        Log(f"Warning: Too few keyframes ({len(kf_ids)}) for ATE evaluation, skipping", tag="Eval")
        return 0.0
    
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    if BA:
        label_evo = "after BA"
    elif final:
        label_evo = "final"
    else:
        label_evo = "{:04}".format(iterations)
    #label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

def project_3d_to_2d(points_3d, viewpoint):
    """
    将3D点投影到2D图像坐标
    
    Args:
        points_3d: [N, 3] numpy array, 世界坐标系下的3D点
        viewpoint: Camera对象，包含相机参数
    
    Returns:
        points_2d: [N, 2] numpy array, 图像坐标 (u, v)
        valid_mask: [N] bool array, 标记哪些点在图像范围内
    """
    if points_3d is None or len(points_3d) == 0:
        return np.array([]), np.array([], dtype=bool)
    
    # 转换为torch tensor
    points_3d_torch = torch.from_numpy(points_3d).float().cuda()
    
    # 世界坐标转视图坐标
    W2C = getWorld2View2(viewpoint.R, viewpoint.T)
    points_view = (W2C[:3, :3] @ points_3d_torch.T + W2C[:3, 3:4]).T
    
    # 检查深度（z > 0）
    valid_depth = points_view[:, 2] > 0.1
    
    # 投影到图像平面
    fx, fy = viewpoint.fx, viewpoint.fy
    cx, cy = viewpoint.cx, viewpoint.cy
    
    u = (points_view[:, 0] / points_view[:, 2]) * fx + cx
    v = (points_view[:, 1] / points_view[:, 2]) * fy + cy
    
    points_2d = torch.stack([u, v], dim=1).cpu().numpy()
    
    # 检查是否在图像范围内
    valid_range = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < viewpoint.image_width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < viewpoint.image_height)
    )
    
    valid_mask = valid_depth.cpu().numpy() & valid_range
    
    return points_2d, valid_mask

def draw_edges_on_image(rgb_image, edge_points_3d, viewpoint, 
                        edge_color=(0, 255, 0), edge_thickness=2, max_points=10000,
                        draw_lines=True, line_color=(255, 0, 0), line_thickness=1):
    """
    使用STAR-Edge 3D边缘点投影在RGB图像上绘制轮廓
    
    将STAR-Edge提取的3D边缘点投影到当前相机位姿的2D图像上，
    以绿色圆点的方式叠加在渲染的RGB图像上。
    
    Args:
        rgb_image: [H, W, 3] numpy array, RGB图像 (0-255)
        edge_points_3d: [N, 3] numpy array, STAR-Edge提取的边缘点3D坐标
        viewpoint: Camera对象
        edge_color: tuple, 轮廓颜色 (R, G, B)，默认绿色
        edge_thickness: int, 轮廓点大小
        max_points: int, 最大显示点数
        draw_lines: bool, 是否连接相邻边缘点形成轮廓线
        line_color: tuple, 轮廓线颜色 (R, G, B)，默认红色
        line_thickness: int, 轮廓线粗细
    
    Returns:
        rgb_with_edges: [H, W, 3] numpy array, 带轮廓的RGB图像
    """
    if edge_points_3d is None or len(edge_points_3d) == 0:
        return rgb_image.copy()
    
    # 投影3D点到2D
    points_2d, valid_mask = project_3d_to_2d(edge_points_3d, viewpoint)
    
    if not valid_mask.any():
        return rgb_image.copy()
    
    # 创建图像副本
    rgb_with_edges = rgb_image.copy()
    
    # 只绘制有效的点
    valid_points_2d = points_2d[valid_mask].astype(np.int32)
    
    # 如果点数太多，随机采样
    if len(valid_points_2d) > max_points:
        indices = np.random.choice(len(valid_points_2d), max_points, replace=False)
        valid_points_2d = valid_points_2d[indices]
    
    # 绘制点（使用更鲜艳的颜色和稍大的点）
    for point in valid_points_2d:
        u, v = int(point[0]), int(point[1])
        if 0 <= u < rgb_with_edges.shape[1] and 0 <= v < rgb_with_edges.shape[0]:
            # 绘制带边框的点，使其更明显
            cv2.circle(rgb_with_edges, (u, v), edge_thickness + 1, (0, 0, 0), -1)  # 黑色边框
            cv2.circle(rgb_with_edges, (u, v), edge_thickness, edge_color, -1)  # 绿色填充
    
    return rgb_with_edges

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    datatype,
    kf_indices,
    iteration="final",
):
    interval = 1
    img_pred, img_gt, saved_frame_idx, img_residual = [], [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # Define directory to save images (uncomment if needed)
    viz_dir = os.path.join(save_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    render_dir = os.path.join(save_dir, "render_rgb")
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    depth_dir = os.path.join(save_dir, "render_depth")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    depth_dir1 = os.path.join(save_dir, "render_depth_npy")
    if not os.path.exists(depth_dir1):
        os.makedirs(depth_dir1)
    
    # 添加轮廓RGB输出目录
    render_rgb_with_edges_dir = os.path.join(save_dir, "render_rgb_with_edges")
    if not os.path.exists(render_rgb_with_edges_dir):
        os.makedirs(render_rgb_with_edges_dir)
    
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _, _ = dataset[idx]

        attach_time_to_viewpoint(frame, frame_idx=idx, num_frames=len(dataset))

        render_pkg = render(frame, gaussians, pipe, background)
        # Check if rendering failed (e.g., no points initialized yet)
        if render_pkg is None:
            # Skip this frame if rendering failed
            saved_frame_idx.pop()  # Remove the idx we just added
            continue
        
        rendering = render_pkg["render"]
        
        # Save depth map
        depth = render_pkg["depth"]
        depth = depth.squeeze()
        depth_np = depth.detach().cpu().numpy()
        depth_max = depth_np.max()
        depth_min = depth_np.min()
        depth_nor = (depth_np-depth_min)/(depth_max-depth_min)
        depth_nor = (depth_nor * 255 ).astype(np.uint8)
        
        img_depth= Image.fromarray(depth_nor)
        #save_path = os.path.join(depth_dir, f"{idx}_pred.png")
        #img_depth.save(save_path, dpi=(300, 300))
        
        # Save depth map as .npy file (unnormalized depth matrix)
        #save_path_npy = os.path.join(depth_dir1, f"{idx}_pred.npy")
        #np.save(save_path_npy, depth_np)
        
        image = torch.clamp(rendering, 0.0, 1.0)
        # Calculate metrics
        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        residual = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
        residual = np.clip(residual, 0, 255).astype(np.uint8)  
        img_pred.append(pred)
        img_gt.append(gt)
        img_residual.append(residual)
        # Render comparison image
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(pred)
        plt.title('Rendered rgb')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(img_depth, cmap='gray')  
        plt.title('Depth Map')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(residual)
        plt.title('Residual')
        plt.axis('off')
        
        plt.figtext(0.5, 0.01, f"PSNR: {psnr_score.item():.2f}", ha="center", fontsize=14)

        save_path = os.path.join(viz_dir, f"{idx}.png")
        plt.tight_layout() 
        plt.savefig(save_path, bbox_inches='tight')
        plt.close() 

        # Save rendered image
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        pred_image = Image.fromarray(pred)
        save_path = os.path.join(render_dir, f"{idx}_pred.png")
        #pred_image.save(save_path, dpi=(300, 300))
        
        # ========== 保存带轮廓信息的RGB图像 ==========
        # 使用STAR-Edge 3D边缘点投影方法
        # 将3D边缘点投影到当前相机位姿的2D图像上
        try:
            # 获取边缘点
            edge_points_3d = None
            if hasattr(gaussians, '_edge_points') and gaussians._edge_points is not None:
                edge_points_3d = gaussians._edge_points
            
            if edge_points_3d is not None and len(edge_points_3d) > 0:
                rgb_with_edges = draw_edges_on_image(
                    pred,  # 渲染的RGB图像
                    edge_points_3d,  # STAR-Edge提取的3D边缘点
                    frame,  # 当前视角
                    edge_color=(0, 255, 0),  # 绿色轮廓 (RGB格式)
                    edge_thickness=2,  # 边缘点大小
                    max_points=10000  # 最大显示点数
                )
            else:
                # 如果没有边缘点，使用原始图像
                rgb_with_edges = pred.copy()
            
            # 保存带轮廓的RGB图像
            rgb_with_edges_image = Image.fromarray(rgb_with_edges)
            save_path = os.path.join(render_rgb_with_edges_dir, f"{idx}_rgb_with_edges.png")
            rgb_with_edges_image.save(save_path)
        except Exception as e:
            Log(f"Warning: Failed to draw edges on image {idx}: {e}", tag="Eval")
            # 如果绘制失败，保存原始RGB图像
            pred_image = Image.fromarray(pred)
            save_path = os.path.join(render_rgb_with_edges_dir, f"{idx}_rgb_with_edges.png")
            pred_image.save(save_path)

    output = dict()
    # Check if we have any successful renderings
    if len(psnr_array) == 0:
        Log("Warning: No frames were successfully rendered during evaluation", tag="Eval")
        output["mean_psnr"] = 0.0
        output["mean_ssim"] = 0.0
        output["mean_lpips"] = 0.0
    else:
        output["mean_psnr"] = float(np.mean(psnr_array))
        output["mean_ssim"] = float(np.mean(ssim_array))
        output["mean_lpips"] = float(np.mean(lpips_array))

        Log(
            f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
            tag="Eval",
        )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    # 如果模型支持时间形变，同时保存形变参数
    if hasattr(gaussians, 'save_deformation_params') and gaussians._w_pos.numel() > 0:
        deformation_path = os.path.join(point_cloud_path, "deformation_params.npz")
        gaussians.save_deformation_params(deformation_path)
        Log(f"Saved deformation parameters to {deformation_path}", tag="Eval")