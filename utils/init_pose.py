from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import os
import numpy as np
import time
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import PIL.Image
from PIL.ImageOps import exif_transpose
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from gaussian_splatting.gaussian_renderer import render_with_custom_resolution
from utils.edge_extraction import EdgeExtractor

import torchvision.transforms as tvf

# 尝试导入Open3D用于点云保存
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("[轮廓可视化] 警告: Open3D不可用，无法保存点云PLY文件")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _to_uint8_rgb(img):
    """Convert torch(1x3xHxW / 3xHxW) or numpy(HWC/CHW) to uint8 HWC RGB."""
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        if x.dtype.is_floating_point:
            x = (x * 0.5 + 0.5).clamp(0, 1)
            x = (x * 255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).contiguous().numpy()
        return x
    arr = img
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr

def _save_rgb_edge_pnp_viz(viz_dir,
                           tag,
                           rgb1_u8, rgb2_u8,
                           edge1, edge2,
                           matches1, matches2,
                           weights=None,
                           inliers=None,
                           max_matches=300):
    os.makedirs(viz_dir, exist_ok=True)

    # edge masks
    e1 = (np.clip(edge1, 0, 1) * 255).astype(np.uint8)
    e2 = (np.clip(edge2, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge1.png"), e1)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge2.png"), e2)

    # overlays
    ov1 = rgb1_u8.copy()
    ov2 = rgb2_u8.copy()
    ov1[..., 2] = np.maximum(ov1[..., 2], e1)  # red channel
    ov2[..., 2] = np.maximum(ov2[..., 2], e2)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_overlay1.png"), cv2.cvtColor(ov1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_overlay2.png"), cv2.cvtColor(ov2, cv2.COLOR_RGB2BGR))

    # weighted matches visualization
    H1, W1 = rgb1_u8.shape[:2]
    H2, W2 = rgb2_u8.shape[:2]
    canvas_h = max(H1, H2)
    canvas_w = W1 + W2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:H1, :W1] = rgb1_u8
    canvas[:H2, W1:W1+W2] = rgb2_u8

    N = len(matches1)
    if N == 0:
        cv2.imwrite(os.path.join(viz_dir, f"{tag}_matches.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    # subsample matches for drawing
    if N > max_matches:
        idx = np.random.choice(N, size=max_matches, replace=False)
    else:
        idx = np.arange(N)

    inlier_set = set(inliers.reshape(-1).tolist()) if inliers is not None else None

    if weights is None:
        w = np.ones(N, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        w = (w - w.min()) / (w.max() - w.min() + 1e-6)  # 0-1 for color mapping

    for i in idx:
        p1 = matches1[i]
        p2 = matches2[i].copy()
        p2[0] += W1
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        x1 = np.clip(x1, 0, W1 - 1)
        y1 = np.clip(y1, 0, H1 - 1)
        x2 = np.clip(x2, W1, W1 + W2 - 1)
        y2 = np.clip(y2, 0, H2 - 1)

        # color by weight: low=blue, high=yellow/red
        ww = float(w[i])
        color = (int(255 * (1 - ww)), int(255 * ww), int(255 * ww))  # RGB-ish
        thickness = 1 + int(2 * ww)

        if inlier_set is not None and i in inlier_set:
            # highlight inliers: green
            color = (0, 255, 0)
            thickness = max(thickness, 2)

        cv2.line(canvas, (x1, y1), (x2, y2), color=color, thickness=thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(os.path.join(viz_dir, f"{tag}_matches.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_3d_edge_projection_viz(viz_dir,
                                 tag,
                                 rgb_u8,
                                 pts3d,
                                 edge_mask_2d,
                                 K,
                                 dist_coeffs=None,
                                 max_points=10000,
                                 edge_scores_2d=None):
    """
    保存3D轮廓点在图像上的投影可视化（改进版，更清晰显示轮廓点）
    
    Args:
        viz_dir: 保存目录
        tag: 文件名标签
        rgb_u8: RGB图像 (H, W, 3) uint8
        pts3d: 3D点云 (H, W, 3)
        edge_mask_2d: 轮廓mask (H, W) bool
        K: 相机内参 (3, 3)
        dist_coeffs: 畸变系数，可选
        max_points: 最大显示点数
        edge_scores_2d: 轮廓分数 (H, W)，可选，用于热力图显示
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    H, W = rgb_u8.shape[:2]
    
    # 1. 保存轮廓mask（二值图）
    edge_mask_vis = (edge_mask_2d.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge_mask_3d.png"), edge_mask_vis)
    
    # 2. 创建叠加图像：在RGB图像上标记轮廓点（使用更明显的标记）
    overlay = rgb_u8.copy()
    
    # 获取轮廓点的2D坐标
    edge_y, edge_x = np.where(edge_mask_2d)
    
    # 如果轮廓点太多，随机采样显示（但保留更多点以便看到完整轮廓）
    display_points = min(len(edge_y), max_points)
    if len(edge_y) > max_points:
        indices = np.random.choice(len(edge_y), max_points, replace=False)
        edge_y_display = edge_y[indices]
        edge_x_display = edge_x[indices]
    else:
        edge_y_display = edge_y
        edge_x_display = edge_x
    
    # 在图像上标记轮廓点（使用更明显的红色，更大的点）
    for y, x in zip(edge_y_display, edge_x_display):
        # 绘制红色圆点标记轮廓点（半径3，更明显）
        cv2.circle(overlay, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
        # 添加白色边框以便在深色区域也能看到
        cv2.circle(overlay, (int(x), int(y)), radius=3, color=(255, 255, 255), thickness=1)
    
    # 保存叠加图像
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge_overlay_3d.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 3. 如果有轮廓分数，创建热力图可视化
    if edge_scores_2d is not None and edge_scores_2d.shape == edge_mask_2d.shape:
        # 创建热力图：轮廓分数越高，颜色越红
        heatmap = np.zeros((H, W, 3), dtype=np.uint8)
        # 归一化分数到0-255
        scores_norm = np.clip((edge_scores_2d / (edge_scores_2d.max() + 1e-6)) * 255, 0, 255).astype(np.uint8)
        # 红色通道表示轮廓分数
        heatmap[:, :, 0] = scores_norm  # 红色
        heatmap[:, :, 1] = 0
        heatmap[:, :, 2] = 0
        # 叠加到原图（半透明）
        heatmap_overlay = cv2.addWeighted(rgb_u8, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge_heatmap_3d.png"), cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    
    # 4. 投影3D轮廓点到图像空间（使用相机参数）
    valid_mask = np.isfinite(pts3d).all(axis=2) & (pts3d[:, :, 2] > 0)
    edge_3d_points = pts3d[edge_mask_2d & valid_mask]  # 只取有效的轮廓3D点
    
    if len(edge_3d_points) > 0:
        # 投影3D点到2D
        if dist_coeffs is not None:
            projected_points, _ = cv2.projectPoints(
                edge_3d_points.reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3), K, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
        else:
            # 无畸变情况下的简单投影
            projected_points = edge_3d_points[:, :2] / edge_3d_points[:, 2:3]
            projected_points = projected_points @ K[:2, :2].T + K[:2, 2]
        
        # 创建投影可视化
        proj_overlay = rgb_u8.copy()
        
        # 限制显示点数
        display_proj_points = min(len(projected_points), max_points)
        if len(projected_points) > max_points:
            indices = np.random.choice(len(projected_points), max_points, replace=False)
            projected_points_display = projected_points[indices]
        else:
            projected_points_display = projected_points
        
        # 绘制投影点（使用更明显的绿色，更大的点）
        for pt in projected_points_display:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < W and 0 <= y < H:
                # 绘制绿色圆点（半径2）
                cv2.circle(proj_overlay, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                # 添加白色边框
                cv2.circle(proj_overlay, (x, y), radius=2, color=(255, 255, 255), thickness=1)
        
        cv2.imwrite(os.path.join(viz_dir, f"{tag}_edge_projection_3d.png"), cv2.cvtColor(proj_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"[轮廓可视化] 已保存3D轮廓投影图像到 {viz_dir}/{tag}_*.png (轮廓点数: {len(edge_y)})")


def _save_colored_pointcloud_ply(points_3d: np.ndarray,
                                  edge_mask: np.ndarray,
                                  viz_dir: str,
                                  tag: str,
                                  rgb_u8: np.ndarray = None,
                                  pts3d_2d: np.ndarray = None,
                                  K: np.ndarray = None,
                                  dist_coeffs: np.ndarray = None,
                                  max_points: int = 500000,
                                  edge_color: tuple = (255, 0, 0),
                                  valid_indices_2d: tuple = None):
    """
    保存带颜色标记的点云PLY文件（保留原始颜色，轮廓点用特殊颜色标记）
    
    Args:
        points_3d: 3D点云 (N, 3)
        edge_mask: 轮廓mask (N,) bool
        viz_dir: 保存目录
        tag: 文件名标签
        rgb_u8: RGB图像 (H, W, 3) uint8，用于获取原始颜色
        pts3d_2d: 3D点云对应的2D坐标 (H, W, 3)，用于映射到图像
        K: 相机内参 (3, 3)，用于投影
        dist_coeffs: 畸变系数，可选
        max_points: 最大保存点数（如果点太多，随机采样）
        edge_color: 轮廓点标记颜色 (R, G, B)，默认红色(255, 0, 0)
        valid_indices_2d: 2D索引元组 (y_indices, x_indices)，如果提供则直接使用索引获取颜色（最快）
    """
    if not O3D_AVAILABLE:
        print("[轮廓可视化] 警告: Open3D不可用，跳过点云PLY保存")
        return
    
    os.makedirs(viz_dir, exist_ok=True)
    
    N = len(points_3d)
    if N == 0:
        print("[轮廓可视化] 警告: 点云为空，跳过PLY保存")
        return
    
    # 如果点太多，随机采样
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        points_3d = points_3d[indices]
        edge_mask = edge_mask[indices]
        print(f"[轮廓可视化] 点云点数过多({N})，随机采样到{max_points}个点")
    
    # 创建颜色数组
    colors = np.zeros((len(points_3d), 3), dtype=np.uint8)
    
    # 如果有RGB图像和投影信息，从图像中获取原始颜色
    if rgb_u8 is not None and K is not None:
        H, W = rgb_u8.shape[:2]
        
        # 如果提供了valid_indices_2d，直接使用索引获取颜色（最快最准确）
        if valid_indices_2d is not None and len(valid_indices_2d) == 2:
            y_indices, x_indices = valid_indices_2d
            if len(y_indices) == len(points_3d):
                # 直接通过索引从RGB图像获取颜色（向量化操作，最快）
                for i, (y, x) in enumerate(zip(y_indices, x_indices)):
                    if 0 <= x < W and 0 <= y < H:
                        colors[i] = rgb_u8[y, x]
                    else:
                        colors[i] = [128, 128, 128]
            else:
                # 索引数量不匹配，使用投影方法
                if dist_coeffs is not None:
                    projected_points, _ = cv2.projectPoints(
                        points_3d.reshape(-1, 1, 3),
                        np.zeros(3), np.zeros(3), K, dist_coeffs
                    )
                    projected_points = projected_points.reshape(-1, 2)
                else:
                    projected_points = points_3d[:, :2] / points_3d[:, 2:3]
                    projected_points = projected_points @ K[:2, :2].T + K[:2, 2]
                
                for i, (x, y) in enumerate(projected_points):
                    x_int, y_int = int(round(x)), int(round(y))
                    if 0 <= x_int < W and 0 <= y_int < H:
                        colors[i] = rgb_u8[y_int, x_int]
                    else:
                        colors[i] = [128, 128, 128]
        # 如果有pts3d_2d（H, W, 3格式），尝试通过最近邻查找匹配点
        elif pts3d_2d is not None and pts3d_2d.shape[:2] == (H, W):
            # 使用投影方法（更可靠）
            if dist_coeffs is not None:
                projected_points, _ = cv2.projectPoints(
                    points_3d.reshape(-1, 1, 3),
                    np.zeros(3), np.zeros(3), K, dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
            else:
                projected_points = points_3d[:, :2] / points_3d[:, 2:3]
                projected_points = projected_points @ K[:2, :2].T + K[:2, 2]
            
            for i, (x, y) in enumerate(projected_points):
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= x_int < W and 0 <= y_int < H:
                    colors[i] = rgb_u8[y_int, x_int]
                else:
                    colors[i] = [128, 128, 128]
        else:
            # 使用投影方法
            if dist_coeffs is not None:
                projected_points, _ = cv2.projectPoints(
                    points_3d.reshape(-1, 1, 3),
                    np.zeros(3), np.zeros(3), K, dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
            else:
                # 无畸变情况下的简单投影
                projected_points = points_3d[:, :2] / points_3d[:, 2:3]
                projected_points = projected_points @ K[:2, :2].T + K[:2, 2]
            
            # 从RGB图像中采样颜色
            for i, (x, y) in enumerate(projected_points):
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= x_int < W and 0 <= y_int < H:
                    colors[i] = rgb_u8[y_int, x_int]
                else:
                    colors[i] = [128, 128, 128]
        
        # 轮廓点用特殊颜色标记（覆盖原始颜色）
        colors[edge_mask] = edge_color
        print(f"[轮廓可视化] 已从RGB图像获取原始颜色，轮廓点用特殊颜色({edge_color})标记")
    else:
        # 没有RGB图像信息，使用默认策略：轮廓点红色，非轮廓点灰色
        colors[edge_mask] = edge_color  # 红色 - 轮廓点
        colors[~edge_mask] = [128, 128, 128]  # 灰色 - 非轮廓点
        print(f"[轮廓可视化] 未提供RGB图像信息，使用默认颜色方案")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    
    # 保存PLY文件
    ply_path = os.path.join(viz_dir, f"{tag}_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    
    num_edge_points = edge_mask.sum()
    num_total_points = len(points_3d)
    print(f"[轮廓可视化] ✓ 已保存带颜色标记的点云PLY: {ply_path}")
    print(f"  - 总点数: {num_total_points}, 轮廓点(特殊颜色): {num_edge_points} ({num_edge_points/num_total_points*100:.1f}%), 非轮廓点(原始颜色): {num_total_points-num_edge_points}")


def _weighted_ransac_pnp(objectPoints: np.ndarray,
                         imagePoints: np.ndarray,
                         K: np.ndarray,
                         dist_coeffs: np.ndarray,
                         weights: np.ndarray,
                         iterationsCount: int = 100,
                         reprojectionError: float = 5.0,
                         min_sample: int = 6,
                         seed: int = 0):
    """
    加权RANSAC-PnP（偏置采样 + 加权inlier评分）。

    OpenCV 的 solvePnPRansac 不支持 per-point 权重，这里用：
    - 采样：按 weights 概率抽样最小集
    - 评分：用 inliers 的 weight 之和作为 score（同时也能返回 inlier mask）
    """
    N = len(objectPoints)
    if N < min_sample:
        return False, None, None, None

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 1e-6, None)
    p = w / w.sum()

    rng = np.random.default_rng(seed)

    best_score = -1.0
    best_rvec, best_tvec = None, None
    best_inliers = None

    # 用 EPNP 做 minimal set 更稳（SQPNP 也可，但对极小样本偶尔不稳定）
    for _ in range(iterationsCount):
        try:
            idx = rng.choice(N, size=min_sample, replace=False, p=p)
        except ValueError:
            idx = rng.choice(N, size=min_sample, replace=False)

        ok, rvec, tvec = cv2.solvePnP(
            objectPoints[idx], imagePoints[idx], K, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            continue

        proj, _ = cv2.projectPoints(objectPoints, rvec, tvec, K, dist_coeffs)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - imagePoints, axis=1)
        inlier_mask = err < reprojectionError
        if inlier_mask.sum() < min_sample:
            continue

        score = float(w[inlier_mask].sum())
        if score > best_score:
            best_score = score
            best_rvec, best_tvec = rvec, tvec
            best_inliers = np.flatnonzero(inlier_mask).astype(np.int32)

    if best_inliers is None:
        return False, None, None, None

    # 用所有inliers做一次迭代PnP精修
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints[best_inliers], imagePoints[best_inliers],
        K, dist_coeffs,
        rvec=best_rvec, tvec=best_tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        # 退回最优解（通常也能用）
        rvec, tvec = best_rvec, best_tvec
        ok = True
    return ok, rvec, tvec, best_inliers

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def torch_images_to_dust3r_format(tensor_images, size, square_ok=False, verbose=False):
    """
    Convert a list of torch tensor images to the format required by the DUSt3R/MASt3R model.
    
    Args:
    - tensor_images (list of torch.Tensor): List of RGB images in torch tensor format.
    - size (int): Target size for the images.
    - square_ok (bool): Whether square images are acceptable.
    - verbose (bool): Whether to print verbose messages.

    Returns:
    - list of dict: Converted images in the required format.
    """
    imgs = []
    for idx, image in enumerate(tensor_images):
        image = image.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HWC format and scale to [0, 255]
        image = image.astype(np.uint8)
        
        img = PIL.Image.fromarray(image, 'RGB')
        img = exif_transpose(img).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx)))

    assert imgs, 'no images found'
    return imgs

def depth_to_3d(depth_map, K, dist_coeffs):
    """
    Convert a depth map to 3D points, taking camera distortion into account.

    Args:
        - depth_map: Depth image
        - K: Camera intrinsic matrix
        - dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        - points_3d: 3D point cloud
    """

    if len(depth_map.shape) == 3:
        # If shape is (C, H, W), remove the channel dimension
        depth_map = depth_map.squeeze(0)  
    
    h, w = depth_map.shape

    # Generate pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 2).astype(np.float32)  

    # Undistort pixel coordinates
    undistorted_pixels = cv2.undistortPoints(pixels, K, dist_coeffs, P=K).reshape(h, w, 2)
    u_undistorted = undistorted_pixels[..., 0]
    v_undistorted = undistorted_pixels[..., 1]
    
    # Project undistorted pixels to 3D space using depth and camera intrinsics
    Z = depth_map
    X = (u_undistorted - K[0, 2]) * Z / K[0, 0]
    Y = (v_undistorted - K[1, 2]) * Z / K[1, 1]
    points_3d = np.stack((X, Y, Z), axis=-1)

    return points_3d

def depth_to_3d1(depth_map, K):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d

# Estimate relative pose and return rendered depth
def get_pose(img1, img2, model, dist_coeffs, viewpoint, gaussians, pipeline_params, background,
             rgb_edge_pnp=None):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    # Extract features from images and perform point matching
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()    
    
    # find 2D-2D matches between the two images
    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    
    H1 = view1['img'].shape[2]     
    W1 = view1['img'].shape[3]
    scale_H = H1 / viewpoint.image_height
    scale_W = W1 / viewpoint.image_width
    render_pkg = render_with_custom_resolution(viewpoint, gaussians, pipeline_params, background, target_width=W1, target_height=H1)
    # Check if rendering failed (e.g., no points initialized yet)
    if render_pkg is None:
        # Use MASt3R depth as fallback when no gaussians are available
        mono_depth = get_depth(img2, img2, model, return_conf=False)
        # Convert to tensor with same format as rendered depth: [1, H, W]
        render_depth = torch.from_numpy(mono_depth).float().to(device)
        if len(render_depth.shape) == 2:
            render_depth = render_depth.unsqueeze(0)  # Add batch dimension if needed
    else:
        render_depth = render_pkg["depth"]

    # Adjust camera intrinsic matrix
    fx_new = viewpoint.fx * scale_W
    fy_new = viewpoint.fy * scale_H
    cx_new = viewpoint.cx * scale_W
    cy_new = viewpoint.cy * scale_H
    
    K_new = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ])

    pts3d = depth_to_3d(render_depth.detach().cpu().numpy(), K_new, dist_coeffs=dist_coeffs)

    # ========== 在MASt3R输出的原始点云上提取轮廓（下采样前）==========
    edge_mask_3d = None
    edge_scores_3d = None
    
    # 检查是否启用轮廓引导的匹配点采样
    enable_edge_guided_matching = rgb_edge_pnp is not None and isinstance(rgb_edge_pnp, dict) and rgb_edge_pnp.get("edge_guided_matching", False)
    
    # 调试信息
    if rgb_edge_pnp is not None:
        print(f"[轮廓提取调试] rgb_edge_pnp存在: {isinstance(rgb_edge_pnp, dict)}, edge_guided_matching: {rgb_edge_pnp.get('edge_guided_matching', 'NOT_SET')}")
    
    if enable_edge_guided_matching:
        try:
            H, W = pts3d.shape[:2]
            
            # 将pts3d从(H, W, 3)转换为点云数组(N, 3)，只取有效点
            valid_mask = np.isfinite(pts3d).all(axis=2) & (pts3d[:, :, 2] > 0)  # 深度>0且所有坐标有效
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) > 0:
                # 提取有效点云（原始点云，下采样前）
                original_points_3d = pts3d[valid_indices].reshape(-1, 3)
                
                # 创建轮廓提取器（使用STAR-Edge方法）
                edge_config = rgb_edge_pnp.get("edge_extraction", {})
                # 确保使用STAR-Edge方法
                if "method" not in edge_config:
                    edge_config["method"] = "star_edge"  # 或 "localsh" 或 "fast_curvature"
                extractor = EdgeExtractor(edge_config)
                
                # 在原始点云上提取轮廓（使用STAR-Edge的LocalSH方法）
                edge_mask_3d, edge_scores_3d = extractor.extract_edges(original_points_3d)
                
                # 将轮廓mask映射回(H, W)格式，用于后续匹配点筛选
                edge_mask_2d = np.zeros((H, W), dtype=bool)
                edge_mask_2d[valid_indices] = edge_mask_3d
                
                # 统计轮廓点数量
                num_edge_points = edge_mask_3d.sum()
                num_valid_points = len(original_points_3d)
                edge_ratio_actual = num_edge_points / num_valid_points if num_valid_points > 0 else 0.0
                
                print(f"[轮廓提取] 点云尺寸: {H}x{W}, 有效点: {num_valid_points}, 轮廓点: {num_edge_points} ({edge_ratio_actual*100:.2f}%)")
                
                # ========== 保存3D轮廓投影可视化 ==========
                # 检查是否启用可视化保存（复用RGB边缘PNP的可视化配置）
                cfg = rgb_edge_pnp or {}
                viz_cfg = cfg.get("viz", {}) if isinstance(cfg, dict) else {}
                viz_enabled = bool(viz_cfg.get("enabled", False))
                viz_every = int(viz_cfg.get("viz_every", 1))
                viz_dir = viz_cfg.get("dir", None)
                tag = viz_cfg.get("tag", None)
                
                # 调试信息
                print(f"[轮廓可视化调试] viz_enabled: {viz_enabled}, viz_dir: {viz_dir}, tag: {tag}, viz_every: {viz_every}")
                
                # 检查是否需要保存（需要viz_enabled、viz_dir、tag，且满足viz_every条件）
                if viz_enabled and viz_dir is not None and tag is not None:
                    try:
                        # 获取当前帧索引
                        frame_idx = int(viz_cfg.get("frame_idx", 0))
                        
                        # 调试信息：显示是否满足保存条件
                        should_save = (frame_idx % viz_every == 0)
                        print(f"[轮廓可视化调试] frame_idx: {frame_idx}, viz_every: {viz_every}, should_save: {should_save}")
                        
                        if should_save:
                            print(f"[轮廓可视化] 开始保存轮廓可视化图片...")
                            # 获取RGB图像（使用view1的图像，因为pts3d对应的是view1的深度）
                            rgb1 = view1["img"]  # 1x3xH1xW1, normalized
                            rgb1_u8 = _to_uint8_rgb(rgb1)
                            
                            # pts3d的尺寸应该与view1["img"]匹配（都是H1, W1）
                            # 但edge_mask_2d是基于pts3d的原始尺寸(H, W)创建的
                            # 需要调整edge_mask_2d以匹配RGB图像尺寸
                            H1, W1 = rgb1_u8.shape[:2]
                            
                            if H != H1 or W != W1:
                                # 需要调整尺寸以匹配RGB图像
                                edge_mask_2d_resized = cv2.resize(
                                    edge_mask_2d.astype(np.uint8), 
                                    (W1, H1), 
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                                
                                # pts3d也需要调整尺寸
                                pts3d_resized = pts3d.reshape(H, W, 3)
                                pts3d_resized = cv2.resize(
                                    pts3d_resized.astype(np.float32),
                                    (W1, H1),
                                    interpolation=cv2.INTER_LINEAR
                                )
                            else:
                                edge_mask_2d_resized = edge_mask_2d
                                pts3d_resized = pts3d.reshape(H, W, 3)
                            
                            # 将轮廓分数映射回2D（用于热力图显示）
                            edge_scores_2d_resized = None
                            if edge_scores_3d is not None:
                                edge_scores_2d = np.zeros((H, W), dtype=np.float32)
                                edge_scores_2d[valid_indices] = edge_scores_3d
                                if H != H1 or W != W1:
                                    edge_scores_2d_resized = cv2.resize(
                                        edge_scores_2d.astype(np.float32),
                                        (W1, H1),
                                        interpolation=cv2.INTER_LINEAR
                                    )
                                else:
                                    edge_scores_2d_resized = edge_scores_2d
                            
                            # 保存可视化（包含轮廓分数用于热力图）
                            _save_3d_edge_projection_viz(
                                viz_dir=viz_dir,
                                tag=f"{tag}_3d_edge",
                                rgb_u8=rgb1_u8,
                                pts3d=pts3d_resized,
                                edge_mask_2d=edge_mask_2d_resized,
                                K=K_new,
                                dist_coeffs=dist_coeffs,
                                max_points=10000,  # 增加显示点数以便看到完整轮廓
                                edge_scores_2d=edge_scores_2d_resized
                            )
                            
                            # 保存带颜色标记的点云PLY文件（保留原始颜色，轮廓点用特殊颜色标记）
                            try:
                                # 从调整尺寸后的pts3d和edge_mask中提取点云，以便与RGB图像对应
                                valid_mask_resized = np.isfinite(pts3d_resized).all(axis=2) & (pts3d_resized[:, :, 2] > 0)
                                valid_indices_resized = np.where(valid_mask_resized)
                                
                                if len(valid_indices_resized[0]) > 0:
                                    points_3d_for_ply = pts3d_resized[valid_indices_resized].reshape(-1, 3)
                                    edge_mask_2d_flat = edge_mask_2d_resized[valid_indices_resized]
                                    
                                    _save_colored_pointcloud_ply(
                                        points_3d=points_3d_for_ply,
                                        edge_mask=edge_mask_2d_flat,
                                        viz_dir=viz_dir,
                                        tag=f"{tag}_3d_edge",
                                        rgb_u8=rgb1_u8,  # RGB图像，用于获取原始颜色
                                        pts3d_2d=pts3d_resized,  # 3D点云对应的2D坐标（用于快速查找）
                                        K=K_new,  # 相机内参
                                        dist_coeffs=dist_coeffs,  # 畸变系数
                                        max_points=500000,  # 最多保存50万个点
                                        edge_color=(255, 0, 0),  # 轮廓点标记颜色：红色
                                        valid_indices_2d=valid_indices_resized  # 直接传递索引，用于快速获取颜色
                                    )
                                else:
                                    print("[轮廓可视化] 警告: 调整尺寸后的点云无效，跳过PLY保存")
                            except Exception as e:
                                print(f"[轮廓可视化] 警告: 保存点云PLY失败: {e}")
                                import traceback
                                traceback.print_exc()
                            except Exception as e:
                                print(f"[轮廓可视化] 警告: 保存点云PLY失败: {e}")
                                import traceback
                                traceback.print_exc()
                    except Exception as e:
                        print(f"[轮廓可视化] 警告: 保存3D轮廓可视化失败: {e}")
            else:
                print("[轮廓提取] 警告: 没有有效的3D点")
                edge_mask_2d = None
        except Exception as e:
            print(f"[轮廓提取] 警告: 轮廓提取失败: {e}, 使用常规匹配")
            edge_mask_3d = None
            edge_mask_2d = None
            edge_scores_3d = None

    # Extract 3D points from image 1 and corresponding 2D points from image 2 for PnP
    objectPoints = pts3d[matches_im1[:, 1].astype(int), matches_im1[:, 0].astype(int), :]
    objectPoints = objectPoints.astype(np.float32)
    imagePoints = matches_im2.astype(np.float32)
    
    # ========== 使用轮廓信息筛选或加权匹配点 ==========
    if enable_edge_guided_matching and edge_mask_2d is not None:
        try:
            # 获取匹配点在图像1中的位置
            match_y = matches_im1[:, 1].astype(int)
            match_x = matches_im1[:, 0].astype(int)
            
            # 检查匹配点是否在轮廓区域
            valid_coords = (match_y >= 0) & (match_y < H) & (match_x >= 0) & (match_x < W)
            edge_match_mask = np.zeros(len(matches_im1), dtype=bool)
            edge_match_mask[valid_coords] = edge_mask_2d[match_y[valid_coords], match_x[valid_coords]]
            
            # 根据配置决定是筛选还是加权
            edge_match_mode = rgb_edge_pnp.get("edge_match_mode", "weight")  # "filter" 或 "weight"
            edge_match_ratio = rgb_edge_pnp.get("edge_match_ratio", 0.5)  # 如果filter模式，保留的轮廓匹配点比例
            
            if edge_match_mode == "filter":
                # 筛选模式：优先保留轮廓区域的匹配点
                num_edge_matches = int(len(matches_im1) * edge_match_ratio)
                num_non_edge_matches = len(matches_im1) - num_edge_matches
                
                edge_match_indices = np.where(edge_match_mask)[0]
                non_edge_match_indices = np.where(~edge_match_mask)[0]
                
                # 保留所有轮廓匹配点，如果不够则补充非轮廓点
                if len(edge_match_indices) >= num_edge_matches:
                    selected_edge = edge_match_indices[:num_edge_matches]
                else:
                    selected_edge = edge_match_indices
                
                # 补充非轮廓点
                if len(non_edge_match_indices) > 0:
                    num_needed = num_non_edge_matches - (len(edge_match_indices) - len(selected_edge))
                    if num_needed > 0:
                        selected_non_edge = non_edge_match_indices[:min(num_needed, len(non_edge_match_indices))]
                        selected_indices = np.concatenate([selected_edge, selected_non_edge])
                    else:
                        selected_indices = selected_edge
                else:
                    selected_indices = selected_edge
                
                # 筛选匹配点
                objectPoints = objectPoints[selected_indices]
                imagePoints = imagePoints[selected_indices]
                # 同步更新匹配点索引（用于后续可视化）
                matches_im1 = matches_im1[selected_indices]
                matches_im2 = matches_im2[selected_indices]
                
                print(f"[轮廓引导匹配] 筛选后: {len(selected_indices)}/{len(matches_im1)} 匹配点 (轮廓点: {len(selected_edge)})")
                # 筛选模式下不需要3D轮廓权重（已经通过筛选实现了）
                weights_3d = None
            else:
                # 加权模式：轮廓区域的匹配点权重更高（与RGB边缘加权合并）
                edge_weights_3d = np.ones(len(matches_im1), dtype=np.float32)
                edge_weights_3d[edge_match_mask] = rgb_edge_pnp.get("edge_3d_weight", 2.0)
                
                # 保存3D轮廓权重，稍后与RGB边缘权重合并
                weights_3d = edge_weights_3d
                
                print(f"[轮廓引导匹配] 加权模式: {edge_match_mask.sum()}/{len(matches_im1)} 匹配点在轮廓区域")
        except Exception as e:
            print(f"[轮廓引导匹配] 警告: 轮廓引导匹配失败: {e}, 使用常规匹配")
            weights_3d = None
    else:
        weights_3d = None

    # ---------- RGB(+可选深度) 轮廓加权 ----------
    weights = None
    cfg = rgb_edge_pnp or {}
    viz_cfg = cfg.get("viz", {}) if isinstance(cfg, dict) else {}
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_every = int(viz_cfg.get("viz_every", 1))
    viz_dir = viz_cfg.get("dir", None)
    tag = viz_cfg.get("tag", None)
    if cfg.get("enabled", False):
        try:
            extractor = EdgeExtractor()

            # matches 的坐标系对应 view['img'] 的分辨率（H1,W1）
            rgb1 = view1["img"]  # 1x3xH1xW1, normalized
            rgb2 = view2["img"]

            use_depth = bool(cfg.get("use_depth", False))
            depth_np = render_depth.detach().cpu().numpy()

            edge_mask1 = extractor.extract_edges_from_rgb_and_depth(
                rgb1,
                depth_map=depth_np if use_depth else None,
                rgb_method=cfg.get("rgb_method", "canny"),
                depth_method=cfg.get("depth_method", "combined"),
                fuse=cfg.get("fuse", "max"),
                rgb_weight=float(cfg.get("rgb_weight", 1.0)),
                depth_weight=float(cfg.get("depth_weight", 1.0)),
                canny_low=int(cfg.get("canny_low", 50)),
                canny_high=int(cfg.get("canny_high", 150)),
                dilate_kernel_size=int(cfg.get("dilate_kernel_size", 3)),
                blur_ksize=int(cfg.get("blur_ksize", 3)),
            )
            edge_mask2 = extractor.extract_edges_from_rgb_and_depth(
                rgb2,
                depth_map=depth_np if use_depth else None,
                rgb_method=cfg.get("rgb_method", "canny"),
                depth_method=cfg.get("depth_method", "combined"),
                fuse=cfg.get("fuse", "max"),
                rgb_weight=float(cfg.get("rgb_weight", 1.0)),
                depth_weight=float(cfg.get("depth_weight", 1.0)),
                canny_low=int(cfg.get("canny_low", 50)),
                canny_high=int(cfg.get("canny_high", 150)),
                dilate_kernel_size=int(cfg.get("dilate_kernel_size", 3)),
                blur_ksize=int(cfg.get("blur_ksize", 3)),
            )

            weights, _ = extractor.enhance_matches_with_edges(
                matches_im1.astype(np.float32),
                matches_im2.astype(np.float32),
                edge_mask1.astype(np.float32),
                edge_mask2.astype(np.float32),
                weight=float(cfg.get("edge_weight", 2.0)),
            )
        except Exception as e:
            # 轮廓增强失败时自动回退（不影响主流程）
            weights = None
            edge_mask1, edge_mask2 = None, None
    else:
        edge_mask1, edge_mask2 = None, None
    
    # ========== 合并3D轮廓权重和RGB边缘权重 ==========
    if enable_edge_guided_matching and 'weights_3d' in locals() and weights_3d is not None:
        if weights is not None:
            # 合并RGB边缘权重和3D轮廓权重
            weights = weights * weights_3d
            print(f"[轮廓引导匹配] 已合并RGB边缘权重和3D轮廓权重")
        elif len(objectPoints) == len(weights_3d):
            # 如果只有3D轮廓权重，直接使用
            weights = weights_3d

    # Skip PnP if there are not enough points
    if len(objectPoints) < 6 or len(imagePoints) < 6:
        print("Warning: Not enough points to perform PnP estimation.")
        print("Number of points:", len(objectPoints))
        success = False
    else:
        if weights is None:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints, imagePoints, K_new, dist_coeffs,
                iterationsCount=int(cfg.get("iterations", 100)),
                reprojectionError=float(cfg.get("reproj_error", 5.0)),
                flags=cv2.SOLVEPNP_SQPNP
            )
        else:
            success, rvec, tvec, inliers = _weighted_ransac_pnp(
                objectPoints, imagePoints, K_new, dist_coeffs,
                weights=weights,
                iterationsCount=int(cfg.get("iterations", 100)),
                reprojectionError=float(cfg.get("reproj_error", 5.0)),
                min_sample=int(cfg.get("min_sample", 6)),
                seed=int(cfg.get("seed", 0)),
            )

    # ---------- 可视化输出 ----------
    try:
        if viz_enabled and viz_dir is not None and tag is not None and (int(viz_cfg.get("frame_idx", 0)) % viz_every == 0):
            if edge_mask1 is None or edge_mask2 is None:
                # 若未启用加权，但要求可视化，则至少做RGB边缘
                extractor = EdgeExtractor()
                rgb1 = view1["img"]
                rgb2 = view2["img"]
                edge_mask1 = extractor.extract_edges_from_rgb(rgb1,
                                                             method=cfg.get("rgb_method", "canny"),
                                                             canny_low=int(cfg.get("canny_low", 50)),
                                                             canny_high=int(cfg.get("canny_high", 150)),
                                                             dilate_kernel_size=int(cfg.get("dilate_kernel_size", 3)),
                                                             blur_ksize=int(cfg.get("blur_ksize", 3)))
                edge_mask2 = extractor.extract_edges_from_rgb(rgb2,
                                                             method=cfg.get("rgb_method", "canny"),
                                                             canny_low=int(cfg.get("canny_low", 50)),
                                                             canny_high=int(cfg.get("canny_high", 150)),
                                                             dilate_kernel_size=int(cfg.get("dilate_kernel_size", 3)),
                                                             blur_ksize=int(cfg.get("blur_ksize", 3)))
            rgb1_u8 = _to_uint8_rgb(view1["img"])
            rgb2_u8 = _to_uint8_rgb(view2["img"])
            _save_rgb_edge_pnp_viz(
                viz_dir=viz_dir,
                tag=tag,
                rgb1_u8=rgb1_u8,
                rgb2_u8=rgb2_u8,
                edge1=edge_mask1,
                edge2=edge_mask2,
                matches1=matches_im1.astype(np.float32),
                matches2=matches_im2.astype(np.float32),
                weights=weights,
                inliers=inliers if success else None,
                max_matches=int(viz_cfg.get("max_matches", 300)),
            )
    except Exception:
        pass
    
    if success:
        R, _ = cv2.Rodrigues(rvec)
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R
        pose_w2c[:3, 3] = tvec[:, 0]
        return pose_w2c, render_depth.detach().cpu().numpy()  
    else:
        print("PnP估计失败")
        pose_w2c = np.eye(4)
        return pose_w2c, render_depth.detach().cpu().numpy()  

# Extract depth and confidence from MASt3R
def get_depth(img1, img2, model, return_conf=False): # <--- 改动：增加参数
    device = 'cuda'
    # ... (原有参数设置不变) ...
    H1 = img1.shape[1]
    W1 = img1.shape[2]
    
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    
    # --- 原有提取深度逻辑 ---
    pts1 = pred1['pts3d'].squeeze(0)
    z1 = pts1[...,2]
    z1 = z1.detach().cpu().numpy()
    z1_resized = cv2.resize(z1, (W1,H1), interpolation=cv2.INTER_NEAREST)
   
    # --- 改动：处理置信度 ---
    if return_conf:
        # 提取置信度 (Confidence)
        conf = pred1['conf'].squeeze(0).detach().cpu().numpy()
        # 缩放至原图大小 (使用线性插值)
        conf_resized = cv2.resize(conf, (W1, H1), interpolation=cv2.INTER_LINEAR)
        return z1_resized, conf_resized # 返回两个值
    else:
        return z1_resized # 保持旧行为，只返回深度
    
# Visualize comparison of rendered depth
def save_depth_comparison(render_depth, mono_depth, rgb, cur_frame_idx, save_dir):
    '''
    Inputs:
        - render_depth: Rendered depth map, (H, W) or (C, H, W) numpy array
        - mono_depth: Monocular depth estimation, (H, W) numpy array
        - rgb: RGB image, (C, H, W) torch tensor
        - cur_frame_idx: Index of the current frame
        - save_dir: Path to save the result
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    if render_depth.ndim == 3:
        render_depth = render_depth.squeeze(0)
    
    rgb_image = rgb.permute(1, 2, 0).cpu().numpy()  
    H, W = render_depth.shape
    if rgb_image.shape[:2] != (H, W):
        rgb_image = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
    
    if mono_depth.shape != (H, W):
        mono_depth = cv2.resize(mono_depth, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Normalize depth maps
    render_depth_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
    mono_depth_norm = (mono_depth - mono_depth.min()) / (mono_depth.max() - mono_depth.min())
    
    # Compute depth error
    depth_error = np.abs(render_depth - mono_depth)
    depth_error_norm = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Frame {cur_frame_idx}", fontsize=20, y=0.93)
    
    render0 = axes[0,0].imshow(render_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,0].set_title("Rendered Depth", fontsize=15)
    axes[0,0].axis("off")
    
    axes[0,1].imshow(mono_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,1].set_title("MASt3R Mono Depth", fontsize=15)
    axes[0,1].axis("off")
    
    # Add a shared colorbar between the two depth maps
    cbar = fig.colorbar(render0, ax=axes[0, :], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Normalized Depth Value", fontsize=12)
    
    # Plot depth error map
    error = axes[1,0].imshow(depth_error_norm, cmap="magma", vmin=0, vmax=1)
    axes[1,0].set_title("Depth Error", fontsize=15)
    axes[1,0].axis("off")
    
    # Add a separate colorbar for the error map
    cbar_error = fig.colorbar(error, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar_error.set_label("Normalized Depth Error", fontsize=12)
    
    axes[1,1].imshow(rgb_image)
    axes[1,1].set_title("RGB", fontsize=15)
    axes[1,1].axis("off")
    
    save_path = os.path.join(save_dir, f"{cur_frame_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path


def save_confidence_map(conf_map, cur_frame_idx, save_dir):
    """
    保存置信度热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # 使用 'plasma' 或 'inferno' 这种高对比度的色图
    # vmax 可以根据你的数据情况调整，通常大于 2.0 就是非常可信了，这里设为自动或固定值
    im = ax.imshow(conf_map, cmap='plasma') 
    
    ax.set_title(f"Confidence Map - Frame {cur_frame_idx}\n(Brighter/Yellow = Higher Confidence)", fontsize=15)
    ax.axis('off')
    
    # 添加色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence Value", fontsize=12)
    
    save_path = os.path.join(save_dir, f"confidence_{cur_frame_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Confidence map saved to: {save_path}")