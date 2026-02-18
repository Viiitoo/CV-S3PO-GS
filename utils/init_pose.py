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
from typing import Optional

from gaussian_splatting.gaussian_renderer import render_with_custom_resolution
from utils.edge_extraction import EdgeExtractor

import torchvision.transforms as tvf
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


def _write_ply_xyzrgb_ascii(save_path: str, xyz: np.ndarray, rgb_u8: np.ndarray):
    """写一个简单的ASCII PLY（x y z + uchar rgb），便于可视化查看。"""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
    assert len(xyz) == len(rgb_u8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(save_path, "w") as f:
        f.write(header)
        for p, c in zip(xyz, rgb_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _save_mast3r_pc_edge_viz(viz_dir: str,
                             tag: str,
                             rgb_u8: np.ndarray,
                             score_map: np.ndarray,
                             mask_map: np.ndarray,
                             ply_xyz: Optional[np.ndarray] = None,
                             ply_rgb_u8: Optional[np.ndarray] = None):
    """
    保存 MASt3R 点云轮廓提取可视化：
    - score_map/mask_map: HxW, float32/bool
    - rgb_u8: HxWx3 uint8 (与 score_map 同分辨率或已对齐)
    """
    os.makedirs(viz_dir, exist_ok=True)

    # 保存 score colormap
    s = np.clip(score_map, 0.0, 1.0)
    s_u8 = (s * 255).astype(np.uint8)
    s_color = cv2.applyColorMap(s_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_score.png"), s_color)

    # 保存 mask
    m = (mask_map.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_mask.png"), m)

    # 保存 overlay（红色覆盖mask）
    ov = rgb_u8.copy()
    ov[..., 0] = np.maximum(ov[..., 0], m)  # R channel (RGB)
    cv2.imwrite(os.path.join(viz_dir, f"{tag}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    # 保存 PLY（可选）
    if ply_xyz is not None and ply_rgb_u8 is not None and len(ply_xyz) > 0:
        _write_ply_xyzrgb_ascii(os.path.join(viz_dir, f"{tag}_pc_edge.ply"), ply_xyz, ply_rgb_u8)

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

    # Extract 3D points from image 1 and corresponding 2D points from image 2 for PnP
    objectPoints = pts3d[matches_im1[:, 1].astype(int), matches_im1[:, 0].astype(int), :]
    objectPoints = objectPoints.astype(np.float32)
    imagePoints = matches_im2.astype(np.float32)

    # ---------- 中心区域裁剪过滤（排除图像边缘的匹配点） ----------
    cfg = rgb_edge_pnp or {}
    center_crop_ratio = float(cfg.get("center_crop_ratio", 1.0))
    if center_crop_ratio < 1.0:
        H_img, W_img = pts3d.shape[:2]
        # center_crop_ratio 是面积比例，每个维度保留 sqrt(ratio) 的范围
        import math
        dim_ratio = math.sqrt(center_crop_ratio)
        margin_x = (1.0 - dim_ratio) / 2.0 * W_img
        margin_y = (1.0 - dim_ratio) / 2.0 * H_img
        ip = imagePoints.reshape(-1, 2)
        center_mask = (
            (ip[:, 0] >= margin_x) & (ip[:, 0] < W_img - margin_x) &
            (ip[:, 1] >= margin_y) & (ip[:, 1] < H_img - margin_y)
        )
        n_before = len(objectPoints)
        objectPoints = objectPoints[center_mask]
        imagePoints = imagePoints[center_mask]
        matches_im1 = matches_im1[center_mask]
        matches_im2 = matches_im2[center_mask]
        print(f"[CenterCrop] area_ratio={center_crop_ratio:.2f}, dim_ratio={dim_ratio:.4f}, "
              f"kept {len(objectPoints)}/{n_before} points")

    # ---------- 3D点云轮廓引导匹配（STAR-Edge / 曲率等） ----------
    weights = None
    # 说明：对“渲染深度生成的3D点云”做轮廓提取，然后把轮廓分数映射到每个匹配的3D点上。
    # - weight: 将每个match的权重乘上 (1 + edge_3d_weight * score)
    # - filter: 保留 score 最高的 edge_match_ratio 部分匹配点
    if cfg.get("edge_guided_matching", False):
        try:
            edge_method_cfg = cfg.get("edge_extraction", {}) if isinstance(cfg, dict) else {}
            extractor3d = EdgeExtractor({"edge_extraction": edge_method_cfg})

            # 全量点云：来自 depth_to_3d 的 (H,W,3)
            H_img, W_img = pts3d.shape[:2]
            pts3d_all = pts3d.reshape(-1, 3).astype(np.float32)
            valid = np.isfinite(pts3d_all).all(axis=1) & (pts3d_all[:, 2] > 1e-6)
            pts3d_all = pts3d_all[valid]

            if len(pts3d_all) >= 50:
                # 下采样以加速（用EdgeExtractor自带体素下采样/点数上限）
                down_pts, _ = extractor3d.voxel_downsample(pts3d_all)
                if len(down_pts) >= 20:
                    edge_mask3d, edge_scores3d = extractor3d.extract_edges(down_pts[:, :3])
                    edge_scores3d = np.asarray(edge_scores3d, dtype=np.float32).reshape(-1)

                    # 在3D层面抑制边界噪声：将下采样点投影回2D，抑制边界区域的轮廓分数
                    border_margin = int(cfg.get("edge_border_margin", 30))
                    if border_margin > 0:
                        z_vals = down_pts[:, 2].clip(min=1e-8)
                        proj_x = (K_new[0, 0] * down_pts[:, 0] / z_vals) + K_new[0, 2]
                        proj_y = (K_new[1, 1] * down_pts[:, 1] / z_vals) + K_new[1, 2]
                        in_border_3d = (
                            (proj_x < border_margin) |
                            (proj_x >= W_img - border_margin) |
                            (proj_y < border_margin) |
                            (proj_y >= H_img - border_margin)
                        )
                        edge_scores3d[in_border_3d] = 0.0

                    # 把downsampled分数映射回每个match的3D点（最近邻）
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(down_pts[:, :3])
                        _, nn = tree.query(objectPoints[:, :3], k=1, workers=-1)
                        score_match = edge_scores3d[np.asarray(nn, dtype=np.int64)]
                    except Exception:
                        score_match = None

                    # 抑制图像边缘区域匹配点的轮廓分数（STAR-Edge 边界噪声后处理）
                    border_margin = int(cfg.get("edge_border_margin", 30))
                    if border_margin > 0 and score_match is not None:
                        ip = imagePoints.reshape(-1, 2)
                        in_border = (
                            (ip[:, 0] < border_margin) |
                            (ip[:, 0] >= W_img - border_margin) |
                            (ip[:, 1] < border_margin) |
                            (ip[:, 1] >= H_img - border_margin)
                        )
                        score_match[in_border] = 0.0

                    if score_match is not None:
                        mode = str(cfg.get("edge_match_mode", "weight")).lower()
                        edge_3d_weight = float(cfg.get("edge_3d_weight", 2.0))

                        if mode == "filter":
                            keep_ratio = float(cfg.get("edge_match_ratio", 0.5))
                            keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
                            keep_n = int(round(len(objectPoints) * keep_ratio))
                            keep_n = max(0, min(keep_n, len(objectPoints)))
                            # 保证PnP最小点数
                            if keep_n >= 6:
                                keep_idx = np.argsort(score_match)[-keep_n:]
                                objectPoints = objectPoints[keep_idx]
                                imagePoints = imagePoints[keep_idx]
                                matches_im1 = matches_im1[keep_idx]
                                matches_im2 = matches_im2[keep_idx]
                                if weights is not None:
                                    weights = np.asarray(weights, dtype=np.float32)[keep_idx]
                                score_match = score_match[keep_idx]

                        # 默认 weight 模式
                        if mode != "filter":
                            w3d = (1.0 + edge_3d_weight * np.clip(score_match, 0.0, 1.0)).astype(np.float32)
                            if weights is None:
                                weights = w3d
                            else:
                                weights = (np.asarray(weights, dtype=np.float32) * w3d).astype(np.float32)
        except Exception:
            # 3D轮廓引导失败时自动回退
            pass

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
def get_depth(img1, img2, model, return_conf=False, mast3r_edge_viz=None): # <--- 改动：增加参数
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

    # --- 新增：对 MASt3R 输出点云做每帧轮廓提取并保存可视化 ---
    cfg = mast3r_edge_viz or {}
    if isinstance(cfg, dict) and cfg.get("enabled", False):
        try:
            viz_every = int(cfg.get("viz_every", 1))
            frame_idx = int(cfg.get("frame_idx", 0))
            viz_dir = cfg.get("dir", None)
            tag = cfg.get("tag", f"f{frame_idx:06d}")
            save_ply = bool(cfg.get("save_ply", True))
            max_ply_points = int(cfg.get("max_ply_points", 50000))
            max_ply_points = max(1000, max_ply_points)

            if viz_dir is not None and (frame_idx % viz_every == 0):
                # MASt3R 点云（在 view1 分辨率）
                pts_map = pred1['pts3d'].squeeze(0).detach().cpu().numpy()  # (h,w,3)
                h, w = pts_map.shape[:2]
                pts_flat = pts_map.reshape(-1, 3)
                valid = np.isfinite(pts_flat).all(axis=1) & (pts_flat[:, 2] > 1e-6)
                valid_idx = np.flatnonzero(valid)

                if len(valid_idx) > 200:
                    pts_valid = pts_flat[valid_idx].astype(np.float32)

                    # edge_extraction 配置：优先使用 cfg.edge_extraction，否则用全局 config 的默认（star_edge）
                    edge_cfg = cfg.get("edge_extraction", {})
                    extractor = EdgeExtractor({"edge_extraction": edge_cfg} if isinstance(edge_cfg, dict) else None)
                    edge_mask, edge_scores = extractor.extract_edges(pts_valid)
                    edge_scores = np.asarray(edge_scores, dtype=np.float32).reshape(-1)
                    edge_mask = np.asarray(edge_mask, dtype=bool).reshape(-1)

                    # 映射回2D map（view1分辨率）
                    score_full = np.zeros((h * w,), dtype=np.float32)
                    mask_full = np.zeros((h * w,), dtype=bool)
                    score_full[valid_idx] = edge_scores
                    mask_full[valid_idx] = edge_mask
                    score_map = score_full.reshape(h, w)
                    mask_map = mask_full.reshape(h, w)

                    # 抑制图像边缘区域的轮廓分数（STAR-Edge 边界噪声后处理）
                    border_margin = int(cfg.get("edge_border_margin", 30))
                    if border_margin > 0:
                        score_map[:border_margin, :] = 0.0
                        score_map[-border_margin:, :] = 0.0
                        score_map[:, :border_margin] = 0.0
                        score_map[:, -border_margin:] = 0.0
                        mask_map[:border_margin, :] = False
                        mask_map[-border_margin:, :] = False
                        mask_map[:, :border_margin] = False
                        mask_map[:, -border_margin:] = False

                    # 用 view1['img'] 生成对应分辨率的 RGB（比原图更对齐）
                    rgb_pred_u8 = _to_uint8_rgb(view1["img"])  # HWC uint8 RGB
                    if rgb_pred_u8.shape[:2] != (h, w):
                        rgb_pred_u8 = cv2.resize(rgb_pred_u8, (w, h), interpolation=cv2.INTER_LINEAR)

                    # 构建 PLY（仅保存一个下采样子集，避免文件过大）
                    ply_xyz = None
                    ply_rgb = None
                    if save_ply:
                        if len(valid_idx) > max_ply_points:
                            sub = np.random.choice(valid_idx, size=max_ply_points, replace=False)
                            sub = np.asarray(sub, dtype=np.int64)
                        else:
                            sub = valid_idx.astype(np.int64)

                        sub_pts = pts_flat[sub]
                        sub_rgb = rgb_pred_u8.reshape(-1, 3)[sub].copy()
                        sub_edge = mask_full[sub]
                        sub_rgb[sub_edge] = np.array([255, 0, 0], dtype=np.uint8)
                        ply_xyz = sub_pts
                        ply_rgb = sub_rgb

                    _save_mast3r_pc_edge_viz(
                        viz_dir=viz_dir,
                        tag=tag,
                        rgb_u8=rgb_pred_u8,
                        score_map=score_map,
                        mask_map=mask_map,
                        ply_xyz=ply_xyz,
                        ply_rgb_u8=ply_rgb,
                    )
        except Exception:
            # 仅可视化失败时不影响主流程
            pass
   
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