from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .init_pose import _resize_pil_image, torch_images_to_dust3r_format

# calculate the scale factor between two depth from different keyframe, based on correspondence
# corresponding to scale remedy strategy in paper
def find_scale(im1, im2, depth1, depth2, model):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    # Extract features from images and perform point matching
    images = torch_images_to_dust3r_format([im1, im2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()   

    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    
    H1 = view1['img'].shape[2]     
    W1 = view1['img'].shape[3]
    
    # Bilinear interpolation
    depth1_resize = cv2.resize(depth1, (W1,H1), interpolation=cv2.INTER_LINEAR)
    depth2_resize = cv2.resize(depth2, (W1,H1), interpolation=cv2.INTER_LINEAR)

    # Nearest-neighbor interpolation
    #depth1_resize = cv2.resize(depth1, (W1, H1), interpolation=cv2.INTER_NEAREST)
    #depth2_resize = cv2.resize(depth2, (W1, H1), interpolation=cv2.INTER_NEAREST)

    # Get column and row indices of correspondences
    u2, v2 = matches_im2[:, 0], matches_im2[:, 1]
    u1, v1 = matches_im1[:, 0], matches_im1[:, 1]

    depth_values_current = depth2_resize[v2, u2]
    depth_values_previous = depth1_resize[v1, u1]

    # Valid mask to ensure both depth arrays have valid pixels
    valid_mask = (depth_values_current > 0) & ~np.isnan(depth_values_current) & (depth_values_previous > 0) & ~np.isnan(depth_values_previous)

    depth_values_current = depth_values_current[valid_mask]
    depth_values_previous = depth_values_previous[valid_mask]

    scale_factor = np.mean(depth_values_previous) / (np.mean(depth_values_current))
    
    return scale_factor

# Patch-based Pointmap Scale Alignment (Algorithm 1 in Paper)
def process_depth(render_depth, mono_depth, last_depth, im1, im2, model, 
                  patch_size=10, mean_threshold=0.25, std_threshold=0.3, 
                  error_threshold=0.1, final_error_threshold=0.15, 
                  max_iter=4, epsilon=0.01, min_accurate_pixels_ratio=0.01,
                  confidence_map=None): # [新增] 传入置信度图
    # Step 1: Ensure render_depth is in (H, W) format
    if render_depth.ndim == 3:
        render_depth = render_depth[0] 

    H, W = render_depth.shape
    scale_factor = 1.0
    prev_scale_factor = 0.0 
    final_mask = np.zeros((H, W), dtype=bool)
    
    total_pixels = H * W
    min_accurate_pixels = int(min_accurate_pixels_ratio * total_pixels)
    
    num_accurate_pixels = 0

    for k in range(max_iter):
        # Exit if the scale factor has already converged
        if (abs(scale_factor - prev_scale_factor) < epsilon) and (scale_factor != 1.0):
            break
        prev_scale_factor = scale_factor
        
        patch_num = 0

        # Step 2: Initial patch filtering
        accurate_pixels = np.zeros((H, W), dtype=bool)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                render_patch = render_depth[i:i+patch_size, j:j+patch_size]
                mono_patch = mono_depth[i:i+patch_size, j:j+patch_size] * scale_factor

                # Check patch size to avoid boundary overflow
                if render_patch.size == 0 or mono_patch.size == 0:
                    continue

                # Filter patches by mean and standard deviation
                mean_condition = abs(np.mean(render_patch) - np.mean(mono_patch)) < mean_threshold * np.mean(mono_patch)
                std_condition = abs(np.std(render_patch) - np.std(mono_patch)) < std_threshold * np.std(mono_patch)

                if mean_condition and std_condition:  
                    patch_num = patch_num + 1
                    # Step 3: Patch normalization
                    render_norm = (render_patch - np.mean(render_patch)) / (np.std(render_patch) + 1e-6)
                    mono_norm = (mono_patch - np.mean(mono_patch)) / (np.std(mono_patch) + 1e-6)

                    # Step 4: Accurate pixel filtering
                    patch_mask = np.abs(render_norm - mono_norm) < error_threshold
                    accurate_pixels[i:i+patch_size, j:j+patch_size] = patch_mask
        # If too few accurate pixels in intermediate patches, apply scale remedy strategy as intermediate scale 
        if (np.sum(accurate_pixels) < min_accurate_pixels) and (k==2):
            num_accurate_pixels = np.sum(accurate_pixels) 
            
            scale_factor = find_scale(im1, im2, last_depth, mono_depth, model)
            continue
        # If too few accurate pixels overall, apply scale remedy strategy as final result
        if (np.sum(accurate_pixels) < min_accurate_pixels) and (k==3):
            num_accurate_pixels = np.sum(accurate_pixels) 
            scale_factor = find_scale(im1, im2, last_depth, mono_depth, model)
            print("Fallback: not enough accurate pixels, apply scale remedy using the previous keyframe")
            break
        # If sufficient accurate pixels exist, compute scale factor based on them
        num_accurate_pixels = 0
        if np.any(accurate_pixels) and ((k<2) or (np.sum(accurate_pixels) >= min_accurate_pixels)):
            scale_factor = np.mean(render_depth[accurate_pixels]) / np.mean(mono_depth[accurate_pixels])
            #scale_factor = np.median(render_depth[accurate_pixels] / (mono_depth[accurate_pixels] + 1e-8))
            final_mask = accurate_pixels.copy()  
            num_accurate_pixels = np.sum(final_mask)

    # Step 7: Soft Weighted Fusion (软加权融合)
    # -------------------------------------------------------
    # 1. 准备对齐后的先验深度 X^p 和 测量深度 X^r
    mono_depth_scaled = mono_depth * scale_factor  # X^p
    X_r = render_depth                             # X^r

    # 2. 准备置信度 c
    # 如果没传置信度，默认全为 1.0 (仅依赖几何误差进行融合)
    if confidence_map is None:
        c = np.ones_like(X_r)
    else:
        # 确保置信度尺寸匹配 (以防传入的尺寸不一致)
        if confidence_map.shape != (H, W):
            c = cv2.resize(confidence_map, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            c = confidence_map

    # 3. 计算相对残差 r = (Xr - Xp) / Xp
    # 加 1e-8 防止除零
    r = (X_r - mono_depth_scaled) / (mono_depth_scaled + 1e-8)

    # 4. 计算鲁棒权重 w(r) (Cauchy Kernel)
    # delta 对应 final_error_threshold (例如 0.15 即 15% 容差)
    delta = final_error_threshold
    w_r = 1.0 / (1.0 + (r / delta)**2)

    # 5. 计算最终融合系数 alpha = c * w(r)
    alpha = c * w_r
    alpha = np.clip(alpha, 0.0, 1.0) # 限制在 [0, 1]

    # 6. 执行融合 X_fused = alpha * Xr + (1-alpha) * Xp
    final_depth = alpha * X_r + (1.0 - alpha) * mono_depth_scaled

    # 7. 特殊情况处理
    # 如果渲染深度本身是 0 (无效区域)，则强制完全信任先验
    mask_zero = (X_r == 0)
    final_depth[mask_zero] = mono_depth_scaled[mask_zero]

    # 为了兼容接口返回 error_mask
    # 定义：融合权重 alpha 过小 (例如小于 0.5) 的区域，可视作“渲染值不可信”的区域
    error_mask = alpha < 0.5 

    print(f"Soft Fusion Applied. Mean Alpha: {np.mean(alpha):.4f}")
    
    return final_depth, scale_factor, error_mask, num_accurate_pixels 
