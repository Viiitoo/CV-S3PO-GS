#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    # ========== 时间形变处理 ==========
    # 从相机对象中获取时间信息（如果存在）
    # 时间用于计算动态场景中高斯点的形变
    t = None
    if hasattr(viewpoint_camera, "time"):
        t = viewpoint_camera.time
    elif hasattr(viewpoint_camera, "t"):
        t = viewpoint_camera.t

    # ========== 获取形变后的高斯属性 ==========
    # 这是关键步骤：根据时间t计算所有高斯点的形变属性
    # 如果支持完整形变（位置+旋转+放缩+不透明度），就用新方法
    if hasattr(pc, "get_deformed_attributes_t") and t is not None:
        # 调试：检查时间值
        print(f"[DEBUG] Render time t={t}")

        # 调用新实现的形变函数，一次性获取所有形变后的属性
        # 这比只变形位置更强大，可以处理旋转、缩放、透明度变化
        means3D, rotations, scales, opacity = pc.get_deformed_attributes_t(t)
        
        # ========== 累积形变量（用于形变点选择）==========
        # 如果支持形变点选择，累积形变量用于后续更新deformation_table
        if hasattr(pc, '_deformation_table') and pc._deformation_table.numel() > 0:
            if hasattr(pc, '_deformation_accum'):
                # 计算形变量（形变后的位置 - 原始位置）
                position_diff = torch.abs(means3D - pc._xyz)  # [N, 3]

                # 调试：检查累积条件
                has_table = hasattr(pc, '_deformation_table')
                table_size = pc._deformation_table.numel() if has_table else 0
                diff_max = position_diff.max().item()
                print(f"[DEBUG] Accum check: has_table={has_table}, table_size={table_size}, diff_max={diff_max:.6f}")

                # 累积形变量：记录每个点的最大形变幅度
                with torch.no_grad():
                    if pc._deformation_accum.numel() == 0 or pc._deformation_accum.shape != position_diff.shape:
                        # 如果accum未初始化或尺寸不匹配，重新初始化为[N, 3]
                        pc._deformation_accum = torch.zeros_like(position_diff)
                        print(f"[DEBUG] Initialized deformation_accum with shape {pc._deformation_accum.shape}")

                    # 累积每个方向的形变量（取最大值）
                    old_max = pc._deformation_accum.max().item()
                    pc._deformation_accum = torch.maximum(
                        pc._deformation_accum,
                        position_diff.detach()
                    )
                    new_max = pc._deformation_accum.max().item()
                    if new_max > old_max:
                        print(f"[DEBUG] Accum updated: {old_max:.6f} -> {new_max:.6f}")
        
        # 处理各向同性放缩的情况（如果scaling只有1维，复制成3维）
        # 各向同性：x、y、z三个方向的放缩相同
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
    else:
        # ========== 向后兼容：回退到旧方法 ==========
        # 如果模型不支持完整形变，或者没有时间信息，就回退到：
        # 1. 只变形位置（如果支持get_xyz_t）
        # 2. 或者完全不变形（使用原始值）
        # 这样保证旧代码不会出错
        means3D = pc.get_xyz_t(t) if hasattr(pc, "get_xyz_t") else pc.get_xyz
        rotations = pc.get_rotation
        scales = pc.get_scaling
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        opacity = pc.get_opacity

    screenspace_points = (
        torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    )

    try:
        screenspace_points.retain_grad()
    except Exception:
        pass


    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # 如果使用形变后的属性，需要重新计算covariance
        # 注意：这里使用形变后的scaling和rotation
        cov3D_precomp = pc.covariance_activation(scales, scaling_modifier, rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )

            dir_pp = means3D - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )

            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if mask is not None:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }

# Render depth map with specified resolution
def render_with_custom_resolution(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    target_width, 
    target_height, 
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene with custom resolution by adjusting camera parameters.

    Background tensor (bg_color) must be on GPU!
    """
    # Save original camera parameters
    original_width = viewpoint_camera.image_width
    original_height = viewpoint_camera.image_height
    original_projection_matrix = viewpoint_camera.projection_matrix.clone()
    
    if pc.get_xyz.shape[0] == 0:
        return None

    # --- time for deformation ---
    t = None
    if hasattr(viewpoint_camera, "time"):
        t = viewpoint_camera.time
    elif hasattr(viewpoint_camera, "t"):
        t = viewpoint_camera.t
    

    # Get original resolution and compute scaling ratio to the new resolution
    scale_x = target_width / viewpoint_camera.image_width
    scale_y = target_height / viewpoint_camera.image_height
    device = bg_color.device
    
    # Adjust the camera intrinsic matrix
    fx_new = viewpoint_camera.fx * scale_x
    fy_new = viewpoint_camera.fy * scale_y
    cx_new = viewpoint_camera.cx * scale_x
    cy_new = viewpoint_camera.cy * scale_y

    # Generate new projection matrix
    new_proj = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx_new, fy=fy_new, cx=cx_new, cy=cy_new, W=target_width, H=target_height
    ).transpose(0, 1)
    
    viewpoint_camera.projection_matrix = new_proj.to(device)

    viewpoint_camera.image_width = target_width
    viewpoint_camera.image_height = target_height

    # Set new rendering configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=target_height,
        image_width=target_width,
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),  
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),  
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取形变后的属性（如果支持完整形变）或仅位置形变
    if hasattr(pc, "get_deformed_attributes_t") and t is not None:
        means3D, rotations, scales, opacity = pc.get_deformed_attributes_t(t)
        # 确保scales的形状正确
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
    else:
        # 回退到仅位置形变或原始值
        means3D = pc.get_xyz_t(t) if hasattr(pc, "get_xyz_t") else pc.get_xyz
        rotations = pc.get_rotation
        scales = pc.get_scaling
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        opacity = pc.get_opacity

    screenspace_points = (
        torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    )
    
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    means2D = screenspace_points

    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # 如果使用形变后的属性，需要重新计算covariance
        cov3D_precomp = pc.build_covariance_from_scaling_rotation(scales, scaling_modifier, rotations)

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = means3D - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )

            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if mask is not None:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Restore original camera parameters
    viewpoint_camera.image_width = original_width
    viewpoint_camera.image_height = original_height
    viewpoint_camera.projection_matrix = original_projection_matrix

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
