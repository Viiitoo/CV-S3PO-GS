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

import os
import torch
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torch.nn.functional as F


from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    # 初始化高斯模型
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        # ========== 时间形变相关参数 ==========
        # K_time: 时间基函数的数量
        # EH-SurGS使用17-20个基函数，我们默认使用17个（与EH-SurGS默认值一致）
        # 更多基函数可以表达更复杂的时间变化，但会增加内存和计算开销
        # 可以通过config中的time_basis_num参数自定义，如果没有配置则使用默认值17
        if config is not None and "time_basis_num" in config.get("model_params", {}):
            self.K_time = config["model_params"]["time_basis_num"]
        else:
            self.K_time = 17  # 默认值：与EH-SurGS的curve_num=17一致
        
        # t_mu: 时间基函数的中心位置，均匀分布在[0.2, 0.8]区间
        # 这些值会在训练过程中自动调整，找到最佳的时间分布
        # 注意：如果K_time很大，这些初始值会均匀分布在整个时间区间
        self.t_mu = nn.Parameter(torch.linspace(0.2, 0.8, self.K_time, device="cuda"))
        
        # t_sigma_raw: 时间基函数的宽度参数（原始值，会通过softplus转换为正数）
        # 控制每个基函数影响的时间范围，值越大影响范围越广
        # 初始值设为0，经过softplus后约为0.693，这样每个基函数有适中的影响范围
        self.t_sigma_raw = nn.Parameter(torch.zeros((self.K_time,), device="cuda"))
        
        # ========== 形变权重参数（每个高斯点都有这些参数）==========
        # _w_pos: 位置形变权重 [N个点, K_time个基函数, 3维坐标(x,y,z)]
        # 用来控制每个点在时间上的位置变化
        self._w_pos = torch.empty(0, self.K_time, 3, device="cuda")
        
        # _w_rot: 旋转形变权重 [N个点, K_time个基函数, 4维四元数(w,x,y,z)]
        # 用来控制每个点在时间上的旋转变化（比如物体转动）
        self._w_rot = torch.empty(0, self.K_time, 4, device="cuda")
        
        # _w_scale: 放缩形变权重 [N个点, K_time个基函数, 3维放缩(sx,sy,sz)]
        # 用来控制每个点在时间上的大小变化（比如物体膨胀或收缩）
        self._w_scale = torch.empty(0, self.K_time, 3, device="cuda")
        
        # _w_opacity: 不透明度形变权重 [N个点, K_time个基函数, 1维不透明度]
        # 用来控制每个点在时间上的透明度变化（比如物体出现或消失）
        self._w_opacity = torch.empty(0, self.K_time, 1, device="cuda")
        
        # ========== 形变点选择相关参数（参考EH-SurGS）==========
        # _deformation_table: 布尔表，标记哪些点需要形变 [N] bool
        # True表示该点需要计算形变，False表示该点是静态的，不需要形变
        self._deformation_table = torch.empty(0, dtype=torch.bool, device="cuda")
        
        # _deformation_accum: 形变累积量 [N, ...]
        # 用于记录每个点的形变量，用于动态更新deformation_table
        self._deformation_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    def _time_phi(self, t: torch.Tensor) -> torch.Tensor:
        """
        # t: shape [] or [1] or [B]; assumed normalized to [0,1]
        #return: phi shape [B, K]
        """
        if t.dim() == 0:
            t = t.view(1)
        t = t.view(-1, 1)  # [B,1]
        mu = self.t_mu.view(1, -1)  # [1,K]
        sigma = F.softplus(self.t_sigma_raw).view(1, -1) + 1e-6
        phi = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)  # [B,K]
        return phi

    _time_deform_logged = False  # 类变量，只打印一次
    
    def get_xyz_t(self, t):
        """
        Return deformed xyz at time t. If t is None, fall back to canonical xyz.
        We assume single-frame render (B=1) for now.
        """
        if t is None or self._w_pos.numel() == 0:
            return self._xyz

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self._xyz.device, dtype=self._xyz.dtype)
        else:
            t = t.to(device=self._xyz.device, dtype=self._xyz.dtype)

        phi = self._time_phi(t)[0]  # [K]
        # w_pos: [N,K,3], phi: [K] => delta: [N,3]
        delta = torch.einsum("k,nkc->nc", phi, self._w_pos)
        
        # 时间形变信息已在backend中输出，这里不再单独打印
        # if not GaussianModel._time_deform_logged:
        #     print(f"[TimeDeform] using time varied gaussian center: t={t.item():.4f}, N={self._xyz.shape[0]}, delta_norm={delta.norm().item():.6f}")
        #     GaussianModel._time_deform_logged = True
        
        return self._xyz + delta

    def get_deformed_attributes_t(self, t):
        """
        返回时间t时的所有形变属性（位置、旋转、放缩、不透明度）
        
        Args:
            t: 时间值，应该在[0,1]范围内，可以是标量或tensor
            
        Returns:
            tuple: (xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed)
        """
        # 如果没有时间或形变参数未初始化，返回原始值
        if t is None or self._w_pos.numel() == 0:
            return self._xyz, self.get_rotation, self.get_scaling, self.get_opacity

        # 确保t是tensor格式
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self._xyz.device, dtype=self._xyz.dtype)
        else:
            t = t.to(device=self._xyz.device, dtype=self._xyz.dtype)

        # 计算时间基函数权重
        phi = self._time_phi(t)[0]  # [K]
        
        # 检查是否使用形变点选择
        use_deformation_table = (
            hasattr(self, '_deformation_table') and 
            self._deformation_table.numel() > 0 and
            self._deformation_table.shape[0] == self._xyz.shape[0]
        )
        
        if use_deformation_table:
            # 只对标记的点计算形变（EH-SurGS策略）
            deformation_mask = self._deformation_table
            
            # 初始化结果（所有点保持原值）
            xyz_deformed = self._xyz.clone()
            # 获取原始值（使用属性访问，然后clone）
            rotation_deformed = self.rotation_activation(self._rotation).clone()
            scaling_deformed = self.scaling_activation(self._scaling).clone()
            opacity_deformed = self.opacity_activation(self._opacity).clone()
            
            # 只对标记的点计算形变
            if deformation_mask.any() and self._w_pos.numel() > 0:
                # 位置形变
                delta_pos = torch.zeros_like(self._xyz)
                delta_pos[deformation_mask] = torch.einsum("k,nkc->nc", phi, self._w_pos[deformation_mask])
                xyz_deformed = xyz_deformed + delta_pos
                
                # 旋转形变
                if self._w_rot.numel() > 0:
                    delta_rot = torch.zeros_like(self._rotation)
                    delta_rot[deformation_mask] = torch.einsum("k,nkc->nc", phi, self._w_rot[deformation_mask])
                    rotation_deformed = self.rotation_activation(self._rotation + delta_rot)
                
                # 放缩形变
                if self._w_scale.numel() > 0:
                    delta_scale = torch.zeros_like(self._scaling)
                    delta_scale[deformation_mask] = torch.einsum("k,nkc->nc", phi, self._w_scale[deformation_mask])
                    scaling_deformed = self.scaling_activation(self._scaling + delta_scale)
                
                # 不透明度形变
                if self._w_opacity.numel() > 0:
                    delta_opacity = torch.zeros_like(self._opacity)
                    w_opacity_flat = self._w_opacity.squeeze(-1)  # [N, K]
                    delta_opacity[deformation_mask] = torch.einsum("k,nk->n", phi, w_opacity_flat[deformation_mask]).unsqueeze(-1)
                    opacity_deformed = self.opacity_activation(self._opacity + delta_opacity)
        else:
            # 所有点都计算形变（向后兼容）
            # 位置形变
            if self._w_pos.numel() > 0:
                delta_pos = torch.einsum("k,nkc->nc", phi, self._w_pos)
                xyz_deformed = self._xyz + delta_pos
            else:
                xyz_deformed = self._xyz
            
            # 旋转形变 (quaternion)
            if self._w_rot.numel() > 0:
                delta_rot = torch.einsum("k,nkc->nc", phi, self._w_rot)
                rotation_deformed = self.rotation_activation(self._rotation + delta_rot)
            else:
                rotation_deformed = self.get_rotation
            
            # 放缩形变
            if self._w_scale.numel() > 0:
                delta_scale = torch.einsum("k,nkc->nc", phi, self._w_scale)
                scaling_deformed = self.scaling_activation(self._scaling + delta_scale)
            else:
                scaling_deformed = self.get_scaling
            
            # 不透明度形变
            if self._w_opacity.numel() > 0:
                delta_opacity = torch.einsum("k,nk->n", phi, self._w_opacity.squeeze(-1)).unsqueeze(-1)
                opacity_deformed = self.opacity_activation(self._opacity + delta_opacity)
            else:
                opacity_deformed = self.get_opacity
        
        return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed


    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    # Process input camera info and image data, then call create_pcd_from_image_and_depth to generate point cloud
    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b    
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)
    # Create point cloud from RGB and depth, apply random downsampling, store as BasicPointCloud, and complete 3DGS initialization
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()     
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())   
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        pts = np.asarray(pcd.points)

        # 使用纯 PyTorch 实现的 KNN 距离计算（避免 distCUDA2 在多进程中的问题）
        # 使用 K=3 最近邻的平均距离，与 distCUDA2 行为一致
        device = torch.device("cuda:0")
        pts_t = torch.from_numpy(pts).float().to(device).contiguous()
        
        N = pts_t.shape[0]
        K = 3  # 与 distCUDA2 一致，使用 3 个最近邻
        avg_knn_dists_sq = torch.zeros(N, device=device)
        batch_size = 1024  # 分批计算以节省内存
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_pts = pts_t[i:end_i]
            diff = batch_pts.unsqueeze(1) - pts_t.unsqueeze(0)
            dists_sq = (diff ** 2).sum(dim=-1)  # [batch, N]
            # 将自身距离设为无穷大
            for j in range(end_i - i):
                dists_sq[j, i + j] = float('inf')
            # 取 K 个最小距离的平均值
            topk_dists, _ = torch.topk(dists_sq, K, dim=1, largest=False)
            avg_knn_dists_sq[i:end_i] = topk_dists.mean(dim=1)
        
        dist2 = torch.clamp_min(avg_knn_dists_sq, 1e-7) * point_size

        # 确保 dist2 有合理的值，避免 log 产生 -inf
        dist2 = torch.clamp(dist2, min=1e-7, max=1e6)
        scales = torch.log(torch.sqrt(dist2))[..., None]
        scales = torch.clamp(scales, min=-10, max=10)
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(         
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities
    
    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        
        # 初始化形变点选择参数（如果还没有初始化）
        if self._deformation_table.numel() == 0:
            N = new_xyz.shape[0]
            self._deformation_table = torch.ones(N, dtype=torch.bool, device="cuda")
            self._deformation_accum = torch.zeros(N, device="cuda")
        
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        
        # ========== 形变参数优化器配置 ==========
        # 形变参数需要单独的学习率，通常比位置参数小10倍
        # 原因：形变是细微的变化，如果学习率太大，会导致训练不稳定
        
        # 获取形变学习率：优先使用配置文件中的deformation_lr_init
        # 如果配置文件中没有，就使用位置学习率的10%（更保守）
        deformation_lr = getattr(
            training_args, 
            "deformation_lr_init",  # 如果配置了就用这个
            training_args.position_lr_init * 0.1  # 否则用位置学习率的10%
        ) * self.spatial_lr_scale  # 乘以空间缩放因子
        
        # 将形变参数添加到优化器配置列表中
        l.extend([
            {
                "params": [self._w_pos],
                "lr": deformation_lr,
                "name": "w_pos",  # 位置形变参数
            },
            {
                "params": [self._w_rot],
                "lr": deformation_lr,
                "name": "w_rot",  # 旋转形变参数
            },
            {
                "params": [self._w_scale],
                "lr": deformation_lr,
                "name": "w_scale",  # 放缩形变参数
            },
            {
                "params": [self._w_opacity],
                "lr": deformation_lr,
                "name": "w_opacity",  # 不透明度形变参数
            },
            {
                "params": [self.t_mu],
                "lr": 0.0,  # 暂时关闭训练
                "name": "t_mu",
            },
            {
                "params": [self.t_sigma_raw],
                "lr": 0.0,  # 暂时关闭训练
                "name": "t_sigma",
            },
        ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_ply_at_time(self, path, t):
        """
        保存指定时刻t的形变后模型为PLY文件
        
        Args:
            path: 保存路径
            t: 时间值，应该在[0,1]范围内
        """
        mkdir_p(os.path.dirname(path))
        
        # 获取形变后的属性（这些是激活后的值）
        xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed = \
            self.get_deformed_attributes_t(t)
        
        # 转换为numpy
        xyz = xyz_deformed.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        
        # 注意：PLY文件保存的是原始参数空间的值，不是激活后的值
        # 所以需要将激活后的值转换回原始空间
        # opacity: sigmoid激活 -> 需要inverse_sigmoid
        opacities = self.inverse_opacity_activation(opacity_deformed).detach().cpu().numpy()
        
        # scaling: exp激活 -> 需要log
        scale = self.scaling_inverse_activation(scaling_deformed).detach().cpu().numpy()
        
        # rotation: 已经是归一化的四元数，可以直接使用
        # 但为了保持一致性，我们使用形变后的旋转值
        rotation = rotation_deformed.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        
    def save_deformation_params(self, path):
        """
        保存形变参数（权重和时间基函数参数）为.npz文件
        
        Args:
            path: 保存路径（.npz文件）
        """
        mkdir_p(os.path.dirname(path))
        
        params = {
            'w_pos': self._w_pos.detach().cpu().numpy() if self._w_pos.numel() > 0 else None,
            'w_rot': self._w_rot.detach().cpu().numpy() if self._w_rot.numel() > 0 else None,
            'w_scale': self._w_scale.detach().cpu().numpy() if self._w_scale.numel() > 0 else None,
            'w_opacity': self._w_opacity.detach().cpu().numpy() if self._w_opacity.numel() > 0 else None,
            't_mu': self.t_mu.detach().cpu().numpy(),
            't_sigma': F.softplus(self.t_sigma_raw).detach().cpu().numpy(),
            'K_time': self.K_time,
        }
        
        np.savez(path, **params)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        # 初始化 per-point 位移系数，默认 0 -> 退化成静态
        self._w_pos = nn.Parameter(
            torch.zeros((self._xyz.shape[0], self.K_time, 3), device="cuda", dtype=torch.float).requires_grad_(True)
        )
        # 初始化其他形变参数
        self._w_rot = nn.Parameter(
            torch.zeros((self._xyz.shape[0], self.K_time, 4), device="cuda", dtype=torch.float).requires_grad_(True)
        )
        self._w_scale = nn.Parameter(
            torch.zeros((self._xyz.shape[0], self.K_time, 3), device="cuda", dtype=torch.float).requires_grad_(True)
        )
        self._w_opacity = nn.Parameter(
            torch.zeros((self._xyz.shape[0], self.K_time, 1), device="cuda", dtype=torch.float).requires_grad_(True)
        )


        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        N = int(mask.numel())


        for group in self.optimizer.param_groups:
            p = group["params"][0]
            name = group.get("name", "NONAME")

            # 只裁剪 per-point 参数：第0维必须等于 N
            # 例如 time_basis 是 (3,) -> p.shape[0]=3 != N(=0 or 当前点数)，应跳过
            is_per_point = (p.ndim >= 1 and p.shape[0] == N)

            if not is_per_point:
                # 全局参数：不参与 prune，也不改 optimizer state
                optimizable_tensors[name] = p
                continue

            stored_state = self.optimizer.state.get(p, None)
            if stored_state is not None:
                # Adam 的一阶/二阶动量也要同步裁剪
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 重要：先删旧 key，再用新 Parameter 作为 key
                del self.optimizer.state[p]
                new_p = nn.Parameter(p[mask].detach(), requires_grad=True)
                group["params"][0] = new_p
                self.optimizer.state[new_p] = stored_state
                optimizable_tensors[name] = new_p
            else:
                new_p = nn.Parameter(p[mask].detach(), requires_grad=True)
                group["params"][0] = new_p
                optimizable_tensors[name] = new_p

        return optimizable_tensors


    def prune_points(self, mask):
        """
        删除不必要的高斯点（比如太透明或太大的点）
        同时也要删除这些点的形变参数，保持数据一致性
        """
        valid_points_mask = ~mask  # mask标记要删除的点，~mask就是保留的点
        
        # 确保mask的长度与当前点数匹配
        current_num_points = self._xyz.shape[0]
        if mask.shape[0] != current_num_points:
            # 如果mask长度不匹配，调整mask（这种情况不应该发生，但为了安全）
            if mask.shape[0] < current_num_points:
                # mask太短，补齐（假设新增的点不需要prune）
                padding = torch.zeros(current_num_points - mask.shape[0], dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, padding])
                valid_points_mask = ~mask
            else:
                # mask太长，截断
                mask = mask[:current_num_points]
                valid_points_mask = ~mask
        
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # ========== 更新形变参数 ==========
        # 删除被prune的点的形变参数，只保留有效点的参数
        # 这样优化器中的参数和实际数据保持一致
        self._w_pos = optimizable_tensors["w_pos"]
        self._w_rot = optimizable_tensors["w_rot"]
        self._w_scale = optimizable_tensors["w_scale"]
        self._w_opacity = optimizable_tensors["w_opacity"]

        # ========== 更新其他参数 ==========
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        
        # 裁剪形变点选择相关参数
        # 确保尺寸匹配：在prune之前，_deformation_table的长度应该等于mask的长度
        # prune之后，长度应该等于valid_points_mask.sum()
        expected_num_points_after_prune = valid_points_mask.sum().item()
        
        # 检查尺寸匹配
        if self._deformation_table.numel() > 0:
            if self._deformation_table.shape[0] == mask.shape[0]:
                # 尺寸匹配，正常裁剪
                self._deformation_table = self._deformation_table[valid_points_mask]
            else:
                # 尺寸不匹配，重新初始化（这种情况不应该发生，但为了安全）
                self._deformation_table = torch.ones(expected_num_points_after_prune, dtype=torch.bool, device="cuda")
        else:
            # 如果未初始化，初始化
            self._deformation_table = torch.ones(expected_num_points_after_prune, dtype=torch.bool, device="cuda")
            
        if self._deformation_accum.numel() > 0:
            if self._deformation_accum.shape[0] == mask.shape[0]:
                # 尺寸匹配，正常裁剪
                self._deformation_accum = self._deformation_accum[valid_points_mask]
            else:
                # 尺寸不匹配，重新初始化
                self._deformation_accum = torch.zeros(expected_num_points_after_prune, device="cuda")
        else:
            # 如果未初始化，初始化
            self._deformation_accum = torch.zeros(expected_num_points_after_prune, device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            
            # 跳过不在 tensors_dict 中的全局参数（如 t_mu, t_sigma）
            if name not in tensors_dict:
                optimizable_tensors[name] = group["params"][0]
                continue
                
            extension_tensor = tensors_dict[name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        """
        当系统创建新的高斯点时（比如从图像中提取新点，或者分裂/克隆现有点），
        需要为这些新点初始化所有参数，包括形变参数
        
        这个函数的作用：把新点的参数添加到优化器中，让它们可以被训练
        """
        # new_xyz: [M,3] M是新点的数量
        M = new_xyz.shape[0]
        
        # ========== 初始化新点的形变参数 ==========
        # 新点的形变参数都初始化为0，意味着：
        # 1. 新点默认是静态的（不随时间变化）
        # 2. 训练过程中，如果发现这个点需要动，优化器会自动调整这些参数
        # 3. 这样初始化是安全的，不会破坏现有场景
        
        # 位置形变权重：新点默认不移动
        new_w_pos = torch.zeros((M, self.K_time, 3), device="cuda", dtype=new_xyz.dtype)
        
        # 旋转形变权重：新点默认不旋转
        new_w_rot = torch.zeros((M, self.K_time, 4), device="cuda", dtype=new_xyz.dtype)
        
        # 放缩形变权重：新点默认大小不变
        new_w_scale = torch.zeros((M, self.K_time, 3), device="cuda", dtype=new_xyz.dtype)
        
        # 不透明度形变权重：新点默认透明度不变
        new_w_opacity = torch.zeros((M, self.K_time, 1), device="cuda", dtype=new_xyz.dtype)
        
        # ========== 初始化新点的形变点选择参数 ==========
        # 新点默认都标记为需要形变（True），这样它们可以学习形变
        # 如果后续发现它们是静态的，update_deformation_table会更新标记
        new_deformation_table = torch.ones(M, dtype=torch.bool, device="cuda")
        new_deformation_accum = torch.zeros(M, device="cuda", dtype=new_xyz.dtype)
        
        # 将新点的形变点选择参数添加到现有参数中
        if self._deformation_table.numel() > 0:
            self._deformation_table = torch.cat([self._deformation_table, new_deformation_table])
        else:
            self._deformation_table = new_deformation_table
            
        if self._deformation_accum.numel() > 0:
            self._deformation_accum = torch.cat([self._deformation_accum, new_deformation_accum])
        else:
            self._deformation_accum = new_deformation_accum

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "w_pos": new_w_pos,
            "w_rot": new_w_rot,
            "w_scale": new_w_scale,
            "w_opacity": new_w_opacity,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._w_pos = optimizable_tensors["w_pos"]
        self._w_rot = optimizable_tensors["w_rot"]
        self._w_scale = optimizable_tensors["w_scale"]
        self._w_opacity = optimizable_tensors["w_opacity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        # 构建prune_filter：标记要删除的点
        # selected_pts_mask标记分裂前的点中哪些要删除（分裂的点要删除）
        # 新添加的点不需要删除（标记为False）
        num_new_points = N * selected_pts_mask.sum()
        prune_filter = torch.cat(
            (
                selected_pts_mask,  # 分裂前的点：选中的点要删除
                torch.zeros(num_new_points, device="cuda", dtype=bool),  # 新点：不删除
            )
        )
        
        # 确保prune_filter的长度与当前点数匹配（densification_postfix后点数已增加）
        current_num_points = self._xyz.shape[0]
        if prune_filter.shape[0] != current_num_points:
            # 如果长度不匹配，调整（这种情况不应该发生，但为了安全）
            if prune_filter.shape[0] < current_num_points:
                # prune_filter太短，补齐（假设新增的点不需要prune）
                padding = torch.zeros(current_num_points - prune_filter.shape[0], dtype=torch.bool, device="cuda")
                prune_filter = torch.cat([prune_filter, padding])
            else:
                # prune_filter太长，截断
                prune_filter = prune_filter[:current_num_points]

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
    
    @torch.no_grad()
    def update_deformation_table(self, threshold=0.01):
        """
        根据形变累积量更新形变表（参考EH-SurGS）
        
        只有形变累积量超过阈值的点才会被标记为需要形变。
        这样可以自动识别静态点，节省计算资源。
        
        Args:
            threshold: 形变阈值，只有形变量超过此值的点才会被标记为需要形变
        """
        if self._deformation_accum.numel() == 0:
            return
        
        if self._deformation_accum.shape[0] != self._xyz.shape[0]:
            # 如果尺寸不匹配，重新初始化
            self._deformation_accum = torch.zeros(self._xyz.shape[0], device=self._xyz.device)
            self._deformation_table = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            return
        
        # 计算每个点的最大形变量（参考EH-SurGS的实现）
        # EH-SurGS使用: max_deform = _deformation_accum.max(dim=-1).values / 100
        # 这里简化处理，直接使用累积的形变量
        max_deform = self._deformation_accum
        
        # 只有形变量超过阈值的点才需要形变
        self._deformation_table = max_deform > threshold
        
        # 可选：输出统计信息
        num_deform_points = self._deformation_table.sum().item()
        total_points = self._deformation_table.shape[0]
        if total_points > 0:
            deform_ratio = num_deform_points / total_points
            # 只在形变点比例变化较大时输出（避免输出过多）
            if not hasattr(self, '_last_deform_ratio') or abs(deform_ratio - self._last_deform_ratio) > 0.05:
                from utils.logging_utils import Log
                Log(f"[DeformTable] Updated: {num_deform_points}/{total_points} points need deformation ({deform_ratio*100:.1f}%)", tag="Deform")
                self._last_deform_ratio = deform_ratio

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def capture(self):
        """
        保存模型状态，用于checkpoint（参考EH-SurGS的实现）
        
        Returns:
            dict: 包含所有模型参数和训练状态的字典
        """
        return {
            'active_sh_degree': self.active_sh_degree,
            'max_sh_degree': self.max_sh_degree,
            '_xyz': self._xyz,
            '_features_dc': self._features_dc,
            '_features_rest': self._features_rest,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_opacity': self._opacity,
            '_w_pos': self._w_pos,
            '_w_rot': self._w_rot,
            '_w_scale': self._w_scale,
            '_w_opacity': self._w_opacity,
            't_mu': self.t_mu,
            't_sigma_raw': self.t_sigma_raw,
            'K_time': self.K_time,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'unique_kfIDs': self.unique_kfIDs,
            'n_obs': self.n_obs,
            'percent_dense': self.percent_dense,
            'spatial_lr_scale': getattr(self, 'spatial_lr_scale', 1.0),
            'lr_init': getattr(self, 'lr_init', None),
            'lr_final': getattr(self, 'lr_final', None),
            'lr_delay_mult': getattr(self, 'lr_delay_mult', None),
            'max_steps': getattr(self, 'max_steps', None),
            # 形变点选择参数
            '_deformation_table': getattr(self, '_deformation_table', torch.empty(0, dtype=torch.bool, device="cuda")),
            '_deformation_accum': getattr(self, '_deformation_accum', torch.empty(0, device="cuda")),
        }

    def restore(self, checkpoint_dict, training_args=None):
        """
        从checkpoint恢复模型状态（参考EH-SurGS的实现）
        
        Args:
            checkpoint_dict: 从capture()保存的字典
            training_args: 训练参数（可选，用于重新设置优化器）
        """
        self.active_sh_degree = checkpoint_dict['active_sh_degree']
        self.max_sh_degree = checkpoint_dict.get('max_sh_degree', self.max_sh_degree)
        
        # 恢复模型参数
        self._xyz = checkpoint_dict['_xyz']
        self._features_dc = checkpoint_dict['_features_dc']
        self._features_rest = checkpoint_dict['_features_rest']
        self._scaling = checkpoint_dict['_scaling']
        self._rotation = checkpoint_dict['_rotation']
        self._opacity = checkpoint_dict['_opacity']
        
        # 恢复形变参数（如果存在）
        if '_w_pos' in checkpoint_dict:
            self._w_pos = checkpoint_dict['_w_pos']
        if '_w_rot' in checkpoint_dict:
            self._w_rot = checkpoint_dict['_w_rot']
        if '_w_scale' in checkpoint_dict:
            self._w_scale = checkpoint_dict['_w_scale']
        if '_w_opacity' in checkpoint_dict:
            self._w_opacity = checkpoint_dict['_w_opacity']
        if 't_mu' in checkpoint_dict:
            self.t_mu = checkpoint_dict['t_mu']
        if 't_sigma_raw' in checkpoint_dict:
            self.t_sigma_raw = checkpoint_dict['t_sigma_raw']
        if 'K_time' in checkpoint_dict:
            self.K_time = checkpoint_dict['K_time']
        
        # 恢复形变点选择参数（如果存在）
        if '_deformation_table' in checkpoint_dict:
            self._deformation_table = checkpoint_dict['_deformation_table']
        elif self._xyz.shape[0] > 0:
            # 如果checkpoint中没有，但有点存在，初始化为全True
            self._deformation_table = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
            
        if '_deformation_accum' in checkpoint_dict:
            self._deformation_accum = checkpoint_dict['_deformation_accum']
        elif self._xyz.shape[0] > 0:
            # 如果checkpoint中没有，但有点存在，初始化为全0
            self._deformation_accum = torch.zeros(self._xyz.shape[0], device="cuda")
        
        # 恢复训练状态
        self.max_radii2D = checkpoint_dict['max_radii2D']
        self.xyz_gradient_accum = checkpoint_dict['xyz_gradient_accum']
        self.denom = checkpoint_dict['denom']
        self.unique_kfIDs = checkpoint_dict['unique_kfIDs']
        self.n_obs = checkpoint_dict['n_obs']
        
        # 恢复超参数
        if 'percent_dense' in checkpoint_dict:
            self.percent_dense = checkpoint_dict['percent_dense']
        if 'spatial_lr_scale' in checkpoint_dict:
            self.spatial_lr_scale = checkpoint_dict['spatial_lr_scale']
        if 'lr_init' in checkpoint_dict:
            self.lr_init = checkpoint_dict['lr_init']
        if 'lr_final' in checkpoint_dict:
            self.lr_final = checkpoint_dict['lr_final']
        if 'lr_delay_mult' in checkpoint_dict:
            self.lr_delay_mult = checkpoint_dict['lr_delay_mult']
        if 'max_steps' in checkpoint_dict:
            self.max_steps = checkpoint_dict['max_steps']
        
        # 恢复优化器状态（如果提供了training_args）
        if training_args is not None and checkpoint_dict.get('optimizer') is not None:
            # 重新设置优化器
            self.training_setup(training_args)
            # 加载优化器状态
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
