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

        # ========== 形变表相关参数（参考EH-SurGS）==========
        # _deformation_table: 布尔表，标记哪些点需要形变 [N] bool
        # True 表示该点需要形变，False 表示静态点（不需要形变）
        # 用于优化：只对需要形变的点计算形变，节省计算资源
        self._deformation_table = torch.empty(0, dtype=torch.bool, device="cuda")
        # _deformation_accum: 累积形变量 [N, 3] (x, y, z 三个方向的累积形变量)
        # 用于动态更新 deformation_table：只有形变量超过阈值的点才需要形变
        # 形状 [N, 3] 与 EH-SurGS 一致
        self._deformation_accum = torch.empty(0, device="cuda")
        # deform_table_threshold: 形变表阈值，从配置读取或使用默认值
        # 默认值从0.01降低到0.0001，让更多点参与形变以改善建模效果
        if config is not None and "deform_table_threshold" in config.get("model_params", {}):
            self.deform_table_threshold = config["model_params"]["deform_table_threshold"]
        else:
            self.deform_table_threshold = 0.0001  # 默认值：比原0.01小100倍

        # ========== 时间形变相关参数（生命周期机制，参考EH-SurGS）==========
        # K_time: 时间基函数的数量（basis_num）
        # EH-SurGS使用17-20个基函数，我们默认使用17个（与EH-SurGS默认值一致）
        # 更多基函数可以表达更复杂的时间变化，但会增加内存和计算开销
        # 可以通过config中的time_basis_num参数自定义，如果没有配置则使用默认值17
        if config is not None and "time_basis_num" in config.get("model_params", {}):
            self.K_time = config["model_params"]["time_basis_num"]
        else:
            self.K_time = 17  # 默认值：与EH-SurGS的curve_num=17一致
        
        # ch_num: 形变通道数
        # 位置(3) + 旋转(4) + 放缩(3) + 不透明度(1) = 11
        # 如果只使用位置形变，则为3
        if config is not None and "deform_ch_num" in config.get("model_params", {}):
            self.ch_num = config["model_params"]["deform_ch_num"]
        else:
            self.ch_num = 11  # 默认值：支持所有形变类型
        
        # ========== 形变系数参数（生命周期机制）==========
        # _coefs: 形变系数 [N个点, ch_num个通道, 3个参数(weights/means/std_devs), K_time个基函数]
        # 这是EH-SurGS的核心：每个点、每个通道、每个基函数都有自己的时间演化参数
        # - weights: 形变权重
        # - means: 时间中心位置（点特定的生命周期）
        # - std_devs: 时间影响范围（点特定的生命周期）
        # 形状: [N, ch_num, 3, K_time]
        self._coefs = torch.empty(0, device="cuda")
        
        # 为了向后兼容，保留旧的参数名（但不再使用）
        # 这些将在后续版本中移除
        self._w_pos = torch.empty(0, device="cuda")
        self._w_rot = torch.empty(0, device="cuda")
        self._w_scale = torch.empty(0, device="cuda")
        self._w_opacity = torch.empty(0, device="cuda")
        

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
        
        # ========== 轮廓提取相关参数 ==========
        # _edge_mask: 边缘点标记 [N] bool
        # True 表示该点是边缘点，False 表示非边缘点
        # 用于PnP位姿估计时只使用边缘点，提高鲁棒性
        self._edge_mask = torch.empty(0, dtype=torch.bool, device="cuda")
        # _edge_points: 边缘点云 [M, 3] (M <= N)
        # 存储提取出的边缘点，用于PnP
        # 注意：现在会累积所有关键帧的边缘点，而不是只保留最后一帧
        self._edge_points = None  # numpy array, 形状 [M, 3]
        self._all_edge_points = []  # 用于累积所有关键帧的边缘点
        # 是否启用轮廓提取
        self.enable_edge_extraction = False
        if config is not None and "enable_edge_extraction" in config.get("model_params", {}):
            self.enable_edge_extraction = config["model_params"]["enable_edge_extraction"]
        
        # 输出轮廓提取配置状态
        if self.enable_edge_extraction:
            print("[轮廓提取] 已启用 - 将在点云创建时提取边缘点（累积模式）")
        else:
            print("[轮廓提取] 未启用 - 如需启用，请在配置文件中设置 model_params.enable_edge_extraction: true")

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
        # 注意：这个方法保留用于向后兼容，但新的生命周期机制使用gaussian_deformation
        """
        if t.dim() == 0:
            t = t.view(1)
        t = t.view(-1, 1)  # [B,1]
        mu = self.t_mu.view(1, -1)  # [1,K]
        sigma = F.softplus(self.t_sigma_raw).view(1, -1) + 1e-6
        phi = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)  # [B,K]
        return phi
    
    def gaussian_deformation(self, time, deformation_point, num_gaussians, ch_num=None, basis_num=None):
        """
        ========== 核心形变计算方法：生命周期机制（参考EH-SurGS）==========
        
        这个方法实现了基于生命周期的高斯形变模型，每个点都有自己的"生命周期"参数，
        控制它在不同时间点的形变行为。这是整个形变系统的核心。
        
        【生命周期机制原理】
        - 每个点、每个通道（位置/旋转/放缩/不透明度）都有K_time个高斯基函数
        - 每个基函数有三个参数：
          * weights: 形变权重，控制该基函数的贡献大小
          * means: 时间中心位置，表示该基函数的"活跃时间"（点特定的生命周期）
          * std_devs: 时间影响范围，表示该基函数在时间上的影响宽度
        
        【形变通道说明】
        ch_num个通道的含义（默认11个）：
        - 通道0-2: 位置形变（平移）[x, y, z]
        - 通道3-6: 旋转形变（四元数）[w, x, y, z]
        - 通道7-9: 放缩形变 [sx, sy, sz]
        - 通道10: 不透明度形变 [opacity]
        
        【计算流程】
        1. 从_coefs中提取每个点的形变系数（weights, means, std_devs）
        2. 对每个基函数，计算高斯函数值：exp(-((time - means) / std_devs)^2)
        3. 将高斯函数值与权重相乘，然后对所有基函数求和
        4. 得到最终的形变值（每个通道一个值）
        
        Args:
            time: 时间值，标量或tensor，应该在[0,1]范围内
                 例如：0.0表示序列开始，1.0表示序列结束
            deformation_point: 布尔mask，标记哪些点需要计算形变 [N]
                               True表示该点需要形变，False表示跳过计算
            num_gaussians: 需要形变的点的数量（用于兼容性，实际从deformation_point计算）
            ch_num: 通道数（如果None则使用self.ch_num，默认11）
            basis_num: 基函数数量（如果None则使用self.K_time，默认17）
            
        Returns:
            deformations: 形变值 [num_deform_points, ch_num]
                         - 前3个通道是位置形变（平移）
                         - 第4-7个通道是旋转形变（四元数）
                         - 第8-10个通道是放缩形变
                         - 第11个通道是不透明度形变
        """
        # ========== 边界情况处理 ==========
        if self._coefs.numel() == 0:
            # 如果没有初始化coefs（形变参数未初始化），返回零形变
            # 这意味着所有点保持原始状态，不进行任何形变
            num_deform = deformation_point.sum().item() if isinstance(deformation_point, torch.Tensor) else num_gaussians
            ch = ch_num if ch_num is not None else self.ch_num
            return torch.zeros((num_deform, ch), device="cuda")
        
        # ========== 参数初始化 ==========
        if ch_num is None:
            ch_num = self.ch_num  # 默认11：位置(3) + 旋转(4) + 放缩(3) + 不透明度(1)
        if basis_num is None:
            basis_num = self.K_time  # 默认17个基函数（与EH-SurGS一致）
        
        # ========== 时间参数处理 ==========
        # 确保time是tensor格式，并移动到正确的设备
        if not torch.is_tensor(time):
            time = torch.tensor(time, device=self._coefs.device, dtype=self._coefs.dtype)
        else:
            time = time.to(device=self._coefs.device, dtype=self._coefs.dtype)
        
        # 确保time是标量（单个时间值）
        # 如果输入是tensor，取第一个值（支持批量处理，但这里只处理单个时间）
        if time.dim() > 0:
            time = time.view(-1)[0]  # 取第一个值，转换为标量
        
        # ========== 数据一致性检查 ==========
        num_points = len(self._xyz)
        # 检查_coefs的形状是否正确：[N, ch_num * 3 * basis_num]
        expected_coefs_size = num_points * ch_num * 3 * basis_num
        if self._coefs.numel() != expected_coefs_size:
            # 如果_coefs的大小不匹配，说明点数不一致（可能是densification后未更新）
            # 输出警告信息（使用颜色输出）
            if not hasattr(self, '_coefs_mismatch_warned') or not self._coefs_mismatch_warned:
                from utils.logging_utils import Log
                Log(f"形变系数大小不匹配，已自动重新初始化 (点数: {num_points})", tag="WARNING")
                self._coefs_mismatch_warned = True
            # 返回零形变，避免计算错误
            num_deform = deformation_point.sum().item() if isinstance(deformation_point, torch.Tensor) else num_gaussians
            return torch.zeros((num_deform, ch_num), device=self._coefs.device, dtype=self._coefs.dtype)
        
        # ========== 提取形变系数 ==========
        # 重塑coefs: [N, ch_num * 3 * basis_num] -> [N, ch_num, 3, basis_num]
        # 其中3表示：weights(0), means(1), std_devs(2)
        coefficients = self._coefs.reshape(num_points, ch_num, 3, basis_num).contiguous()
        
        # 只选择需要形变的点（deformation_point标记的点）
        coefficients = coefficients[deformation_point, :, :, :]
        # coefficients形状: [num_deform, ch_num, 3, basis_num]
        # num_deform = deformation_point.sum()
        
        # ========== 分解形变参数 ==========
        # 将coefficients分解为三个部分：weights, means, std_devs
        weights, means, std_devs = torch.chunk(coefficients, 3, dim=-2)
        # weights: [num_deform, ch_num, 1, basis_num] - 形变权重
        # means: [num_deform, ch_num, 1, basis_num] - 时间中心位置（生命周期中心）
        # std_devs: [num_deform, ch_num, 1, basis_num] - 时间影响范围（生命周期宽度）
        
        # ========== 计算高斯基函数值（生命周期机制的核心）==========
        # 对每个基函数，计算它在当前时间点的激活值
        # 公式：gaussian = exp(-((time - means) / std_devs)^2)
        # 
        # 物理意义：
        # - means: 该基函数的"活跃时间"，例如means=0.5表示在序列中间最活跃
        # - std_devs: 该基函数的"影响范围"，std_devs越大，影响的时间范围越广
        # - 当time接近means时，gaussian值接近1（最大激活）
        # - 当time远离means时，gaussian值接近0（几乎不激活）
        #
        # 1e-4是为了避免std_devs为0时除零错误
        exponent = (time - means) ** 2 / (std_devs ** 2 + 1e-4)
        gaussians = torch.exp(-exponent ** 2)  # [num_deform, ch_num, 1, basis_num]
        # gaussians值域：[0, 1]，表示每个基函数在当前时间的激活程度
        
        # ========== 加权求和得到最终形变 ==========
        # 将每个基函数的高斯值与其权重相乘，然后对所有基函数求和
        # deformations = sum_k (gaussians[k] * weights[k])，k从0到basis_num-1
        # gaussians * weights: [num_deform, ch_num, 1, basis_num]
        # sum(-1): [num_deform, ch_num, 1] (对最后一个维度求和，移除 basis_num 维度)
        # squeeze(-1): [num_deform, ch_num] (移除最后一个维度，即维度2)
        weighted_sum = (gaussians * weights).sum(-1)  # [num_deform, ch_num, 1]
        deformations = weighted_sum.squeeze(-1)  # [num_deform, ch_num] (移除最后一个维度)
        
        # 确保 deformations 是二维的 [num_deform, ch_num]
        # 如果 squeeze 移除了错误的维度（例如当 num_deform=1 或 ch_num=1 时），需要恢复
        if deformations.dim() == 1:
            # 如果变成一维了，可能是 squeeze 移除了错误的维度
            # 需要根据实际情况恢复正确的形状
            num_deform = deformation_point.sum().item() if isinstance(deformation_point, torch.Tensor) else num_gaussians
            if num_deform == 1:
                # 如果只有一个点，reshape为 [1, ch_num]
                deformations = deformations.unsqueeze(0)  # [ch_num] -> [1, ch_num]
            elif deformations.shape[0] == num_deform:
                # 如果第一维是正确的，说明 ch_num=1，需要添加第二维
                deformations = deformations.unsqueeze(-1)  # [num_deform] -> [num_deform, 1]
            else:
                # 尝试根据 ch_num 恢复
                ch = ch_num if ch_num is not None else self.ch_num
                if deformations.shape[0] % ch == 0:
                    deformations = deformations.view(-1, ch)
                else:
                    # 如果无法恢复，报错
                    raise RuntimeError(f"gaussian_deformation returned 1D tensor with shape {deformations.shape}, "
                                     f"expected 2D tensor [{num_deform}, {ch}]. "
                                     f"deformation_point.sum()={num_deform}, ch_num={ch}")
        elif deformations.dim() == 0:
            # 如果变成标量了，说明有严重问题
            raise RuntimeError(f"gaussian_deformation returned scalar, expected 2D tensor [num_deform, ch_num]")
        elif deformations.dim() > 2:
            # 如果维度太多，尝试squeeze所有大小为1的维度
            deformations = deformations.squeeze()
            if deformations.dim() != 2:
                raise RuntimeError(f"gaussian_deformation returned {deformations.dim()}-D tensor with shape {deformations.shape}, "
                                 f"expected 2D tensor [num_deform, ch_num]")
        
        # Debug: 检查形变计算过程（降低频率，更早输出）
        if not hasattr(self, '_gaussian_deform_debug_count'):
            self._gaussian_deform_debug_count = 0
        self._gaussian_deform_debug_count += 1
        if self._gaussian_deform_debug_count <= 10 or self._gaussian_deform_debug_count % 500 == 0:
            weights_abs_max = weights.abs().max().item()
            weights_abs_mean = weights.abs().mean().item()
            gaussians_max = gaussians.max().item()
            gaussians_mean = gaussians.mean().item()
            weighted_sum_max = (gaussians * weights).abs().max().item()
            weighted_sum_mean = (gaussians * weights).abs().mean().item()
            deform_max = deformations.abs().max().item()
            deform_mean = deformations.abs().mean().item()
            # 移除详细的debug输出，只保留关键警告
            pass  # 已清理：这些详细数值对用户没有直观价值
        
        # 结果解释：
        # - deformations[i, 0:3]: 第i个点的位置形变（平移）[dx, dy, dz]
        # - deformations[i, 3:7]: 第i个点的旋转形变（四元数增量）[dw, dx, dy, dz]
        # - deformations[i, 7:10]: 第i个点的放缩形变 [dsx, dsy, dsz]
        # - deformations[i, 10]: 第i个点的不透明度形变 [dopacity]
        
        return deformations

    _time_deform_logged = False  # 类变量，只打印一次
    
    def get_xyz_t(self, t):
        """
        ========== 获取时间t时的形变位置（平移）==========
        
        这个方法计算并返回在时间t时，所有高斯点的形变后位置。
        它只处理位置形变（平移），不涉及旋转、放缩等其他属性。
        
        【功能说明】
        - 输入：时间t（标量，范围[0,1]）
        - 输出：形变后的3D位置 [N, 3]
        - 公式：xyz_deformed = xyz_canonical + delta_position
              其中 delta_position 是通过生命周期机制计算得到的位置偏移量
        
        【位置形变（平移）原理】
        1. 调用 gaussian_deformation 计算所有通道的形变值
        2. 提取前3个通道（通道0-2），这些是位置形变 [dx, dy, dz]
        3. 将位置偏移量加到原始位置：xyz_new = xyz_old + [dx, dy, dz]
        
        【使用场景】
        - 渲染时获取当前帧的点位置
        - 可视化不同时间点的场景状态
        - 计算位置相关的损失函数
        
        Args:
            t: 时间值，应该在[0,1]范围内
               - None: 返回原始位置（不应用形变）
               - 标量或tensor: 计算该时间点的形变位置
        
        Returns:
            xyz_deformed: 形变后的3D位置 [N, 3]
                         - 如果t为None或形变参数未初始化，返回原始位置
                         - 否则返回 xyz_canonical + position_deformation
        """
        # ========== 边界情况处理 ==========
        if t is None or self._coefs.numel() == 0:
            # 如果没有指定时间，或者形变参数未初始化，返回原始位置
            # 这适用于静态场景或形变系统未启用的情况
            return self._xyz

        # ========== 时间参数处理 ==========
        # 确保t是tensor格式，并移动到正确的设备
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self._xyz.device, dtype=self._xyz.dtype)
        else:
            t = t.to(device=self._xyz.device, dtype=self._xyz.dtype)

        # ========== 计算位置形变（平移）==========
        # 创建全True的deformation_point，表示所有点都需要计算形变
        deformation_point = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        
        # 调用核心形变计算方法，获取所有通道的形变值
        # deform形状: [N, ch_num]，其中ch_num默认是11
        deform = self.gaussian_deformation(t, deformation_point, self._xyz.shape[0], 
                                          ch_num=self.ch_num, basis_num=self.K_time)
        
        # ========== 提取位置形变（前3个通道）==========
        # deform[:, 0:3] 是位置形变，表示每个点在x、y、z方向上的偏移量
        # delta形状: [N, 3]，表示位置偏移 [dx, dy, dz]
        if deform.shape[1] >= 3:
            delta = deform[:, :3]  # [N, 3] - 位置形变（平移）
        else:
            # 如果形变通道数不足3，说明位置形变未启用，返回零偏移
            delta = torch.zeros_like(self._xyz)
        
        # ========== 应用位置形变 ==========
        # 将位置偏移量加到原始位置，得到形变后的位置
        # xyz_deformed = xyz_canonical + delta_position
        # 其中：
        # - xyz_canonical: 原始（规范）位置，在训练过程中学习得到
        # - delta_position: 时间相关的偏移量，通过生命周期机制计算
        return self._xyz + delta

    def get_deformed_attributes_t(self, t):
        """
        返回时间t时的所有形变属性（位置、旋转、放缩、不透明度）
        使用生命周期机制（参考EH-SurGS）
        
        Args:
            t: 时间值，应该在[0,1]范围内，可以是标量或tensor
            
        Returns:
            tuple: (xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed)
        """
        # 如果没有时间或形变参数未初始化，返回原始值
        if t is None or self._coefs.numel() == 0:
            return self._xyz, self.get_rotation, self.get_scaling, self.get_opacity

        # 确保t是tensor格式
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self._xyz.device, dtype=self._xyz.dtype)
        else:
            t = t.to(device=self._xyz.device, dtype=self._xyz.dtype)

        # 检查是否使用形变点选择（deformation_table）
        # 如果启用了 deformation_table，只对标记的点计算形变，节省计算资源
        use_deformation_table = (
            hasattr(self, '_deformation_table') and 
            self._deformation_table.numel() > 0 and
            self._deformation_table.shape[0] == self._xyz.shape[0]
        )
        
        # 确保_xyz的形状是[N, 3]，如果不是则尝试修复
        xyz = self._xyz
        if xyz.dim() != 2 or xyz.shape[1] != 3:
            # 如果形状不对，尝试重塑为[N, 3]
            if xyz.numel() % 3 == 0:
                expected_num_points = xyz.numel() // 3
                xyz = xyz.reshape(expected_num_points, 3)
            else:
                # 如果无法重塑，返回原始值（不应用形变）
                return self._xyz, self.get_rotation, self.get_scaling, self.get_opacity
        
        num_points = xyz.shape[0]
        
        # 根据 deformation_table 决定哪些点需要形变
        if use_deformation_table:
            # 只对标记的点计算形变（EH-SurGS策略）
            deformation_mask = self._deformation_table
        else:
            # 如果没有启用 deformation_table，所有点都计算形变（向后兼容）
            deformation_mask = torch.ones(num_points, dtype=torch.bool, device=xyz.device)
        
        # 初始化结果（所有点保持原值）
        xyz_deformed = xyz.clone()
        rotation_deformed = self.rotation_activation(self._rotation).clone()
        scaling_deformed = self.scaling_activation(self._scaling).clone()
        opacity_deformed = self.opacity_activation(self._opacity).clone()
        
        # 只对标记的点计算形变
        if deformation_mask.any():
            num_deform_points = deformation_mask.sum().item()
            deform = self.gaussian_deformation(t, deformation_mask, num_deform_points, 
                                              ch_num=self.ch_num, basis_num=self.K_time)
            
            # 移除详细的shape调试信息，只保留错误警告
            pass  # 已清理：shape信息只在出错时输出
        else:
            # 如果没有点需要形变，直接返回原始值
            return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed
        
        # 检查deform的形状是否正确
        # deform 的形状应该是 [num_deform_points, ch_num]，其中 num_deform_points = deformation_mask.sum()
        num_deform_points = deformation_mask.sum().item()
        
        # 确保 deform 是二维的 [num_deform_points, ch_num]
        if deform.dim() == 1:
            # 如果是一维的，说明可能只有一个点，需要reshape为 [1, ch_num]
            if num_deform_points == 1:
                deform = deform.unsqueeze(0)  # [ch_num] -> [1, ch_num]
            else:
                # 如果应该有多个点但返回的是一维的，说明有问题
                from utils.logging_utils import Log
                Log(f"形变计算错误：返回了1D张量而非预期形状", tag="WARNING")
                return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed
        
        if deform.shape[0] != num_deform_points:
            # 如果形状不匹配，返回原始值（不应用形变）
            from utils.logging_utils import Log
            Log(f"形变张量形状不匹配，已跳过形变应用", tag="WARNING")
            return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed
        
        # 确保 deform 有足够的通道数
        if deform.shape[1] < 3:
            from utils.logging_utils import Log
            Log(f"形变通道数不足（至少需要3个，当前{deform.shape[1]}），已跳过形变应用", tag="WARNING")
            return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed
        
        # 应用形变（参考EH-SurGS的apply_deformations）
        deformation_config = {
            "xyz_scale": 4.0,
            "rotation_scale": 6.0,
            "opacity_scale": 20.0
        }
        
        # 位置形变（前3个通道）
        if deform.shape[1] >= 3:
            xyz_deform = deform[:, :3] * deformation_config["xyz_scale"]
            # 确保 xyz_deform 的形状是 [num_deform_points, 3]
            if xyz_deform.dim() == 1:
                xyz_deform = xyz_deform.unsqueeze(0)
            # 只对标记的点应用形变
            xyz_deformed[deformation_mask] = xyz_deformed[deformation_mask] + xyz_deform
        
        # 旋转形变（第4-7个通道）
        if deform.shape[1] >= 7:
            rot_deform = deform[:, 3:7] * deformation_config["rotation_scale"]
            rotation_deformed[deformation_mask] = self.rotation_activation(
                self._rotation[deformation_mask] + rot_deform
            )
        
        # 放缩形变（第8-10个通道）
        if deform.shape[1] >= 10:
            scale_deform = deform[:, 7:10]
            scaling_deformed[deformation_mask] = self.scaling_activation(
                self._scaling[deformation_mask] + scale_deform
            )
        
        # 不透明度形变（第11个通道）
        if deform.shape[1] >= 11:
            opacity_deform = deform[:, 10:11] * deformation_config["opacity_scale"]
            opacity_deformed[deformation_mask] = self.opacity_activation(
                self._opacity[deformation_mask] + opacity_deform
            )
        
        return xyz_deformed, rotation_deformed, scaling_deformed, opacity_deformed

    @torch.no_grad()
    def update_deformation_table(self, threshold=None):
        """
        根据形变累积量更新形变表（参考EH-SurGS）
        
        只有形变累积量超过阈值的点才会被标记为需要形变。
        这样可以自动识别静态点，节省计算资源。
        
        【更新策略（参考EH-SurGS）】
        1. 计算每个点在 x, y, z 三个方向上的最大形变量
        2. 除以 100 进行归一化（EH-SurGS 的做法）
        3. 如果最大形变量超过阈值，标记为需要形变
        
        Args:
            threshold: 形变阈值，只有形变量超过此值的点才会被标记为需要形变
                      如果为None，则使用self.deform_table_threshold（从配置读取或默认0.0001）
        """
        # 使用传入的阈值，或从配置/默认值获取
        if threshold is None:
            threshold = getattr(self, 'deform_table_threshold', 0.0001)
        
        # 检查 deformation_accum 是否已初始化
        if self._deformation_accum.numel() == 0:
            # 如果还没累积形变量，初始化所有点都需要形变（训练初期）
            if self._xyz.numel() > 0:
                self._deformation_accum = torch.zeros((self._xyz.shape[0], 3), device=self._xyz.device)
                self._deformation_table = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            return
        
        # 检查尺寸是否匹配
        if self._deformation_accum.shape[0] != self._xyz.shape[0]:
            # 如果尺寸不匹配，重新初始化
            self._deformation_accum = torch.zeros((self._xyz.shape[0], 3), device=self._xyz.device)
            # 初始化为全 True（所有点都需要形变）
            self._deformation_table = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            return
        
        # 检查 deformation_accum 的形状是否正确 [N, 3]
        if self._deformation_accum.dim() != 2 or self._deformation_accum.shape[1] != 3:
            # 如果形状不对，重新初始化
            self._deformation_accum = torch.zeros((self._xyz.shape[0], 3), device=self._xyz.device)
            self._deformation_table = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            return
        
        # 计算每个点在 x, y, z 三个方向上的最大形变量（参考EH-SurGS）
        # max_deform: [N]，每个点的最大形变量（在 x, y, z 三个方向上的最大值）
        # EH-SurGS 使用: max_deform = _deformation_accum.max(dim=-1).values / 100
        max_deform = self._deformation_accum.max(dim=-1).values / 100.0
        
        # 只有形变量超过阈值的点才需要形变
        # torch.gt 等同于 > 操作，返回布尔tensor
        self._deformation_table = torch.gt(max_deform, threshold)
        
        # 优化：如果形变点比例太低（<5%），说明阈值可能太高，临时降低阈值让更多点参与
        num_deform_points = self._deformation_table.sum().item()
        total_points = self._deformation_table.shape[0]
        if total_points > 0:
            deform_ratio = num_deform_points / total_points
            # 如果形变点比例低于5%，且形变累积量最大值大于0，则使用更宽松的阈值
            if deform_ratio < 0.05 and max_deform.max().item() > 0:
                # 使用更宽松的阈值：原阈值的1/10，但至少保证有20%的点参与形变
                relaxed_threshold = threshold * 0.1
                relaxed_deform_table = torch.gt(max_deform, relaxed_threshold)
                relaxed_num = relaxed_deform_table.sum().item()
                relaxed_ratio = relaxed_num / total_points
                
                # 如果放宽后能达到至少20%的点，则使用放宽后的结果
                if relaxed_ratio >= 0.20:
                    self._deformation_table = relaxed_deform_table
                    num_deform_points = relaxed_num
                    deform_ratio = relaxed_ratio
                    # 使用颜色输出警告信息
                    from utils.logging_utils import Log, format_percentage
                    Log(f"动态点比例过低 ({format_percentage(deform_ratio, 2)} < 5%)，已自动放宽阈值 "
                        f"({relaxed_threshold:.6f} → {relaxed_ratio*100:.1f}%)", 
                        tag="WARNING")
        
        # 输出直观的统计信息
        if total_points > 0:
            # 只在某些条件下输出（避免输出过多）
            if not hasattr(self, '_deform_table_update_count'):
                self._deform_table_update_count = 0
            self._deform_table_update_count += 1
            # 每100次迭代或前5次输出一次
            if self._deform_table_update_count <= 5 or self._deform_table_update_count % 100 == 0:
                from utils.logging_utils import Log, format_percentage
                from rich import print as rprint
                # 计算静态点比例
                static_ratio = 1.0 - deform_ratio
                # 使用颜色标记动态点和静态点比例
                dynamic_pct = format_percentage(deform_ratio, 1)
                static_pct = format_percentage(static_ratio, 1)
                Log(f"动态点: {dynamic_pct} ({num_deform_points:,}/{total_points:,}) | "
                    f"静态点: {static_pct}", 
                    tag="Deform")

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
        
        # 获取下采样前的完整点云
        original_xyz = np.asarray(pcd_tmp.points)
        
        # ========== 轮廓提取：在下采样之前提取边缘点 ==========
        # 时间点说明：此时点云已经从MASt3R深度图创建完成，但还未下采样
        # 在下采样前提取边缘点的好处：
        # 1. 可以使用完整的点云信息，边缘特征更完整
        # 2. 边缘点提取后再下采样，可以保留更多边缘信息
        # 3. 点云坐标已经转换到世界坐标系
        # 
        # 改进：累积所有关键帧的边缘点，而不是只保留最后一帧
        edge_mask_original = None
        edge_points = None
        if self.enable_edge_extraction:
            try:
                from utils.star_edge_extractor import STAREdgeExtractor
                
                # 创建提取器（使用单例模式避免重复加载模型）
                if not hasattr(self, '_edge_extractor'):
                    # 调整参数以适应点云：
                    # - kk=26: 使用标准K近邻数量
                    # - bw=10: 必须保持为10，因为网络期望10维描述符
                    # - mu=0.15: 稍微增加细化参数，减少噪声
                    # - max_points=70000: STAR-Edge内部临时下采样（1月14日版本使用此值）
                    self._edge_extractor = STAREdgeExtractor(
                        verbose=True,
                        kk=26,  # K近邻数量
                        bw=10,  # 必须保持为10（网络期望10维描述符）
                        sampleNum=40,  # 采样数量 = bw * 4
                        mu=0.15,  # 细化参数
                        max_points=70000  # STAR-Edge临时下采样点数（与1月14日版本一致）
                    )
                
                # 在下采样前的完整点云上提取边缘点
                result = self._edge_extractor.extract_edges(
                    original_xyz, 
                    refine=True,
                    return_details=False
                )
                edge_mask_original, edge_points = result
                
                # 累积边缘点（关键改进）
                if edge_points is not None and len(edge_points) > 0:
                    self._all_edge_points.append(edge_points.copy())
                    # 合并所有累积的边缘点
                    self._edge_points = np.vstack(self._all_edge_points)
                    print(f"[轮廓提取] 累积边缘点: 当前帧 {len(edge_points)} 个, 总计 {len(self._edge_points)} 个")
                
            except Exception as e:
                print(f"[WARNING] 轮廓提取失败: {e}")
                print(f"[WARNING] 将使用全部点云进行后续处理")
                import traceback
                traceback.print_exc()
                edge_mask_original = None
                edge_points = None
        
        # 下采样点云
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        
        # 在下采样后的点云中找出边缘点（通过最近邻匹配）
        edge_mask = None
        if self.enable_edge_extraction and edge_mask_original is not None:
            try:
                # 在下采样后的点云中找出哪些点在边缘点附近
                # 使用距离阈值匹配：如果下采样后的点距离原始边缘点很近，则认为是边缘点
                if edge_points is not None and len(edge_points) > 0:
                    from scipy.spatial import cKDTree
                    # 构建边缘点的KD树
                    edge_tree = cKDTree(edge_points)
                    # 为下采样后的每个点找最近的边缘点
                    distances, indices = edge_tree.query(new_xyz, k=1)
                    # 计算合理的距离阈值：使用点云密度的估计值
                    # 估计点云的平均点间距离（通过点云的范围和点数量）
                    if len(new_xyz) > 1:
                        # 计算点云的包围盒
                        xyz_range = new_xyz.max(axis=0) - new_xyz.min(axis=0)
                        # 估算平均点间距离：使用包围盒体积的立方根除以点数
                        volume = np.prod(xyz_range)
                        if volume > 0:
                            avg_point_distance = np.power(volume / len(new_xyz), 1.0/3.0)
                        else:
                            avg_point_distance = np.median(distances)
                        # 使用平均点间距离的1.5倍作为阈值
                        threshold = avg_point_distance * 1.5
                    else:
                        # 如果点太少，使用中位数距离
                        threshold = np.median(distances) * 1.5 if len(distances) > 0 else 0.01
                    edge_mask = distances < threshold
                    print(f"[轮廓提取] 下采样后边缘点匹配: {edge_mask.sum()}/{len(new_xyz)} 个点被标记为边缘点 (阈值={threshold:.6f})")
                    
                    # 存储当前帧的边缘信息（对应下采样后的点云）
                    self._edge_mask = torch.from_numpy(edge_mask).bool().cuda()
                else:
                    # 如果没有边缘点，则全部标记为False
                    edge_mask = np.zeros(len(new_xyz), dtype=bool)
                    self._edge_mask = torch.from_numpy(edge_mask).bool().cuda()
            except Exception as e:
                print(f"[WARNING] 下采样后边缘点匹配失败: {e}")
                import traceback
                traceback.print_exc()
                # 如果匹配失败，使用空mask
                edge_mask = np.zeros(len(new_xyz), dtype=bool)
                self._edge_mask = torch.from_numpy(edge_mask).bool().cuda()
        elif self.enable_edge_extraction:
            # 如果启用了边缘提取但没有提取到边缘点，创建空mask
            edge_mask = np.zeros(len(new_xyz), dtype=bool)
            self._edge_mask = torch.from_numpy(edge_mask).bool().cuda()
        
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
        
        # 注意：_coefs的初始化由densification_postfix统一处理，避免重复初始化
        
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
        
        # ========== 初始化形变表相关参数（参考EH-SurGS）==========
        # 如果 deformation_table 还没有初始化，或者大小不匹配，则初始化为全 True（所有点都需要形变）
        num_points = self.get_xyz.shape[0]
        if self._deformation_table.numel() == 0 or self._deformation_table.shape[0] != num_points:
            # 初始化为全 True：所有点都需要形变（训练初期）
            self._deformation_table = torch.ones(num_points, dtype=torch.bool, device="cuda")
        
        # 如果 deformation_accum 还没有初始化，或者大小不匹配，则初始化为全零
        # 形状为 [N, 3]，存储每个点在 x, y, z 三个方向上的累积形变量
        if self._deformation_accum.numel() == 0 or self._deformation_accum.shape[0] != num_points:
            # 初始化为全零：累积形变量从零开始
            # 形状 [N, 3] 与 EH-SurGS 一致，存储 x, y, z 三个方向的累积形变量
            self._deformation_accum = torch.zeros((num_points, 3), device="cuda")
        elif self._deformation_accum.dim() == 1 or self._deformation_accum.shape[1] != 3:
            # 如果形状不对（例如是 [N] 而不是 [N, 3]），重新初始化为 [N, 3]
            self._deformation_accum = torch.zeros((num_points, 3), device="cuda")

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
        
        # ========== 形变参数优化器配置（生命周期机制）==========
        # 形变参数需要单独的学习率，通常比位置参数小10倍
        # 原因：形变是细微的变化，如果学习率太大，会导致训练不稳定
        
        # 获取形变学习率：优先使用配置文件中的deformation_lr_init
        # 如果配置文件中没有，就使用位置学习率的10%（更保守）
        deformation_lr = getattr(
            training_args, 
            "deformation_lr_init",  # 如果配置了就用这个
            training_args.position_lr_init * 0.1  # 否则用位置学习率的10%
        ) * self.spatial_lr_scale  # 乘以空间缩放因子
        
        # 将形变系数参数添加到优化器配置列表中（生命周期机制）
        if self._coefs.numel() > 0:
            l.append({
                "params": [self._coefs],
                "lr": deformation_lr,
                "name": "coefs",  # 形变系数（包含weights, means, std_devs）
            })

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
        # 添加形变系数（生命周期机制）
        if self._coefs.numel() > 0:
            for i in range(self._coefs.shape[1]):
                l.append("coefs_{}".format(i))
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
        
        # 添加形变系数（生命周期机制）
        coefs = None
        coefs_valid = False
        if self._coefs.numel() > 0:
            coefs_numpy = self._coefs.detach().cpu().numpy()
            # 检查coefs的形状是否与xyz匹配
            if coefs_numpy.shape[0] == xyz.shape[0]:
                coefs = coefs_numpy
                coefs_valid = True
            else:
                # 如果形状不匹配，跳过coefs（避免保存错误）
                from utils.logging_utils import Log
                Log(f"保存PLY时跳过形变系数（形状不匹配）", tag="WARNING")

        # 构建属性列表（根据coefs是否有效）
        attributes_list = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attributes_list.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attributes_list.append("f_rest_{}".format(i))
        attributes_list.append("opacity")
        for i in range(self._scaling.shape[1]):
            attributes_list.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            attributes_list.append("rot_{}".format(i))
        # 只有在coefs有效时才添加coefs属性
        if coefs_valid:
            for i in range(coefs.shape[1]):
                attributes_list.append("coefs_{}".format(i))
        
        dtype_full = [(attribute, "f4") for attribute in attributes_list]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if coefs_valid:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation, coefs), axis=1
            )
        else:
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
        
        # 构建attributes列表，注意construct_list_of_attributes可能包含coefs
        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        # 如果construct_list_of_attributes包含coefs，也需要添加到attributes中
        if self._coefs.numel() > 0:
            coefs = self._coefs.detach().cpu().numpy()
            attributes_list.append(coefs)
        
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        
    def save_deformation_params(self, path):
        """
        保存形变参数（生命周期机制的系数）为.npz文件
        
        Args:
            path: 保存路径（.npz文件）
        """
        mkdir_p(os.path.dirname(path))
        
        params = {
            'coefs': self._coefs.detach().cpu().numpy() if self._coefs.numel() > 0 else None,
            'ch_num': self.ch_num,
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
        
        # 加载形变系数（生命周期机制）
        coef_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("coefs_")
        ]
        coefs = None
        if len(coef_names) > 0:
            coef_names = sorted(coef_names, key=lambda x: int(x.split("_")[-1]))
            coefs = np.zeros((xyz.shape[0], len(coef_names)))
            for idx, attr_name in enumerate(coef_names):
                coefs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        # 初始化形变系数（生命周期机制）
        if coefs is not None:
            # 从PLY文件加载
            self._coefs = nn.Parameter(
                torch.tensor(coefs, dtype=torch.float, device="cuda").requires_grad_(True)
            )
        else:
            # 初始化新的形变系数
            N = self._xyz.shape[0]
            # 使用小的随机初始化，而不是全0，这样更容易学习
            weight_coefs = torch.randn((N, self.ch_num, self.K_time), device="cuda") * 0.01
            position_coefs = torch.linspace(0, 1, self.K_time, device="cuda").view(1, 1, -1).repeat(N, self.ch_num, 1)
            shape_coefs = torch.full((N, self.ch_num, self.K_time), 0.01, device="cuda")
            _coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(N, -1)
            self._coefs = nn.Parameter(_coefs.requires_grad_(True))
        
        # 为了向后兼容，保留旧的参数名（但不再使用）
        self._w_pos = torch.empty(0, device="cuda")
        self._w_rot = torch.empty(0, device="cuda")
        self._w_scale = torch.empty(0, device="cuda")
        self._w_opacity = torch.empty(0, device="cuda")


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

            # Debug: 检查coefs的处理
            if name == "coefs":
                # 移除详细的shape debug信息
                pass  # 已清理：shape信息对用户没有直观价值
                # 如果coefs的大小不匹配，需要重新初始化
                if not is_per_point and p.ndim >= 1:
                    # _coefs 应该是 per-point 参数，但大小不匹配
                    # 需要重新初始化以匹配当前点数
                    from utils.logging_utils import Log
                    Log(f"形变系数大小不匹配，已重新初始化（点数变化: {p.shape[0]} → {N}）", tag="WARNING")
                    # 这里不处理，让后续代码处理

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

        # ========== 更新形变系数（生命周期机制）==========
        # 删除被prune的点的形变系数，只保留有效点的参数
        # 这样优化器中的参数和实际数据保持一致
        if "coefs" in optimizable_tensors:
            self._coefs = optimizable_tensors["coefs"]
        else:
            # 如果coefs没有被_prune_optimizer处理（因为大小不匹配），需要重新初始化
            num_points_after_prune = valid_points_mask.sum().item()
            if num_points_after_prune > 0:
                from utils.logging_utils import Log
                Log(f"形变系数未被优化器处理，已重新初始化（当前点数: {num_points_after_prune:,}）", tag="WARNING")
                # 重新初始化_coefs以匹配当前点数
                # 使用小的随机初始化，而不是全0，这样更容易学习
                weight_coefs = torch.randn((num_points_after_prune, self.ch_num, self.K_time), device="cuda") * 0.01
                position_coefs = torch.linspace(0, 1, self.K_time, device="cuda").view(1, 1, -1).repeat(num_points_after_prune, self.ch_num, 1)
                shape_coefs = torch.full((num_points_after_prune, self.ch_num, self.K_time), 0.01, device="cuda")
                _coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(num_points_after_prune, -1)
                self._coefs = nn.Parameter(_coefs.requires_grad_(True))
                # 如果优化器已经初始化，需要手动添加coefs参数组
                if self.optimizer is not None:
                    # 检查优化器中是否已有coefs参数组
                    has_coefs_group = any(group.get("name") == "coefs" for group in self.optimizer.param_groups)
                    if not has_coefs_group:
                        # 添加coefs参数组到优化器
                        deformation_lr = getattr(self, 'spatial_lr_scale', 1.0) * 0.001  # 使用默认学习率
                        self.optimizer.add_param_group({
                            "params": [self._coefs],
                            "lr": deformation_lr,
                            "name": "coefs",
                        })
                    else:
                        # 更新现有参数组
                        for group in self.optimizer.param_groups:
                            if group.get("name") == "coefs":
                                group["params"][0] = self._coefs
                                break

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
        
        # ========== 更新形变表相关参数（参考EH-SurGS）==========
        # 删除被prune的点的 deformation_table 和 deformation_accum
        if hasattr(self, '_deformation_table') and self._deformation_table.numel() > 0:
            # 确保 valid_points_mask 的长度与 deformation_table 匹配
            if self._deformation_table.shape[0] == valid_points_mask.shape[0]:
                self._deformation_table = self._deformation_table[valid_points_mask]
            elif self._deformation_table.shape[0] == self._xyz.shape[0]:
                # 如果点数匹配，使用新的点数
                num_valid = valid_points_mask.sum().item()
                if num_valid > 0:
                    self._deformation_table = self._deformation_table[valid_points_mask]
                else:
                    self._deformation_table = torch.empty(0, dtype=torch.bool, device="cuda")
            else:
                # 如果不匹配，重新初始化
                num_valid = valid_points_mask.sum().item()
                if num_valid > 0:
                    self._deformation_table = torch.ones(num_valid, dtype=torch.bool, device="cuda")
                else:
                    self._deformation_table = torch.empty(0, dtype=torch.bool, device="cuda")
        
        if hasattr(self, '_deformation_accum') and self._deformation_accum.numel() > 0:
            # 确保 valid_points_mask 的长度与 deformation_accum 匹配
            if self._deformation_accum.shape[0] == valid_points_mask.shape[0]:
                self._deformation_accum = self._deformation_accum[valid_points_mask]
            elif self._deformation_accum.shape[0] == self._xyz.shape[0]:
                # 如果点数匹配，使用新的点数
                num_valid = valid_points_mask.sum().item()
                if num_valid > 0:
                    self._deformation_accum = self._deformation_accum[valid_points_mask]
                else:
                    self._deformation_accum = torch.empty(0, device="cuda")
            else:
                # 如果不匹配，重新初始化
                num_valid = valid_points_mask.sum().item()
                if num_valid > 0:
                    self._deformation_accum = torch.zeros((num_valid, 3), device="cuda")
                else:
                    self._deformation_accum = torch.empty(0, device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        
        # 如果优化器还没有初始化，直接返回tensors_dict中的值（转换为Parameter）
        if self.optimizer is None:
            for name, tensor in tensors_dict.items():
                if isinstance(tensor, nn.Parameter):
                    optimizable_tensors[name] = tensor
                else:
                    optimizable_tensors[name] = nn.Parameter(tensor.requires_grad_(True))
            return optimizable_tensors
        
        # 处理优化器中已存在的参数组
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
        
        # 处理tensors_dict中存在但优化器中还没有的参数组（如coefs在优化器初始化前）
        for name, tensor in tensors_dict.items():
            if name not in optimizable_tensors:
                # 如果优化器中没有这个参数组，直接返回tensor（转换为Parameter）
                if isinstance(tensor, nn.Parameter):
                    optimizable_tensors[name] = tensor
                else:
                    optimizable_tensors[name] = nn.Parameter(tensor.requires_grad_(True))

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
        new_deformation_table=None,
    ):
        """
        当系统创建新的高斯点时（比如从图像中提取新点，或者分裂/克隆现有点），
        需要为这些新点初始化所有参数，包括形变参数
        
        这个函数的作用：把新点的参数添加到优化器中，让它们可以被训练
        """
        # new_xyz: [M,3] M是新点的数量
        M = new_xyz.shape[0]
        
        # 如果没有新点，直接返回（避免空张量操作）
        if M == 0:
            return
        
        # ========== 初始化新点的形变系数（生命周期机制）==========
        # 新点的形变系数初始化（参考EH-SurGS）：
        # - weights: 使用小的随机初始化（而不是全0），这样更容易学习
        #   使用小的随机值（std=0.01）可以让权重有小的初始形变，梯度更容易传播
        # - means: 均匀分布在[0,1]（时间中心位置）
        # - std_devs: 初始化为0.01（时间影响范围）
        # 注意：虽然EH-SurGS使用全0初始化，但小的随机初始化可以让训练更稳定
        weight_coefs = torch.randn((M, self.ch_num, self.K_time), device="cuda", dtype=new_xyz.dtype) * 0.01
        position_coefs = torch.linspace(0, 1, self.K_time, device="cuda").view(1, 1, -1).repeat(M, self.ch_num, 1)
        shape_coefs = torch.full((M, self.ch_num, self.K_time), 0.01, device="cuda", dtype=new_xyz.dtype)
        # 堆叠为 [M, ch_num, 3, K_time] 然后reshape为 [M, ch_num * 3 * K_time]
        new_coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(M, -1)
        

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "coefs": new_coefs,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # 安全处理coefs：如果优化器中还没有coefs参数组，使用传入的new_coefs
        if "coefs" in optimizable_tensors:
            self._coefs = optimizable_tensors["coefs"]
        else:
            # 如果优化器还没有初始化coefs参数组，直接使用new_coefs
            # 这会在training_setup时被添加到优化器
            if isinstance(new_coefs, nn.Parameter):
                self._coefs = new_coefs
            else:
                self._coefs = nn.Parameter(new_coefs.requires_grad_(True))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # ========== 处理新点的形变表相关参数（参考EH-SurGS）==========
        # 为新点初始化 deformation_table 和 deformation_accum
        # 新点的 deformation_table 继承自原点的状态（如果是从原点和分裂/克隆的）
        # 如果是全新点（从图像提取），初始化为 True（需要形变）
        if hasattr(self, '_deformation_table'):
            # 检查当前 deformation_table 是否存在
            if self._deformation_table.numel() > 0:
                # 如果原有点的 deformation_table 存在，需要扩展它
                old_num_points = self._deformation_table.shape[0]
                new_num_points = self.get_xyz.shape[0]
                if new_num_points > old_num_points:
                    # 有新点添加，为它们初始化 deformation_table
                    # 如果提供了 new_deformation_table（从原点分裂/克隆），则使用它
                    # 否则初始化为 True（需要形变），参考EH-SurGS的策略
                    if new_deformation_table is not None:
                        # 使用传入的 deformation_table（继承自原点）
                        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], dim=0)
                    else:
                        # 新点初始化为 True（需要形变）
                        new_deform_table = torch.ones(new_num_points - old_num_points, 
                                                       dtype=torch.bool, 
                                                       device=self._deformation_table.device)
                        self._deformation_table = torch.cat([self._deformation_table, new_deform_table], dim=0)
                elif new_num_points < old_num_points:
                    # 点数减少（可能被prune了），需要调整
                    self._deformation_table = self._deformation_table[:new_num_points]
            else:
                # 如果 deformation_table 还未初始化，初始化为全 True
                if new_deformation_table is not None:
                    # 使用传入的 deformation_table
                    self._deformation_table = new_deformation_table.clone()
                else:
                    # 初始化为全 True
                    self._deformation_table = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        # 为新点初始化 deformation_accum（累积形变量）
        if hasattr(self, '_deformation_accum'):
            # 检查当前 deformation_accum 是否存在
            if self._deformation_accum.numel() > 0:
                # 如果原有点的 deformation_accum 存在，需要扩展它
                old_num_points = self._deformation_accum.shape[0]
                new_num_points = self.get_xyz.shape[0]
                if new_num_points > old_num_points:
                    # 有新点添加，为它们初始化 deformation_accum
                    # 新点初始化为全零（累积形变量从零开始）
                    # 形状为 [M, 3]，存储 x, y, z 三个方向的累积形变量
                    new_deformation_accum = torch.zeros((new_num_points - old_num_points, 3), 
                                                       device=self._deformation_accum.device,
                                                       dtype=self._deformation_accum.dtype)
                    self._deformation_accum = torch.cat([self._deformation_accum, new_deformation_accum], dim=0)
                elif new_num_points < old_num_points:
                    # 点数减少（可能被prune了），需要调整
                    self._deformation_accum = self._deformation_accum[:new_num_points]
            else:
                # 如果 deformation_accum 还未初始化，初始化为全零
                # 形状为 [N, 3]，存储 x, y, z 三个方向的累积形变量
                self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

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
        
        # 新点的 deformation_table 继承自原点的状态（参考EH-SurGS）
        new_deformation_table = None
        if hasattr(self, '_deformation_table') and self._deformation_table.numel() > 0:
            new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_deformation_table=new_deformation_table,
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
        
        # 新点的 deformation_table 继承自原点的状态（参考EH-SurGS）
        new_deformation_table = None
        if hasattr(self, '_deformation_table') and self._deformation_table.numel() > 0:
            new_deformation_table = self._deformation_table[selected_pts_mask]
        
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_deformation_table=new_deformation_table,
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
            '_coefs': getattr(self, '_coefs', torch.empty(0, device="cuda")),
            't_mu': getattr(self, 't_mu', None),
            't_sigma_raw': getattr(self, 't_sigma_raw', None),
            'K_time': self.K_time,
            'ch_num': getattr(self, 'ch_num', 11),
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
        
        # 恢复模型参数（确保它们是Parameter并且requires_grad=True）
        self._xyz = nn.Parameter(checkpoint_dict['_xyz'].requires_grad_(True))
        self._features_dc = nn.Parameter(checkpoint_dict['_features_dc'].requires_grad_(True))
        self._features_rest = nn.Parameter(checkpoint_dict['_features_rest'].requires_grad_(True))
        self._scaling = nn.Parameter(checkpoint_dict['_scaling'].requires_grad_(True))
        self._rotation = nn.Parameter(checkpoint_dict['_rotation'].requires_grad_(True))
        self._opacity = nn.Parameter(checkpoint_dict['_opacity'].requires_grad_(True))
        
        # 恢复形变参数（如果存在）
        if '_w_pos' in checkpoint_dict:
            self._w_pos = checkpoint_dict['_w_pos']
        if '_w_rot' in checkpoint_dict:
            self._w_rot = checkpoint_dict['_w_rot']
        if '_w_scale' in checkpoint_dict:
            self._w_scale = checkpoint_dict['_w_scale']
        if '_w_opacity' in checkpoint_dict:
            self._w_opacity = checkpoint_dict['_w_opacity']
        if '_coefs' in checkpoint_dict and checkpoint_dict['_coefs'].numel() > 0:
            self._coefs = nn.Parameter(checkpoint_dict['_coefs'].requires_grad_(True))
        elif self._xyz.shape[0] > 0:
            # 如果checkpoint中没有_coefs，但有点存在，需要初始化
            N = self._xyz.shape[0]
            ch_num = checkpoint_dict.get('ch_num', self.ch_num)
            K_time = checkpoint_dict.get('K_time', self.K_time)
            # 使用小的随机初始化，而不是全0，这样更容易学习
            weight_coefs = torch.randn((N, ch_num, K_time), device="cuda") * 0.01
            position_coefs = torch.linspace(0, 1, K_time, device="cuda").view(1, 1, -1).repeat(N, ch_num, 1)
            shape_coefs = torch.full((N, ch_num, K_time), 0.01, device="cuda")
            _coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(N, -1)
            self._coefs = nn.Parameter(_coefs.requires_grad_(True))
        if 't_mu' in checkpoint_dict and checkpoint_dict['t_mu'] is not None:
            self.t_mu = checkpoint_dict['t_mu']
        if 't_sigma_raw' in checkpoint_dict and checkpoint_dict['t_sigma_raw'] is not None:
            self.t_sigma_raw = checkpoint_dict['t_sigma_raw']
        if 'K_time' in checkpoint_dict:
            self.K_time = checkpoint_dict['K_time']
        if 'ch_num' in checkpoint_dict:
            self.ch_num = checkpoint_dict['ch_num']
        
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
