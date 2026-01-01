#!/usr/bin/env python3
"""
测试形变功能的简单脚本
"""

import torch
from gaussian_splatting.scene.gaussian_model import GaussianModel

def test_deformation():
    print("=== 测试形变功能 ===")

    # 创建一个简单的GaussianModel
    model = GaussianModel(sh_degree=3)

    # 模拟一些点
    N = 10
    model._xyz = torch.randn((N, 3), device="cuda") * 0.1
    model._features_dc = torch.randn((N, 1, 3), device="cuda") * 0.1
    model._features_rest = torch.randn((N, 15, 3), device="cuda") * 0.1
    model._opacity = torch.randn((N, 1), device="cuda") * 0.1
    model._scaling = torch.randn((N, 3), device="cuda") * 0.1
    model._rotation = torch.randn((N, 4), device="cuda") * 0.1

    # 初始化形变参数（随机初始化）
    model.ch_num = 11
    model.K_time = 17
    weight_coefs = torch.randn((N, model.ch_num, model.K_time), device="cuda") * 0.1
    position_coefs = torch.linspace(0, 1, model.K_time, device="cuda").view(1, 1, -1).repeat(N, model.ch_num, 1)
    shape_coefs = torch.full((N, model.ch_num, model.K_time), 0.01, device="cuda")
    _coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(N, -1)
    model._coefs = torch.nn.Parameter(_coefs)

    print(f"模型创建完成，有 {N} 个点")
    print(f"coefs形状: {model._coefs.shape}")

    # 测试时间t=0.5
    t = 0.5
    print(f"\n测试时间 t={t}")

    # 测试1: gaussian_deformation
    print("\n1. 测试 gaussian_deformation:")
    deformation_point = torch.ones(N, dtype=torch.bool, device="cuda")
    deform = model.gaussian_deformation(t, deformation_point, N)
    print(f"   形变输出形状: {deform.shape}")
    print(f"   形变最大值: {deform.abs().max().item():.6f}")
    if deform.dim() >= 2:
        print(f"   位置形变 (前3通道): {deform[:, :3].abs().max().item():.6f}")
    else:
        print("   形变输出维度不足，无法提取位置形变")

    # 测试2: get_xyz_t
    print("\n2. 测试 get_xyz_t:")
    xyz_original = model._xyz.clone()
    xyz_deformed = model.get_xyz_t(t)
    position_diff = (xyz_deformed - xyz_original).abs().max().item()
    print(f"   原始位置范围: [{xyz_original.min().item():.6f}, {xyz_original.max().item():.6f}]")
    print(f"   形变后位置范围: [{xyz_deformed.min().item():.6f}, {xyz_deformed.max().item():.6f}]")
    print(f"   最大位置变化: {position_diff:.6f}")

    # 测试3: get_deformed_attributes_t
    print("\n3. 测试 get_deformed_attributes_t:")
    model._deformation_table = torch.ones(N, dtype=torch.bool, device="cuda")  # 所有点都是动态的
    xyz_def, rot_def, scale_def, opacity_def = model.get_deformed_attributes_t(t)
    pos_diff_full = (xyz_def - xyz_original).abs().max().item()
    print(f"   完整形变最大位置变化: {pos_diff_full:.6f}")

    # 测试4: 权重统计
    print("\n4. 权重统计:")
    coefs_reshaped = model._coefs.reshape(N, model.ch_num, 3, model.K_time)
    weights = coefs_reshaped[:, :, 0, :]  # weights
    means = coefs_reshaped[:, :, 1, :]    # means
    std_devs = coefs_reshaped[:, :, 2, :] # std_devs

    print(f"   weights范围: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
    print(f"   means范围: [{means.min().item():.6f}, {means.max().item():.6f}]")
    print(f"   std_devs范围: [{std_devs.min().item():.6f}, {std_devs.max().item():.6f}]")

    # 测试5: 高斯基函数值
    print("\n5. 高斯基函数测试:")
    time_tensor = torch.tensor(t, device="cuda")
    gaussians = torch.exp(-((time_tensor - means[:, 0, :]) / (std_devs[:, 0, :] + 1e-4)) ** 2)
    gaussian_max = gaussians.max().item()
    print(f"   高斯基函数最大值: {gaussian_max:.6f}")

    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_deformation()
