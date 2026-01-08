#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证 _deformation_accum 的形状修复逻辑
"""
import torch
import sys
sys.path.insert(0, '.')

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.cameras import Camera
import numpy as np

def test_deformation_accum_shape():
    """测试 _deformation_accum 形状修复"""
    print("=" * 60)
    print("测试 _deformation_accum 形状修复")
    print("=" * 60)
    
    # 创建高斯模型
    sh_degree = 0
    config = {
        "model_params": {
            "time_basis_num": 17,
            "ch_num": 11
        }
    }
    gaussians = GaussianModel(sh_degree, config=config)
    
    # 初始化一些点
    N = 100
    gaussians._xyz = torch.nn.Parameter(torch.randn(N, 3, device="cuda"))
    gaussians._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device="cuda"))
    gaussians._features_rest = torch.nn.Parameter(torch.randn(N, 1, 3, device="cuda"))
    gaussians._scaling = torch.nn.Parameter(torch.randn(N, 1, device="cuda"))
    gaussians._rotation = torch.nn.Parameter(torch.randn(N, 4, device="cuda"))
    gaussians._opacity = torch.nn.Parameter(torch.randn(N, 1, device="cuda"))
    
    # 初始化形变参数
    gaussians.K_time = 17
    gaussians.ch_num = 11
    gaussians._coefs = torch.nn.Parameter(torch.randn(N, 11 * 3 * 17, device="cuda") * 0.01)
    
    # 测试1: 正常情况 - accum为空
    print("\n[测试1] accum为空，应该自动初始化")
    gaussians._deformation_accum = torch.empty(0, device="cuda")
    gaussians._deformation_table = torch.empty(0, dtype=torch.bool, device="cuda")
    
    # 创建虚拟相机
    class DummyCamera:
        def __init__(self):
            self.image_height = 480
            self.image_width = 640
            self.FoVx = 1.0
            self.FoVy = 1.0
            self.world_view_transform = torch.eye(4, device="cuda")
            self.full_proj_transform = torch.eye(4, device="cuda")
            self.projection_matrix = torch.eye(4, device="cuda")
            self.camera_center = torch.zeros(3, device="cuda")
            self.t = 0.5
            self.cam_rot_delta = None
            self.cam_trans_delta = None
    
    camera = DummyCamera()
    pipe = type('obj', (object,), {
        'compute_cov3D_python': False,
        'convert_SHs_python': False
    })()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    try:
        result = render(camera, gaussians, pipe, bg_color)
        print(f"  ✓ 渲染成功")
        print(f"  accum形状: {gaussians._deformation_accum.shape}")
        print(f"  xyz形状: {gaussians._xyz.shape}")
        assert gaussians._deformation_accum.shape[0] == gaussians._xyz.shape[0], \
            f"形状不匹配: accum={gaussians._deformation_accum.shape[0]}, xyz={gaussians._xyz.shape[0]}"
        print(f"  ✓ 形状匹配")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试2: 形状不匹配 - accum比xyz大
    print("\n[测试2] accum比xyz大，应该截断")
    gaussians._xyz = torch.nn.Parameter(torch.randn(N, 3, device="cuda"))
    gaussians._deformation_accum = torch.zeros(N * 2, 3, device="cuda")  # 两倍大小
    gaussians._deformation_table = torch.ones(N * 2, dtype=torch.bool, device="cuda")
    
    try:
        result = render(camera, gaussians, pipe, bg_color)
        print(f"  ✓ 渲染成功")
        print(f"  accum形状: {gaussians._deformation_accum.shape}")
        print(f"  xyz形状: {gaussians._xyz.shape}")
        assert gaussians._deformation_accum.shape[0] == gaussians._xyz.shape[0], \
            f"形状不匹配: accum={gaussians._deformation_accum.shape[0]}, xyz={gaussians._xyz.shape[0]}"
        print(f"  ✓ 形状已修复")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试3: 形状不匹配 - accum比xyz小
    print("\n[测试3] accum比xyz小，应该重新初始化")
    gaussians._xyz = torch.nn.Parameter(torch.randn(N, 3, device="cuda"))
    gaussians._deformation_accum = torch.zeros(N // 2, 3, device="cuda")  # 一半大小
    gaussians._deformation_table = torch.ones(N // 2, dtype=torch.bool, device="cuda")
    
    try:
        result = render(camera, gaussians, pipe, bg_color)
        print(f"  ✓ 渲染成功")
        print(f"  accum形状: {gaussians._deformation_accum.shape}")
        print(f"  xyz形状: {gaussians._xyz.shape}")
        assert gaussians._deformation_accum.shape[0] == gaussians._xyz.shape[0], \
            f"形状不匹配: accum={gaussians._deformation_accum.shape[0]}, xyz={gaussians._xyz.shape[0]}"
        print(f"  ✓ 形状已修复")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_deformation_accum_shape()
    sys.exit(0 if success else 1)

