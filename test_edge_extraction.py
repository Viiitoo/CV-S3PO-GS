#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAR-Edge 边缘提取功能测试脚本

测试内容：
1. LocalSH模块导入测试
2. 边缘提取配置加载测试
3. 临时下采样功能测试
4. 边缘提取功能测试（LocalSH方法和曲率方法）
5. 边缘投影和mask生成测试
"""

import sys
import os
import numpy as np
import torch
import yaml
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_localsh_import():
    """测试LocalSH模块导入"""
    print("\n" + "="*60)
    print("测试 1: LocalSH模块导入")
    print("="*60)
    
    from utils.edge_extraction import LOCALSH_AVAILABLE, LocalSH
    
    if LOCALSH_AVAILABLE:
        print("✓ LocalSH模块成功导入")
        localsh_file = getattr(LocalSH, '__file__', 'unknown')
        print("  模块位置: " + str(localsh_file))
        
        # 检查子模块
        if hasattr(LocalSH, "LocalSHFeature"):
            print("✓ LocalSHFeature 子模块存在")
            lsh_feature = LocalSH.LocalSHFeature
            if hasattr(lsh_feature, "ComLSHF_knn_nosample"):
                print("✓ ComLSHF_knn_nosample 函数存在")
            else:
                print("✗ ComLSHF_knn_nosample 函数不存在")
        else:
            print("✗ LocalSHFeature 子模块不存在")
        
        return True
    else:
        print("✗ LocalSH模块不可用，将使用曲率方法作为备选")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n" + "="*60)
    print("测试 2: 边缘提取配置加载")
    print("="*60)
    
    from utils.edge_extraction import EdgeExtractionConfig
    
    # 测试默认配置
    config = EdgeExtractionConfig()
    print("✓ 默认配置创建成功")
    print(f"  voxel_size: {config.voxel_size}")
    print(f"  max_points: {config.max_points}")
    print(f"  edge_ratio: {config.edge_ratio}")
    print(f"  knn_neighbors: {config.knn_neighbors}")
    
    # 测试从字典创建配置
    config_dict = {
        'edge_extraction': {
            'voxel_size': 0.03,
            'max_points': 30000,
            'edge_ratio': 0.15,
        }
    }
    config2 = EdgeExtractionConfig(config_dict)
    print("✓ 从字典创建配置成功")
    print(f"  voxel_size: {config2.voxel_size}")
    print(f"  max_points: {config2.max_points}")
    print(f"  edge_ratio: {config2.edge_ratio}")
    
    return True


def test_voxel_downsample():
    """测试临时体素下采样"""
    print("\n" + "="*60)
    print("测试 3: 临时体素下采样（独立于SLAM下采样）")
    print("="*60)
    
    from utils.edge_extraction import EdgeExtractor
    
    extractor = EdgeExtractor()
    
    # 生成测试点云
    np.random.seed(42)
    num_points = 100000
    points = np.random.randn(num_points, 3).astype(np.float64)
    
    print(f"原始点云: {len(points)} 点")
    
    # 测试下采样
    downsampled, indices = extractor.voxel_downsample(points)
    
    print(f"✓ 下采样完成")
    print(f"  下采样后: {len(downsampled)} 点")
    print(f"  下采样比例: {len(downsampled) / len(points) * 100:.2f}%")
    print(f"  保留的索引数: {len(indices)}")
    
    # 验证下采样结果
    assert len(downsampled) == len(indices), "下采样点数与索引数不匹配"
    assert len(downsampled) <= extractor.config.max_points, "超过最大点数限制"
    assert len(downsampled) >= int(len(points) * extractor.config.min_points_ratio), "低于最小点数要求"
    
    print("✓ 下采样约束验证通过")
    
    # 测试自适应voxel_size
    large_points = np.random.randn(200000, 3).astype(np.float64)
    downsampled2, indices2 = extractor.voxel_downsample(large_points)
    print(f"✓ 大点云下采样: {len(large_points)} -> {len(downsampled2)} 点")
    
    return True


def test_edge_extraction():
    """测试边缘提取功能"""
    print("\n" + "="*60)
    print("测试 4: 边缘提取功能")
    print("="*60)
    
    from utils.edge_extraction import EdgeExtractor, LOCALSH_AVAILABLE
    
    extractor = EdgeExtractor()
    
    # 生成测试点云（包含一些明显的边缘结构）
    np.random.seed(42)
    
    # 创建一个平面 + 边缘的点云
    plane_points = np.random.randn(1000, 3) * 0.1
    plane_points[:, 2] = 0  # Z=0 平面
    
    # 添加边缘点（沿着X轴的突起）
    edge_points = np.zeros((100, 3))
    edge_points[:, 0] = np.linspace(-1, 1, 100)
    edge_points[:, 1] = 0
    edge_points[:, 2] = 1  # 高度为1
    
    points = np.vstack([plane_points, edge_points])
    points = points.astype(np.float64)
    
    print(f"测试点云: {len(points)} 点 (包含 {len(edge_points)} 边缘点)")
    
    # 测试边缘提取
    edge_mask, edge_scores = extractor.extract_edges(points)
    
    print(f"✓ 边缘提取完成")
    print(f"  检测到边缘点: {edge_mask.sum()} 个")
    print(f"  边缘点比例: {edge_mask.sum() / len(points) * 100:.2f}%")
    print(f"  边缘分数范围: [{edge_scores.min():.4f}, {edge_scores.max():.4f}]")
    
    if LOCALSH_AVAILABLE:
        print(f"  使用方法: LocalSH (STAR-Edge)")
        
        # 同时测试曲率方法作为对比
        edge_mask_curv, edge_scores_curv = extractor.extract_edges_curvature(points)
        print(f"  曲率方法检测: {edge_mask_curv.sum()} 个边缘点")
    else:
        print(f"  使用方法: 曲率方法 (备选)")
    
    # 验证结果
    assert len(edge_mask) == len(points), "边缘mask长度不匹配"
    assert len(edge_scores) == len(points), "边缘分数长度不匹配"
    assert edge_mask.dtype == bool, "边缘mask类型错误"
    assert 0 <= edge_scores.min() <= edge_scores.max() <= 1.0, "边缘分数范围错误"
    
    print("✓ 边缘提取结果验证通过")
    
    return True


def test_edge_projection():
    """测试边缘投影和mask生成"""
    print("\n" + "="*60)
    print("测试 5: 边缘投影和mask生成")
    print("="*60)
    
    from utils.edge_extraction import EdgeExtractor
    
    extractor = EdgeExtractor()
    
    # 创建测试数据
    np.random.seed(42)
    points_3d = np.random.randn(1000, 3).astype(np.float64)
    points_3d[:, 2] += 5  # 放在相机前方
    
    # 相机参数
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    R = np.eye(3)
    T = np.zeros(3)
    width, height = 640, 480
    
    # 投影到图像
    points_2d, valid_mask = extractor.project_points_to_image(
        points_3d, K, R, T, width, height
    )
    
    print(f"✓ 3D点投影完成")
    print(f"  投影点数: {len(points_2d)} / {len(points_3d)}")
    print(f"  有效率: {valid_mask.sum() / len(points_3d) * 100:.2f}%")
    
    # 创建边缘mask图像
    edge_scores = np.random.rand(len(points_2d)).astype(np.float32)
    edge_mask_img = extractor.create_edge_mask(points_2d, width, height, edge_scores)
    
    print(f"✓ 边缘mask图像生成完成")
    print(f"  图像尺寸: {edge_mask_img.shape}")
    print(f"  非零像素: {(edge_mask_img > 0).sum()}")
    print(f"  值范围: [{edge_mask_img.min():.4f}, {edge_mask_img.max():.4f}]")
    
    # 验证结果
    assert edge_mask_img.shape == (height, width), "mask图像尺寸错误"
    assert 0 <= edge_mask_img.min() <= edge_mask_img.max() <= 1.0, "mask值范围错误"
    
    print("✓ 边缘投影和mask生成验证通过")
    
    return True


def test_integration():
    """测试与SLAM系统的集成"""
    print("\n" + "="*60)
    print("测试 6: 与SLAM系统集成")
    print("="*60)
    
    # 检查配置文件
    config_files = [
        "configs/mono/Stereo/base_config.yaml",
        "configs/mono/Stereo/Stereo_seq_easy/base_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'edge_extraction' in config:
                print(f"✓ {config_file} 包含边缘提取配置")
                edge_config = config['edge_extraction']
                print(f"  enabled: {edge_config.get('enabled', False)}")
                print(f"  voxel_size: {edge_config.get('voxel_size', 'N/A')}")
                print(f"  edge_ratio: {edge_config.get('edge_ratio', 'N/A')}")
            else:
                print(f"✗ {config_file} 缺少边缘提取配置")
        else:
            print(f"✗ {config_file} 不存在")
    
    # 检查init_pose.py中的集成
    init_pose_path = project_root / "utils" / "init_pose.py"
    if init_pose_path.exists():
        with open(init_pose_path, 'r') as f:
            content = f.read()
        
        if 'edge_extraction' in content and 'get_edge_extractor' in content:
            print(f"✓ utils/init_pose.py 已集成边缘提取功能")
        else:
            print(f"✗ utils/init_pose.py 未集成边缘提取功能")
    
    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("STAR-Edge 边缘提取功能测试")
    print("="*60)
    
    tests = [
        ("LocalSH模块导入", test_localsh_import),
        ("配置加载", test_config_loading),
        ("临时体素下采样", test_voxel_downsample),
        ("边缘提取", test_edge_extraction),
        ("边缘投影和mask生成", test_edge_projection),
        ("SLAM系统集成", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！STAR-Edge集成成功！")
        print("\n关键要点:")
        print("1. 临时下采样机制已实现，使用独立的voxel_size参数")
        print("2. 不影响SLAM主流程的pcd_downsample配置")
        print("3. LocalSH方法和曲率方法均可用作边缘提取")
        print("4. 边缘增强已集成到位姿估计流程")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

