"""
高效点云轮廓/边缘提取工具

该模块提供点云边缘/轮廓提取功能，用于增强形变场景下的图像匹配。
采用多种高效算法：
1. 深度不连续性检测 - 基于渲染深度图的梯度（最快，推荐）
2. 基于KDTree的快速曲率估计 - 使用scipy加速
3. 混合方法 - 结合2D和3D特征

主要优化：
- 使用scipy.spatial.cKDTree加速邻域搜索
- 支持直接从深度图提取边缘（无需3D点云处理）
- 向量化操作，避免Python循环
"""

import numpy as np
import torch
import cv2
import sys
import os
import time
import glob
from typing import Optional, Tuple, Union

# 尝试导入scipy用于快速KNN
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[EdgeExtraction] scipy不可用，将使用基础方法")

# 尝试导入LocalSH（保持兼容性）
LOCALSH_AVAILABLE = False
LocalSH = None

try:
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # 兼容多种LocalSH编译产物目录布局：
    # - STAR-Edge/LocalSH/build
    # - STAR-Edge/LocalSH/build/lib.linux-*/ (setuptools生成)
    # - STAR-Edge/LocalSH (inplace build)
    # - STAR-Edge/pre_process (已有预编译so)
    candidate_paths = [
        os.path.join(repo_root, 'STAR-Edge', 'LocalSH', 'build'),
        os.path.join(repo_root, 'STAR-Edge', 'LocalSH'),
        os.path.join(repo_root, 'STAR-Edge', 'pre_process'),
    ]
    candidate_paths.extend(glob.glob(os.path.join(repo_root, 'STAR-Edge', 'LocalSH', 'build', 'lib.*')))

    for p in candidate_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    import LocalSH as _LocalSH  # pybind11 module
    LocalSH = _LocalSH
    LOCALSH_AVAILABLE = True
    print("[EdgeExtraction] LocalSH 模块加载成功")
except ImportError as e:
    pass  # 静默处理，使用高效备选方法


class EdgeExtractionConfig:
    """边缘提取配置参数"""
    def __init__(self, config_dict=None):
        # 方法选择: 'depth_gradient', 'fast_curvature', 'hybrid', 'star_edge', 'localsh'
        self.method = 'depth_gradient'  # 默认使用最快的深度梯度方法
        
        # 深度梯度方法参数
        self.depth_gradient_threshold = 0.1  # 深度梯度阈值（相对值）
        self.depth_sobel_ksize = 3  # Sobel算子核大小
        self.depth_canny_low = 50  # Canny低阈值
        self.depth_canny_high = 150  # Canny高阈值
        
        # 快速曲率方法参数
        self.knn_neighbors = 15  # KNN邻域大小（减少以加速）
        self.curvature_threshold = 0.1  # 曲率阈值
        self.edge_ratio = 0.1  # 保留的边缘点比例
        
        # 下采样参数
        self.voxel_size = 0.02  # 体素大小
        self.max_points = 20000  # 最大点数限制（减少以加速）
        self.min_points_ratio = 0.05
        
        # 边缘mask参数
        self.dilate_kernel_size = 5  # 膨胀核大小
        self.gaussian_blur_size = 3  # 高斯模糊核大小
        self.edge_weight = 2.0  # 边缘区域权重

        # STAR-Edge(LocalSH + MLP) 参数（“满血版”）
        # 说明：STAR-Edge 的预处理脚本默认 bw=10, kk=26, sampleNum=bw*4
        self.star_edge_bw = 10
        self.star_edge_kk = 26
        self.star_edge_sample_num = 40
        self.star_edge_threshold = 0.5
        self.star_edge_model_path = None  # 默认使用 STAR-Edge/model/best.ckpt
        self.star_edge_use_cuda = False   # 默认CPU推理（更稳）
        self.star_edge_max_points = 50000  # LocalSH最大点数（过大将自动子采样）
        self.star_edge_fallback = 'fast_curvature'
        self.star_edge_profile = False  # 打印耗时统计（用于benchmark）
        
        # 从配置字典更新
        if config_dict is not None:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict):
        """从配置字典更新参数"""
        edge_config = config_dict.get('edge_extraction', {})
        for key, value in edge_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ==================== STAR-Edge 推理辅助（带缓存）====================
_STAR_EDGE_NET = None
_STAR_EDGE_NET_DEVICE = None
_STAR_EDGE_NET_PATH = None
_STAR_EDGE_WARNED = False


def _load_star_edge_classifier(model_path: str, use_cuda: bool = False):
    """
    加载 STAR-Edge 的 MLP 分类器（DescClassifier）。
    - model_path: ckpt路径（默认 STAR-Edge/model/best.ckpt）
    - use_cuda: 是否把MLP放到CUDA（LocalSH仍在CPU侧计算）
    """
    global _STAR_EDGE_NET, _STAR_EDGE_NET_DEVICE, _STAR_EDGE_NET_PATH

    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    device_key = str(device)

    if _STAR_EDGE_NET is not None and _STAR_EDGE_NET_PATH == model_path and _STAR_EDGE_NET_DEVICE == device_key:
        return _STAR_EDGE_NET, device

    repo_root = os.path.dirname(os.path.dirname(__file__))
    star_edge_root = os.path.join(repo_root, "STAR-Edge")
    if os.path.isdir(star_edge_root) and star_edge_root not in sys.path:
        sys.path.insert(0, star_edge_root)

    try:
        from net import DescClassifier  # STAR-Edge/net.py
    except Exception as e:
        raise ImportError(f"无法导入 STAR-Edge/net.py 的 DescClassifier: {e}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"STAR-Edge 模型权重不存在: {model_path}")

    net = DescClassifier()
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # 允许strict=False以兼容不同保存格式（例如Lightning额外前缀）
    net.load_state_dict(state, strict=False)
    net.to(device=device)
    net.eval()

    _STAR_EDGE_NET = net
    _STAR_EDGE_NET_DEVICE = device_key
    _STAR_EDGE_NET_PATH = model_path
    return net, device


class EdgeExtractor:
    """
    高效点云边缘提取器
    
    支持多种边缘检测方法，针对实时SLAM优化。
    """
    
    def __init__(self, config=None):
        """
        初始化边缘提取器
        
        Args:
            config: EdgeExtractionConfig 或 dict
        """
        if config is None:
            self.config = EdgeExtractionConfig()
        elif isinstance(config, dict):
            self.config = EdgeExtractionConfig(config)
        else:
            self.config = config
        
    # ==================== 深度梯度方法（最快）====================
    
    def extract_edges_from_depth(self, depth_map: np.ndarray, 
                                  method: str = 'combined') -> np.ndarray:
        """
        从深度图直接提取边缘mask（最快的方法）
        
        Args:
            depth_map: 深度图 (H, W) 或 (1, H, W)
            method: 'sobel', 'canny', 'laplacian', 'combined'
            
        Returns:
            edge_mask: 边缘mask (H, W)，值为0-1
        """
        # #region agent log
        import json
        log_path = '/home/sjw/data0/lsx/S3PO_baseline/.cursor/debug.log'
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A,B",
                    "location": "edge_extraction.py:111",
                    "message": "extract_edges_from_depth entry",
                    "data": {"method": method, "depth_shape": str(depth_map.shape), "depth_dtype": str(depth_map.dtype)},
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        # 处理输入格式
        if len(depth_map.shape) == 3:
            depth_map = depth_map.squeeze(0)
        
        # 转换为float32
        depth = depth_map.astype(np.float32)
        
        # 处理无效深度值
        valid_mask = (depth > 0) & np.isfinite(depth)
        if not valid_mask.any():
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "edge_extraction.py:133",
                        "message": "no valid depth values",
                        "data": {"depth_min": float(depth.min()), "depth_max": float(depth.max()), "valid_count": int(valid_mask.sum())},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            return np.zeros_like(depth)
        
        # 归一化深度图
        depth_min = depth[valid_mask].min()
        depth_max = depth[valid_mask].max()
        if depth_max - depth_min < 1e-6:
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "edge_extraction.py:139",
                        "message": "depth range too small",
                        "data": {"depth_min": float(depth_min), "depth_max": float(depth_max), "diff": float(depth_max - depth_min)},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            return np.zeros_like(depth)
        
        depth_norm = np.zeros_like(depth)
        depth_norm[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min)
        
        # 转换为8位图像用于边缘检测
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        edge_sobel = None
        edge_canny = None
        
        if method == 'sobel' or method == 'combined':
            # Sobel边缘检测
            sobel_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=self.config.depth_sobel_ksize)
            sobel_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=self.config.depth_sobel_ksize)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_norm = sobel_mag / (sobel_mag.max() + 1e-8)
            
            if method == 'sobel':
                edge_mask = sobel_norm
            else:
                edge_sobel = sobel_norm
        
        if method == 'canny' or method == 'combined':
            # Canny边缘检测
            edges_canny = cv2.Canny(depth_uint8, 
                                     self.config.depth_canny_low, 
                                     self.config.depth_canny_high)
            edge_canny = edges_canny.astype(np.float32) / 255.0
            
            if method == 'canny':
                edge_mask = edge_canny
        
        if method == 'laplacian':
            # Laplacian边缘检测
            laplacian = cv2.Laplacian(depth_uint8, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            edge_mask = laplacian_abs / (laplacian_abs.max() + 1e-8)
        
        if method == 'combined':
            # 组合多种方法
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "edge_extraction.py:177",
                        "message": "combining edges",
                        "data": {"edge_sobel_is_none": edge_sobel is None, "edge_canny_is_none": edge_canny is None},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            if edge_sobel is None or edge_canny is None:
                # 回退到单一方法
                if edge_sobel is not None:
                    edge_mask = edge_sobel
                elif edge_canny is not None:
                    edge_mask = edge_canny
                else:
                    edge_mask = np.zeros_like(depth)
            else:
                edge_mask = np.maximum(edge_sobel, edge_canny)
        
        # 膨胀边缘
        if self.config.dilate_kernel_size > 1:
            kernel = np.ones((self.config.dilate_kernel_size, 
                            self.config.dilate_kernel_size), np.uint8)
            edge_mask = cv2.dilate(edge_mask.astype(np.float32), kernel, iterations=1)
        
        # 高斯模糊平滑
        if self.config.gaussian_blur_size > 1:
            edge_mask = cv2.GaussianBlur(edge_mask, 
                                         (self.config.gaussian_blur_size, 
                                          self.config.gaussian_blur_size), 0)
        
        # 归一化
        if edge_mask.max() > 0:
            edge_mask = edge_mask / edge_mask.max()
        
        # #region agent log
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A,B,C",
                    "location": "edge_extraction.py:195",
                    "message": "extract_edges_from_depth exit",
                    "data": {"edge_shape": str(edge_mask.shape), "edge_min": float(edge_mask.min()), "edge_max": float(edge_mask.max()), "edge_nonzero": int((edge_mask > 0.1).sum())},
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        return edge_mask.astype(np.float32)
    
    # ==================== 快速曲率方法 ====================
    
    def voxel_downsample(self, points: np.ndarray, 
                         voxel_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速体素下采样
        
        Args:
            points: 点云 (N, 3) 或 (N, D)
            voxel_size: 体素大小
            
        Returns:
            downsampled_points: 下采样后的点云
            indices: 保留点的原始索引
        """
        if voxel_size is None:
            voxel_size = self.config.voxel_size
        
        if len(points) == 0:
            return points, np.array([], dtype=np.int64)
        
        xyz = points[:, :3] if points.shape[1] > 3 else points
        
        # 计算体素索引
        min_bound = xyz.min(axis=0)
        voxel_indices = ((xyz - min_bound) / voxel_size).astype(np.int32)
        
        # 使用numpy的unique进行快速去重
        _, unique_indices = np.unique(
            voxel_indices[:, 0] * 1000000 + voxel_indices[:, 1] * 1000 + voxel_indices[:, 2],
            return_index=True
        )
        
        indices = unique_indices
        downsampled_points = points[indices]
        
        # 限制最大点数
        if len(downsampled_points) > self.config.max_points:
            sample_indices = np.random.choice(
                len(downsampled_points), 
                self.config.max_points, 
                replace=False
            )
            indices = indices[sample_indices]
            downsampled_points = downsampled_points[sample_indices]
        
        return downsampled_points, indices
    
    def compute_normals_fast(self, points: np.ndarray, k: int = None) -> np.ndarray:
        """
        使用KDTree快速计算点云法线
        
        Args:
            points: 点云 (N, 3)
            k: KNN邻域大小
            
        Returns:
            normals: 法线 (N, 3)
        """
        if k is None:
            k = self.config.knn_neighbors
        
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        if not SCIPY_AVAILABLE or N < k:
            # 回退到简单方法
            return self._compute_normals_simple(points, k)
        
        # 构建KDTree
        tree = cKDTree(points)
        
        # 批量查询最近邻
        distances, indices = tree.query(points, k=k, workers=-1)
        
        # 向量化计算法线
        for i in range(N):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
        
            # 使用SVD计算法线（最小特征值对应的特征向量）
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normals[i] = vh[-1]
            except:
                normals[i] = np.array([0, 0, 1])
        
        return normals
    
    def _compute_normals_simple(self, points: np.ndarray, k: int) -> np.ndarray:
        """简单的法线计算（无scipy时使用，使用批处理加速）"""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        k = min(k, N)
        
        # 使用批处理减少内存占用
        batch_size = min(500, N)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = points[start:end]
            
            # 计算批次内每个点到所有点的距离
            # shape: (batch_size, N)
            dists = np.sqrt(((batch[:, None, :] - points[None, :, :])**2).sum(axis=2))
            
            for i in range(end - start):
                knn_indices = np.argsort(dists[i])[:k]
                neighbors = points[knn_indices]
                
                centered = neighbors - neighbors.mean(axis=0)
                try:
                    _, s, vh = np.linalg.svd(centered, full_matrices=False)
                    normals[start + i] = vh[-1]
                except:
                    normals[start + i] = np.array([0, 0, 1])
        
        return normals
    
    def compute_curvature_fast(self, points: np.ndarray, 
                                normals: Optional[np.ndarray] = None,
                                k: int = None) -> np.ndarray:
        """
        使用KDTree快速计算点云曲率
        
        Args:
            points: 点云 (N, 3)
            normals: 法线 (N, 3)
            k: KNN邻域大小
            
        Returns:
            curvatures: 曲率 (N,)
        """
        if k is None:
            k = self.config.knn_neighbors
        
        N = len(points)
        
        if normals is None:
            normals = self.compute_normals_fast(points, k)
        
        curvatures = np.zeros(N, dtype=np.float32)
        
        if not SCIPY_AVAILABLE or N < k:
            return self._compute_curvature_simple(points, normals, k)
        
        # 构建KDTree
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k, workers=-1)
        
        # 向量化计算曲率
        for i in range(N):
            neighbor_normals = normals[indices[i][1:]]  # 排除自身
            normal_diffs = np.abs(np.dot(neighbor_normals, normals[i]))
            curvatures[i] = 1.0 - normal_diffs.mean()
        
        return curvatures
    
    def _compute_curvature_simple(self, points: np.ndarray, 
                                   normals: np.ndarray, k: int) -> np.ndarray:
        """简单的曲率计算（使用批处理加速）"""
        N = len(points)
        curvatures = np.zeros(N, dtype=np.float32)
        k = min(k, N)
        
        batch_size = min(500, N)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = points[start:end]
            batch_normals = normals[start:end]
            
            # 计算距离
            dists = np.sqrt(((batch[:, None, :] - points[None, :, :])**2).sum(axis=2))
            
            for i in range(end - start):
                knn_indices = np.argsort(dists[i])[1:k]  # 排除自身
                if len(knn_indices) == 0:
                    continue
                neighbor_normals = normals[knn_indices]
                # 计算法线夹角的余弦值
                normal_diffs = np.abs(np.dot(neighbor_normals, batch_normals[i]))
                # 曲率 = 1 - 平均法线一致性
                curvatures[start + i] = 1.0 - np.clip(normal_diffs.mean(), 0, 1)
        
        return curvatures
    
    def extract_edges_curvature_fast(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用快速曲率方法提取边缘点
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        N = len(points)
        
        if N < 10:
            return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32)
        
        # 对于大点云，先进行下采样以加速计算
        if N > 5000:
            # 随机采样子集进行曲率计算
            sample_size = min(5000, N)
            sample_indices = np.random.choice(N, sample_size, replace=False)
            sample_points = points[sample_indices]
            
            # 计算采样点的曲率
            normals = self.compute_normals_fast(sample_points)
            curvatures_sample = self.compute_curvature_fast(sample_points, normals)
            
            # 将曲率传播到所有点（使用最近邻）
            curvatures = np.zeros(N, dtype=np.float32)
            curvatures[sample_indices] = curvatures_sample
            
            # 对未采样的点，使用最近采样点的曲率
            non_sample_mask = np.ones(N, dtype=bool)
            non_sample_mask[sample_indices] = False
            non_sample_indices = np.where(non_sample_mask)[0]
            
            if len(non_sample_indices) > 0:
                # 简单方法：使用采样点曲率的平均值
                curvatures[non_sample_indices] = curvatures_sample.mean()
        else:
            # 小点云直接计算
            normals = self.compute_normals_fast(points)
            curvatures = self.compute_curvature_fast(points, normals)
        
        # 归一化曲率到0-1
        curv_min, curv_max = curvatures.min(), curvatures.max()
        if curv_max - curv_min > 1e-6:
            edge_scores = (curvatures - curv_min) / (curv_max - curv_min)
        else:
            edge_scores = np.zeros(N, dtype=np.float32)
            
        # 选择边缘点：取前edge_ratio的高曲率点
        num_edges = max(10, int(N * self.config.edge_ratio))
        num_edges = min(num_edges, N)
        
        if num_edges >= N:
            edge_mask = np.ones(N, dtype=bool)
        else:
            # 使用argpartition更快地找到top-k
            threshold_idx = N - num_edges
            threshold = np.partition(curvatures, threshold_idx)[threshold_idx]
            edge_mask = curvatures >= threshold
        
        return edge_mask, edge_scores
    
    # ==================== 主接口 ====================
    
    def extract_edges(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取边缘点（自动选择方法）
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        method = getattr(self.config, "method", "fast_curvature")

        if method == "fast_curvature":
            return self.extract_edges_curvature_fast(points)

        if method in ("star_edge", "localsh"):
            edge_mask, edge_scores = self.extract_edges_star_edge(points)
            if edge_mask is not None and edge_scores is not None:
                return edge_mask, edge_scores

            # star-edge不可用时回退
            fallback = getattr(self.config, "star_edge_fallback", "fast_curvature")
            if fallback == "fast_curvature":
                return self.extract_edges_curvature_fast(points)
            # 兜底
            return self.extract_edges_curvature_fast(points)

        if method == "hybrid":
            # 当前工程的“hybrid”主要用于2D(RGB/Depth)融合，这里对3D点云先回退到曲率
            return self.extract_edges_curvature_fast(points)

        # 未知方法：回退
        return self.extract_edges_curvature_fast(points)

    def extract_edges_star_edge(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        STAR-Edge “满血”3D轮廓提取：
        - LocalSHFeature: 计算 Local Spherical Curve 描述子（默认 bw=10）
        - DescClassifier(MLP): 对每点输出边缘概率

        Returns:
            edge_mask, edge_scores
            若 LocalSH/模型不可用，则返回 (None, None) 让上层回退。
        """
        if points is None:
            return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)
        pts = np.asarray(points)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

        global _STAR_EDGE_WARNED
        if (not LOCALSH_AVAILABLE) or (LocalSH is None):
            if not _STAR_EDGE_WARNED:
                print("[EdgeExtraction] STAR-Edge(LocalSH) 不可用：未成功导入 LocalSH.so，将回退到 fast_curvature（可检查fftw依赖/LocalSH编译版本）")
                _STAR_EDGE_WARNED = True
            return None, None

        # LocalSH 需要 float64
        pts64 = np.asarray(pts[:, :3], dtype=np.float64)
        N = pts64.shape[0]
        profile = bool(getattr(self.config, "star_edge_profile", False))
        t_total0 = time.perf_counter() if profile else None

        # 过大点云自动子采样，避免LocalSH拖慢
        maxN = int(getattr(self.config, "star_edge_max_points", 50000))
        if N > maxN:
            # 避免递归时重复打印profile
            prev_profile = getattr(self.config, "star_edge_profile", False)
            if profile:
                setattr(self.config, "star_edge_profile", False)
            idx = np.random.choice(N, size=maxN, replace=False)
            sub_pts = pts64[idx]
            t_sub0 = time.perf_counter() if profile else None
            sub_mask, sub_scores = self.extract_edges_star_edge(sub_pts)
            t_sub1 = time.perf_counter() if profile else None
            if profile:
                setattr(self.config, "star_edge_profile", prev_profile)
            if sub_mask is None or sub_scores is None:
                return None, None

            # 将子集分数传播到全量点云（优先使用cKDTree，避免粗暴填充）
            t_prop0 = time.perf_counter() if profile else None
            if SCIPY_AVAILABLE:
                try:
                    tree = cKDTree(sub_pts)
                    _, nn = tree.query(pts64, k=1, workers=-1)
                    scores = sub_scores[np.asarray(nn, dtype=np.int64)].astype(np.float32)
                except Exception:
                    scores = np.zeros((N,), dtype=np.float32)
                    scores[idx] = sub_scores.astype(np.float32)
            else:
                # 无scipy时退化：未采样点填充均值（比填0更稳）
                scores = np.zeros((N,), dtype=np.float32)
                scores[idx] = sub_scores.astype(np.float32)
                fill = float(sub_scores.mean()) if len(sub_scores) > 0 else 0.0
                scores[scores == 0] = fill
            t_prop1 = time.perf_counter() if profile else None

            thr = float(getattr(self.config, "star_edge_threshold", 0.5))
            mask = scores >= thr
            if profile:
                t_total1 = time.perf_counter()
                print(
                    "[STAR-Edge] N={} maxN={} sub_ms={:.2f} prop_ms={:.2f} total_ms={:.2f} scipy={}".format(
                        int(N),
                        int(maxN),
                        (t_sub1 - t_sub0) * 1000.0 if (t_sub0 is not None and t_sub1 is not None) else -1.0,
                        (t_prop1 - t_prop0) * 1000.0 if (t_prop0 is not None and t_prop1 is not None) else -1.0,
                        (t_total1 - t_total0) * 1000.0 if t_total0 is not None else -1.0,
                        bool(SCIPY_AVAILABLE),
                    )
                )
            return mask.astype(bool), scores.astype(np.float32)

        bw = int(getattr(self.config, "star_edge_bw", 10))
        kk = int(getattr(self.config, "star_edge_kk", 26))
        sample_num = int(getattr(self.config, "star_edge_sample_num", bw * 4))

        # 计算LocalSH描述子
        try:
            localsh_feature = getattr(LocalSH, "LocalSHFeature", None)
            if localsh_feature is None:
                return None, None
            t_lsh0 = time.perf_counter() if profile else None
            result = localsh_feature.ComLSHF_knn_upsample(pts64, bw, kk, sample_num)
            t_lsh1 = time.perf_counter() if profile else None
            desc = result["Descs"]  # (N, bw)
        except Exception:
            return None, None

        # MLP 推理
        try:
            repo_root = os.path.dirname(os.path.dirname(__file__))
            default_ckpt = os.path.join(repo_root, "STAR-Edge", "model", "best.ckpt")
            model_path = getattr(self.config, "star_edge_model_path", None) or default_ckpt
            use_cuda = bool(getattr(self.config, "star_edge_use_cuda", False))
            net, device = _load_star_edge_classifier(model_path, use_cuda=use_cuda)

            t_mlp0 = time.perf_counter() if profile else None
            x = torch.from_numpy(np.asarray(desc, dtype=np.float32)).to(device=device)
            with torch.no_grad():
                prob = net(x).detach().cpu().numpy().astype(np.float32)
            t_mlp1 = time.perf_counter() if profile else None
        except Exception:
            return None, None

        thr = float(getattr(self.config, "star_edge_threshold", 0.5))
        edge_scores = np.clip(prob, 0.0, 1.0).astype(np.float32)
        edge_mask = (edge_scores >= thr).astype(bool)
        if profile:
            t_total1 = time.perf_counter()
            print(
                "[STAR-Edge] N={} bw={} kk={} sample={} localsh_ms={:.2f} mlp_ms={:.2f} total_ms={:.2f} device={}".format(
                    int(N),
                    int(bw),
                    int(kk),
                    int(sample_num),
                    (t_lsh1 - t_lsh0) * 1000.0 if (t_lsh0 is not None and t_lsh1 is not None) else -1.0,
                    (t_mlp1 - t_mlp0) * 1000.0 if (t_mlp0 is not None and t_mlp1 is not None) else -1.0,
                    (t_total1 - t_total0) * 1000.0 if t_total0 is not None else -1.0,
                    str(device),
                )
            )
        return edge_mask, edge_scores
    
    def project_points_to_image(self, points_3d: np.ndarray, 
                                 K: np.ndarray, R: np.ndarray, T: np.ndarray,
                                 width: int, height: int,
                                 dist_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将3D点投影到图像空间
        
        Args:
            points_3d: 3D点 (N, 3)
            K: 相机内参矩阵 (3, 3)
            R: 旋转矩阵 (3, 3)
            T: 平移向量 (3,)
            width: 图像宽度
            height: 图像高度
            dist_coeffs: 畸变系数
            
        Returns:
            points_2d: 2D点 (M, 2)
            valid_mask: 有效点mask (N,)
        """
        # 转换到相机坐标系
        points_cam = (R @ points_3d.T).T + T
        
        # 过滤深度为负的点
        valid_depth = points_cam[:, 2] > 0.01
        
        if not valid_depth.any():
            return np.array([]).reshape(0, 2), np.zeros(len(points_3d), dtype=bool)
        
        # 投影
        if dist_coeffs is not None and np.any(dist_coeffs != 0):
            rvec, _ = cv2.Rodrigues(R)
            points_2d, _ = cv2.projectPoints(
                points_3d[valid_depth], rvec, T, K, dist_coeffs
            )
            points_2d = points_2d.squeeze(1)
        else:
            points_2d = (K @ points_cam[valid_depth].T).T
            points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)
        
        # 检查是否在图像内
        in_image = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        )
        
        # 构建完整的有效mask
        valid_mask = np.zeros(len(points_3d), dtype=bool)
        valid_indices = np.where(valid_depth)[0][in_image]
        valid_mask[valid_indices] = True
        
        return points_2d[in_image], valid_mask
    
    def create_edge_mask(self, edge_points_2d: np.ndarray, 
                         width: int, height: int,
                         edge_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建边缘mask图像
        
        Args:
            edge_points_2d: 2D边缘点 (N, 2)
            width: 图像宽度
            height: 图像高度
            edge_scores: 边缘分数 (N,)
            
        Returns:
            edge_mask: 边缘mask图像 (H, W)
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        if len(edge_points_2d) == 0:
            return mask
        
        # 将点坐标转换为整数
        points_int = np.round(edge_points_2d).astype(np.int32)
        
        # 使用向量化操作设置边缘点
        valid = (points_int[:, 0] >= 0) & (points_int[:, 0] < width) & \
                (points_int[:, 1] >= 0) & (points_int[:, 1] < height)
        
        valid_points = points_int[valid]
        if edge_scores is not None:
            valid_scores = edge_scores[valid]
            for i, (x, y) in enumerate(valid_points):
                mask[y, x] = max(mask[y, x], valid_scores[i])
        else:
            mask[valid_points[:, 1], valid_points[:, 0]] = 1.0
        
        # 膨胀边缘区域
        if self.config.dilate_kernel_size > 1:
            kernel = np.ones((self.config.dilate_kernel_size, 
                            self.config.dilate_kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 高斯模糊平滑
        if self.config.gaussian_blur_size > 1:
            mask = cv2.GaussianBlur(mask, 
                                    (self.config.gaussian_blur_size, 
                                     self.config.gaussian_blur_size), 0)
        
        # 归一化
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask

    # ==================== RGB(+Depth 可选) 轮廓/边缘方法 ====================

    def extract_edges_from_rgb(self,
                               rgb_image: Union[np.ndarray, torch.Tensor],
                               method: str = 'canny',
                               canny_low: int = 50,
                               canny_high: int = 150,
                               sobel_ksize: int = 3,
                               dilate_kernel_size: int = 3,
                               blur_ksize: int = 3) -> np.ndarray:
        """
        从RGB图像直接提取边缘mask（0-1 float32）。

        支持输入：
        - np.ndarray: HWC(uint8/float) 或 CHW(float)
        - torch.Tensor: CHW 或 1CHW（来自view['img']）
        """
        # torch -> numpy
        if isinstance(rgb_image, torch.Tensor):
            x = rgb_image.detach().cpu()
            if x.ndim == 4:
                x = x[0]
            # dust3r/ImgNorm: (x - 0.5)/0.5 => [-1,1]，这里反归一化
            if x.dtype.is_floating_point:
                x = (x * 0.5 + 0.5).clamp(0, 1)
                x = (x * 255.0).to(torch.uint8)
            x = x.permute(1, 2, 0).contiguous().numpy()  # HWC
            rgb = x
        else:
            rgb = rgb_image

        # 统一为HWC uint8 RGB
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255)
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0)
            rgb = rgb.astype(np.uint8)

        # to gray
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if method == 'canny':
            edges = cv2.Canny(gray, canny_low, canny_high)
        elif method == 'sobel':
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
            mag = cv2.magnitude(sx, sy)
            mag = mag / (mag.max() + 1e-6)
            edges = (mag * 255.0).astype(np.uint8)
        elif method == 'laplacian':
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            lap = np.abs(lap)
            lap = lap / (lap.max() + 1e-6)
            edges = (lap * 255.0).astype(np.uint8)
        elif method == 'combined':
            e1 = cv2.Canny(gray, canny_low, canny_high).astype(np.float32) / 255.0
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
            mag = cv2.magnitude(sx, sy)
            mag = mag / (mag.max() + 1e-6)
            edges = (np.maximum(e1, mag) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Unknown rgb edge method: {method}")

        edge_mask = edges.astype(np.float32) / 255.0

        if dilate_kernel_size and dilate_kernel_size > 1:
            kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        if blur_ksize and blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            edge_mask = cv2.GaussianBlur(edge_mask, (blur_ksize, blur_ksize), 0)

        if edge_mask.max() > 0:
            edge_mask = edge_mask / edge_mask.max()

        return edge_mask.astype(np.float32)

    def extract_edges_from_rgb_and_depth(self,
                                         rgb_image: Union[np.ndarray, torch.Tensor],
                                         depth_map: Optional[np.ndarray] = None,
                                         rgb_method: str = 'canny',
                                         depth_method: str = 'combined',
                                         fuse: str = 'max',
                                         rgb_weight: float = 1.0,
                                         depth_weight: float = 1.0,
                                         **rgb_kwargs) -> np.ndarray:
        """
        从RGB（可选融合深度）提取边缘mask（0-1 float32）。

        - depth_map=None 时只用RGB
        - fuse: 'max' 或 'sum'
        """
        rgb_edge = self.extract_edges_from_rgb(rgb_image, method=rgb_method, **rgb_kwargs)
        if depth_map is None:
            return rgb_edge

        depth_edge = self.extract_edges_from_depth(depth_map, method=depth_method)
        # 尺寸对齐（以rgb为准）
        if depth_edge.shape != rgb_edge.shape:
            depth_edge = cv2.resize(depth_edge, (rgb_edge.shape[1], rgb_edge.shape[0]), interpolation=cv2.INTER_LINEAR)

        if fuse == 'max':
            fused = np.maximum(rgb_weight * rgb_edge, depth_weight * depth_edge)
        elif fuse == 'sum':
            fused = rgb_weight * rgb_edge + depth_weight * depth_edge
        else:
            raise ValueError(f"Unknown fuse mode: {fuse}")

        if fused.max() > 0:
            fused = fused / fused.max()
        return fused.astype(np.float32)
    
    def process_frame(self, gaussians, viewpoint, 
                      K: Optional[np.ndarray] = None,
                      dist_coeffs: Optional[np.ndarray] = None,
                      render_depth: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        处理单帧：提取边缘并生成mask
        
        支持两种模式：
        1. 从深度图直接提取边缘（快速，推荐）
        2. 从3D点云提取边缘（较慢但更准确）
        
        Args:
            gaussians: GaussianModel 实例
            viewpoint: Camera 视点信息
            K: 相机内参矩阵
            dist_coeffs: 畸变系数
            render_depth: 渲染深度图（如果提供，使用深度梯度方法）
            
        Returns:
            edge_mask: 边缘mask图像 (H, W)
            edge_points_2d: 2D边缘点（仅点云方法）
            edge_points_3d: 3D边缘点（仅点云方法）
        """
        # 优先使用深度梯度方法（最快）
        if render_depth is not None and self.config.method == 'depth_gradient':
            edge_mask = self.extract_edges_from_depth(render_depth, method='combined')
            return edge_mask, None, None
        
        # 从高斯模型提取点云
        if hasattr(gaussians, 'get_xyz'):
            xyz = gaussians.get_xyz
            if isinstance(xyz, torch.Tensor):
                points_3d = xyz.detach().cpu().numpy()
            else:
                points_3d = xyz
        else:
            return None, None, None
        
        if len(points_3d) == 0:
            return None, None, None
        
        # 下采样
        downsampled_points, _ = self.voxel_downsample(points_3d)
        
        if len(downsampled_points) < 10:
            return None, None, None
        
        # 提取边缘
        edge_mask_3d, edge_scores = self.extract_edges(downsampled_points[:, :3])
        edge_points_3d = downsampled_points[edge_mask_3d]
        edge_scores_filtered = edge_scores[edge_mask_3d]
        
        if len(edge_points_3d) == 0:
            return None, None, None
        
        # 获取相机参数
        if K is None:
            K = np.array([
                [viewpoint.fx, 0, viewpoint.cx],
                [0, viewpoint.fy, viewpoint.cy],
                [0, 0, 1]
            ])
        
        R = viewpoint.R.cpu().numpy() if isinstance(viewpoint.R, torch.Tensor) else viewpoint.R
        T = viewpoint.T.cpu().numpy() if isinstance(viewpoint.T, torch.Tensor) else viewpoint.T
        
        # 投影到图像空间
        edge_points_2d, valid_mask = self.project_points_to_image(
            edge_points_3d, K, R, T,
            viewpoint.image_width, viewpoint.image_height,
            dist_coeffs
        )
        
        # 创建边缘mask
        valid_scores = edge_scores_filtered[valid_mask] if len(edge_scores_filtered) > 0 else None
        edge_mask_img = self.create_edge_mask(
            edge_points_2d, 
            viewpoint.image_width, 
            viewpoint.image_height,
            valid_scores
        )
        
        return edge_mask_img, edge_points_2d, edge_points_3d
    
    def process_frame_from_depth(self, render_depth: np.ndarray,
                                  width: int, height: int) -> np.ndarray:
        """
        从渲染深度图直接提取边缘mask（最快的方法）
        
        Args:
            render_depth: 渲染深度图 (H, W) 或 (1, H, W)
            width: 目标宽度
            height: 目标高度
            
        Returns:
            edge_mask: 边缘mask图像 (H, W)
        """
        edge_mask = self.extract_edges_from_depth(render_depth, method='combined')
        
        # 如果尺寸不匹配，调整大小
        if edge_mask.shape != (height, width):
            edge_mask = cv2.resize(edge_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return edge_mask
    
    def enhance_matches_with_edges(self, matches_im1: np.ndarray, 
                                    matches_im2: np.ndarray,
                                    edge_mask1: np.ndarray, 
                                    edge_mask2: np.ndarray,
                                    weight: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用边缘mask增强匹配点权重
        
        Args:
            matches_im1: 图像1的匹配点 (N, 2)
            matches_im2: 图像2的匹配点 (N, 2)
            edge_mask1: 图像1的边缘mask (H1, W1)
            edge_mask2: 图像2的边缘mask (H2, W2)
            weight: 边缘区域权重倍数
            
        Returns:
            weights: 匹配点权重 (N,)
            edge_matches_mask: 边缘区域匹配点mask (N,) bool
        """
        if weight is None:
            weight = self.config.edge_weight
        
        N = len(matches_im1)
        
        if edge_mask1 is None or edge_mask2 is None:
            return np.ones(N), np.zeros(N, dtype=bool)
        
        H1, W1 = edge_mask1.shape
        H2, W2 = edge_mask2.shape
        
        # 向量化操作
        x1 = np.clip(np.round(matches_im1[:, 0]).astype(int), 0, W1 - 1)
        y1 = np.clip(np.round(matches_im1[:, 1]).astype(int), 0, H1 - 1)
        x2 = np.clip(np.round(matches_im2[:, 0]).astype(int), 0, W2 - 1)
        y2 = np.clip(np.round(matches_im2[:, 1]).astype(int), 0, H2 - 1)
        
        scores1 = edge_mask1[y1, x1]
        scores2 = edge_mask2[y2, x2]
                
        max_scores = np.maximum(scores1, scores2)
        edge_matches_mask = max_scores > 0.1
        weights = 1.0 + weight * max_scores
        
        return weights, edge_matches_mask


# ==================== 工厂函数和便捷接口 ====================

def create_edge_extractor(config=None) -> EdgeExtractor:
    """
    创建边缘提取器实例
    
    Args:
        config: 配置字典或EdgeExtractionConfig
        
    Returns:
        EdgeExtractor 实例
    """
    return EdgeExtractor(config)


def extract_edges_from_pointcloud(points: np.ndarray, 
                                   config=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从点云提取边缘点（便捷函数）
    
    Args:
        points: 点云 (N, 3)
        config: 配置
        
    Returns:
        edge_mask: 边缘点mask
        edge_scores: 边缘分数
    """
    extractor = EdgeExtractor(config)
    return extractor.extract_edges(points)


def extract_edges_from_depth_map(depth_map: np.ndarray, 
                                  config=None) -> np.ndarray:
    """
    从深度图提取边缘mask（便捷函数，最快）
    
    Args:
        depth_map: 深度图 (H, W)
        config: 配置
        
    Returns:
        edge_mask: 边缘mask (H, W)
    """
    extractor = EdgeExtractor(config)
    return extractor.extract_edges_from_depth(depth_map)


def create_edge_mask_from_gaussians(gaussians, viewpoint, 
                                     config=None, 
                                     dist_coeffs=None,
                                     render_depth=None) -> Optional[np.ndarray]:
    """
    从高斯模型创建边缘mask图像（便捷函数）
    
    Args:
        gaussians: GaussianModel 实例
        viewpoint: Camera 视点信息
        config: 配置
        dist_coeffs: 畸变系数
        render_depth: 渲染深度图（如果提供，使用快速深度梯度方法）
        
    Returns:
        edge_mask: 边缘mask图像 (H, W)
    """
    extractor = EdgeExtractor(config)
    edge_mask, _, _ = extractor.process_frame(
        gaussians, viewpoint, 
        dist_coeffs=dist_coeffs,
        render_depth=render_depth
    )
    return edge_mask
