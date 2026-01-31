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
import importlib.util
from typing import Optional, Tuple, Union

# 尝试导入scipy用于快速KNN
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[EdgeExtraction] scipy不可用，将使用基础方法")

# 强制使用STAR-Edge的LocalSH（无回退机制）
LOCALSH_AVAILABLE = False
LocalSH = None
LocalSHFeature = None

# 尝试从pre_process目录导入（.so文件所在位置）
localsh_preprocess_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'STAR-Edge', 'pre_process')
# 尝试多个可能的.so文件名（Python 3.8, 3.11等）
import sys
python_version = f"{sys.version_info.major}{sys.version_info.minor}"
localsh_so_path = os.path.join(localsh_preprocess_path, f'LocalSH.cpython-{python_version}-x86_64-linux-gnu.so')
# 如果找不到，尝试Python 3.8版本（向后兼容）
if not os.path.exists(localsh_so_path):
    localsh_so_path = os.path.join(localsh_preprocess_path, 'LocalSH.cpython-38-x86_64-linux-gnu.so')
# 如果还是找不到，尝试从LocalSH目录直接加载
if not os.path.exists(localsh_so_path):
    localsh_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'STAR-Edge', 'LocalSH')
    localsh_so_path = os.path.join(localsh_dir, f'LocalSH.cpython-{python_version}-x86_64-linux-gnu.so')

if os.path.exists(localsh_so_path):
    sys.path.insert(0, localsh_preprocess_path)
    try:
        # 使用importlib直接加载.so文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("LocalSH", localsh_so_path)
        if spec is not None and spec.loader is not None:
            _LocalSH = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_LocalSH)
            LocalSH = _LocalSH
            
            # 尝试访问LocalSHFeature子模块
            if hasattr(_LocalSH, 'LocalSHFeature'):
                LocalSHFeature = _LocalSH.LocalSHFeature
                LOCALSH_AVAILABLE = True
                print("[EdgeExtraction] ✓ LocalSH 模块加载成功（强制使用STAR-Edge）")
            else:
                raise ImportError("LocalSH模块缺少LocalSHFeature子模块")
        else:
            raise ImportError(f"无法创建LocalSH模块规范: {localsh_so_path}")
    except Exception as e:
        raise RuntimeError(f"[EdgeExtraction] ✗ LocalSH 模块加载失败: {e}\n"
                          f"请确保STAR-Edge的LocalSH模块已正确编译。\n"
                          f"路径: {localsh_so_path}")
else:
    raise RuntimeError(f"[EdgeExtraction] ✗ LocalSH .so文件不存在: {localsh_so_path}\n"
                      f"请确保STAR-Edge已正确安装。")


class EdgeExtractionConfig:
    """边缘提取配置参数"""
    def __init__(self, config_dict=None):
        # 方法选择: 'depth_gradient', 'fast_curvature', 'hybrid'
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
        
        # 从配置字典更新
        if config_dict is not None:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict):
        """从配置字典更新参数"""
        edge_config = config_dict.get('edge_extraction', {})
        for key, value in edge_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


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
            # 使用体素下采样而不是随机采样，保持空间分布
            downsampled_points, sample_indices = self.voxel_downsample(points, voxel_size=self.config.voxel_size)
            sample_size = len(downsampled_points)
            
            if sample_size < 10:
                # 如果下采样后点数太少，回退到随机采样
                sample_size = min(5000, N)
                sample_indices = np.random.choice(N, sample_size, replace=False)
                sample_points = points[sample_indices]
            else:
                sample_points = downsampled_points
            
            # 计算采样点的曲率
            normals = self.compute_normals_fast(sample_points)
            curvatures_sample = self.compute_curvature_fast(sample_points, normals)
            
            # 将曲率传播到所有点（使用KNN找到最近邻）
            curvatures = np.zeros(N, dtype=np.float32)
            curvatures[sample_indices] = curvatures_sample
            
            # 对未采样的点，使用KNN找到最近采样点的曲率
            non_sample_mask = np.ones(N, dtype=bool)
            non_sample_mask[sample_indices] = False
            non_sample_indices = np.where(non_sample_mask)[0]
            
            if len(non_sample_indices) > 0 and SCIPY_AVAILABLE:
                # 使用KDTree找到最近邻采样点
                from scipy.spatial import cKDTree
                tree = cKDTree(sample_points)
                _, nearest_indices = tree.query(points[non_sample_indices], k=1, workers=-1)
                # 使用最近邻采样点的曲率
                curvatures[non_sample_indices] = curvatures_sample[nearest_indices]
            elif len(non_sample_indices) > 0:
                # 回退方法：使用采样点曲率的中位数（比平均值更稳健）
                curvatures[non_sample_indices] = np.median(curvatures_sample)
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
            
        # 选择边缘点：优先使用阈值，如果阈值无效则使用比例
        # 计算曲率的统计信息
        curv_mean = curvatures.mean()
        curv_std = curvatures.std()
        
        # 使用阈值方法：曲率 > mean + threshold * std
        threshold_value = curv_mean + self.config.curvature_threshold * curv_std
        edge_mask_by_threshold = curvatures > threshold_value
        
        # 使用比例方法：取前edge_ratio的高曲率点
        num_edges = max(10, int(N * self.config.edge_ratio))
        num_edges = min(num_edges, N)
        
        if num_edges >= N:
            edge_mask_by_ratio = np.ones(N, dtype=bool)
        else:
            threshold_idx = N - num_edges
            threshold_ratio = np.partition(curvatures, threshold_idx)[threshold_idx]
            edge_mask_by_ratio = curvatures >= threshold_ratio
        
        # 智能选择策略：优先使用阈值方法，但如果结果不合理则调整
        edge_mask = edge_mask_by_threshold
        
        # 如果阈值方法选的点太少（<0.5%），降低阈值
        if edge_mask.sum() < N * 0.005:
            # 使用更宽松的阈值：mean + 0.5 * threshold * std
            threshold_value_relaxed = curv_mean + 0.5 * self.config.curvature_threshold * curv_std
            edge_mask = curvatures > threshold_value_relaxed
            # 如果还是太少，使用比例方法作为下限
            if edge_mask.sum() < N * 0.005:
                edge_mask = edge_mask_by_ratio
        # 如果阈值方法选的点太多（>30%），提高阈值
        elif edge_mask.sum() > N * 0.3:
            # 使用更严格的阈值：mean + 1.5 * threshold * std
            threshold_value_strict = curv_mean + 1.5 * self.config.curvature_threshold * curv_std
            edge_mask = curvatures > threshold_value_strict
            # 如果还是太多，使用比例方法作为上限
            if edge_mask.sum() > N * 0.3:
                edge_mask = edge_mask_by_ratio
        
        return edge_mask, edge_scores
    
    # ==================== 主接口 ====================
    
    def extract_edges(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取边缘点（强制使用STAR-Edge方法，无回退）
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        # 强制使用STAR-Edge的LocalSH方法（无回退机制）
        if not LOCALSH_AVAILABLE:
            raise RuntimeError("[轮廓提取] ✗ LocalSH不可用，无法使用STAR-Edge方法。"
                             "请确保LocalSH模块已正确加载。")
        
        method_name = getattr(self.config, 'method', 'star_edge')
        if method_name not in ['star_edge', 'localsh']:
            print(f"[轮廓提取] 警告: method={method_name}，强制使用STAR-Edge方法")
        
        print(f"[轮廓提取] ✓ 使用STAR-Edge LocalSH方法提取轮廓")
        return self.extract_edges_star_edge(points)
    
    def extract_edges_star_edge(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用STAR-Edge的LocalSH方法提取边缘点（强制方法，无回退）
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        if not LOCALSH_AVAILABLE or LocalSHFeature is None:
            raise RuntimeError("[轮廓提取] ✗ LocalSH模块不可用，无法使用STAR-Edge方法")
        
        N = len(points)
        if N < 10:
            return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float32)
        
        # STAR-Edge参数
        bw = 10  # 球面谐波带宽
        kk = 26  # KNN邻居数
        sampleNum = bw * 4  # 采样数
        
        # 调用LocalSH计算特征（强制使用，无回退）
        if LocalSHFeature is None:
            raise RuntimeError("[轮廓提取] ✗ LocalSHFeature不可用")
        
        result = LocalSHFeature.ComLSHF_knn_upsample(
            points.astype(np.float64),
            bw, kk, sampleNum
        )
        
        # 获取特征描述符和法线
        descs = result["Descs"]  # (N, bw) - 局部球面曲线特征
        normals = result["normals"]  # (N, 3) - 法线
        
        # 使用特征描述符的方差或熵来识别边缘点
        # 边缘点的特征描述符通常有更高的变化
        if descs.shape[1] > 0:
            # 计算每个点特征描述符的方差
            desc_var = np.var(descs, axis=1)
            
            # 归一化到0-1
            if desc_var.max() - desc_var.min() > 1e-6:
                edge_scores = (desc_var - desc_var.min()) / (desc_var.max() - desc_var.min())
            else:
                edge_scores = np.zeros(N, dtype=np.float32)
            
            # 使用阈值方法选择边缘点
            curv_mean = desc_var.mean()
            curv_std = desc_var.std()
            threshold_value = curv_mean + self.config.curvature_threshold * curv_std
            edge_mask = desc_var > threshold_value
            
            # 如果阈值方法选的点太多或太少，使用比例方法调整
            # 计算目标轮廓点数量（考虑比例和绝对上限）
            num_edges_by_ratio = max(10, int(N * self.config.edge_ratio))
            # 如果有max_points配置，限制绝对数量上限
            if hasattr(self.config, 'max_points') and self.config.max_points > 0:
                num_edges_threshold = min(num_edges_by_ratio, self.config.max_points)
            else:
                num_edges_threshold = num_edges_by_ratio
            
            if edge_mask.sum() < num_edges_threshold * 0.5:
                # 太少，降低阈值（但不超过edge_ratio的2倍）
                threshold_value = curv_mean + 0.5 * self.config.curvature_threshold * curv_std
                edge_mask = desc_var > threshold_value
                # 如果还是太少，使用比例方法确保至少有一些边缘点
                if edge_mask.sum() < num_edges_threshold * 0.5:
                    threshold_idx = N - num_edges_threshold
                    threshold_ratio = np.partition(desc_var, threshold_idx)[threshold_idx]
                    edge_mask = desc_var >= threshold_ratio
            elif edge_mask.sum() > num_edges_threshold * 2.0:
                # 太多，使用比例方法限制到目标数量（考虑max_points上限）
                threshold_idx = N - num_edges_threshold
                threshold_ratio = np.partition(desc_var, threshold_idx)[threshold_idx]
                edge_mask = desc_var >= threshold_ratio
            
            # 最终检查：确保不超过max_points绝对上限
            if hasattr(self.config, 'max_points') and self.config.max_points > 0:
                if edge_mask.sum() > self.config.max_points:
                    # 如果超过上限，只保留分数最高的max_points个点
                    edge_scores_sorted = np.argsort(desc_var)[::-1]  # 从高到低排序
                    edge_mask = np.zeros(N, dtype=bool)
                    edge_mask[edge_scores_sorted[:self.config.max_points]] = True
                    print(f"[轮廓提取] 轮廓点数量({edge_mask.sum()})超过上限({self.config.max_points})，已限制到上限")
            
            # 如果阈值方法选的点在合理范围内，保持原结果
        else:
            # 回退到法线变化方法
            # 计算法线变化（使用KNN）
            edge_mask = np.zeros(N, dtype=bool)
            edge_scores = np.zeros(N, dtype=np.float32)
        
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
