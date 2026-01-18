"""
STAR-Edge 点云轮廓提取工具封装

该模块提供点云边缘/轮廓提取功能，用于增强形变场景下的图像匹配。
使用独立的"临时"下采样机制，不影响SLAM主流程的下采样。

主要功能：
1. 临时体素下采样 - 针对边缘提取的独立下采样
2. 边缘提取 - 使用LocalSH库或备选的曲率方法
3. 边缘投影 - 将3D边缘点投影到图像空间
4. 边缘mask生成 - 生成用于增强特征匹配的边缘mask
"""

import numpy as np
import torch
import cv2
import sys
import os

# 尝试导入LocalSH
LOCALSH_AVAILABLE = False
LocalSH = None

try:
    # 添加LocalSH路径
    localsh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'STAR-Edge', 'LocalSH', 'build')
    if os.path.exists(localsh_path):
        sys.path.insert(0, localsh_path)
    
    import LocalSH as _LocalSH
    LocalSH = _LocalSH
    LOCALSH_AVAILABLE = True
    print("[EdgeExtraction] LocalSH 模块加载成功")
except ImportError as e:
    print(f"[EdgeExtraction] LocalSH 模块不可用，将使用曲率方法: {e}")


class EdgeExtractionConfig:
    """边缘提取配置参数"""
    def __init__(self, config_dict=None):
        # 临时下采样参数（独立于SLAM下采样）
        self.voxel_size = 0.05  # 体素大小（米），用于临时下采样
        self.min_points_ratio = 0.1  # 下采样后最少保留的点比例
        self.max_points = 50000  # 最大点数限制
        
        # 边缘提取参数
        self.knn_neighbors = 20  # KNN邻域大小
        self.curvature_threshold = 0.01  # 曲率阈值
        self.normal_angle_threshold = 15.0  # 法线角度阈值（度）
        self.edge_ratio = 0.2  # 保留的边缘点比例
        
        # LocalSH参数
        self.sh_order = 4  # 球谐阶数
        self.search_radius = 0.1  # 搜索半径
        
        # 边缘mask参数
        self.dilate_kernel_size = 5  # 膨胀核大小
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
    点云边缘提取器
    
    提供独立的边缘提取功能，不影响SLAM主流程的下采样机制。
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
        
        self.use_localsh = LOCALSH_AVAILABLE
        
    def voxel_downsample(self, points, voxel_size=None):
        """
        临时体素下采样（独立于SLAM下采样）
        
        Args:
            points: 点云 numpy array, shape (N, 3) 或 (N, 6) 包含颜色
            voxel_size: 体素大小，如果None则使用配置值
            
        Returns:
            downsampled_points: 下采样后的点云
            indices: 保留点的原始索引
        """
        if voxel_size is None:
            voxel_size = self.config.voxel_size
        
        if len(points) == 0:
            return points, np.array([], dtype=np.int64)
        
        # 只取位置信息
        xyz = points[:, :3] if points.shape[1] > 3 else points
        
        # 计算体素索引
        min_bound = xyz.min(axis=0)
        voxel_indices = ((xyz - min_bound) / voxel_size).astype(np.int32)
        
        # 使用字典进行去重，每个体素保留一个点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = i
        
        indices = np.array(list(voxel_dict.values()))
        downsampled_points = points[indices]
        
        # 检查是否满足最小点数要求
        min_points = int(len(points) * self.config.min_points_ratio)
        if len(downsampled_points) < min_points:
            # 如果下采样后点数太少，减小voxel_size重试
            new_voxel_size = voxel_size * 0.5
            if new_voxel_size > 0.01:  # 防止无限递归
                return self.voxel_downsample(points, new_voxel_size)
        
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
    
    def compute_normals(self, points, k=None):
        """
        计算点云法线
        
        Args:
            points: 点云 (N, 3)
            k: KNN邻域大小
            
        Returns:
            normals: 法线 (N, 3)
        """
        if k is None:
            k = self.config.knn_neighbors
        
        N = len(points)
        normals = np.zeros((N, 3))
        
        # 使用批处理计算KNN
        batch_size = min(1000, N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = points[start:end]
            
            # 计算到所有点的距离
            dists = np.linalg.norm(batch[:, None] - points[None, :], axis=2)
            
            # 获取K个最近邻
            for i, dist in enumerate(dists):
                knn_indices = np.argsort(dist)[:k]
                neighbors = points[knn_indices]
                
                # PCA计算法线
                centered = neighbors - neighbors.mean(axis=0)
                cov = centered.T @ centered
                _, _, vh = np.linalg.svd(cov)
                normal = vh[-1]  # 最小特征值对应的特征向量
                normals[start + i] = normal
        
        return normals
    
    def compute_curvature(self, points, normals=None, k=None):
        """
        计算点云曲率
        
        Args:
            points: 点云 (N, 3)
            normals: 法线 (N, 3)，如果None则先计算
            k: KNN邻域大小
            
        Returns:
            curvatures: 曲率 (N,)
        """
        if k is None:
            k = self.config.knn_neighbors
        
        if normals is None:
            normals = self.compute_normals(points, k)
        
        N = len(points)
        curvatures = np.zeros(N)
        
        batch_size = min(1000, N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = points[start:end]
            batch_normals = normals[start:end]
            
            dists = np.linalg.norm(batch[:, None] - points[None, :], axis=2)
            
            for i, (dist, n) in enumerate(zip(dists, batch_normals)):
                knn_indices = np.argsort(dist)[1:k]  # 排除自身
                neighbor_normals = normals[knn_indices]
                
                # 曲率估计：法线变化量
                normal_diffs = np.abs(np.dot(neighbor_normals, n))
                curvature = 1.0 - normal_diffs.mean()
                curvatures[start + i] = curvature
        
        return curvatures
    
    def extract_edges_curvature(self, points):
        """
        使用曲率方法提取边缘点
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        normals = self.compute_normals(points)
        curvatures = self.compute_curvature(points, normals)
        
        # 基于曲率阈值和比例提取边缘
        threshold_mask = curvatures > self.config.curvature_threshold
        
        # 取前 edge_ratio 的高曲率点
        num_edges = int(len(points) * self.config.edge_ratio)
        top_indices = np.argsort(curvatures)[-num_edges:]
        ratio_mask = np.zeros(len(points), dtype=bool)
        ratio_mask[top_indices] = True
        
        # 结合两种mask
        edge_mask = threshold_mask | ratio_mask
        
        return edge_mask, curvatures
    
    def extract_edges_localsh(self, points):
        """
        使用LocalSH库提取边缘点
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        if not LOCALSH_AVAILABLE:
            return self.extract_edges_curvature(points)
        
        try:
            # 调用LocalSH进行边缘提取
            # 注意：具体API需要根据LocalSH的实际接口调整
            N = len(points)
            
            # 尝试使用LocalSH的边缘提取功能
            # 典型的调用方式可能是：
            # edge_features = LocalSH.extract_edge_features(points, self.config.sh_order)
            # edge_scores = LocalSH.compute_edge_scores(edge_features)
            
            # 由于API未知，使用曲率方法作为后备
            return self.extract_edges_curvature(points)
            
        except Exception as e:
            print(f"[EdgeExtraction] LocalSH 提取失败，使用曲率方法: {e}")
            return self.extract_edges_curvature(points)
    
    def extract_edges(self, points):
        """
        提取边缘点（自动选择方法）
        
        Args:
            points: 点云 (N, 3)
            
        Returns:
            edge_mask: 边缘点mask (N,) bool
            edge_scores: 边缘分数 (N,)
        """
        if self.use_localsh and LOCALSH_AVAILABLE:
            return self.extract_edges_localsh(points)
        else:
            return self.extract_edges_curvature(points)
    
    def project_points_to_image(self, points_3d, K, R, T, width, height, dist_coeffs=None):
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
            points_2d: 2D点 (M, 2)，只包含在图像内的点
            valid_mask: 有效点mask (N,)
        """
        # 转换到相机坐标系
        points_cam = (R @ points_3d.T).T + T
        
        # 过滤深度为负的点
        valid_depth = points_cam[:, 2] > 0
        
        # 投影
        if dist_coeffs is not None and np.any(dist_coeffs != 0):
            # 使用OpenCV处理畸变
            rvec, _ = cv2.Rodrigues(R)
            points_2d, _ = cv2.projectPoints(
                points_3d[valid_depth], rvec, T, K, dist_coeffs
            )
            points_2d = points_2d.squeeze(1)
        else:
            # 简单投影
            points_2d = (K @ points_cam[valid_depth].T).T
            points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
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
    
    def create_edge_mask(self, edge_points_2d, width, height, edge_scores=None):
        """
        创建边缘mask图像
        
        Args:
            edge_points_2d: 2D边缘点 (N, 2)
            width: 图像宽度
            height: 图像高度
            edge_scores: 边缘分数 (N,)，可选
            
        Returns:
            edge_mask: 边缘mask图像 (H, W)，值为0-1
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        if len(edge_points_2d) == 0:
            return mask
        
        # 将点坐标转换为整数
        points_int = np.round(edge_points_2d).astype(np.int32)
        
        # 设置边缘点
        for i, (x, y) in enumerate(points_int):
            if 0 <= x < width and 0 <= y < height:
                if edge_scores is not None and i < len(edge_scores):
                    mask[y, x] = max(mask[y, x], edge_scores[i])
                else:
                    mask[y, x] = 1.0
        
        # 膨胀边缘区域
        kernel_size = self.config.dilate_kernel_size
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 归一化到0-1
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def process_frame(self, gaussians, viewpoint, K=None, dist_coeffs=None):
        """
        处理单帧：从高斯模型提取边缘并投影到图像
        
        Args:
            gaussians: GaussianModel 实例
            viewpoint: Camera 视点信息
            K: 相机内参矩阵（如果None则从viewpoint获取）
            dist_coeffs: 畸变系数
            
        Returns:
            edge_mask: 边缘mask图像 (H, W)
            edge_points_2d: 2D边缘点
            edge_points_3d: 3D边缘点
        """
        # 获取点云位置
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
        
        # 临时下采样（独立于SLAM下采样）
        downsampled_points, ds_indices = self.voxel_downsample(points_3d)
        
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
        
        # 获取相机位姿
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
    
    def enhance_matches_with_edges(self, matches_im1, matches_im2, edge_mask1, edge_mask2, weight=None):
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
        
        if edge_mask1 is None or edge_mask2 is None:
            return np.ones(len(matches_im1)), np.zeros(len(matches_im1), dtype=bool)
        
        N = len(matches_im1)
        weights = np.ones(N)
        edge_matches_mask = np.zeros(N, dtype=bool)
        
        H1, W1 = edge_mask1.shape
        H2, W2 = edge_mask2.shape
        
        for i, (p1, p2) in enumerate(zip(matches_im1, matches_im2)):
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            x2, y2 = int(round(p2[0])), int(round(p2[1]))
            
            # 检查边界
            if 0 <= x1 < W1 and 0 <= y1 < H1 and 0 <= x2 < W2 and 0 <= y2 < H2:
                score1 = edge_mask1[y1, x1]
                score2 = edge_mask2[y2, x2]
                
                # 如果匹配点在边缘区域，增加权重
                if score1 > 0.1 or score2 > 0.1:
                    edge_matches_mask[i] = True
                    weights[i] = 1.0 + weight * max(score1, score2)
        
        return weights, edge_matches_mask


def create_edge_extractor(config=None):
    """
    创建边缘提取器实例的工厂函数
    
    Args:
        config: 配置字典或EdgeExtractionConfig
        
    Returns:
        EdgeExtractor 实例
    """
    return EdgeExtractor(config)


# 便捷函数
def extract_edges_from_pointcloud(points, config=None):
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


def create_edge_mask_from_gaussians(gaussians, viewpoint, config=None, dist_coeffs=None):
    """
    从高斯模型创建边缘mask图像（便捷函数）
    
    Args:
        gaussians: GaussianModel 实例
        viewpoint: Camera 视点信息
        config: 配置
        dist_coeffs: 畸变系数
        
    Returns:
        edge_mask: 边缘mask图像 (H, W)
    """
    extractor = EdgeExtractor(config)
    edge_mask, _, _ = extractor.process_frame(gaussians, viewpoint, dist_coeffs=dist_coeffs)
    return edge_mask

