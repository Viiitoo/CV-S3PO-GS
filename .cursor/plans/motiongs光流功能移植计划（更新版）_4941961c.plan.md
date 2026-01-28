# MotionGS光流功能移植计划（更新版）

## 概述

本计划将MotionGS中的光流功能移植到S3PO_baseline项目，包括：

1. 移植optical_flow（使用GMFlow计算2D光流）
2. 移植camera_flow（基于深度和相机位姿计算相机流）**使用Mast3r预测的深度**
3. 添加光流可视化输出
4. 添加使用真值位姿的测试模式

## 重要说明

**在计算camera_flow时，必须使用Mast3r预测的深度值（`viewpoint.mono_depth`），而不是其他深度源。**

## 架构设计

### 数据流

```javascript
当前帧图像 + 下一帧图像
    ↓
[GMFlow] → optical_flow (2D光流)
    ↓
Mast3r预测深度 (viewpoint.mono_depth) + 相机位姿
    ↓
[camera_flow] → camera_flow (相机流)
    ↓
motion_flow = optical_flow - camera_flow
    ↓
[可视化] → 保存光流图像
```



## 实施步骤

### 阶段1: 移植optical_flow功能

#### 1.1 复制GMFlow相关代码

- 复制 `MotionGS/gmflow/` 目录到项目根目录
- 复制 `MotionGS/core_flow/utils_former/flow_viz.py` 到 `utils/flow_viz.py`
- 确保所有依赖模块（如transformer、backbone等）都已包含

#### 1.2 创建光流工具模块

- 在 `utils/` 目录下创建 `flow_utils.py`
- 实现 `init_optical_flow_model()` 函数：
- 加载GMFlow配置
- 初始化GMFlow模型
- 加载预训练权重
- 设置为eval模式
- 实现 `compute_optical_flow(img1, img2, flownet)` 函数：
- 输入：当前帧和下一帧图像（tensor格式，范围0-1）
- 输出：光流张量 (2, H, W)
- 处理尺寸不匹配的情况（插值调整）

#### 1.3 集成到训练流程

- 在 `slam.py` 或 `slam_frontend.py` 中初始化光流模型
- 在需要计算光流的地方调用 `compute_optical_flow()`
- 缓存光流结果（类似MotionGS中的flow_2d_gt_list）

### 阶段2: 移植camera_flow功能

#### 2.1 复制warp_utils相关代码

- 从 `MotionGS/utils/warp_utils.py` 复制以下内容到 `utils/warp_utils.py`：
- `BackprojectDepth` 类
- `Project3D` 类
- `calculate_camera_flow(depth1, cam1, cam2)` 函数
- 适配项目的Camera类接口（确保extrinsic和intrinsic属性兼容）

#### 2.2 适配Camera类

- 检查 `utils/camera_utils.py` 中的Camera类
- **添加extrinsic和intrinsic属性或方法**：
- `extrinsic`: 4x4外参矩阵（world to camera），从R和T构建
- `intrinsic`: 4x4内参矩阵，从fx, fy, cx, cy构建
- 实现方式：
  ```python
    @property
    def extrinsic(self):
        # 从R和T构建4x4外参矩阵
        T_matrix = torch.eye(4, device=self.device)
        T_matrix[:3, :3] = self.R
        T_matrix[:3, 3] = self.T
        return T_matrix
    
    @property
    def intrinsic(self):
        # 从fx, fy, cx, cy构建4x4内参矩阵
        K = torch.eye(4, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K
  ```




#### 2.3 集成camera_flow计算

- 在需要计算camera_flow的地方调用 `calculate_camera_flow()`
- **关键：使用Mast3r预测的深度**
- 输入深度：`viewpoint.mono_depth`（numpy格式，H x W）
- 需要转换为tensor格式：`torch.from_numpy(viewpoint.mono_depth).float().cuda()`
- 如果需要添加batch和channel维度：`.unsqueeze(0).unsqueeze(0)` → (1, 1, H, W)
- 输入：当前帧Mast3r深度、当前帧相机、下一帧相机
- 输出：相机流张量 (2, H, W)

#### 2.4 深度格式处理

- `viewpoint.mono_depth` 是numpy数组，形状为 (H, W)
- 转换为tensor并适配 `calculate_camera_flow` 的输入格式：
  ```python
    mono_depth_tensor = torch.from_numpy(viewpoint.mono_depth).float().cuda()
    # calculate_camera_flow期望输入为 (B) (1) H W 格式
    if len(mono_depth_tensor.shape) == 2:
        mono_depth_tensor = mono_depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    camera_flow = calculate_camera_flow(mono_depth_tensor, cam1, cam2)
  ```




### 阶段3: 添加光流可视化

#### 3.1 实现可视化函数

- 在 `utils/flow_viz.py` 中已有 `flow_to_image()` 函数
- 创建 `save_flow_visualization()` 函数：
- 输入：光流张量、保存路径、标题
- 使用 `flow_to_image()` 转换为RGB图像
- 保存为PNG文件

#### 3.2 集成可视化输出

- 在计算optical_flow和camera_flow后，调用可视化函数
- 保存路径：`{save_dir}/flow_viz/optical_flow_{frame_idx}.png`
- 保存路径：`{save_dir}/flow_viz/camera_flow_{frame_idx}.png`
- 可选：同时保存motion_flow可视化

### 阶段4: 添加真值位姿测试模式

#### 4.1 配置选项

- 在配置文件中添加 `use_gt_pose` 选项（默认False）
- 位置：`config["Training"]["use_gt_pose"]` 或 `config["Dataset"]["use_gt_pose"]`

#### 4.2 修改位姿使用逻辑

- 在 `slam_frontend.py` 中：
- 检查 `use_gt_pose` 配置
- 如果为True，在计算camera_flow时使用 `R_gt` 和 `T_gt`
- 创建临时Camera对象或修改现有Camera的R和T为真值
- **注意：即使使用真值位姿，深度仍然使用Mast3r预测值**
- 在 `slam_backend.py` 中（如果需要）：
- 如果 `use_gt_pose=True`，跳过位姿优化，直接使用真值

#### 4.3 测试模式入口

- 在 `slam.py` 中添加命令行参数或配置项
- 当启用 `use_gt_pose` 时，系统将：
- 使用真值位姿进行光流计算
- 使用Mast3r预测的深度
- 跳过位姿估计和优化
- 仅测试光流功能本身

## 文件修改清单

### 新建文件

- `utils/flow_utils.py` - 光流计算工具
- `utils/warp_utils.py` - 相机流计算工具（从MotionGS复制并适配）
- `utils/flow_viz.py` - 光流可视化工具（从MotionGS复制）

### 修改文件

- `utils/camera_utils.py` - **添加extrinsic和intrinsic属性**
- `utils/slam_frontend.py` - 集成光流计算和可视化，**使用mono_depth计算camera_flow**
- `slam.py` - 初始化光流模型，添加配置选项
- 配置文件 - 添加 `use_gt_pose` 和光流相关配置

### 依赖检查

- 确保GMFlow的预训练权重文件路径正确
- 检查是否需要额外的依赖包（如yacs等）

## 配置示例

```yaml
Training:
  use_gt_pose: false  # 使用真值位姿模式
  use_optical_flow: true
  use_camera_flow: true
  flow_visualization: true  # 是否保存光流可视化

Flow:
  model_path: "./gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
  model_type: "gmflow"  # 可选: gmflow, raft, flowformer
```



## 测试验证

1. **单元测试**：

- 测试optical_flow计算（给定两张图像）
- 测试camera_flow计算（给定Mast3r深度和位姿）
- 测试可视化函数

2. **集成测试**：

- 使用真值位姿模式运行一个序列
- 验证光流图像正确保存
- 检查camera_flow和optical_flow的合理性
- **验证使用的是Mast3r预测的深度**

3. **可视化检查**：

- 检查生成的光流图像是否符合预期
- 对比optical_flow和camera_flow的差异