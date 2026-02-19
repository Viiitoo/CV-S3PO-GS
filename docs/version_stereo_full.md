# stereo_full / exp_decouple 版本说明

> 最后更新：2026-02-18（commit `2ca1536`）

## 配置文件说明

| 配置文件 | 用途 |
|---------|------|
| `configs/mono/Stereo/Stereo_seq_easy/stereo_full.yaml` | 基线对比版本（无形变、无光流）|
| `configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml` | **当前主实验版本**（形变+解耦+光流）|

`exp_decouple.yaml` 继承自 `base_config.yaml`，在此基础上启用了所有实验特性。

## Feature 启用状态（exp_decouple）

| Feature | 状态 | 说明 |
|---------|------|------|
| STAR-Edge 轮廓增强 PnP | **启用** | 3D 边缘引导匹配，提升前端位姿估计精度 |
| 边界噪声抑制 (`edge_border_margin`) | **启用** | 排除图像边缘 30px 的 STAR-Edge 噪声点 |
| 中心裁剪 (`center_crop_ratio`) | **禁用** | 设为 1（不裁剪），使用全图匹配 |
| 轮廓引导采样 (`edge_guided_sampling`) | **启用** | 下采样前保留轮廓区域点，提升重建质量 |
| MASt3R 点云轮廓可视化 (`mast3r_edge_viz`) | **启用** | 每帧输出 STAR-Edge 轮廓 PNG，保存到 `viz_mast3r_pc_edge/` |
| 高斯时间形变 (Deformation) | **启用** | RBF 基函数建模动态场景，提升渲染质量 |
| 位姿-形变解耦优化 (`decouple_pose_deform`) | **启用** | 交替更新策略，防止 `_coefs` 吸收位姿误差 |
| 光流监督 (Optical Flow) | **启用** | `use_optical_flow: True`，`flow_loss_weight: 0.1` |
| 完整光流光栅化器 (`use_flow_rasterizer`) | **启用** | MotionGS 各向异性光流，返回 10 个输出 |
| 三维 ATE 轨迹可视化 | **启用** | 评估后生成 `evo_3dplot_*.png`（颜色编码误差）|

## 核心改动

### 1. 启用时间形变 (`use_deformation: true`)

旧 `stereo_full` 关闭了形变（`use_deformation: false`），轨迹精度好（ATE=0.000932）但渲染质量差（PSNR=23.73）。直接开启形变后 `_coefs` 会吸收位姿误差导致轨迹退化（ATE=0.001496）。

### 2. 位姿-形变解耦优化 (`decouple_pose_deform`)

**问题根因**：后端 `map()` 中 `backward()` 对位姿参数和形变系数 `_coefs` 同时计算梯度，`_coefs` 的高自由度使其能"吸收"位姿误差——渲染 loss 下降但轨迹退化。

**解决方案**：在 `backward()` 之后、`optimizer.step()` 之前，按 N:M 交替策略选择性清零一方梯度。

```yaml
decouple_pose_deform:
    enabled: true
    pose_steps: 3    # 连续 3 步只更新位姿 + 高斯几何
    deform_steps: 1  # 连续 1 步只更新 _coefs + 高斯几何
```

**交替周期**（cycle=4）：

| 迭代 | Phase | 位姿更新 | _coefs 更新 | 高斯几何更新 |
|------|-------|---------|------------|------------|
| 1    | P     | Y       | -          | Y          |
| 2    | P     | Y       | -          | Y          |
| 3    | P     | Y       | -          | Y          |
| 4    | D     | -       | Y          | Y          |

### 3. 光流监督（新增）

在 `exp_decouple.yaml` 中启用光流 loss：

```yaml
use_optical_flow: True
use_camera_flow: True
flow_visualization: True
flow_loss_weight: 0.1
use_flow_rasterizer: True   # MotionGS 完整光栅化器（10个输出，各向异性）
flow_rasterizer_fallback: True
flow_warm_up: 1000          # 预热后才开始计算 flow loss
flow_loss_interval: 20      # 每 20 次 map 迭代计算一次
```

光流 loss 使用 `optical_flow_gt`（总光流，含相机运动），而非仅 `motion_flow`（动态物体光流）。

### 4. 光流归一化 Bug 修复

`utils/slam_utils.py` 中 `flow_loss()` 函数之前错误地将 x 方向（水平）除以 height、y 方向（垂直）除以 width，现已修正：

```python
# 修复前（错误）
flow_pred[0] /= height   # x 方向
flow_pred[1] /= width    # y 方向

# 修复后（正确）
flow_pred[0] /= width    # x 方向（水平位移）除以宽度
flow_pred[1] /= height   # y 方向（垂直位移）除以高度
```

### 5. STAR-Edge 轮廓增强 PnP（沿用并扩展）

前端使用 STAR-Edge 提取 3D 点云轮廓，在 PnP 匹配中对轮廓点加权。新增 `edge_border_margin` 抑制边缘噪声：

```yaml
rgb_edge_pnp:
    edge_guided_matching: true
    edge_match_mode: "weight"
    edge_match_ratio: 0.5
    edge_3d_weight: 2.0
    edge_border_margin: 30    # 排除图像边缘 30px（新增）
    center_crop_ratio: 1      # 不裁剪，使用全图（从 0.5 改为 1）
```

### 6. 评估改进

`utils/eval_utils.py` 新增三维 ATE 轨迹可视化，使用 jet colormap 编码每步误差大小，输出到 `evo_3dplot_<label>.png`。

## 代码修改位置

| 文件 | 行号 | 改动 |
|------|------|------|
| `utils/slam_backend.py` | 196-205 | `map()` 循环前读取解耦配置 |
| `utils/slam_backend.py` | 477-489 | `backward()` 后按阶段清零梯度 |
| `utils/slam_backend.py` | ~417 | flow loss 改用 `optical_flow_gt` 而非 `motion_flow` |
| `utils/slam_utils.py` | 146-154 | `flow_loss()` 修复 x/y 归一化方向 |
| `utils/eval_utils.py` | 109-140 | 新增三维 ATE 轨迹可视化 |
| `configs/.../exp_decouple.yaml` | 25-28 | 启用光流（`use_optical_flow: True` 等）|
| `configs/.../exp_decouple.yaml` | 48 | `center_crop_ratio: 1`（从 0.5 改为 1）|
| `configs/mono/Stereo/base_config.yaml` | - | `use_flow_rasterizer: True` |

## 实验对比

| 配置 | ATE RMSE | PSNR | 备注 |
|------|----------|------|------|
| 无形变 (`stereo_full`) | 0.000932 | 23.73 | 轨迹好、渲染差 |
| 有形变、无解耦 | 0.001496 | - | _coefs 吸收位姿误差 |
| 冻结 _coefs (freeze) | 0.000954 | 26.42 | 形变代码路径有益但不学形变 |
| 有形变 + 解耦（无光流）| 待确认 | 待确认 | `exp_decouple` 初始版本 |
| **有形变 + 解耦 + 光流（当前版本）** | **待确认** | **待确认** | `exp_decouple`，flow_loss_weight=0.1 |

## 运行命令

```bash
# 当前主实验（形变 + 解耦 + 光流）
CUDA_VISIBLE_DEVICES=X python3 slam.py --config configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml

# 基线对比（无形变、无光流）
CUDA_VISIBLE_DEVICES=X python3 slam.py --config configs/mono/Stereo/Stereo_seq_easy/stereo_full.yaml
```
