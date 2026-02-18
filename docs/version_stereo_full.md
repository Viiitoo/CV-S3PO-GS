# stereo_full 版本说明

## Feature 启用状态

| Feature | 状态 | 说明 |
|---------|------|------|
| STAR-Edge 轮廓增强 PnP | **启用** | 3D 边缘引导匹配，提升前端位姿估计精度 |
| 高斯时间形变 (Deformation) | **启用** | RBF 基函数建模动态场景，提升渲染质量 |
| 位姿-形变解耦优化 | **启用** | 交替更新策略，防止 `_coefs` 吸收位姿误差 |
| 光流 (Optical Flow) | 禁用 | `use_optical_flow: False`, `flow_loss_weight: 0` |

## 核心改动

### 1. 启用时间形变 (`use_deformation: true`)

之前 `stereo_full` 关闭了形变（`use_deformation: false`），轨迹精度好（ATE=0.000932）但渲染质量差（PSNR=23.73）。直接开启形变后 `_coefs` 会吸收位姿误差导致轨迹退化（ATE=0.001496）。

### 2. 新增位姿-形变解耦优化 (`decouple_pose_deform`)

**问题根因**：后端 `map()` 中 `backward()` 对位姿参数和形变系数 `_coefs` 同时计算梯度，`_coefs` 的高自由度使其能"吸收"位姿误差——渲染 loss 下降但轨迹退化。

**解决方案**：在 `backward()` 之后、`optimizer.step()` 之前，按 N:M 交替策略选择性清零一方梯度，使位姿和 `_coefs` 不在同一步中同时更新。

```
decouple_pose_deform:
    enabled: true
    pose_steps: 3    # 连续 3 步只更新位姿 + 高斯几何
    deform_steps: 1  # 连续 1 步只更新 _coefs + 高斯几何
```

**交替周期**（以 cycle=4 为例）：

| 迭代 | Phase | 位姿更新 | _coefs 更新 | 高斯几何更新 |
|------|-------|---------|------------|------------|
| 1    | P     | Y       | -          | Y          |
| 2    | P     | Y       | -          | Y          |
| 3    | P     | Y       | -          | Y          |
| 4    | D     | -       | Y          | Y          |

位姿获得 3/4 的优化机会，形变获得 1/4，高斯几何属性（xyz、opacity、scaling 等）始终参与更新。

### 3. STAR-Edge 轮廓增强 PnP（沿用）

前端使用 STAR-Edge 提取 3D 点云轮廓，在 PnP 匹配中对轮廓点加权，提升位姿估计对结构边缘的敏感度。

```yaml
rgb_edge_pnp:
    edge_guided_matching: true
    edge_match_mode: "weight"
    edge_3d_weight: 2.0
    center_crop_ratio: 0.5
```

## 代码修改位置

| 文件 | 行号 | 改动 |
|------|------|------|
| `utils/slam_backend.py` | 181-191 | `map()` 循环前读取解耦配置 |
| `utils/slam_backend.py` | 624-637 | `backward()` 后按阶段清零梯度 |
| `configs/.../stereo_full.yaml` | 35-38, 41 | 新增 `decouple_pose_deform`，`use_deformation` 改为 `true` |

## 实验对比

| 配置 | ATE RMSE | PSNR | 备注 |
|------|----------|------|------|
| 无形变 (旧 stereo_full) | 0.000932 | 23.73 | 轨迹好、渲染差 |
| 有形变、无解耦 | 0.001496 | - | _coefs 吸收位姿误差 |
| 冻结 _coefs (freeze) | 0.000954 | 26.42 | 形变代码路径有益但不学形变 |
| **有形变 + 解耦优化 (当前版本)** | **待确认** | **待确认** | 预期 ATE≤0.001, PSNR≥26.42 |

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=X python3 slam.py --config configs/mono/Stereo/Stereo_seq_easy/stereo_full.yaml
```
