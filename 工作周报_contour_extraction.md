# Contour-Extraction 分支周工作汇报

> 分支: `contour-extraction`
> 时间范围: 2026-02-07 ~ 2026-02-08
> 涵盖版本: 最近 4 个 commit

---

## 版本总览

| # | Commit | 日期 | 标题 | 核心内容 |
|---|--------|------|------|----------|
| 1 | `e8d31e5` | 02-07 | `2Dpnp_deleted&Timer` | 删除 2D RGB 轮廓 PnP 路径，统一为 3D STAR-Edge；添加全链路计时系统 |
| 2 | `418c099` | 02-08 | `GPU_STAR-Edge` | 实现 STAR-Edge CUDA Kernel 加速，新增 `star_edge_cuda` 子模块 |
| 3 | `efd59ef` | 02-08 | `flow_loss_bug_fixed` | 修复光流损失的帧对齐 bug，重构前后端 flow 数据传递架构 |
| 4 | `3a8849a` | 02-08 | `wiz_full_MotionGS` | 集成完整 MotionGS 光流光栅化器，支持各向异性 GS 光流计算 |

---

## 版本 1: `e8d31e5` — 2Dpnp_deleted & Timer

### 主要改动

**1. 移除 2D RGB 轮廓加权 PnP 路径**

- 删除 `init_pose.py` 中基于 Canny/Sobel 等 2D 边缘检测的 RGB 轮廓加权逻辑（约 170 行）
- 删除 `_save_rgb_edge_pnp_viz()` 可视化函数及其相关调用
- 移除所有 2D 边缘相关的配置项（`enabled`, `use_depth`, `rgb_method`, `canny_low/high`, `dilate_kernel_size`, `blur_ksize`, `edge_weight`, `viz` 等）
- **统一为 3D 点云轮廓引导匹配（STAR-Edge）**，保留 `edge_guided_matching`, `edge_match_mode`, `edge_3d_weight` 等 3D 轮廓配置

**2. 添加前端全链路计时系统**

- 在 `slam_frontend.py` 中新增精确计时属性，覆盖 SLAM 前端核心模块：
  - `get_pose`: MASt3R 位姿估计耗时
  - `get_depth`: MASt3R 深度估计耗时
  - `optical_flow`: 光流计算耗时
  - `pose_opt`: 位姿优化耗时（含实际迭代次数）
  - `process_depth`: 深度处理耗时
  - `cam_init / tracking / kf_creation`: 每帧各阶段耗时
- 使用 `torch.cuda.synchronize()` 确保 GPU 计时准确
- 在 `logging_utils.py` 中添加 `Timing` 日志标签，每帧输出格式化的耗时汇总

**3. 配置清理**

- 精简 `base_config.yaml` 和 `stereo_*.yaml`，移除所有 2D 边缘相关冗余配置
- 将配置结构统一为仅包含 3D STAR-Edge 轮廓引导的参数

---

## 版本 2: `418c099` — GPU STAR-Edge

### 主要改动

**1. 新增 `star_edge_cuda` CUDA 子模块**（约 2860 行新增代码）

完整实现了 STAR-Edge 3D 轮廓提取的 GPU 加速版本，包含 5 个 CUDA Kernel：

| Kernel | 文件 | 功能 |
|--------|------|------|
| Kernel 1 | `knn_morton.cu` | 基于 Morton Code 的 KNN 搜索（空间排序 + 分区） |
| Kernel 2 | `direction_sampling.cu` | 方向向量计算（fused gather + normalize） |
| Kernel 3 | `direction_sampling.cu` | PCA 投影 + 角度排序 + 曲线采样 |
| Kernel 4 | `sphere_kde.cu` | 球面核密度估计（KDE） |
| Kernel 5 | `sht_power.cu` | 球谐变换 + 功率谱计算 |

- Python 封装层 `star_edge_cuda/__init__.py` 提供 `compute_descriptors()` 高级接口
- 附带 `test_benchmark.py` 和 `test_validate.py` 用于性能和正确性验证
- 使用 `setup.py` + `torch.utils.cpp_extension.BuildExtension` 编译，适配 RTX 3090 (SM 8.6)

**2. 集成 CUDA 加速路径到 `edge_extraction.py`**

- 新增 `_extract_edges_star_edge_cuda()` 方法，调用 `star_edge_cuda.compute_descriptors()` 计算描述子
- 在 `extract_edges_star_edge()` 入口处添加 CUDA 路径优先分支，失败时自动回退到 CPU 路径
- 新增 `star_edge_cuda: true` 配置项控制是否启用 CUDA 加速
- 修复 GPU 设备选择：使用 `torch.cuda.current_device()` 替代硬编码 `cuda:0`，正确支持多 GPU 和 `CUDA_VISIBLE_DEVICES`

**3. 配置更新**

- CUDA 加速后可大幅提升处理点数：`star_edge_max_points` 从 3000 提升到 30000（约 15ms/帧）
- 恢复邻域参数到高质量设置：`star_edge_kk` 从 16 恢复到 26，`star_edge_sample_num` 从 20 恢复到 30
- 在所有 `edge_extraction` 配置段中默认启用 `star_edge_cuda: true`

---

## 版本 3: `efd59ef` — Flow Loss Bug Fixed

### 主要改动

**1. 修复关键帧索引不匹配 bug**

这是 flow loss 功能的核心 bug 修复：

- **问题**: 旧代码在后端使用 `motion_flow_dict[kf_idx]` 查找光流数据，但存储的是相邻帧间（如帧 5→帧 6）的 motion flow，而后端实际需要的是关键帧间（如帧 5→帧 10）的光流。这导致 flow loss 使用了错误的帧对，优化方向偏离，RMSE 反而增大。
- **修复**: 重构了前后端数据传递架构：
  - 前端改为存储原始 `optical_flow`（而非预计算的 `motion_flow`），并附带目标关键帧索引
  - 后端接收 `(prev_kf_idx, optical_flow, target_kf_idx)` 三元组
  - 后端使用当前优化后的位姿重新计算 `camera_flow`，再由 `motion_flow = optical_flow - camera_flow` 得到运动光流

**2. 重构前后端光流传递架构**

- `motion_flow_dict` → `optical_flow_dict`：语义更清晰
- 前端 `request_keyframe()` 改为"回溯计算"模式：当创建关键帧 K' 时，计算前一个关键帧 K → K' 的光流（而非尝试预测 K' → 未来帧的光流）
- 后端新增帧对齐验证：`stored_target_kf` 必须与实际的 `actual_next_kf` 一致，否则跳过

**3. 后端 Flow Loss 精简重构**（-339 行 / +122 行）

- 移除大量调试日志、冗余注释和 debug.log 写入代码
- 新增 flow loss 截断机制：`torch.clamp(flow_loss_value, max=0.5)`，防止异常大的梯度主导优化
- 降低默认 `flow_loss_weight` 从 0.5 到 0.1
- 清理 `slam_utils.py` 中 `flow_loss()` 函数的调试代码

---

## 版本 4: `3a8849a` — 完整 MotionGS 集成

### 主要改动

**1. 集成 MotionGS 光流光栅化器（FlowRasterizer）**

- 在 `gaussian_renderer/__init__.py` 中添加光流光栅化器导入（`flow_diff_gaussian_rasterization`）
- `render()` 函数新增 `use_flow_rasterizer` 参数，支持在标准光栅化器（5输出）和光流光栅化器（10输出）之间切换
- 光流光栅化器额外返回：`proj_2D`, `conic_2D`, `conic_2D_inv`, `gs_per_pixel`, `weight_per_gs_pixel`, `x_mu`
- 处理 API 差异：光流光栅化器不支持 `projmatrix_raw`, `theta`, `rho` 参数，通过条件判断适配

**2. 后端智能光栅化器调度**

- 仅在满足所有条件时启用光流光栅化器：配置启用 + 有光流数据 + 超过预热期 + 到达计算间隔
- 新增 `flow_on_all_frames` 和 `flow_max_pairs` 配置项：控制在几对关键帧上计算 flow loss（默认仅最近 1 对），避免不必要的开销
- 新增完整模式张量形状验证：检测 `gs_per_pixel` 与渲染输出的空间维度是否一致，不匹配时自动降级到简化模式
- 处理 `n_touched` 为 None 的情况（FlowRasterizer 不返回该值），假设所有点可见

**3. 新增完整迁移方案文档**

- 新增 `MotionGS完整迁移方案.md`（418行），详细记录：
  - 简化模式与完整模式的差异分析
  - 光流光栅化器的编译、集成方案
  - 后端调度策略、性能基准预期
  - 故障排除指南（编译、形状不匹配、NaN、性能下降）
- 删除已完成的 `STAR_EDGE_CUDA_PLAN.md` 计划文档

**4. 配置更新**

- 所有配置文件新增：`use_flow_rasterizer`, `flow_rasterizer_fallback`
- Stereo 配置启用完整模式：`use_flow_rasterizer: True`
- 新增 `flow_loss_interval: 20`（从 5 提升，降低计算频率以提速）

---

## 本周工作总结

| 工作方向 | 具体成果 |
|----------|----------|
| **架构统一** | 移除 2D RGB 轮廓路径，全面统一为 3D STAR-Edge 点云轮廓方案 |
| **性能优化** | 实现 STAR-Edge CUDA Kernel（5 个 Kernel），处理能力从 3K 点提升到 30K 点/帧 |
| **Bug 修复** | 修复 flow loss 帧对齐核心 bug，重构前后端光流数据传递架构 |
| **功能集成** | 完成完整 MotionGS 光流光栅化器集成，支持各向异性 GS 光流计算 |
| **工程质量** | 添加全链路计时系统、配置清理、调试代码移除、自动降级机制 |
