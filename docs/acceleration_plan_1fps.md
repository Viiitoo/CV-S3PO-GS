# S3PO-GS 加速方案：目标 1 FPS

## Context

当前系统每帧处理耗时约 2200ms（~0.45 FPS）。目标是降低到 1000ms（1 FPS）。
通过分析运行日志中 198 帧的时间数据，各组件平均耗时如下：

| 组件 | 平均耗时 | 占比 |
|------|---------|------|
| get_pose (MASt3R + PnP) | ~650ms | 30% |
| 位姿优化 (render+loss×N) | ~600ms | 27% |
| get_depth (MASt3R 深度) | ~320ms | 15% |
| 光流 (GMFlow) | ~300ms | 14% |
| STAR-Edge | ~60ms | 3% |
| 相机初始化+其他 | ~30ms | 1% |
| **总计** | **~2200ms** | |

目标预算：1000ms → 需削减 ~1200ms。

---

## 优化方案（按实施顺序）

### 步骤 1：减少位姿优化迭代次数（节省 ~400ms）

**原理**：当前 `tracking_itr_num=100`，但多数帧在 40-60 次迭代即收敛（早停条件 `tau.norm() < 1e-4`，`pose_utils.py:70`）。由于 `get_pose` 已通过 MASt3R+PnP 提供了高质量初始位姿，25 次迭代足以精细优化。

**修改**：`configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml`（或 CLI `--iter 25`）
```yaml
Training:
  tracking_itr_num: 25  # 从 100 改为 25
```

**精度影响**：极低。PnP 初始位姿已非常准确，后续优化收益递减。

---

### 步骤 2：非关键帧跳过光流计算（节省 ~300ms）

**原理**：Frontend 当前每帧都调用 `compute_optical_flow()`（`slam_frontend.py:299-306`，耗时 ~300ms）。但 Backend 实际只使用关键帧之间的光流（通过 `request_keyframe()` 在 line 615-642 的回溯计算获得），每帧的光流计算是冗余的。

**修改文件**：`utils/slam_frontend.py`

在 `tracking()` 方法中（line 274-395），将光流计算块设置为仅关键帧执行：

```python
# slam_frontend.py:278 - 用条件跳过整个块
# 光流仅在 request_keyframe() 的回溯机制中计算（line 615-642）
# 非 KF 帧无需计算
self._timing_optical_flow = 0.0
# 注释掉或用 if False 包裹 line 278-395 的光流计算代码
```

**精度影响**：零。Backend 使用的光流完全来自 `request_keyframe()` 的回溯计算。

---

### 步骤 3：非关键帧跳过 get_depth（节省 ~320ms）

**原理**：`get_depth()` 调用 MASt3R 推理（`init_pose.py:483-484`），生成 `viewpoint.mono_depth`。但 `mono_depth` 仅在以下场景使用：
- `process_depth()` - 仅关键帧处理时调用（`slam_frontend.py:163`）
- `camera_flow` 计算 - 已在步骤 2 中跳过

位姿优化循环（line 445-492）**不使用** `mono_depth`（它用的是 `render_pkg["depth"]`）。`is_keyframe()` 判定也不依赖 `mono_depth`。

**修改文件**：`utils/slam_frontend.py`

**修改 1**：在 `tracking()` 中跳过 get_depth（line 266-272）：
```python
# 替换 line 266-272:
# 延迟 depth 计算到关键帧处理时
viewpoint.mono_depth = None  # 标记为待计算
self._timing_get_depth = 0.0
```

**修改 2**：在 `run()` 中，创建关键帧前补充计算（line 816-818）：
```python
if create_kf:
    # 延迟计算：仅为关键帧计算 MASt3R 深度
    if viewpoint.mono_depth is None:
        t_depth_start = time.time()
        viewpoint.mono_depth = get_depth(
            viewpoint.original_image, viewpoint.original_image,
            self.model, return_conf=False)
        self._timing_get_depth = (time.time() - t_depth_start) * 1000
    self.current_window, removed = self.add_to_window(...)
    depth_map = self.add_new_keyframe(...)
    self.request_keyframe(...)
```

**精度影响**：零。关键帧仍然获得完整的 MASt3R 深度估计。

---

### 步骤 4：关闭调试可视化（节省 ~10ms）

**修改**：`configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml` 或对应 base_config

```yaml
mast3r_edge_viz:
  enabled: false       # 关闭每帧 STAR-Edge 点云可视化

Training:
  flow_visualization: false  # 关闭光流可视化输出
```

---

## 预期效果

| 组件 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| get_pose | ~650ms | ~650ms | 0ms |
| 位姿优化 | ~600ms | **~200ms** | **400ms** |
| get_depth | ~320ms | **~0ms** (非KF) | **320ms** |
| 光流 | ~300ms | **~0ms** (非KF) | **300ms** |
| 其他 | ~30ms | ~30ms | 0ms |
| **总计** | **~2200ms** | **~880ms** | **~1320ms** |

非关键帧：~880ms（1.14 FPS）✓
关键帧（~每5-20帧一次）：~880ms + 320ms(depth) + 300ms(flow) + 700ms(process_depth) ≈ 2200ms

---

## 关键文件

- `utils/slam_frontend.py` — tracking() 方法 (line 232-495)、run() 主循环 (line 673-849)
- `utils/init_pose.py` — get_pose (line 277-474)、get_depth (line 476-586)
- `configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml` — 实验配置
- `configs/mono/Stereo/base_config.yaml` — 基础配置 (tracking_itr_num line 83)
- `utils/pose_utils.py` — 收敛判定 (line 70, threshold=1e-4)

## 验证方法

```bash
# 运行实验，观察日志中 Timing 行的帧处理时间
CUDA_VISIBLE_DEVICES=X python3 slam.py \
  --config configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml \
  --iter 25

# 关注输出中的：
# 1. Timing 行 - 非KF帧总计应 < 1000ms
# 2. Eval: RMSE ATE - 精度回归检查（对比基线 0.0009m）
# 3. Eval: mean psnr - 渲染质量检查（对比基线 ~25.6）
```

每步优化后逐步验证精度是否退化，如果某步影响过大则回退该步。

---

## 各方案加速效果排序（从高到低）

| 排名 | 方案 | 节省耗时 | 精度影响 |
|:----:|------|---------|---------|
| 1 | 步骤 1：减少位姿优化迭代次数（100→25） | **~400ms** | 极低 |
| 2 | 步骤 3：非关键帧跳过 get_depth | **~320ms** | 零 |
| 3 | 步骤 2：非关键帧跳过光流计算 | **~300ms** | 零 |
| 4 | 步骤 4：关闭调试可视化 | **~10ms** | 零 |
