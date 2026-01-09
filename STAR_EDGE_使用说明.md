# STAR-Edge 集成使用说明

## 快速开始

STAR-Edge点云轮廓提取工具已成功集成到SLAM系统中，用于增强形变场景下的图像匹配和位姿估计。

## 核心特性

### ✓ 独立的临时下采样机制

**重要**：边缘提取使用**独立的下采样参数**，**不影响**SLAM主流程的下采样！

- **SLAM下采样**：`pcd_downsample` = 256, `pcd_downsample_init` = 128  
  → 用于3D高斯建图，在 `gaussian_model.py` 中
  
- **边缘提取临时下采样**：`voxel_size` = 0.05  
  → 仅用于边缘检测，在 `edge_extraction.py` 中  
  → 对原始点云进行临时下采样，不修改SLAM点云

## 配置说明

配置文件位置：
- `configs/mono/Stereo/base_config.yaml`
- `configs/mono/Stereo/Stereo_seq_easy/base_config.yaml`

关键参数：

```yaml
edge_extraction:
  enabled: True              # 启用边缘增强
  voxel_size: 0.05          # 临时下采样体素大小（关键参数！）
  max_points: 50000         # 最大点数限制
  edge_ratio: 0.2           # 保留20%最显著的边缘
  edge_weight: 2.0          # 边缘匹配权重倍数
```

## 下采样阈值选择

根据场景选择合适的 `voxel_size`：

| 场景 | voxel_size | 点云密度 | 计算速度 |
|------|-----------|---------|---------|
| 室内小物体 | 0.03 | 密集 | 慢 |
| 室内场景（推荐）| 0.05 | 适中 | 中等 |
| 室内大场景 | 0.08 | 稀疏 | 快 |
| 室外场景 | 0.10-0.20 | 很稀疏 | 很快 |

**原则**：
- `voxel_size` ↓（变小）→ 更多点 → 更精细边缘 → 更慢
- `voxel_size` ↑（变大）→ 更少点 → 更粗糙边缘 → 更快

## 使用方法

### 方法1：直接使用（已配置好）

配置文件已包含边缘提取配置，直接运行即可：

```bash
python slam.py --config configs/mono/Stereo/Stereo_seq_easy/stereo_full.yaml
```

### 方法2：禁用边缘增强（对比实验）

在配置文件中修改：

```yaml
edge_extraction:
  enabled: False
```

### 方法3：调整下采样阈值

根据场景调整 `voxel_size`：

```yaml
edge_extraction:
  enabled: True
  voxel_size: 0.08  # 场景较大时增大此值
```

## 测试验证

运行测试脚本：

```bash
cd /home/sjw/data0/lsx/S3PO_baseline
python3 test_edge_extraction.py
```

预期结果：5/6 测试通过（LocalSH可能因Python版本不匹配而失败，但不影响功能）

## 技术细节

### 边缘提取流程

1. **获取点云**：从高斯模型获取3D点
2. **临时下采样**：使用 `voxel_size` 进行体素下采样（独立操作）
3. **边缘检测**：
   - 优先使用LocalSH方法（STAR-Edge）
   - 如果失败，自动回退到曲率方法
4. **投影到图像**：3D边缘点投影为2D边缘mask
5. **增强匹配**：对边缘区域的匹配点增加权重
6. **位姿估计**：使用加权匹配点进行PnP求解

### LocalSH状态

- **位置**：`STAR-Edge/pre_process/LocalSH.cpython-38-x86_64-linux-gnu.so`
- **状态**：已编译，但为Python 3.8（当前环境是3.6）
- **影响**：LocalSH暂不可用，系统自动使用曲率方法
- **解决**：如需使用LocalSH，请重新编译（见完整文档）

## 参数调优建议

### 优先调整 voxel_size

这是影响最大的参数：

```yaml
# 快速模式（粗糙边缘，适合大场景）
voxel_size: 0.10

# 平衡模式（推荐）
voxel_size: 0.05

# 精细模式（细节边缘，适合小物体）
voxel_size: 0.03
```

### 其次调整 edge_ratio

控制保留多少边缘：

```yaml
# 明显边缘（10%）
edge_ratio: 0.1

# 平衡（20%，推荐）
edge_ratio: 0.2

# 复杂场景（30%）
edge_ratio: 0.3
```

### 最后调整 edge_weight

控制边缘匹配的重要性：

```yaml
# 较低权重
edge_weight: 1.5

# 中等权重（推荐）
edge_weight: 2.0

# 较高权重
edge_weight: 3.0
```

## 对比实验建议

### 实验1：基准（无边缘增强）

```yaml
edge_extraction:
  enabled: False
```

### 实验2：边缘增强（不同阈值）

```yaml
edge_extraction:
  enabled: True
  voxel_size: 0.03  # 测试 0.03, 0.05, 0.08
```

### 评估指标

- ATE（绝对轨迹误差）
- RPE（相对位姿误差）
- 位姿估计成功率
- 每帧处理时间

## 常见问题

### Q: 边缘提取会影响SLAM的下采样吗？

**A: 不会！** 边缘提取使用独立的临时下采样（`voxel_size`），不修改SLAM中的点云。SLAM仍使用 `pcd_downsample` 参数。

### Q: LocalSH不可用怎么办？

**A: 没关系！** 系统会自动回退到曲率方法，功能完全正常。如需使用LocalSH，请重新编译。

### Q: 如何知道使用了哪种边缘提取方法？

**A:** 查看日志输出：
- `[EdgeExtraction] LocalSH 模块加载成功` → 使用LocalSH
- `[EdgeExtraction] LocalSH 模块不可用，将使用曲率方法` → 使用曲率方法

### Q: 边缘提取太慢怎么办？

**A:** 增大 `voxel_size`（如从0.05改为0.08），或减小 `max_points`（如从50000改为30000）。

### Q: 如何完全禁用边缘提取？

**A:** 在配置文件中设置 `enabled: False`，或在代码中传递 `use_edge_enhancement=False`。

## 文件说明

- `utils/edge_extraction.py` - 边缘提取核心代码
- `utils/init_pose.py` - 位姿估计中的边缘增强集成
- `test_edge_extraction.py` - 功能测试脚本
- `STAR_EDGE_INTEGRATION.md` - 完整技术文档（英文）
- `STAR-Edge/` - STAR-Edge原始代码库

## 总结

✓ **独立下采样**：不影响SLAM主流程  
✓ **自动回退**：LocalSH失败时使用曲率方法  
✓ **灵活配置**：通过配置文件调整参数  
✓ **增强位姿**：提升形变场景下的位姿估计精度  

祝实验顺利！有问题请参考 `STAR_EDGE_INTEGRATION.md` 完整文档。



