# 轮廓RGB输出功能说明

## 功能概述

本功能在评估渲染时，自动输出带有轮廓信息的RGB图像。轮廓信息来自STAR-Edge提取的边缘点，这些边缘点会被投影到每个相机位姿的2D图像上，并以绿色圆点的方式叠加在渲染的RGB图像上。

## 实现位置

- **文件**: `utils/eval_utils.py`
- **函数**: 
  - `project_3d_to_2d()`: 将3D边缘点投影到2D图像坐标
  - `draw_edges_on_image()`: 在RGB图像上绘制轮廓
  - `eval_rendering()`: 评估渲染函数（已集成轮廓绘制功能）

## 输出目录

带轮廓的RGB图像保存在：
```
{save_dir}/render_rgb_with_edges/{frame_idx}_rgb_with_edges.png
```

## 使用方法

### 1. 启用轮廓提取

在配置文件中启用轮廓提取功能：

```yaml
model_params:
  enable_edge_extraction: true  # 启用轮廓提取
```

### 2. 运行评估

运行SLAM系统，当调用 `eval_rendering()` 函数时，会自动：
1. 检查是否存在边缘点（`gaussians._edge_points`）
2. 如果存在，将边缘点投影到当前相机位姿
3. 在渲染的RGB图像上绘制绿色轮廓点
4. 保存到 `render_rgb_with_edges` 目录

### 3. 查看结果

评估完成后，在 `{save_dir}/render_rgb_with_edges/` 目录下可以找到所有带轮廓的RGB图像。

## 功能特点

1. **自动检测**: 如果边缘点不存在，会自动保存原始RGB图像（不会报错）
2. **性能优化**: 
   - 最多显示5000个边缘点（避免图像混乱）
   - 自动过滤不可见的点（深度<0.1或超出图像范围）
3. **错误处理**: 如果绘制失败，会自动回退到保存原始RGB图像
4. **颜色设置**: 默认使用绿色轮廓（BGR格式: (0, 255, 0)）

## 参数说明

### `draw_edges_on_image()` 函数参数

- `edge_color`: 轮廓颜色，默认 `(0, 255, 0)` (绿色，BGR格式)
- `edge_thickness`: 轮廓点大小，默认 `2` 像素
- `max_points`: 最大显示点数，默认 `5000`（如果边缘点过多，会随机采样）

### 修改轮廓颜色

如果需要修改轮廓颜色，可以在 `eval_rendering()` 函数中修改：

```python
rgb_with_edges = draw_edges_on_image(
    pred,
    edge_points_3d,
    frame,
    edge_color=(255, 0, 0),  # 改为红色 (BGR格式)
    edge_thickness=3,        # 增大点的大小
    max_points=10000          # 增加最大点数
)
```

## 注意事项

1. **轮廓提取**: 需要先启用轮廓提取功能（`enable_edge_extraction: true`），否则只会保存原始RGB图像
2. **边缘点来源**: 边缘点来自初始点云创建时提取的轮廓，可能不完全匹配当前视图
3. **性能影响**: 轮廓绘制对性能影响很小（每帧约1-5ms）

## 示例输出

评估完成后，会在以下目录生成文件：

```
results/
  └── {dataset_name}_{timestamp}/
      └── render_rgb_with_edges/
          ├── 0_rgb_with_edges.png
          ├── 1_rgb_with_edges.png
          ├── 2_rgb_with_edges.png
          └── ...
```

每个图像都是渲染的RGB图像，上面叠加了绿色轮廓点，表示从该视角可见的边缘点。

## 故障排除

### 问题1：没有生成轮廓图像

**可能原因**:
- 轮廓提取功能未启用
- 边缘点不存在（`gaussians._edge_points` 为 None）

**解决方案**:
- 检查配置文件中是否启用了 `enable_edge_extraction`
- 检查终端输出是否有轮廓提取的相关信息

### 问题2：轮廓点太少或太多

**解决方案**:
- 调整 `max_points` 参数
- 检查边缘点提取的质量（可能需要调整STAR-Edge参数）

### 问题3：轮廓点位置不准确

**可能原因**:
- 边缘点来自初始点云，可能与当前视图不完全匹配
- 相机位姿估计不准确

**解决方案**:
- 这是正常现象，边缘点主要用于可视化
- 如果需要更准确的轮廓，可以考虑实时提取边缘点（会增加计算开销）

## 技术细节

### 投影流程

1. **世界坐标 → 视图坐标**: 使用 `getWorld2View2()` 将3D点从世界坐标系转换到相机视图坐标系
2. **视图坐标 → 图像坐标**: 使用相机内参（fx, fy, cx, cy）将3D点投影到2D图像平面
3. **有效性检查**: 
   - 深度检查（z > 0.1）
   - 图像范围检查（0 <= u < width, 0 <= v < height）

### 绘制方法

使用OpenCV的 `cv2.circle()` 函数在图像上绘制圆形点，表示边缘位置。


