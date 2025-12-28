# S3PO Baseline with Time-Varying Deformation

## 核心功能更新

### 2025/12/28 - ⭐ Time-Varying Gaussian Deformation (时间形变高斯)

实现了完整的时间形变功能，支持动态场景建模。参考EH-SurGS方法，使用时间基函数对高斯点的位置、旋转、缩放和不透明度进行时间相关的形变。

#### 核心公式

**1. 时间基函数（Gaussian RBF）:**

$$
\phi_k(t) = \exp\left(-\frac{1}{2}\left(\frac{t - \mu_k}{\sigma_k}\right)^2\right)
$$

其中：
- $t \in [0,1]$ 是归一化的时间
- $\mu_k$ 是第 $k$ 个基函数的中心位置（可学习参数）
- $\sigma_k = \text{softplus}(\sigma_{k,\text{raw}})$ 是基函数的宽度（可学习参数）
- $K$ 是基函数数量（默认17，与EH-SurGS一致）

**2. 位置形变:**

$$
\mathbf{x}(t) = \mathbf{x}_0 + \sum_{k=1}^{K} \phi_k(t) \cdot \mathbf{w}_{k,\text{pos}}
$$

其中 $\mathbf{w}_{k,\text{pos}} \in \mathbb{R}^3$ 是位置形变权重（每个点、每个基函数都有独立的权重）。

**3. 完整形变（位置+旋转+缩放+不透明度）:**

$$
\begin{align}
\mathbf{x}(t) &= \mathbf{x}_0 + \sum_{k=1}^{K} \phi_k(t) \cdot \mathbf{w}_{k,\text{pos}} \\
\mathbf{q}(t) &= \text{normalize}\left(\mathbf{q}_0 + \sum_{k=1}^{K} \phi_k(t) \cdot \mathbf{w}_{k,\text{rot}}\right) \\
\mathbf{s}(t) &= \exp\left(\log(\mathbf{s}_0) + \sum_{k=1}^{K} \phi_k(t) \cdot \mathbf{w}_{k,\text{scale}}\right) \\
\alpha(t) &= \text{sigmoid}\left(\text{inverse\_sigmoid}(\alpha_0) + \sum_{k=1}^{K} \phi_k(t) \cdot w_{k,\text{opacity}}\right)
\end{align}
$$

#### 核心特性

1. **形变点选择（Deformation Table）**
   - 自动识别哪些点需要形变，哪些点是静态的
   - 使用 `_deformation_table` 布尔表标记形变点
   - 通过 `_deformation_accum` 累积形变量，动态更新形变表
   - 阈值：`threshold = 0.01`（只有形变量超过此值的点才计算形变）

2. **正则化损失**
   - L2正则化防止位移权重过大：$L_{\text{reg}} = 0.1 \cdot \|\mathbf{w}_{\text{pos}}\|^2$

3. **性能优化**
   - 替换 `distCUDA2` 为纯PyTorch实现的KNN距离计算，避免多进程CUDA上下文问题
   - 分批处理（batch_size=1024）以节省内存

#### 技术细节

- **基函数数量**: $K=17$（可通过 `config.model_params.time_basis_num` 配置）
- **学习率**: 形变参数使用独立的学习率（`deformation_lr_init`，默认是位置学习率的10%）
- **更新频率**: 形变表每200次迭代更新一次
- **初始化**: 所有形变权重初始化为0（静态场景），训练过程中自动学习

---

### 2025/12/8 - 🔧 Robust Huber Loss for Depth Supervision

Replaced standard $L_1$ loss with **Huber Loss** to handle depth outliers (flying pixels) and improve geometric consistency during tracking and mapping.

**Core Formula:**

$$
L_{depth} = \begin{cases} 
\frac{1}{2}(D_{render} - D_{gt})^2 & \text{if } |D_{render} - D_{gt}| \le \delta \\
\delta (|D_{render} - D_{gt}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

**Parameters & Configuration:**
- **Loss Type**: Huber Loss (Smooth L1)
- **Threshold ($\delta$)**: `0.2`
- **Config**: Set `depth_loss: True` in `config.yaml` (under `Dataset` section).

---

## 历史记录

2025/12/8: 12月4号meeting后开的新分支，意在在loss里把置信度先拿掉，用上鲁棒核函数；把点云选择部分的修改拿掉