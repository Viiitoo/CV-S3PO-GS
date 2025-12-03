2025/11/30 增加了一个输出：置信度图

2025/12/1 把所有帧trj都输出了（原本是只输出关键帧）

2025/12/2 
引入了 MASt3R 输出的置信度图（Confidence Map）参与 Loss 计算
1. Mathematical Formulation 核心变化在于 Depth Loss 增加了置信度权重 $C$：
   
   * **Before (Standard L1):**
       $$L_{depth} = | D_{render} - D_{gt} |_1$$
   
   * **After (Confidence-Weighted):**
       $$L_{depth} = \mathbf{C} \cdot | D_{render} - D_{gt} |_1$$
2. Core Implementation

FrontEnd: 获取 MASt3R 的 confidence 并存储至 viewpoint.confidence_map (GPU Tensor)。

BackEnd / Tracking: 在 get_loss_mapping 和 get_loss_tracking 中提取该 Map。

Loss Calculation: 修改 slam_utils.py，将置信度作为 Soft Mask 乘入深度误差项。

2025/12/3
引入鲁棒核函数

**Core Formulation:**
Instead of a binary selection, the final depth is computed as:

$$
X_{fused} = \alpha X^r + (1 - \alpha) X^p
$$

Where the fusion weight $\alpha$ is determined by the confidence $c$ and the relative residual $r$:

$$
\alpha = c \cdot w(r), \quad w(r) = \frac{1}{1 + (r / \delta)^2}
$$

* $r = (X^r - X^p) / X^p$: Relative geometric error.
* $\delta$: Error tolerance threshold (e.g., 0.15).
* $w(r)$: **Cauchy Robust Kernel** that automatically down-weights large errors.