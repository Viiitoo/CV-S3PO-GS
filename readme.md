2025/12/8 12月4号meeting后开的新分支，意在在loss里把置信度先拿掉，用上鲁棒核函数；把点云选择部分的修改拿掉

2025/12/8 
### 🔧 Update: Robust Huber Loss for Depth Supervision

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