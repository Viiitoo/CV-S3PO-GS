# MotionGS：高斯运动相关光流 Loss（关键步骤 / 公式 / 符号）

## 1. 符号（Notation）

- **相邻两帧图像**：\(I_t, I_{t+1}\)
- **像素坐标域**：\(\mathbf{x}\in\Omega\subset\mathbb{R}^2\)
- **2D 光流**：\(\mathbf{F}(\mathbf{x})=[u(\mathbf{x}), v(\mathbf{x})]^\top\in\mathbb{R}^2\)
- **深度图**：\(D_t(\mathbf{x})\)
- **相机位姿**：\(\mathbf{T}_t,\mathbf{T}_{t+1}\in SE(3)\)
- **相机内参矩阵**：\(\mathbf{K}\)
- **投影 / 反投影**：
  - \(\pi(\mathbf{X})\): 3D 点 \(\mathbf{X}\in\mathbb{R}^3\) 投影到像素平面
  - \(\Pi^{-1}(\mathbf{x}, d)\): 像素 \(\mathbf{x}\) 与深度 \(d\) 反投影到 3D
- **对像素有贡献的高斯集合**：\(\mathcal{K}(\mathbf{x})\)
- **渲染权重**：\(w_k(\mathbf{x})\ge 0\)，归一化权重 \(\tilde w_k(\mathbf{x})\)
- **第 \(k\) 个高斯中心的 2D 投影**：\(\mathbf{p}_{k,t}, \mathbf{p}_{k,t+1}\in\mathbb{R}^2\)
- **像素相对偏移**：\(\boldsymbol\delta_k(\mathbf{x})=\mathbf{x}-\mathbf{p}_{k,t}\)
- **2D conic / 协方差相关矩阵（各向异性椭圆）**：\(\mathbf{C}_{k,t},\mathbf{C}_{k,t+1}\in\mathbb{R}^{2\times 2}\)，及其逆 \(\mathbf{C}^{-1}_{k,t}\)
- **图像高宽**：\(H,W\)
- **可选运动 mask**：\(m(\mathbf{x})\in[0,1]\)

---

## 2. 监督目标：运动光流（Motion Flow）

### 2.1 实际光流（由 2D 光流网络估计）

\[
\mathbf{F}_{\text{gt}}(\mathbf{x}) \approx \text{FlowNet}(I_t, I_{t+1})
\]

### 2.2 相机光流（由深度 + 位姿产生）

反投影到 3D：

\[
\mathbf{X}=\Pi^{-1}(\mathbf{x}, D_t(\mathbf{x}))
= D_t(\mathbf{x})\;\mathbf{K}^{-1}\begin{bmatrix}\mathbf{x}\\1\end{bmatrix}
\]

将 \(\mathbf{X}\) 从 \(t\) 帧坐标变换到 \(t{+}1\)：

\[
\mathbf{X}' = \mathbf{T}_{t+1}\mathbf{T}_t^{-1}\mathbf{X}
\]

重投影到像素：

\[
\mathbf{x}' = \pi(\mathbf{X}')
\]

相机光流：

\[
\mathbf{F}_{\text{cam}}(\mathbf{x})=\mathbf{x}'-\mathbf{x}
\]

### 2.3 运动光流（监督信号）

\[
\mathbf{F}_{\text{motion}}(\mathbf{x})
=
\mathbf{F}_{\text{gt}}(\mathbf{x})
-
\mathbf{F}_{\text{cam}}(\mathbf{x})
\]

若使用 mask（示例写法）：

\[
\mathbf{F}_{\text{motion}}(\mathbf{x}) \leftarrow m(\mathbf{x})\,\mathbf{F}_{\text{motion}}(\mathbf{x})
\]

---

## 3. 预测：由高斯运动得到 GS 光流 \(\mathbf{F}_{\text{gs}}\)

### 3.1 conic / 协方差乘积相关的局部线性映射

\[
\mathbf{A}_k = \mathbf{C}_{k,t+1}\,\mathbf{C}^{-1}_{k,t}
\]

### 3.2 单个高斯对像素的局部光流贡献

\[
\mathbf{f}_k(\mathbf{x})
=
\underbrace{\left(\mathbf{A}_k\boldsymbol\delta_k(\mathbf{x})-\boldsymbol\delta_k(\mathbf{x})\right)}_{\text{各向异性形状/朝向引起的局部变形项}}
+
\underbrace{\left(\mathbf{p}_{k,t+1}-\mathbf{p}_{k,t}\right)}_{\text{高斯中心投影的平移项}}
\]

其中：
\[
\boldsymbol\delta_k(\mathbf{x})=\mathbf{x}-\mathbf{p}_{k,t}
\]

### 3.3 像素级 GS 光流（按渲染权重加权平均）

权重归一化：

\[
\tilde w_k(\mathbf{x})=
\frac{w_k(\mathbf{x})}{\sum_{j\in\mathcal{K}(\mathbf{x})} w_j(\mathbf{x})+\epsilon}
\]

最终预测光流：

\[
\mathbf{F}_{\text{gs}}(\mathbf{x})
=
\sum_{k\in\mathcal{K}(\mathbf{x})}
\tilde w_k(\mathbf{x})\,\mathbf{f}_k(\mathbf{x})
\]

---

## 4. 光流 Loss：归一化 + 截断 + L1

### 4.1 分量归一化（分辨率无关）

\[
\mathcal{N}(\mathbf{F})(\mathbf{x})
=
\begin{bmatrix}
u(\mathbf{x})/H\\
v(\mathbf{x})/W
\end{bmatrix}
\]

### 4.2 截断（抑制极端 outlier）

\[
\mathcal{C}(\mathbf{y})=\mathrm{clip}(\mathbf{y},-1,1)
\]

### 4.3 运动光流监督的 L1 损失

\[
\mathcal{L}_{\text{flow}}
=
\frac{1}{|\Omega|}
\sum_{\mathbf{x}\in\Omega}
\left\|
\mathcal{C}\big(\mathcal{N}(\mathbf{F}_{\text{gs}})(\mathbf{x})\big)
-
\mathcal{C}\big(\mathcal{N}(\mathbf{F}_{\text{motion}})(\mathbf{x})\big)
\right\|_1
\]

### 4.4 总目标中的加权（示意）

\[
\mathcal{L} = \mathcal{L}_{\text{img}} + \lambda_{\text{flow}}\,\mathcal{L}_{\text{flow}}
\]


