# STAR-Edge GPU CUDA Kernel 实现方案

## 0. 背景与目标

### 当前性能瓶颈

STAR-Edge 轮廓提取的核心是 **LocalSH 特征计算**（C++ pybind11, CPU），占整条流水线 95%+ 的耗时：

| 点云规模 N | CPU LocalSH 耗时 | PyTorch GPU (参考) | CUDA Kernel (目标) |
|-----------|-----------------|-------------------|-------------------|
| 1,000 | 340 ms | ~13 ms | ~6 ms |
| 5,000 | 1,660 ms | ~60 ms | ~25 ms |
| 15,000 | 5,110 ms | ~188 ms | ~80 ms |

### 目标

用自定义 CUDA Kernel 替代 `LocalSH.LocalSHFeature.ComLSHF_knn_upsample()` 全部计算，实现 **50-65x 加速**（相对 CPU），比纯 PyTorch 方案再快约 **2x**。

### 环境约束

| 项目 | 值 |
|------|-----|
| Docker | 5d0fdd8cb92d (s3po-baseline:latest) |
| Python | 3.11.0 |
| PyTorch | 2.1.0+cu118 |
| CUDA | 11.8 (nvcc 可用: `/usr/local/cuda/bin/nvcc`) |
| GPU | 4x RTX 3090 (24GB, SM 8.6) |
| g++ | 11.4.0 |
| ninja | 不可用（无外网）|
| 编译方式 | `setup.py` + `torch.utils.cpp_extension.BuildExtension` |

---

## 1. 整体架构

### 1.1 算法流水线

原始 C++ `ComLSHF_knn_upsample` 对每个点执行以下步骤，用 CUDA kernel 全部替代：

```
输入: points (N, 3) float32 on GPU

┌─────────────────────────────────────────────────────────────┐
│ Kernel 1: KNN Search (Morton code + spatial partitioning)   │
│   points (N,3) → knn_idx (N,K), knn_dist (N,K)            │
│   K=26, 基于 simple-knn 的 Morton code 排序框架扩展         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Kernel 2: Direction Vectors (fused gather + normalize)      │
│   points, knn_idx → directions (N, K, 3)                   │
│   每线程处理1个点, 循环K个邻居                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Kernel 3: PCA + Angular Sort + Curve Sampling (fused)       │
│   directions (N,K,3) → sample_curves (N, S, 3)             │
│   S=30, 替代凸包: 3x3协方差→特征分解→2D投影→角度排序→插值    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Kernel 4: Sphere KDE (fused compute + max reduction)        │
│   sample_curves (N,S,3) + grid_xyz (G,3)                   │
│   → sphere_func (N, G)                                     │
│   G=400, 不物化中间量(N,S,G), 逐grid cell循环S个采样点      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Kernel 5: SHT + Power Spectrum (cuBLAS GEMM + custom reduce)│
│   sphere_func (N,G) × SH_basis (G, bw²) → coeffs (N, bw²) │
│   coeffs → descriptors (N, bw)                             │
│   bw=10, G=400, cuBLAS sgemm + 功率谱归约                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ MLP Inference (已有 PyTorch DescClassifier, GPU)             │
│   descriptors (N,10) → edge_scores (N,)                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 文件结构

```
submodules/
└── star_edge_cuda/
    ├── setup.py                  # 编译配置
    ├── star_edge_cuda/
    │   └── __init__.py           # Python 包入口
    ├── ext.cpp                   # pybind11 绑定
    ├── star_edge.h               # 头文件: 函数声明
    ├── star_edge.cu              # 主入口: 调度各kernel
    ├── knn_morton.cu             # Kernel 1: Morton code KNN
    ├── knn_morton.h              # Kernel 1 头文件
    ├── direction_sampling.cu     # Kernel 2+3: 方向向量 + 采样
    ├── direction_sampling.h      # Kernel 2+3 头文件
    ├── sphere_kde.cu             # Kernel 4: 球面核密度估计
    ├── sphere_kde.h              # Kernel 4 头文件
    ├── sht_power.cu              # Kernel 5: SHT + 功率谱
    ├── sht_power.h               # Kernel 5 头文件
    └── sh_basis_precompute.py    # SH基矩阵离线预计算脚本

utils/
└── edge_extraction.py            # 修改: 添加CUDA路径调用
```

---

## 2. 编译系统

### 2.1 setup.py

沿用项目中 `simple-knn` 和 `diff-gaussian-rasterization` 的模式：

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="star_edge_cuda",
    packages=["star_edge_cuda"],
    ext_modules=[
        CUDAExtension(
            name="star_edge_cuda._C",
            sources=[
                "ext.cpp",
                "star_edge.cu",
                "knn_morton.cu",
                "direction_sampling.cu",
                "sphere_kde.cu",
                "sht_power.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "--std=c++17",
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### 2.2 安装方式

```bash
# 在 Docker 容器内 (/workspace)
cd submodules/star_edge_cuda
pip install -e .
```

### 2.3 ext.cpp (pybind11 入口)

```cpp
#include <torch/extension.h>
#include "star_edge.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_localsh_descriptors", &compute_localsh_descriptors,
          "STAR-Edge LocalSH descriptor computation (CUDA)",
          py::arg("points"),
          py::arg("sh_basis"),
          py::arg("grid_xyz"),
          py::arg("bw") = 10,
          py::arg("kk") = 26,
          py::arg("num_samples") = 30);

    m.def("knn_search", &knn_search,
          "Morton code based KNN search (CUDA)",
          py::arg("points"),
          py::arg("k") = 26);
}
```

---

## 3. Kernel 1: KNN Search (Morton Code)

### 3.1 设计思路

扩展 `simple-knn` 的 Morton code + 空间分区方案，从 K=3 扩展到 K=26：

**原 simple-knn 流程：**
1. CUB DeviceReduce 求点云 AABB (min/max)
2. 每点计算 Morton code (30-bit, 10 bits/axis)
3. CUB RadixSort 按 Morton code 排序
4. 每 BOX_SIZE=1024 个连续点为一个 box，计算 box 的 AABB
5. 每个点用 box AABB 剪枝，在 box 内逐点比较更新 K-best

**扩展到 K=26 的改动：**
- `updateKBest` 模板参数从 `K=3` 改为 `K=26`
- 增大局部搜索窗口从 `idx±3` 到 `idx±30`
- 返回 KNN **索引** (不仅是距离)，供后续 kernel 使用

### 3.2 数据结构

```cpp
// 输入
float3* points;     // (N,) 点云坐标

// 输出
int*    knn_idx;     // (N * K,) 每点的K近邻索引, row-major
float*  knn_dist;    // (N * K,) 每点的K近邻距离²

// 中间量
uint32_t* morton_codes;      // (N,) Morton codes
uint32_t* sorted_indices;    // (N,) 排序后的点索引
MinMax*   boxes;             // (num_boxes,) 空间分区AABB
```

### 3.3 Kernel 实现

```cpp
#define KNN_K 26
#define BOX_SIZE 1024

// K-best 维护: 插入排序, K=26 时仍然高效(26次比较)
template<int K>
__device__ void updateKBestWithIdx(
    const float3& ref, const float3& point, int point_idx,
    float* best_dist, int* best_idx)
{
    float3 d = {point.x - ref.x, point.y - ref.y, point.z - ref.z};
    float dist = d.x * d.x + d.y * d.y + d.z * d.z;

    // 插入排序: 从后往前找插入位置
    if (dist < best_dist[K-1]) {
        best_dist[K-1] = dist;
        best_idx[K-1] = point_idx;
        // bubble up
        for (int j = K-2; j >= 0; --j) {
            if (best_dist[j] > best_dist[j+1]) {
                // swap
                float td = best_dist[j]; best_dist[j] = best_dist[j+1]; best_dist[j+1] = td;
                int ti = best_idx[j]; best_idx[j] = best_idx[j+1]; best_idx[j+1] = ti;
            } else break;
        }
    }
}

__global__ void knnMortonSearch(
    uint32_t P, float3* points, uint32_t* sorted_indices,
    MinMax* boxes, int* out_idx, float* out_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    uint32_t orig_idx = sorted_indices[idx];
    float3 point = points[orig_idx];

    float best_dist[KNN_K];
    int   best_idx[KNN_K];
    for (int j = 0; j < KNN_K; j++) {
        best_dist[j] = FLT_MAX;
        best_idx[j] = -1;
    }

    // 局部邻域先扫描 (Morton排序后空间上相邻)
    int local_range = 40;  // K=26时适当扩大窗口
    for (int i = max(0, idx - local_range);
         i <= min((int)P - 1, idx + local_range); i++) {
        if (i == idx) continue;
        uint32_t ni = sorted_indices[i];
        updateKBestWithIdx<KNN_K>(point, points[ni], ni, best_dist, best_idx);
    }

    // 用当前第K个距离作为剪枝半径
    float reject = best_dist[KNN_K - 1];

    // box级别剪枝扫描
    uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    for (uint32_t b = 0; b < num_boxes; b++) {
        float box_dist = distBoxPoint(boxes[b], point);
        if (box_dist > reject)
            continue;

        // 搜索box内的点
        uint32_t start = b * BOX_SIZE;
        uint32_t end = min(P, start + BOX_SIZE);
        for (uint32_t i = start; i < end; i++) {
            if (i == (uint32_t)idx) continue;
            uint32_t ni = sorted_indices[i];
            updateKBestWithIdx<KNN_K>(point, points[ni], ni, best_dist, best_idx);
        }
        reject = best_dist[KNN_K - 1];  // 更新剪枝半径
    }

    // 写出结果 (按原始索引)
    for (int j = 0; j < KNN_K; j++) {
        out_idx[orig_idx * KNN_K + j] = best_idx[j];
        out_dist[orig_idx * KNN_K + j] = best_dist[j];
    }
}
```

### 3.4 线程配置

```
Block: 256 threads
Grid:  ceil(N / 256) blocks
寄存器: 每线程 KNN_K*2 = 52 个 float/int (208 bytes) → SM 8.6 寄存器充足
共享内存: 不需要 (box AABB 从全局内存读取, 访问次数少)
```

### 3.5 性能预估

| 操作 | 耗时 (N=15000) |
|------|----------------|
| Morton code + sort | ~2 ms (CUB radix sort) |
| Box AABB | ~0.1 ms |
| KNN search | ~5-10 ms (box剪枝大幅减少比较次数) |
| **总计** | **~8-12 ms** |

对比 `torch.cdist+topk` 的 71ms，预期 **6-9x 加速**。

---

## 4. Kernel 2: Direction Vectors

### 4.1 设计

最简单的 kernel，每线程处理一个点的所有 K 个邻居：

```cpp
__global__ void computeDirectionVectors(
    int N, int K,
    const float3* __restrict__ points,
    const int*    __restrict__ knn_idx,    // (N*K,)
    float3*       __restrict__ directions) // (N*K,)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    float3 p = points[pid];

    for (int k = 0; k < K; k++) {
        int nid = knn_idx[pid * K + k];
        float3 q = points[nid];
        float3 d = {q.x - p.x, q.y - p.y, q.z - p.z};
        float norm = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);

        if (norm > 1e-10f) {
            d.x /= norm; d.y /= norm; d.z /= norm;
        }
        directions[pid * K + k] = d;
    }
}
```

### 4.2 线程配置

```
Block: 256
Grid:  ceil(N / 256)
```

### 4.3 性能预估

纯内存受限操作，预计 **~1-2 ms** (N=15000)。
主要开销在 `points[nid]` 的随机内存访问。

---

## 5. Kernel 3: PCA + Angular Sort + Curve Sampling

### 5.1 设计思路

这是替代原始 C++ 中 `FittingLineSample::projectPointCloud` + `convexHull` + `fitCurve` 的核心 kernel。

**原始算法：**
1. PCA投影: 将K个3D方向向量投影到2D平面
2. 凸包: Graham scan 提取2D外边界
3. 曲线拟合: 在凸包点上分段线性插值, 均匀采样30个点
4. 归一化: 采样点归一化到单位球

**GPU替代方案（等效近似）：**
1. 3×3 协方差矩阵 → 解析特征分解 (不需要迭代SVD)
2. 投影到前2个主方向 → 2D坐标
3. 按极角排序 (K=26, 插入排序足够)
4. 取最外层点作为边界 (按角度分bins, 每bin取最远点 → 近似凸包)
5. 等角度间隔线性插值采样 S=30 个3D方向
6. 归一化到单位球

### 5.2 Kernel 实现

```cpp
// 3x3对称矩阵的解析特征分解 (Cardano公式)
// 适用于协方差矩阵, 保证实特征值
__device__ void eigen3x3_symmetric(
    float cov[6],  // 上三角: c00,c01,c02,c11,c12,c22
    float eigvec[9], float eigval[3]);

__global__ void pcaAngularSampling(
    int N, int K, int S,
    const float3* __restrict__ directions,  // (N*K,)
    float3*       __restrict__ samples)     // (N*S,) output
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    const float3* dirs = directions + pid * K;
    float3* out = samples + pid * S;

    // ---- Step 1: 计算3x3协方差矩阵 ----
    // cov = (1/K) * sum(d * d^T), d已中心化(均值≈0因为方向分布近似对称)
    float cov[6] = {0}; // 上三角: xx,xy,xz,yy,yz,zz
    float mx=0, my=0, mz=0;
    for (int k = 0; k < K; k++) {
        mx += dirs[k].x; my += dirs[k].y; mz += dirs[k].z;
    }
    mx /= K; my /= K; mz /= K;

    for (int k = 0; k < K; k++) {
        float dx = dirs[k].x - mx;
        float dy = dirs[k].y - my;
        float dz = dirs[k].z - mz;
        cov[0] += dx*dx; cov[1] += dx*dy; cov[2] += dx*dz;
        cov[3] += dy*dy; cov[4] += dy*dz; cov[5] += dz*dz;
    }

    // ---- Step 2: 特征分解, 取前2个主方向 ----
    float eigvec[9], eigval[3];
    eigen3x3_symmetric(cov, eigvec, eigval);
    // eigvec 按特征值降序排列, 取前2列作为投影方向
    float3 axis1 = {eigvec[0], eigvec[1], eigvec[2]};
    float3 axis2 = {eigvec[3], eigvec[4], eigvec[5]};

    // ---- Step 3: 投影到2D + 计算极角 ----
    float angles[26];  // 最大K=26, 放在寄存器/local memory
    float radii[26];
    for (int k = 0; k < K; k++) {
        float dx = dirs[k].x - mx, dy = dirs[k].y - my, dz = dirs[k].z - mz;
        float u = dx*axis1.x + dy*axis1.y + dz*axis1.z;
        float v = dx*axis2.x + dy*axis2.y + dz*axis2.z;
        angles[k] = atan2f(v, u);
        radii[k] = sqrtf(u*u + v*v);
    }

    // ---- Step 4: 按角度排序 (插入排序, K=26) ----
    int sorted_idx[26];
    for (int k = 0; k < K; k++) sorted_idx[k] = k;
    for (int i = 1; i < K; i++) {
        int key = sorted_idx[i];
        float key_angle = angles[key];
        int j = i - 1;
        while (j >= 0 && angles[sorted_idx[j]] > key_angle) {
            sorted_idx[j+1] = sorted_idx[j];
            j--;
        }
        sorted_idx[j+1] = key;
    }

    // ---- Step 5: 近似凸包 → 取每个角度区间最远的点 ----
    // 将 [-pi, pi] 分成 S=30 个bins, 每bin取最远点
    // 然后在这些点之间线性插值得到均匀采样

    // 简化: 直接等角度间隔从排序后的方向中采样
    float angle_start = angles[sorted_idx[0]];
    float angle_end = angles[sorted_idx[K-1]];
    float angle_range = angle_end - angle_start;
    if (angle_range < 0) angle_range += 2.0f * M_PI;

    for (int s = 0; s < S; s++) {
        float target = angle_start + angle_range * s / (float)(S - 1);
        // 在排序后的方向中找两个最近的, 线性插值
        int lo = 0, hi = 0;
        // ... (binary search or linear scan for interpolation bounds)

        // 线性插值3D方向
        float3 interp;
        // ... (interpolate between dirs[sorted_idx[lo]] and dirs[sorted_idx[hi]])

        // 归一化到单位球
        float norm = sqrtf(interp.x*interp.x + interp.y*interp.y + interp.z*interp.z);
        out[s] = {interp.x/norm, interp.y/norm, interp.z/norm};
    }
}
```

### 5.3 3x3 对称矩阵特征分解

K=26 的协方差矩阵是 3×3 对称实矩阵，使用 **Cardano 公式** 可以 O(1) 解析求解（无需迭代），完全适合 GPU：

```cpp
// 基于 Cardano 公式的解析特征分解
// 参考: Kopp, J. "Efficient numerical diagonalization of hermitian 3x3 matrices"
__device__ void eigen3x3_cardano(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float* eigval,   // [3] 降序
    float* eigvec)   // [9] 列优先, 对应3个特征向量
{
    // de = det(A - lambda*I) = -lambda^3 + c2*lambda^2 + c1*lambda + c0
    float c2 = a00 + a11 + a22;  // trace
    float c1 = a01*a01 + a02*a02 + a12*a12 - a00*a11 - a00*a22 - a11*a22;
    float c0 = a00*a11*a22 + 2*a01*a02*a12 - a00*a12*a12 - a11*a02*a02 - a22*a01*a01;

    // Cardano's formula for depressed cubic
    float p = c2*c2/3.0f + c1;      // 注意: 不是标准形式，需要调整
    float q = 2.0f*c2*c2*c2/27.0f + c1*c2/3.0f + c0;
    // ... (标准 Cardano 求解)

    // 求特征向量: (A - lambda_i * I) v_i = 0
    // 用叉积法: v_i = (A-lambda_i*I)[row_j] × (A-lambda_i*I)[row_k]
    // ...
}
```

### 5.4 线程配置

```
Block: 128 (寄存器压力较大, 每线程需要 ~200 bytes local memory)
Grid:  ceil(N / 128)
```

### 5.5 性能预估

每线程的计算量：26次乘加(协方差) + 特征分解(~50 flops) + 26次投影 + 排序(~26²/2=338比较) + 30次插值。

预计 **~3-5 ms** (N=15000)。

---

## 6. Kernel 4: Sphere KDE (融合核密度 + Max归约)

### 6.1 设计思路

这是最关键的优化点。PyTorch 方案需要物化 `(N, S, G)` = `(15000, 30, 400)` 的中间 tensor (687MB)。CUDA kernel 将计算和归约融合，**零中间显存开销**。

**算法复现（对应 `LocalSHFeature::LocalSH()`）：**
1. 球面网格 `G = 2*bw × 2*bw = 400` 个cell
2. 每个cell的球面坐标 `(theta_i, phi_j)` → 笛卡尔坐标 `grid_xyz[g]`
3. 对每个 `(point, grid_cell)` 对：
   - 遍历 S=30 个 curve sample 点
   - 计算 sample 与 grid_cell 的测地距离 `= arccos(dot(sample, grid_cell))`
   - 高斯核 `K = exp(-0.5 * (dist/h)²) / sqrt(2*pi)`
   - 取 S 个值中的 **最大值** (对应原始代码的 `ddd_max`)
4. 输出 `sphere_func[point][grid_cell] = max_k KDE(sample_k, grid_cell)`

### 6.2 Kernel 实现

```cpp
// 预计算的球面网格坐标 (常量内存, 400*3 = 4.8KB)
__constant__ float c_grid_xyz[400 * 3];  // (G, 3)

__global__ void sphereKDE(
    int N, int S, int G,  // G=400, S=30
    float h,              // bandwidth = pi / (2*bw)
    const float3* __restrict__ samples,       // (N*S,)
    float*        __restrict__ sphere_func)   // (N*G,) output
{
    // 每线程处理一个 (point, grid_cell) 对
    // 2D grid: blockIdx.x → grid_cell batches, blockIdx.y → point batches
    int g = blockIdx.x * blockDim.x + threadIdx.x;  // grid cell index
    int pid = blockIdx.y * blockDim.y + threadIdx.y; // point index

    if (g >= G || pid >= N) return;

    // 从常量内存读取 grid cell 坐标
    float gx = c_grid_xyz[g * 3 + 0];
    float gy = c_grid_xyz[g * 3 + 1];
    float gz = c_grid_xyz[g * 3 + 2];

    float inv_h = 1.0f / h;
    float inv_sqrt2pi = 0.3989422804f;  // 1/sqrt(2*pi)
    float inv_hS = 1.0f / (h * S);

    float kde_max = 0.0f;

    const float3* my_samples = samples + pid * S;

    // 遍历 S=30 个 curve sample 点
    for (int s = 0; s < S; s++) {
        float3 sv = my_samples[s];

        // cos(geodesic distance) = dot product (both unit vectors)
        float cos_dist = sv.x * gx + sv.y * gy + sv.z * gz;
        cos_dist = fminf(fmaxf(cos_dist, -1.0f), 1.0f);

        float geodesic = acosf(cos_dist);
        float x = geodesic * inv_h;

        // Gaussian kernel
        float K_val = inv_sqrt2pi * expf(-0.5f * x * x);
        float D = K_val * inv_hS;

        kde_max = fmaxf(kde_max, D);
    }

    sphere_func[pid * G + g] = kde_max;
}
```

### 6.3 线程配置

```
Block: (20, 16) = 320 threads  (20 grid cells × 16 points per block)
Grid:  (ceil(400/20), ceil(N/16)) = (20, 938) for N=15000

每线程: 30次 dot+acos+exp = ~900 flops
总 flops: 15000 * 400 * 900 ≈ 5.4 G flops
RTX 3090 FP32: 35.6 TFLOPS → 理论 ~0.15ms
实际(内存+指令延迟): ~3-5 ms
```

### 6.4 优化: 利用常量内存

球面网格坐标 `grid_xyz` 仅 400×3×4 = 4.8KB，放入 **CUDA 常量内存** (`__constant__`)。常量内存通过广播机制对同一 warp 内读取相同地址有 ~1 cycle 延迟，非常适合这种所有线程都读取相同网格点的模式。

### 6.5 替代线程布局 (按点分配)

如果 N 较大，可以改为每线程处理一个点的所有 G 个 grid cell：

```cpp
__global__ void sphereKDE_v2(
    int N, int S, int G, float h,
    const float3* __restrict__ samples,
    float*        __restrict__ sphere_func)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    const float3* my_samples = samples + pid * S;

    // 预加载采样点到寄存器/shared memory
    float3 local_samples[30];  // S=30, 放在 local memory
    for (int s = 0; s < S; s++)
        local_samples[s] = my_samples[s];

    for (int g = 0; g < G; g++) {
        float gx = c_grid_xyz[g*3+0];
        float gy = c_grid_xyz[g*3+1];
        float gz = c_grid_xyz[g*3+2];

        float kde_max = 0.0f;
        for (int s = 0; s < S; s++) {
            float cos_dist = local_samples[s].x*gx + local_samples[s].y*gy + local_samples[s].z*gz;
            cos_dist = fminf(fmaxf(cos_dist, -1.0f), 1.0f);
            float x = acosf(cos_dist) / h;
            float K_val = 0.3989422804f * expf(-0.5f * x * x);
            kde_max = fmaxf(kde_max, K_val / (h * S));
        }
        sphere_func[pid * G + g] = kde_max;
    }
}
```

此方案 **每线程处理 400×30 = 12000 次乘加**，适合 occupancy 较低但指令级并行度高的场景。

### 6.6 性能预估

| 方案 | 预估耗时 (N=15000) |
|------|-------------------|
| v1 (2D grid) | ~3-5 ms |
| v2 (per-point) | ~4-7 ms |

对比 PyTorch 方案的 41ms，预期 **8-14x 加速**。

---

## 7. Kernel 5: SHT + Power Spectrum

### 7.1 SH 基矩阵预计算

SHT 本质是 `coeffs = sphere_func × SH_basis^T`，其中 SH 基矩阵可以离线预计算。

**预计算脚本 `sh_basis_precompute.py`：**

```python
import numpy as np
from scipy.special import sph_harm_y

def precompute_sh_basis(bw=10):
    """
    预计算 S2kit 兼容的 SHT 矩阵。

    S2kit 的 FSTSemiMemo 使用 Driscoll-Healy 采样:
      theta_j = pi * (2*j + 1) / (4*bw),  j = 0..2*bw-1
      phi_k   = 2*pi * k / (2*bw),        k = 0..2*bw-1

    但原始 LocalSH 代码中 KDE 使用的网格是:
      theta_j = (j + 0.5) * pi / (2*bw)   (与S2kit相同!)
      phi_k   = (k + 0.5) * 2*pi / (2*bw) (有0.5偏移!)

    为了与原始代码数值一致, 使用与 S2kit FSTSemiMemo
    等效的矩阵形式, 通过测试用例验证。
    """
    lat = lon = 2 * bw
    G = lat * lon   # 400
    C = bw * bw     # 100

    # S2kit 的采样网格
    theta = np.array([(2*j + 1) * np.pi / (4*bw) for j in range(lat)])
    phi   = np.array([2*np.pi * k / (2*bw) for k in range(lon)])

    # S2kit 的积分权重 (Driscoll-Healy)
    weights = compute_dh_weights(bw)  # (2*bw,)

    # SH 基矩阵: (C, G) complex
    basis = np.zeros((C, G), dtype=np.complex128)
    for l in range(bw):
        for m in range(-l, l+1):
            c_idx = index_of_sh(m, l, bw)  # S2kit 的系数索引
            for j in range(lat):
                for k in range(lon):
                    g_idx = j * lon + k
                    Y = sph_harm_y(l, m, theta[j], phi[k])
                    basis[c_idx, g_idx] = Y * weights[j] * (2*np.pi / lon)

    # 保存为 float32 (实部和虚部分开)
    np.save("sh_basis_real.npy", basis.real.astype(np.float32))
    np.save("sh_basis_imag.npy", basis.imag.astype(np.float32))
    return basis

def index_of_sh(m, l, bw):
    """S2kit 的 SH 系数索引 (与 IndexOfHarmonicCoeff 一致)"""
    if m >= 0:
        return m * bw + l
    else:
        return (bw + m) * bw + l  # 需要验证

def compute_dh_weights(bw):
    """Driscoll-Healy 积分权重"""
    weights = np.zeros(2 * bw)
    for j in range(2 * bw):
        w = 0.0
        for k in range(bw):
            w += 1.0 / (2*k + 1) * np.sin((2*j+1) * (2*k+1) * np.pi / (4*bw))
        weights[j] = w * 2.0 * np.pi / (bw * bw) * np.sin((2*j+1) * np.pi / (4*bw))
    return weights
```

**验证方法：** 用相同输入分别跑 CPU S2kit 和 GPU 矩阵乘法，对比系数的 L2 相对误差 < 1%。

### 7.2 SHT: cuBLAS GEMM

```cpp
// sphere_func: (N, G) float32,  row-major
// basis_real:  (C, G) float32,  预计算, 常驻GPU
// basis_imag:  (C, G) float32
// coeffs_real: (N, C) float32,  output
// coeffs_imag: (N, C) float32,  output

// coeffs = sphere_func × basis^T
// 即 (N, C) = (N, G) × (G, C)

#include <cublas_v2.h>

void computeSHT(
    cublasHandle_t handle,
    int N, int G, int C,
    const float* sphere_func,   // (N, G)
    const float* basis_real,    // (C, G) → 转置为 (G, C)
    const float* basis_imag,    // (C, G)
    float* coeffs_real,         // (N, C)
    float* coeffs_imag)         // (N, C)
{
    float alpha = 1.0f, beta = 0.0f;

    // coeffs_real = sphere_func × basis_real^T
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // basis transposed, sphere_func not
        C, N, G,                    // M=C, N=N, K=G
        &alpha,
        basis_real, G,              // (G, C) after transpose
        sphere_func, G,             // (N, G) → lda=G
        &beta,
        coeffs_real, C);            // (N, C)

    // coeffs_imag = sphere_func × basis_imag^T
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        C, N, G,
        &alpha,
        basis_imag, G,
        sphere_func, G,
        &beta,
        coeffs_imag, C);
}
```

### 7.3 Power Spectrum Kernel

```cpp
// power[n][l] = sqrt( sum_{m=-l}^{l} (real[idx]² + imag[idx]²) )
// 每线程处理一个 (point, degree l) 对

// 预计算: 每个degree l 对应的系数索引范围
// l=0: 1个系数, l=1: 3个, ..., l=9: 19个, 共 100 个
__constant__ int c_degree_offset[11];  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
__constant__ int c_coeff_indices[100]; // S2kit 索引映射

__global__ void powerSpectrum(
    int N, int bw,
    const float* __restrict__ coeffs_real,  // (N, bw*bw)
    const float* __restrict__ coeffs_imag,
    float*       __restrict__ descriptors)  // (N, bw)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int l   = blockIdx.y;  // degree

    if (pid >= N || l >= bw) return;

    int start = c_degree_offset[l];
    int end   = c_degree_offset[l + 1];

    float power = 0.0f;
    for (int i = start; i < end; i++) {
        int idx = c_coeff_indices[i];
        float r = coeffs_real[pid * bw * bw + idx];
        float im = coeffs_imag[pid * bw * bw + idx];
        power += r * r + im * im;
    }

    descriptors[pid * bw + l] = sqrtf(power);
}
```

### 7.4 线程配置

```
Power Spectrum:
  Block: 256
  Grid:  (ceil(N/256), bw) = (59, 10)
```

### 7.5 性能预估

| 操作 | 耗时 (N=15000) |
|------|----------------|
| cuBLAS GEMM (×2) | ~1 ms (小矩阵, cuBLAS 有启动开销) |
| Power spectrum | ~0.5 ms |
| **总计** | **~1.5 ms** |

---

## 8. 主入口函数

### 8.1 star_edge.cu

```cpp
#include <torch/extension.h>
#include <cublas_v2.h>
#include "knn_morton.h"
#include "direction_sampling.h"
#include "sphere_kde.h"
#include "sht_power.h"

torch::Tensor compute_localsh_descriptors(
    torch::Tensor points,       // (N, 3) float32, CUDA
    torch::Tensor sh_basis_real,// (C, G) float32, CUDA (预计算)
    torch::Tensor sh_basis_imag,// (C, G) float32, CUDA
    torch::Tensor grid_xyz,     // (G, 3) float32, CUDA (球面网格)
    int bw,
    int kk,
    int num_samples)
{
    const int N = points.size(0);
    const int K = kk;
    const int S = num_samples;
    const int G = 4 * bw * bw;  // 400
    const int C = bw * bw;       // 100

    auto opts_f = points.options();
    auto opts_i = points.options().dtype(torch::kInt32);

    // ---- Kernel 1: KNN ----
    auto knn_idx = torch::empty({N, K}, opts_i);
    auto knn_dist = torch::empty({N, K}, opts_f);
    knn_morton_search(
        N, K,
        (float3*)points.contiguous().data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        knn_dist.data_ptr<float>());

    // ---- Kernel 2: Direction Vectors ----
    auto directions = torch::empty({N, K, 3}, opts_f);
    compute_direction_vectors(
        N, K,
        (float3*)points.data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        (float3*)directions.data_ptr<float>());

    // ---- Kernel 3: PCA + Angular Sampling ----
    auto samples = torch::empty({N, S, 3}, opts_f);
    pca_angular_sampling(
        N, K, S,
        (float3*)directions.data_ptr<float>(),
        (float3*)samples.data_ptr<float>());

    // ---- Kernel 4: Sphere KDE ----
    // 将 grid_xyz 拷贝到常量内存 (仅首次)
    upload_grid_to_constant(grid_xyz.data_ptr<float>(), G);

    auto sphere_func = torch::empty({N, G}, opts_f);
    float h = M_PI / (2.0f * bw);
    sphere_kde_fused(
        N, S, G, h,
        (float3*)samples.data_ptr<float>(),
        sphere_func.data_ptr<float>());

    // ---- Kernel 5: SHT (cuBLAS) ----
    auto coeffs_real = torch::empty({N, C}, opts_f);
    auto coeffs_imag = torch::empty({N, C}, opts_f);

    cublasHandle_t handle;
    cublasCreate(&handle);
    compute_sht(
        handle, N, G, C,
        sphere_func.data_ptr<float>(),
        sh_basis_real.data_ptr<float>(),
        sh_basis_imag.data_ptr<float>(),
        coeffs_real.data_ptr<float>(),
        coeffs_imag.data_ptr<float>());
    cublasDestroy(handle);

    // ---- Kernel 6: Power Spectrum ----
    auto descriptors = torch::empty({N, bw}, opts_f);
    power_spectrum(
        N, bw,
        coeffs_real.data_ptr<float>(),
        coeffs_imag.data_ptr<float>(),
        descriptors.data_ptr<float>());

    return descriptors;
}

torch::Tensor knn_search(torch::Tensor points, int k) {
    const int N = points.size(0);
    auto knn_idx = torch::empty({N, k}, points.options().dtype(torch::kInt32));
    auto knn_dist = torch::empty({N, k}, points.options());
    knn_morton_search(
        N, k,
        (float3*)points.contiguous().data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        knn_dist.data_ptr<float>());
    return knn_idx;
}
```

### 8.2 显存使用量

| Tensor | 大小 (N=15000) | 显存 |
|--------|----------------|------|
| points (N,3) | 15000×3×4 | 0.18 MB |
| knn_idx (N,K) | 15000×26×4 | 1.5 MB |
| knn_dist (N,K) | 15000×26×4 | 1.5 MB |
| directions (N,K,3) | 15000×26×3×4 | 4.5 MB |
| samples (N,S,3) | 15000×30×3×4 | 5.1 MB |
| sphere_func (N,G) | 15000×400×4 | 23 MB |
| coeffs_real (N,C) | 15000×100×4 | 5.7 MB |
| coeffs_imag (N,C) | 15000×100×4 | 5.7 MB |
| descriptors (N,bw) | 15000×10×4 | 0.6 MB |
| sh_basis (2×C×G) | 2×100×400×4 | 0.3 MB |
| Morton sort temp | ~N×4×3 | 0.7 MB |
| **总计** | | **~49 MB** |

对比 PyTorch 方案的 ~700 MB 峰值，CUDA 方案显存占用减少 **~14x**。

---

## 9. 与现有代码集成

### 9.1 Python 包装层

在 `star_edge_cuda/__init__.py` 中提供高级接口：

```python
import torch
import numpy as np

# 延迟导入, 仅在首次调用时import
_C = None
_sh_basis_cache = {}

def _ensure_loaded():
    global _C
    if _C is None:
        from star_edge_cuda import _C as mod
        _C = mod

def compute_descriptors(
    points: np.ndarray,
    bw: int = 10,
    kk: int = 26,
    num_samples: int = 30,
    device: str = "cuda:0"
) -> np.ndarray:
    """
    GPU 版 LocalSH 描述子计算。

    Args:
        points: (N, 3) float64/float32 numpy array
        bw, kk, num_samples: 与原始 LocalSH 一致

    Returns:
        descriptors: (N, bw) float32 numpy array
    """
    _ensure_loaded()

    # 准备输入
    pts = torch.from_numpy(np.asarray(points[:, :3], dtype=np.float32))
    pts = pts.to(device).contiguous()

    # 加载/缓存 SH 基矩阵
    cache_key = (bw, device)
    if cache_key not in _sh_basis_cache:
        basis_r, basis_i, grid = _load_sh_basis(bw, device)
        _sh_basis_cache[cache_key] = (basis_r, basis_i, grid)
    basis_r, basis_i, grid = _sh_basis_cache[cache_key]

    # 调用 CUDA kernel
    desc = _C.compute_localsh_descriptors(
        pts, basis_r, basis_i, grid, bw, kk, num_samples)

    return desc.cpu().numpy()

def _load_sh_basis(bw, device):
    """加载预计算的 SH 基矩阵"""
    import os
    basis_dir = os.path.join(os.path.dirname(__file__), "data")
    basis_r = torch.from_numpy(
        np.load(os.path.join(basis_dir, f"sh_basis_real_bw{bw}.npy"))
    ).to(device)
    basis_i = torch.from_numpy(
        np.load(os.path.join(basis_dir, f"sh_basis_imag_bw{bw}.npy"))
    ).to(device)
    grid = torch.from_numpy(
        np.load(os.path.join(basis_dir, f"grid_xyz_bw{bw}.npy"))
    ).to(device)
    return basis_r, basis_i, grid
```

### 9.2 修改 edge_extraction.py

在 `extract_edges_star_edge()` 方法开头添加 GPU 分支：

```python
def extract_edges_star_edge(self, points):
    # ... existing null/empty checks (lines 657-661) ...

    # GPU CUDA kernel 路径
    use_cuda_kernel = bool(getattr(self.config, "star_edge_cuda", False))
    if use_cuda_kernel and torch.cuda.is_available():
        try:
            return self._extract_edges_star_edge_cuda(points)
        except ImportError:
            pass  # star_edge_cuda 未编译, 回退

    # ... existing CPU path ...

def _extract_edges_star_edge_cuda(self, points):
    """CUDA kernel 版 STAR-Edge"""
    import star_edge_cuda

    pts = np.asarray(points[:, :3], dtype=np.float32)
    N = pts.shape[0]

    bw = int(getattr(self.config, "star_edge_bw", 10))
    kk = int(getattr(self.config, "star_edge_kk", 26))
    sample_num = int(getattr(self.config, "star_edge_sample_num", bw * 4))

    # 大点云子采样 (与 CPU 版逻辑一致)
    maxN = int(getattr(self.config, "star_edge_max_points", 50000))
    if N > maxN:
        idx = np.random.choice(N, size=maxN, replace=False)
        sub_desc = star_edge_cuda.compute_descriptors(
            pts[idx], bw=bw, kk=kk, num_samples=sample_num)
        # KDTree 传播
        tree = cKDTree(pts[idx])
        _, nn = tree.query(pts, k=1, workers=-1)
        desc = sub_desc[nn.astype(np.int64)]
    else:
        desc = star_edge_cuda.compute_descriptors(
            pts, bw=bw, kk=kk, num_samples=sample_num)

    # MLP 推理 (沿用现有的 _load_star_edge_classifier)
    device = torch.device("cuda:0")
    net, device = _load_star_edge_classifier(
        getattr(self.config, "star_edge_model_path", None) or
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "STAR-Edge", "model", "best.ckpt"),
        use_cuda=True)

    x = torch.from_numpy(desc).to(device)
    with torch.no_grad():
        prob = net(x).detach().cpu().numpy().astype(np.float32)

    thr = float(getattr(self.config, "star_edge_threshold", 0.5))
    edge_scores = np.clip(prob, 0.0, 1.0)
    edge_mask = (edge_scores >= thr).astype(bool)

    return edge_mask, edge_scores
```

### 9.3 配置新增项

```yaml
edge_extraction:
  method: 'star_edge'
  star_edge_cuda: true          # 启用 CUDA kernel 版
  # 以下参数不变
  star_edge_bw: 10
  star_edge_kk: 26
  star_edge_sample_num: 40
  star_edge_threshold: 0.5
  star_edge_max_points: 50000
```

---

## 10. 测试与验证

### 10.1 数值一致性验证

```python
def validate_cuda_vs_cpu():
    """对比 CUDA kernel 与 C++ CPU 的描述子输出"""
    import sys
    sys.path.insert(0, '/workspace/STAR-Edge/LocalSH')
    import LocalSH
    import star_edge_cuda

    np.random.seed(42)
    pts = np.random.randn(5000, 3).astype(np.float64)

    # CPU 基准
    result_cpu = LocalSH.LocalSHFeature.ComLSHF_knn_upsample(pts, 10, 26, 40)
    desc_cpu = np.array(result_cpu['Descs'])

    # CUDA
    desc_gpu = star_edge_cuda.compute_descriptors(pts, bw=10, kk=26, num_samples=30)

    # 对比
    rel_error = np.linalg.norm(desc_cpu - desc_gpu) / (np.linalg.norm(desc_cpu) + 1e-10)
    print(f"Relative L2 error: {rel_error:.6f}")
    # 目标: < 5% (允许凸包近似+SHT实现差异)

    # 边缘检测一致性 (更重要)
    # 用同一个MLP, 对比分类结果
    net = load_mlp()
    pred_cpu = net(torch.from_numpy(desc_cpu.astype(np.float32))).numpy() > 0.5
    pred_gpu = net(torch.from_numpy(desc_gpu.astype(np.float32))).numpy() > 0.5
    agreement = (pred_cpu == pred_gpu).mean()
    print(f"Edge classification agreement: {agreement:.4f}")
    # 目标: > 95%
```

### 10.2 性能基准测试

```python
def benchmark():
    """端到端性能对比"""
    for N in [1000, 5000, 15000, 30000, 50000]:
        pts = np.random.randn(N, 3)

        # CPU
        t0 = time.perf_counter()
        LocalSH.LocalSHFeature.ComLSHF_knn_upsample(pts.astype(np.float64), 10, 26, 40)
        t_cpu = time.perf_counter() - t0

        # CUDA (warmup后)
        star_edge_cuda.compute_descriptors(pts, bw=10)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        star_edge_cuda.compute_descriptors(pts, bw=10)
        torch.cuda.synchronize()
        t_cuda = time.perf_counter() - t0

        print(f"N={N:6d}  CPU={t_cpu*1000:8.1f}ms  CUDA={t_cuda*1000:8.1f}ms  speedup={t_cpu/t_cuda:.1f}x")
```

### 10.3 SLAM集成测试

验证完整 SLAM 流水线在使用 CUDA 版 STAR-Edge 后：
1. 轨迹精度 (ATE/RPE) 与 CPU 版无显著差异
2. tracking 阶段延迟明显下降
3. 无 CUDA 内存泄漏 (多帧连续运行)

---

## 11. 性能目标汇总

| | CPU LocalSH | PyTorch GPU | CUDA Kernel (目标) |
|--|-------------|-------------|-------------------|
| N=1,000 | 340 ms | ~13 ms | **~6 ms** |
| N=5,000 | 1,660 ms | ~60 ms | **~25 ms** |
| N=15,000 | 5,110 ms | ~188 ms | **~80 ms** |
| N=50,000 | ~17,000 ms | ~600 ms | **~250 ms** |
| 显存峰值 (N=15K) | 0 (CPU) | ~700 MB | **~49 MB** |
| 额外依赖 | fftw, pybind11 | 无 | 无 (PyTorch自带nvcc) |
| SHT 方式 | S2kit (fftw) | 预计算基矩阵+matmul | cuBLAS GEMM |
| KNN 方式 | nanoflann (CPU) | torch.cdist+topk | Morton code+空间分区 |

---

## 12. 实施路线

### Phase 1: 基础设施 + KNN Kernel
1. 搭建 `submodules/star_edge_cuda/` 目录和编译系统
2. 实现 Kernel 1 (KNN Morton)，独立测试
3. 预计算 SH 基矩阵，独立验证

### Phase 2: 特征计算 Kernel
4. 实现 Kernel 2 (方向向量)
5. 实现 Kernel 3 (PCA + 采样)
6. 实现 Kernel 4 (球面 KDE)
7. 实现 Kernel 5 (SHT + 功率谱)

### Phase 3: 集成与优化
8. 组合所有 kernel 为 `compute_localsh_descriptors`
9. 数值一致性验证
10. 集成到 `edge_extraction.py`
11. SLAM 端到端测试
12. 性能profiling + 热点优化

### Phase 4: 可选进阶优化
13. 多 kernel 融合 (Kernel 2+3 融合, Kernel 4+5 融合)
14. CUDA Stream 流水线化 (KNN 与后续计算重叠)
15. 多 GPU 支持 (大点云切分到多卡)
