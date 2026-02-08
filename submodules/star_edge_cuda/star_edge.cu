/*
 * Main entry point: orchestrates all CUDA kernels for LocalSH descriptor computation.
 *
 * Pipeline:
 *   1. KNN search (Morton code)
 *   2. Direction vectors
 *   3. PCA + angular sampling (curve extraction)
 *   4. Sphere KDE (fused)
 *   5. SHT via torch::mm (cuBLAS)
 *   6. Power spectrum
 */

#include <torch/extension.h>
#include "star_edge.h"
#include "knn_morton.h"
#include "direction_sampling.h"
#include "sphere_kde.h"
#include "sht_power.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

torch::Tensor compute_localsh_descriptors(
    torch::Tensor points,         // (N, 3) float32, CUDA
    torch::Tensor sh_basis_real,  // (C, G) float32, CUDA
    torch::Tensor sh_basis_imag,  // (C, G) float32, CUDA
    torch::Tensor grid_xyz,       // (G, 3) float32, CUDA
    int bw,
    int kk,
    int num_samples)
{
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be (N, 3)");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");

    const int N = points.size(0);
    const int K = kk;
    const int S = num_samples;
    const int G = 4 * bw * bw;   // grid size: (2*bw)*(2*bw)
    const int C = bw * bw;        // number of SH coefficients

    TORCH_CHECK(sh_basis_real.size(0) == C && sh_basis_real.size(1) == G,
                "sh_basis_real must be (C, G)");

    auto opts_f = points.options();                          // float32, same device
    auto opts_i = points.options().dtype(torch::kInt32);

    // Ensure contiguous
    points = points.contiguous();
    sh_basis_real = sh_basis_real.contiguous();
    sh_basis_imag = sh_basis_imag.contiguous();
    grid_xyz = grid_xyz.contiguous();

    // ---- Kernel 1: KNN search ----
    auto knn_idx = torch::empty({N, K}, opts_i);
    auto knn_dist = torch::empty({N, K}, opts_f);
    knn_morton_search(
        N, K,
        points.data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        knn_dist.data_ptr<float>());

    // ---- Kernel 2: Direction vectors ----
    auto directions = torch::empty({N, K, 3}, opts_f);
    compute_direction_vectors(
        N, K,
        points.data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        directions.data_ptr<float>());

    // ---- Kernel 3: PCA + angular sampling ----
    auto samples = torch::empty({N, S, 3}, opts_f);
    pca_angular_sampling(
        N, K, S,
        directions.data_ptr<float>(),
        samples.data_ptr<float>());

    // ---- Upload grid to constant memory (once) ----
    upload_grid_to_constant(grid_xyz.data_ptr<float>(), G);

    // ---- Kernel 4: Sphere KDE ----
    auto sphere_func = torch::empty({N, G}, opts_f);
    float h = (float)M_PI / (float)(2 * bw);
    sphere_kde_fused(
        N, S, G, h,
        samples.data_ptr<float>(),
        sphere_func.data_ptr<float>());

    // ---- Kernel 5: SHT via cuBLAS (torch::mm) ----
    // coeffs_real = sphere_func (N, G) × sh_basis_real^T (G, C) → (N, C)
    // coeffs_imag = sphere_func (N, G) × sh_basis_imag^T (G, C) → (N, C)
    auto coeffs_real = torch::mm(sphere_func, sh_basis_real.t());  // (N, C)
    auto coeffs_imag = torch::mm(sphere_func, sh_basis_imag.t());  // (N, C)

    // ---- Kernel 6: Power spectrum ----
    auto descriptors = torch::empty({N, bw}, opts_f);
    power_spectrum_kernel(
        N, bw,
        coeffs_real.data_ptr<float>(),
        coeffs_imag.data_ptr<float>(),
        descriptors.data_ptr<float>());

    cudaDeviceSynchronize();

    return descriptors;
}

torch::Tensor knn_search(torch::Tensor points, int k) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be (N, 3)");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");

    const int N = points.size(0);
    points = points.contiguous();

    auto opts_i = points.options().dtype(torch::kInt32);
    auto knn_idx = torch::empty({N, k}, opts_i);
    auto knn_dist = torch::empty({N, k}, points.options());

    knn_morton_search(
        N, k,
        points.data_ptr<float>(),
        knn_idx.data_ptr<int>(),
        knn_dist.data_ptr<float>());

    cudaDeviceSynchronize();
    return knn_idx;
}
