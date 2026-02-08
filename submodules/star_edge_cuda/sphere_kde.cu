/*
 * Kernel 4: Sphere KDE (fused kernel density estimation + max reduction)
 *
 * For each (point, grid_cell) pair, loops over S curve samples,
 * computes geodesic distance → Gaussian kernel, takes max.
 * Zero intermediate memory — no (N, S, G) tensor materialization.
 *
 * Grid layout stored in constant memory (400 * 3 = 4.8 KB).
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sphere_kde.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constant memory for sphere grid coordinates (max 1200 floats = 512 grid cells * 3)
#define MAX_GRID_SIZE 512
__constant__ float c_grid_xyz[MAX_GRID_SIZE * 3];

static bool grid_uploaded = false;
static int  grid_size_cached = 0;

void upload_grid_to_constant(const float* grid_xyz, int G) {
    if (grid_uploaded && grid_size_cached == G) return;
    if (G > MAX_GRID_SIZE) {
        // Fallback: should not happen for bw <= 16
        G = MAX_GRID_SIZE;
    }
    cudaMemcpyToSymbol(c_grid_xyz, grid_xyz, G * 3 * sizeof(float));
    grid_uploaded = true;
    grid_size_cached = G;
}

// ===================== Sphere KDE kernel (per-point version) =====================
// Each thread handles one point, loops over all G grid cells and S samples.
// Avoids inter-thread communication, maximizes instruction-level parallelism.

__global__ void sphereKDEKernel(
    int N, int S, int G,
    float inv_h,           // 1/h where h = pi/(2*bw)
    float inv_hS,          // 1/(h * S)
    const float* __restrict__ samples,       // (N, S, 3) flat
    float*       __restrict__ sphere_func)   // (N, G) flat
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    const float* my_samples = samples + pid * S * 3;
    float* my_output = sphere_func + pid * G;

    // Preload sample points to registers/local memory
    // S is typically 30, so 30*3 = 90 floats in local memory
    float sx[64], sy[64], sz[64];  // max S=64
    for (int s = 0; s < S; s++) {
        sx[s] = my_samples[s * 3 + 0];
        sy[s] = my_samples[s * 3 + 1];
        sz[s] = my_samples[s * 3 + 2];
    }

    float inv_sqrt2pi = 0.3989422804014327f;  // 1/sqrt(2*pi)

    for (int g = 0; g < G; g++) {
        float gx = c_grid_xyz[g * 3 + 0];
        float gy = c_grid_xyz[g * 3 + 1];
        float gz = c_grid_xyz[g * 3 + 2];

        float kde_max = 0.0f;

        for (int s = 0; s < S; s++) {
            // Dot product = cos(geodesic distance) for unit vectors
            float cos_dist = sx[s] * gx + sy[s] * gy + sz[s] * gz;
            cos_dist = fminf(fmaxf(cos_dist, -1.0f), 1.0f);

            float geodesic = acosf(cos_dist);
            float x = geodesic * inv_h;

            // Gaussian kernel: K(x) = exp(-0.5 * x²) / sqrt(2*pi)
            float K_val = inv_sqrt2pi * expf(-0.5f * x * x);
            float D = K_val * inv_hS;

            kde_max = fmaxf(kde_max, D);
        }

        my_output[g] = kde_max;
    }
}

void sphere_kde_fused(int N, int S, int G, float h,
                       const float* samples, float* sphere_func) {
    float inv_h = 1.0f / h;
    float inv_hS = 1.0f / (h * (float)S);

    // Each thread processes one point (400 grid cells × 30 samples = 12000 iterations)
    // Use 128 threads/block to balance occupancy vs register pressure
    sphereKDEKernel<<<(N + 127) / 128, 128>>>(N, S, G, inv_h, inv_hS, samples, sphere_func);
}
