/*
 * Kernel 5: Power spectrum computation
 *
 * SHT is done via torch::mm() (cuBLAS) on the host side.
 * This file provides the power spectrum kernel:
 *   descriptor[l] = sqrt( sum_{m=-l}^{l} |c_{lm}|² )
 *
 * Coefficient indexing (sequential by degree):
 *   (0,0) → 0
 *   (1,-1),(1,0),(1,1) → 1,2,3
 *   (2,-2),...,(2,2) → 4,...,8
 *   ...
 *   Degree l starts at index l², has 2l+1 coefficients.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sht_power.h"
#include <cmath>

__global__ void powerSpectrumKernel(
    int N, int bw,
    const float* __restrict__ coeffs_real,   // (N, bw*bw)
    const float* __restrict__ coeffs_imag,   // (N, bw*bw)
    float*       __restrict__ descriptors)   // (N, bw)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int l   = blockIdx.y;

    if (pid >= N || l >= bw) return;

    int C = bw * bw;
    int start = l * l;          // first coeff index for degree l
    int count = 2 * l + 1;      // number of coeffs for degree l

    float power = 0.0f;
    for (int i = 0; i < count; i++) {
        int idx = start + i;
        float r = coeffs_real[pid * C + idx];
        float im = coeffs_imag[pid * C + idx];
        power += r * r + im * im;
    }

    descriptors[pid * bw + l] = sqrtf(power);
}

void power_spectrum_kernel(int N, int bw,
                            const float* coeffs_real,
                            const float* coeffs_imag,
                            float* descriptors) {
    dim3 block(256);
    dim3 grid((N + 255) / 256, bw);
    powerSpectrumKernel<<<grid, block>>>(N, bw, coeffs_real, coeffs_imag, descriptors);
}
