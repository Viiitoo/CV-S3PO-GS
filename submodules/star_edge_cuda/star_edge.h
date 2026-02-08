#ifndef STAR_EDGE_H
#define STAR_EDGE_H

#include <torch/extension.h>

torch::Tensor compute_localsh_descriptors(
    torch::Tensor points,         // (N, 3) float32, CUDA
    torch::Tensor sh_basis_real,  // (C, G) float32, CUDA
    torch::Tensor sh_basis_imag,  // (C, G) float32, CUDA
    torch::Tensor grid_xyz,       // (G, 3) float32, CUDA
    int bw,
    int kk,
    int num_samples);

torch::Tensor knn_search(
    torch::Tensor points,         // (N, 3) float32, CUDA
    int k);

#endif
