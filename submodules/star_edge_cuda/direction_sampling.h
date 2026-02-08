#ifndef DIRECTION_SAMPLING_H
#define DIRECTION_SAMPLING_H

void compute_direction_vectors(
    int N, int K,
    const float* points,      // (N, 3)
    const int* knn_idx,       // (N, K)
    float* directions);       // (N, K, 3)

void pca_angular_sampling(
    int N, int K, int S,
    const float* directions,  // (N, K, 3)
    float* samples);          // (N, S, 3)

#endif
