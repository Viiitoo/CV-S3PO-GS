#ifndef KNN_MORTON_H
#define KNN_MORTON_H

#include <cstdint>

void knn_morton_search(
    int N, int K,
    const float* points,   // (N, 3)
    int* out_idx,          // (N, K)
    float* out_dist);      // (N, K)

#endif
