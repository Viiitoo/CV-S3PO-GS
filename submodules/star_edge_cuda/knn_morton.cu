/*
 * KNN search using Morton code spatial partitioning.
 * Extended from simple-knn (3DGS) to support arbitrary K (default K=26).
 *
 * Algorithm:
 * 1. Compute AABB of point cloud (CUB reduction)
 * 2. Compute Morton codes for spatial sorting
 * 3. Radix sort points by Morton code (CUB)
 * 4. Partition sorted points into boxes of BOX_SIZE
 * 5. Compute AABB per box (shared memory reduction)
 * 6. For each point, scan boxes with AABB pruning, maintain K-best list
 */

#define BOX_SIZE 1024
#define MAX_KNN 64  // maximum supported K

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "knn_morton.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cfloat>

namespace cg = cooperative_groups;

// ===================== Helper structs =====================

struct Float3Min {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
    }
};

struct Float3Max {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
    }
};

struct BoxMinMax {
    float3 minn;
    float3 maxx;
};

// ===================== Morton code =====================

__host__ __device__ uint32_t prepMorton(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx) {
    float3 range = {maxx.x - minn.x, maxx.y - minn.y, maxx.z - minn.z};
    // Avoid division by zero
    if (range.x < 1e-10f) range.x = 1.0f;
    if (range.y < 1e-10f) range.y = 1.0f;
    if (range.z < 1e-10f) range.z = 1.0f;

    uint32_t x = prepMorton((uint32_t)(((coord.x - minn.x) / range.x) * ((1 << 10) - 1)));
    uint32_t y = prepMorton((uint32_t)(((coord.y - minn.y) / range.y) * ((1 << 10) - 1)));
    uint32_t z = prepMorton((uint32_t)(((coord.z - minn.z) / range.z) * ((1 << 10) - 1)));
    return x | (y << 1) | (z << 2);
}

__global__ void computeMortonCodes(int P, const float3* points,
                                    float3 minn, float3 maxx,
                                    uint32_t* codes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    codes[idx] = coord2Morton(points[idx], minn, maxx);
}

// ===================== Box AABB computation =====================

__global__ void computeBoxAABB(uint32_t P, const float3* points,
                                const uint32_t* sorted_indices,
                                BoxMinMax* boxes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    BoxMinMax me;
    if (idx < (int)P) {
        me.minn = points[sorted_indices[idx]];
        me.maxx = points[sorted_indices[idx]];
    } else {
        me.minn = {FLT_MAX, FLT_MAX, FLT_MAX};
        me.maxx = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    }

    __shared__ BoxMinMax shared_data[BOX_SIZE];

    for (int off = BOX_SIZE / 2; off >= 1; off /= 2) {
        if (threadIdx.x < 2 * off)
            shared_data[threadIdx.x] = me;
        __syncthreads();
        if (threadIdx.x < off) {
            BoxMinMax other = shared_data[threadIdx.x + off];
            me.minn.x = fminf(me.minn.x, other.minn.x);
            me.minn.y = fminf(me.minn.y, other.minn.y);
            me.minn.z = fminf(me.minn.z, other.minn.z);
            me.maxx.x = fmaxf(me.maxx.x, other.maxx.x);
            me.maxx.y = fmaxf(me.maxx.y, other.maxx.y);
            me.maxx.z = fmaxf(me.maxx.z, other.maxx.z);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        boxes[blockIdx.x] = me;
}

// ===================== Distance helpers =====================

__device__ __host__ float distBoxPoint(const BoxMinMax& box, const float3& p) {
    float dx = 0, dy = 0, dz = 0;
    if (p.x < box.minn.x || p.x > box.maxx.x)
        dx = fminf(fabsf(p.x - box.minn.x), fabsf(p.x - box.maxx.x));
    if (p.y < box.minn.y || p.y > box.maxx.y)
        dy = fminf(fabsf(p.y - box.minn.y), fabsf(p.y - box.maxx.y));
    if (p.z < box.minn.z || p.z > box.maxx.z)
        dz = fminf(fabsf(p.z - box.minn.z), fabsf(p.z - box.maxx.z));
    return dx * dx + dy * dy + dz * dz;
}

// ===================== K-best maintenance =====================

__device__ void knn_insert(const float3& ref, const float3& cand, int cand_idx,
                            float* best_dist, int* best_idx, int K) {
    float3 d = {cand.x - ref.x, cand.y - ref.y, cand.z - ref.z};
    float dist = d.x * d.x + d.y * d.y + d.z * d.z;

    if (dist >= best_dist[K - 1] || dist < 1e-12f)
        return;

    // Insert into sorted list (insertion sort from back)
    best_dist[K - 1] = dist;
    best_idx[K - 1] = cand_idx;
    for (int j = K - 2; j >= 0; --j) {
        if (best_dist[j] > best_dist[j + 1]) {
            float td = best_dist[j];
            best_dist[j] = best_dist[j + 1];
            best_dist[j + 1] = td;
            int ti = best_idx[j];
            best_idx[j] = best_idx[j + 1];
            best_idx[j + 1] = ti;
        } else {
            break;
        }
    }
}

// ===================== Main KNN kernel =====================

__global__ void knnSearchKernel(
    uint32_t P, int K,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ sorted_indices,
    const BoxMinMax* __restrict__ boxes,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)P) return;

    uint32_t orig_idx = sorted_indices[idx];
    float3 point = points[orig_idx];

    // Use local memory for K-best (K <= MAX_KNN)
    float best_dist[MAX_KNN];
    int best_idx[MAX_KNN];
    for (int j = 0; j < K; j++) {
        best_dist[j] = FLT_MAX;
        best_idx[j] = -1;
    }

    // Phase 1: Local neighborhood scan (Morton-order neighbors are spatially close)
    int local_range = K * 2 + 16;  // heuristic window
    int lo = max(0, idx - local_range);
    int hi = min((int)P - 1, idx + local_range);
    for (int i = lo; i <= hi; i++) {
        if (i == idx) continue;
        uint32_t ni = sorted_indices[i];
        knn_insert(point, points[ni], (int)ni, best_dist, best_idx, K);
    }

    // Phase 2: Box-level pruned scan
    float reject = best_dist[K - 1];
    uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;

    for (uint32_t b = 0; b < num_boxes; b++) {
        float bd = distBoxPoint(boxes[b], point);
        if (bd > reject) continue;

        uint32_t start = b * BOX_SIZE;
        uint32_t end = min(P, start + BOX_SIZE);

        // Skip local range we already scanned
        for (uint32_t i = start; i < end; i++) {
            if ((int)i >= lo && (int)i <= hi) continue;  // already scanned
            if (i == (uint32_t)idx) continue;
            uint32_t ni = sorted_indices[i];
            knn_insert(point, points[ni], (int)ni, best_dist, best_idx, K);
        }
        reject = best_dist[K - 1];
    }

    // Write results (in original point order)
    for (int j = 0; j < K; j++) {
        out_idx[orig_idx * K + j] = best_idx[j];
        out_dist[orig_idx * K + j] = best_dist[j];
    }
}

// ===================== Host entry point =====================

void knn_morton_search(int N, int K,
                       const float* points_raw,
                       int* out_idx, float* out_dist) {
    const float3* points = (const float3*)points_raw;

    // --- Step 1: Compute AABB ---
    float3* d_result;
    cudaMalloc(&d_result, sizeof(float3));
    size_t temp_bytes;

    float3 init = {0, 0, 0}, minn, maxx;

    cub::DeviceReduce::Reduce(nullptr, temp_bytes, points, d_result, N, Float3Min(), init);
    thrust::device_vector<char> temp_storage(temp_bytes);

    // Min
    float3 init_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_bytes,
                               points, d_result, N, Float3Min(), init_min);
    cudaMemcpy(&minn, d_result, sizeof(float3), cudaMemcpyDeviceToHost);

    // Max
    float3 init_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_bytes,
                               points, d_result, N, Float3Max(), init_max);
    cudaMemcpy(&maxx, d_result, sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    // --- Step 2: Morton codes ---
    thrust::device_vector<uint32_t> morton(N);
    thrust::device_vector<uint32_t> morton_sorted(N);
    computeMortonCodes<<<(N + 255) / 256, 256>>>(N, points, minn, maxx,
                                                   morton.data().get());

    // --- Step 3: Sort by Morton code ---
    thrust::device_vector<uint32_t> indices(N);
    thrust::sequence(indices.begin(), indices.end());
    thrust::device_vector<uint32_t> indices_sorted(N);

    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        morton.data().get(), morton_sorted.data().get(),
        indices.data().get(), indices_sorted.data().get(), N);
    temp_storage.resize(temp_bytes);
    cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_bytes,
        morton.data().get(), morton_sorted.data().get(),
        indices.data().get(), indices_sorted.data().get(), N);

    // --- Step 4: Box AABBs ---
    uint32_t num_boxes = (N + BOX_SIZE - 1) / BOX_SIZE;
    thrust::device_vector<BoxMinMax> boxes(num_boxes);
    computeBoxAABB<<<num_boxes, BOX_SIZE>>>(N, points, indices_sorted.data().get(),
                                             boxes.data().get());

    // --- Step 5: KNN search ---
    knnSearchKernel<<<(N + 255) / 256, 256>>>(
        N, K, points, indices_sorted.data().get(),
        boxes.data().get(), out_idx, out_dist);

    cudaDeviceSynchronize();
}
