/*
 * Kernel 2: Direction vector computation (fused gather + normalize)
 * Kernel 3: PCA projection + angular sorting + curve sampling
 *
 * Replaces the C++ pipeline:
 *   direction vectors → FittingLineSample::projectPointCloud
 *   → convexHull → fitCurve → normalize
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "direction_sampling.h"
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_KNN 64

// ===================== Kernel 2: Direction vectors =====================

__global__ void directionVectorKernel(
    int N, int K,
    const float* __restrict__ points,      // (N, 3) flat
    const int*   __restrict__ knn_idx,     // (N, K) flat
    float*       __restrict__ directions)  // (N, K, 3) flat
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    float px = points[pid * 3 + 0];
    float py = points[pid * 3 + 1];
    float pz = points[pid * 3 + 2];

    for (int k = 0; k < K; k++) {
        int nid = knn_idx[pid * K + k];
        float dx, dy, dz;
        if (nid >= 0 && nid < N) {
            dx = points[nid * 3 + 0] - px;
            dy = points[nid * 3 + 1] - py;
            dz = points[nid * 3 + 2] - pz;
        } else {
            dx = dy = dz = 0.0f;
        }
        float norm = sqrtf(dx * dx + dy * dy + dz * dz);
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            dx *= inv; dy *= inv; dz *= inv;
        } else {
            dx = dy = dz = 0.0f;
        }
        int out_base = (pid * K + k) * 3;
        directions[out_base + 0] = dx;
        directions[out_base + 1] = dy;
        directions[out_base + 2] = dz;
    }
}

void compute_direction_vectors(int N, int K,
                                const float* points, const int* knn_idx,
                                float* directions) {
    directionVectorKernel<<<(N + 255) / 256, 256>>>(N, K, points, knn_idx, directions);
}

// ===================== 3x3 Symmetric Eigendecomposition =====================
// Analytic solution using Cardano's formula for 3x3 symmetric matrices
// Reference: Kopp, "Efficient numerical diagonalization of hermitian 3x3 matrices"

__device__ void eigen3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float* eigval,    // [3] descending order
    float eigvec[9])  // [9] column-major: eigvec[0..2] = first eigenvector
{
    // Shift to improve numerical stability
    float shift = (a00 + a11 + a22) / 3.0f;
    a00 -= shift; a11 -= shift; a22 -= shift;

    float p1 = a01 * a01 + a02 * a02 + a12 * a12;

    if (p1 < 1e-12f) {
        // Already diagonal
        eigval[0] = a00 + shift;
        eigval[1] = a11 + shift;
        eigval[2] = a22 + shift;
        // Sort descending
        if (eigval[0] < eigval[1]) { float t = eigval[0]; eigval[0] = eigval[1]; eigval[1] = t; }
        if (eigval[0] < eigval[2]) { float t = eigval[0]; eigval[0] = eigval[2]; eigval[2] = t; }
        if (eigval[1] < eigval[2]) { float t = eigval[1]; eigval[1] = eigval[2]; eigval[2] = t; }
        // Identity eigenvectors (reordered)
        for (int i = 0; i < 9; i++) eigvec[i] = 0;
        eigvec[0] = eigvec[4] = eigvec[8] = 1.0f;
        return;
    }

    float q = (a00 + a11 + a22) / 3.0f;  // should be ~0 after shift
    float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q)
               + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);
    float inv_p = (p > 1e-12f) ? (1.0f / p) : 0.0f;

    // B = (1/p) * (A - q*I)
    float b00 = (a00 - q) * inv_p;
    float b01 = a01 * inv_p;
    float b02 = a02 * inv_p;
    float b11 = (a11 - q) * inv_p;
    float b12 = a12 * inv_p;
    float b22 = (a22 - q) * inv_p;

    // det(B) / 2
    float detB_half = (b00 * (b11 * b22 - b12 * b12)
                     - b01 * (b01 * b22 - b12 * b02)
                     + b02 * (b01 * b12 - b11 * b02)) * 0.5f;

    detB_half = fmaxf(-1.0f, fminf(1.0f, detB_half));

    float phi = acosf(detB_half) / 3.0f;

    // Eigenvalues (descending)
    eigval[0] = q + 2.0f * p * cosf(phi) + shift;
    eigval[2] = q + 2.0f * p * cosf(phi + (2.0f * M_PI / 3.0f)) + shift;
    eigval[1] = (a00 + a11 + a22 + 3.0f * shift) - eigval[0] - eigval[2];  // trace identity

    // Compute eigenvectors via cross products of rows of (A - lambda*I)
    for (int ev = 0; ev < 3; ev++) {
        float lam = eigval[ev] - shift;  // eigenvalue in shifted system
        // M = A_shifted - lam * I
        float m00 = a00 - lam, m01 = a01,       m02 = a02;
        float m10 = a01,       m11 = a11 - lam,  m12 = a12;
        float m20 = a02,       m21 = a12,        m22 = a22 - lam;

        // Cross product of row0 and row1
        float vx = m01 * m12 - m02 * m11;
        float vy = m02 * m10 - m00 * m12;
        float vz = m00 * m11 - m01 * m10;
        float vnorm = sqrtf(vx * vx + vy * vy + vz * vz);

        if (vnorm < 1e-8f) {
            // Try row0 x row2
            vx = m01 * m22 - m02 * m21;
            vy = m02 * m20 - m00 * m22;
            vz = m00 * m21 - m01 * m20;
            vnorm = sqrtf(vx * vx + vy * vy + vz * vz);
        }
        if (vnorm < 1e-8f) {
            // Try row1 x row2
            vx = m11 * m22 - m12 * m21;
            vy = m12 * m20 - m10 * m22;
            vz = m10 * m21 - m11 * m20;
            vnorm = sqrtf(vx * vx + vy * vy + vz * vz);
        }
        if (vnorm < 1e-8f) {
            // Degenerate: use canonical direction
            vx = (ev == 0) ? 1.0f : 0.0f;
            vy = (ev == 1) ? 1.0f : 0.0f;
            vz = (ev == 2) ? 1.0f : 0.0f;
            vnorm = 1.0f;
        }

        float inv_norm = 1.0f / vnorm;
        eigvec[ev * 3 + 0] = vx * inv_norm;
        eigvec[ev * 3 + 1] = vy * inv_norm;
        eigvec[ev * 3 + 2] = vz * inv_norm;
    }
}

// ===================== Kernel 3: PCA + Angular sort + Curve sampling =====================

__global__ void pcaAngularSamplingKernel(
    int N, int K, int S,
    const float* __restrict__ directions,  // (N, K, 3)
    float*       __restrict__ samples)     // (N, S, 3)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    const float* dirs = directions + pid * K * 3;
    float* out = samples + pid * S * 3;

    // ---- Step 1: Compute mean ----
    float mx = 0, my = 0, mz = 0;
    int valid_count = 0;
    for (int k = 0; k < K; k++) {
        float dx = dirs[k * 3 + 0];
        float dy = dirs[k * 3 + 1];
        float dz = dirs[k * 3 + 2];
        float n2 = dx * dx + dy * dy + dz * dz;
        if (n2 > 1e-10f) {
            mx += dx; my += dy; mz += dz;
            valid_count++;
        }
    }
    if (valid_count < 3) {
        // Not enough valid neighbors, output zeros
        for (int s = 0; s < S; s++) {
            out[s * 3 + 0] = 0; out[s * 3 + 1] = 0; out[s * 3 + 2] = 1.0f;
        }
        return;
    }
    float inv_vc = 1.0f / (float)valid_count;
    mx *= inv_vc; my *= inv_vc; mz *= inv_vc;

    // ---- Step 2: Covariance matrix (3x3 symmetric) ----
    float c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
    for (int k = 0; k < K; k++) {
        float dx = dirs[k * 3 + 0] - mx;
        float dy = dirs[k * 3 + 1] - my;
        float dz = dirs[k * 3 + 2] - mz;
        float n2 = dirs[k * 3 + 0] * dirs[k * 3 + 0] +
                   dirs[k * 3 + 1] * dirs[k * 3 + 1] +
                   dirs[k * 3 + 2] * dirs[k * 3 + 2];
        if (n2 < 1e-10f) continue;
        c00 += dx * dx; c01 += dx * dy; c02 += dx * dz;
        c11 += dy * dy; c12 += dy * dz; c22 += dz * dz;
    }

    // ---- Step 3: Eigendecomposition ----
    float eigval[3];
    float eigvec[9]; // column major
    eigen3x3(c00, c01, c02, c11, c12, c22, eigval, eigvec);

    // First two eigenvectors as projection axes (largest eigenvalues)
    float ax1x = eigvec[0], ax1y = eigvec[1], ax1z = eigvec[2];
    float ax2x = eigvec[3], ax2y = eigvec[4], ax2z = eigvec[5];

    // ---- Step 4: Collect valid direction vectors ----
    float dir3d[MAX_KNN * 3];
    int valid_k = 0;

    for (int k = 0; k < K; k++) {
        float dx = dirs[k * 3 + 0];
        float dy = dirs[k * 3 + 1];
        float dz = dirs[k * 3 + 2];
        float n2 = dx * dx + dy * dy + dz * dz;
        if (n2 < 1e-10f) continue;

        dir3d[valid_k * 3 + 0] = dx;
        dir3d[valid_k * 3 + 1] = dy;
        dir3d[valid_k * 3 + 2] = dz;
        valid_k++;
    }

    if (valid_k < 3) {
        for (int s = 0; s < S; s++) {
            out[s * 3 + 0] = 0; out[s * 3 + 1] = 0; out[s * 3 + 2] = 1.0f;
        }
        return;
    }

    // ---- Step 5: Graham scan convex hull on 2D projected points ----
    // Matches original C++ FittingLineSample::convexHull exactly.
    // For K=26 this is ~700 operations, trivial per thread.
    float proj_u[MAX_KNN], proj_v[MAX_KNN];
    for (int k = 0; k < valid_k; k++) {
        float dx = dirs[k * 3 + 0] - mx, dy = dirs[k * 3 + 1] - my, dz = dirs[k * 3 + 2] - mz;
        // Use raw (not centered) directions for 3D, but centered for 2D projection
        proj_u[k] = dx * ax1x + dy * ax1y + dz * ax1z;
        proj_v[k] = dx * ax2x + dy * ax2y + dz * ax2z;
    }

    // Find bottom-most point (min v, then min u as tiebreak)
    int pivot = 0;
    for (int i = 1; i < valid_k; i++) {
        if (proj_v[i] < proj_v[pivot] ||
            (proj_v[i] == proj_v[pivot] && proj_u[i] < proj_u[pivot])) {
            pivot = i;
        }
    }
    // Swap pivot to index 0
    { float t;
      t = proj_u[0]; proj_u[0] = proj_u[pivot]; proj_u[pivot] = t;
      t = proj_v[0]; proj_v[0] = proj_v[pivot]; proj_v[pivot] = t;
      for (int d = 0; d < 3; d++) {
          t = dir3d[d]; dir3d[d] = dir3d[pivot*3+d]; dir3d[pivot*3+d] = t;
      }
    }

    // Sort remaining points by polar angle from pivot (insertion sort)
    int order[MAX_KNN];
    for (int i = 0; i < valid_k; i++) order[i] = i;
    float px0 = proj_u[0], py0 = proj_v[0];
    for (int i = 2; i < valid_k; i++) {
        int key = order[i];
        float ka_u = proj_u[key] - px0, ka_v = proj_v[key] - py0;
        int j = i - 1;
        while (j >= 1) {
            int oj = order[j];
            float ob_u = proj_u[oj] - px0, ob_v = proj_v[oj] - py0;
            // cross product: ka × ob > 0 means ka has smaller angle
            float cross = ka_u * ob_v - ka_v * ob_u;
            if (cross > 0 || (cross == 0 &&
                (ka_u*ka_u + ka_v*ka_v) < (ob_u*ob_u + ob_v*ob_v))) {
                order[j + 1] = order[j];
                j--;
            } else break;
        }
        order[j + 1] = key;
    }

    // Graham scan: build hull on stack
    int hull[MAX_KNN];
    int hull_size = 0;
    for (int i = 0; i < valid_k; i++) {
        int idx_i = order[i];
        while (hull_size >= 2) {
            int a = hull[hull_size - 2], b = hull[hull_size - 1];
            float cross = (proj_u[b] - proj_u[a]) * (proj_v[idx_i] - proj_v[a])
                        - (proj_v[b] - proj_v[a]) * (proj_u[idx_i] - proj_u[a]);
            if (cross <= 0) hull_size--;  // right turn or collinear → pop
            else break;
        }
        hull[hull_size++] = idx_i;
    }

    // hull[] now contains convex hull vertex indices in CCW order
    int nh = hull_size;
    if (nh < 3) nh = min(valid_k, 3);  // degenerate

    // ---- Step 6: Piecewise linear curve sampling (matches C++ fitCurve) ----
    // Close the polygon: hull[nh] = hull[0]
    // Compute cumulative arc length along hull edges (in 3D direction space)
    float arc_len[MAX_KNN + 1];  // cumulative arc length
    arc_len[0] = 0.0f;
    for (int i = 0; i < nh; i++) {
        int cur = hull[i];
        int nxt = hull[(i + 1) % nh];
        float dx = dir3d[nxt*3+0] - dir3d[cur*3+0];
        float dy = dir3d[nxt*3+1] - dir3d[cur*3+1];
        float dz = dir3d[nxt*3+2] - dir3d[cur*3+2];
        arc_len[i + 1] = arc_len[i] + sqrtf(dx*dx + dy*dy + dz*dz);
    }
    float total_len = arc_len[nh];
    if (total_len < 1e-10f) total_len = 1.0f;

    // ---- Step 7: Sample S points uniformly along arc length ----
    for (int s = 0; s < S; s++) {
        float target = total_len * (float)s / (float)S;

        // Find the edge segment containing this arc length
        int seg = 0;
        for (int i = 0; i < nh; i++) {
            if (arc_len[i + 1] >= target) { seg = i; break; }
        }
        float seg_len = arc_len[seg + 1] - arc_len[seg];
        float local_t = (seg_len > 1e-10f) ? (target - arc_len[seg]) / seg_len : 0.0f;

        int i0 = hull[seg];
        int i1 = hull[(seg + 1) % nh];

        // Linear interpolation in 3D direction space
        float ix = dir3d[i0*3+0] * (1.0f - local_t) + dir3d[i1*3+0] * local_t;
        float iy = dir3d[i0*3+1] * (1.0f - local_t) + dir3d[i1*3+1] * local_t;
        float iz = dir3d[i0*3+2] * (1.0f - local_t) + dir3d[i1*3+2] * local_t;

        // Normalize to unit sphere
        float norm = sqrtf(ix*ix + iy*iy + iz*iz);
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            ix *= inv; iy *= inv; iz *= inv;
        } else {
            ix = 0; iy = 0; iz = 1.0f;
        }

        out[s*3+0] = ix;
        out[s*3+1] = iy;
        out[s*3+2] = iz;
    }
}

void pca_angular_sampling(int N, int K, int S,
                           const float* directions, float* samples) {
    // Use 128 threads per block (high register usage per thread)
    pcaAngularSamplingKernel<<<(N + 127) / 128, 128>>>(N, K, S, directions, samples);
}
