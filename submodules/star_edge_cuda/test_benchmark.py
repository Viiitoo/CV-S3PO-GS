"""
Performance benchmark: CPU LocalSH vs CUDA kernel.

Usage (inside Docker container):
    cd /workspace/submodules/star_edge_cuda
    python3 test_benchmark.py
"""

import sys
import time
import numpy as np
import torch

# CPU LocalSH
sys.path.insert(0, "/workspace/STAR-Edge/LocalSH")
import LocalSH

# CUDA
import star_edge_cuda


def benchmark(sizes=None, bw=10, kk=26, sample_num=30, n_warmup=2, n_repeat=3):
    if sizes is None:
        sizes = [500, 1000, 5000, 15000, 30000]

    print(f"{'N':>8s}  {'CPU (ms)':>10s}  {'CUDA (ms)':>10s}  {'Speedup':>8s}  {'VRAM (MB)':>10s}")
    print("-" * 56)

    for N in sizes:
        np.random.seed(42)
        pts64 = np.random.randn(N, 3).astype(np.float64)
        pts32 = pts64.astype(np.float32)

        # ---- CPU ----
        t_cpu_list = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            LocalSH.LocalSHFeature.ComLSHF_knn_upsample(pts64, bw, kk, sample_num)
            t_cpu_list.append(time.perf_counter() - t0)
        t_cpu = np.median(t_cpu_list)

        # ---- CUDA warmup ----
        for _ in range(n_warmup):
            star_edge_cuda.compute_descriptors(pts32[:min(500, N)], bw=bw, kk=kk, num_samples=sample_num)
        torch.cuda.synchronize()

        # ---- CUDA ----
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        t_cuda_list = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            star_edge_cuda.compute_descriptors(pts32, bw=bw, kk=kk, num_samples=sample_num)
            torch.cuda.synchronize()
            t_cuda_list.append(time.perf_counter() - t0)
        t_cuda = np.median(t_cuda_list)

        peak_mem = torch.cuda.max_memory_allocated() - mem_before

        speedup = t_cpu / t_cuda if t_cuda > 0 else float("inf")
        print(
            f"{N:8d}  {t_cpu*1000:10.1f}  {t_cuda*1000:10.1f}  {speedup:7.1f}x  {peak_mem/1024/1024:10.1f}"
        )


def benchmark_per_kernel(N=15000, bw=10, kk=26, num_samples=30):
    """Profile individual kernel stages."""
    print(f"\n{'='*60}")
    print(f"Per-kernel profiling (N={N})")
    print(f"{'='*60}")

    from star_edge_cuda._C import compute_localsh_descriptors, knn_search

    np.random.seed(42)
    pts = torch.from_numpy(np.random.randn(N, 3).astype(np.float32)).cuda()

    basis_r, basis_i, grid = star_edge_cuda._get_sh_basis(bw, "cuda:0")

    # Warmup
    compute_localsh_descriptors(pts, basis_r, basis_i, grid, bw, kk, num_samples)
    torch.cuda.synchronize()

    # KNN only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        knn_idx = knn_search(pts, kk)
    torch.cuda.synchronize()
    t_knn = (time.perf_counter() - t0) / 5
    print(f"  KNN search:       {t_knn*1000:8.2f} ms")

    # Full pipeline
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        desc = compute_localsh_descriptors(pts, basis_r, basis_i, grid, bw, kk, num_samples)
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / 5
    print(f"  Full pipeline:    {t_full*1000:8.2f} ms")
    print(f"  Non-KNN portion:  {(t_full-t_knn)*1000:8.2f} ms")


if __name__ == "__main__":
    print("STAR-Edge CUDA Benchmark")
    print("=" * 56)
    benchmark()
    benchmark_per_kernel()
