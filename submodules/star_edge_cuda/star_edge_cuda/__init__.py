"""
STAR-Edge CUDA: GPU-accelerated LocalSH descriptor computation.

Usage:
    import star_edge_cuda
    descriptors = star_edge_cuda.compute_descriptors(points, bw=10, kk=26)
"""

import os
import torch
import numpy as np

_C_module = None
_sh_basis_cache = {}
_scale_factors_cache = {}


def _ensure_loaded():
    global _C_module
    if _C_module is None:
        import importlib
        _C_module = importlib.import_module("star_edge_cuda._C")


def _precompute_sh_basis(bw, device="cpu"):
    """
    Precompute the SH transform matrices for the forward SHT.

    Uses a unified grid for both KDE evaluation and SH basis:
        theta_j = pi * (2j + 1) / (4 * bw),  j = 0..2*bw-1
        phi_k   = pi * (2k + 1) / (2 * bw),  k = 0..2*bw-1

    This matches the original LocalSH C++ code's grid convention.

    Returns:
        basis_real: (C, G) float32 tensor
        basis_imag: (C, G) float32 tensor
        grid_xyz:   (G, 3) float32 tensor (sphere grid cartesian coords for KDE)
    """
    from scipy.special import sph_harm as _sph_harm

    lat = lon = 2 * bw
    G = lat * lon
    C = bw * bw

    # --- Unified grid (matching LocalSH's convention) ---
    # theta_j = (j + 0.5) * pi / lat = (2j+1) * pi / (4*bw)
    # phi_k   = (k + 0.5) * 2*pi / lon = (2k+1) * pi / (2*bw)
    thetas = np.array([(j + 0.5) * np.pi / lat for j in range(lat)])
    phis = np.array([(k + 0.5) * 2.0 * np.pi / lon for k in range(lon)])

    # Grid XYZ for KDE
    grid_xyz_np = np.zeros((G, 3), dtype=np.float32)
    for ii in range(lat):
        for jj in range(lon):
            g = ii * lon + jj
            grid_xyz_np[g, 0] = np.sin(thetas[ii]) * np.cos(phis[jj])
            grid_xyz_np[g, 1] = np.sin(thetas[ii]) * np.sin(phis[jj])
            grid_xyz_np[g, 2] = np.cos(thetas[ii])

    # --- Compute DH quadrature weights ---
    # S2kit convention: w_j includes sin(theta) and normalization
    weights = np.zeros(lat)
    for j in range(lat):
        w = 0.0
        for k in range(bw):
            w += np.sin((2 * j + 1) * (2 * k + 1) * np.pi / (4 * bw)) / (2 * k + 1)
        weights[j] = w * (2.0 * np.pi / (bw * bw)) * np.sin((2 * j + 1) * np.pi / (4 * bw))

    dphi = 2.0 * np.pi / lon

    # --- Build SH basis matrix using the same unified grid ---
    # Coefficient indexing: degree l starts at l^2, m goes from -l to l
    # scipy sph_harm convention: sph_harm(m, l, phi, theta)
    basis = np.zeros((C, G), dtype=np.complex128)

    for l in range(bw):
        for m_idx, m in enumerate(range(-l, l + 1)):
            c_idx = l * l + m_idx

            for jj in range(lat):
                w = weights[jj] * dphi

                for kk in range(lon):
                    g_idx = jj * lon + kk
                    # Evaluate Y_lm at the unified grid point
                    Y_conj = np.conj(_sph_harm(m, l, phis[kk], thetas[jj]))
                    basis[c_idx, g_idx] = Y_conj * w

    basis_real = basis.real.astype(np.float32)
    basis_imag = basis.imag.astype(np.float32)

    return (
        torch.from_numpy(basis_real).to(device),
        torch.from_numpy(basis_imag).to(device),
        torch.from_numpy(grid_xyz_np).to(device),
    )


def _get_sh_basis(bw, device):
    """Get or create cached SH basis matrices."""
    cache_key = (bw, str(device))
    if cache_key not in _sh_basis_cache:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        # Use v2 suffix to distinguish from old cached files
        real_path = os.path.join(data_dir, f"sh_basis_real_bw{bw}_v2.npy")
        imag_path = os.path.join(data_dir, f"sh_basis_imag_bw{bw}_v2.npy")
        grid_path = os.path.join(data_dir, f"grid_xyz_bw{bw}_v2.npy")

        if os.path.exists(real_path) and os.path.exists(imag_path) and os.path.exists(grid_path):
            basis_r = torch.from_numpy(np.load(real_path)).to(device)
            basis_i = torch.from_numpy(np.load(imag_path)).to(device)
            grid = torch.from_numpy(np.load(grid_path)).to(device)
        else:
            # Compute on the fly and cache to disk
            basis_r, basis_i, grid = _precompute_sh_basis(bw, device)
            os.makedirs(data_dir, exist_ok=True)
            np.save(real_path, basis_r.cpu().numpy())
            np.save(imag_path, basis_i.cpu().numpy())
            np.save(grid_path, grid.cpu().numpy())

        _sh_basis_cache[cache_key] = (basis_r, basis_i, grid)

    return _sh_basis_cache[cache_key]


def _get_scale_factors(bw):
    """Load per-degree scale factors for calibration against CPU LocalSH."""
    if bw not in _scale_factors_cache:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        sf_path = os.path.join(data_dir, f"scale_factors_bw{bw}_v2.npy")
        if os.path.exists(sf_path):
            _scale_factors_cache[bw] = np.load(sf_path).astype(np.float32)
        else:
            _scale_factors_cache[bw] = None
    return _scale_factors_cache[bw]


def compute_descriptors(
    points,
    bw=10,
    kk=26,
    num_samples=30,
    device="cuda:0",
):
    """
    GPU-accelerated LocalSH descriptor computation.

    Args:
        points: (N, 3) numpy array or torch tensor
        bw: SH bandwidth (default 10)
        kk: number of KNN neighbors (default 26)
        num_samples: number of curve samples (default 30)
        device: CUDA device string

    Returns:
        descriptors: (N, bw) float32 numpy array
    """
    _ensure_loaded()

    # Convert input
    if isinstance(points, np.ndarray):
        pts = torch.from_numpy(np.ascontiguousarray(points[:, :3]).astype(np.float32))
    else:
        pts = points[:, :3].float()
    pts = pts.to(device).contiguous()

    N = pts.shape[0]
    if N < kk + 1:
        # Not enough points for KNN
        return np.zeros((N, bw), dtype=np.float32)

    # Get precomputed SH basis
    basis_r, basis_i, grid = _get_sh_basis(bw, device)

    # Call CUDA kernels
    desc = _C_module.compute_localsh_descriptors(
        pts, basis_r, basis_i, grid, bw, kk, num_samples
    )

    result = desc.cpu().numpy()

    # Apply per-degree calibration scale factors to match CPU LocalSH output
    sf = _get_scale_factors(bw)
    if sf is not None:
        result *= sf[np.newaxis, :]

    return result
