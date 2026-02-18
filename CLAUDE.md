# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
总是使用中文回答！

S3PO-GS (CV-S3PO-GS) is a real-time monocular/stereo SLAM system built on 3D Gaussian Splatting with time-varying deformation for dynamic scene modeling. It uses MASt3R for depth/pose estimation and supports multiple datasets (KITTI, Waymo, DL3DV, StereoMIS).

## Runtime Environment

- **Docker container**: `5d0fdd8cb92d`
- Common run command:
  ```bash
  CUDA_VISIBLE_DEVICES=X python3 slam.py --config configs/mono/Stereo/Stereo_seq_easy/stereo_full.yaml
  ```
  其中 `X` 取决于哪个 GPU 空闲（通过 `nvidia-smi` 查看）。

## Environment Setup

```bash
conda env create -f environment.yml   # Creates "S3PO-GS" env (Python 3.11, PyTorch 2.1.0, CUDA 11.8)
conda activate S3PO-GS

# Build CUDA extensions (required)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
# Optional: pip install submodules/star_edge_cuda, submodules/localsh_cuda
```

## Running SLAM

```bash
python slam.py --config configs/mono/Stereo/Stereo_seq_easy/<scene>.yaml
```

Key CLI overrides: `--begin/--end` (frame range), `--alpha` (loss weight), `--windowsize` (local BA window), `--iter` (tracking iterations), `--sh` (SH degree), `--patch_size`, `--color` (enable color refinement).

Config hierarchy: dataset-specific YAML inherits from `configs/mono/<dataset>/base_config.yaml`.

## Architecture

### Multi-Process Design

The system runs three processes communicating via `mp.Queue`:

- **Main process** (`slam.py`): Loads config, dataset, MASt3R model, GaussianModel; spawns frontend/backend; runs evaluation after SLAM completes
- **Frontend** (`utils/slam_frontend.py`, ~940 lines): Tracking loop — processes frames sequentially, estimates poses via PnP (with optional STAR-Edge edge guidance), selects keyframes, manages sliding window
- **Backend** (`utils/slam_backend.py`, ~1024 lines): Mapping — optimizes Gaussians (densification, pruning, color/depth losses), runs local/global BA, handles color refinement

Frontend sends keyframe data to backend via queue; backend sends optimized Gaussians back.

### Gaussian Model

`gaussian_splatting/scene/gaussian_model.py` (~1982 lines) is the core data structure:

- Per-point attributes: `_xyz`, `_scaling`, `_rotation`, `_opacity`, `_features_dc` (SH coefficients)
- **Time-varying deformation**: 17 Gaussian RBF basis functions (`time_basis_num` in config) modulate position, rotation, scale, and opacity over time
- `_deformation_table` (bool mask): separates static vs dynamic points
- `_deformation_accum`: accumulates deformation magnitudes; table updated every 200 iterations with threshold from `deform_table_threshold`

### Key Modules

| Module | Role |
|--------|------|
| `utils/init_pose.py` | Pose initialization via PnP with MASt3R point matches |
| `utils/edge_extraction.py` | Edge/contour extraction (STAR-Edge integration) |
| `utils/flow_utils.py` / `gmflow/` | Optical flow computation (GMFlow) |
| `utils/depth_utils.py` | Depth filtering and patch-based validation |
| `utils/dataset.py` | Dataset loading for all supported formats |
| `utils/eval_utils.py` | ATE evaluation and rendering metrics (PSNR, SSIM, LPIPS) |
| `gui/slam_gui.py` | Real-time OpenGL visualization (enabled via `use_gui: True`) |

### External Dependencies (in-tree)

- **MASt3R** (`mast3r/`): Monocular/stereo 3D estimation; model checkpoint at `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- **DUST3R** (`dust3r/`): Dense 3D reconstruction backbone
- **MotionGS** (`MotionGS/`): Motion-guided deformable Gaussian splatting (flow loss)
- **STAR-Edge** (`STAR-Edge/`): Structure-aware edge extraction from point clouds
- **EH-SurGS** (git submodule): Reference for temporal deformation approach
- **GMFlow** (`gmflow/`): Optical flow network

### CUDA Extensions (`submodules/`)

- `diff-gaussian-rasterization/`: Custom differentiable Gaussian rasterizer
- `simple-knn/`: Fast KNN for point cloud operations
- `star_edge_cuda/`: CUDA kernels for edge extraction
- `localsh_cuda/`: Local spherical harmonics CUDA implementation

## Configuration Reference

Key config sections in YAML files:

- `Dataset.sensor_type`: `"monocular"` or `"stereo"`
- `Dataset.depth_loss`: Enable Huber loss depth supervision (threshold δ=0.2)
- `Training.use_optical_flow` / `use_camera_flow`: Enable flow-based supervision
- `Training.use_gt_pose`: Use ground truth poses (for testing)
- `model_params.time_basis_num`: Number of RBF basis functions for deformation (default 17)
- `model_params.deform_table_threshold`: Threshold for marking points as dynamic
- `opt_params.deformation_lr_init`: Learning rate for deformation parameters
- `rgb_edge_pnp`: STAR-Edge guided PnP matching configuration
- `Flow`: GMFlow model path and type

## Utility Scripts

```bash
# Extract GT vs rendered comparison video
python extract_rgb_video.py --viz_dir results/<run>/viz --output output.mp4 --fps 10

# Visualize temporal deformation model
python view_time_models.py <save_dir> --animate --start 0 --end 99 --step 5
```

## Notes

- Uses `mp.set_start_method("spawn")` — CUDA tensors cannot be shared across processes directly
- `distCUDA2` replaced with pure PyTorch KNN to avoid multi-process CUDA context issues (see `docs/distCUDA2_bug_report.md`)
- Logging uses W&B (`wandb`); disable with `use_wandb: False` in config
- Results saved to `results/<dataset_path>/<timestamp>/` with config snapshot
- Documentation and comments are primarily in Chinese
