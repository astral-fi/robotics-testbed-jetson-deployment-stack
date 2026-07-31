# Data Directory

Place the following files here before running `docker compose up`:

1. **`camera_calibration.yaml`** — Per-camera intrinsics (K) and extrinsics (R, t).
   See `ros2_ws/src/gsplat_tracker/config/camera_calibration.yaml` for the expected format.

2. **`model.ckpt`** — Trained gsplat model checkpoint (PyTorch `.ckpt`).
   Must contain keys: `means`, `quats`, `scales`, `opacities`, and either `colors` or `sh_coefficients`.

This directory is mounted read-only into the container at `/workspace/data`.
