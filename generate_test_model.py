#!/usr/bin/env python3
"""
generate_test_model.py

Generates a synthetic gsplat model checkpoint for pipeline testing.
Creates a colorful scene with:
  - A ground plane (green/brown Gaussians)
  - A central floating cube (red)
  - Scattered spherical clusters (blue, yellow, cyan)

This gives the renderer a recognizable scene to rasterise so you can
verify the full pipeline works end-to-end without needing real training data.

Usage:
    python3 generate_test_model.py
    → Writes ./data/model.ckpt
"""

import torch
import numpy as np
import os


def random_quaternions(n: int) -> torch.Tensor:
    """Generate n random unit quaternions (w, x, y, z)."""
    q = torch.randn(n, 4)
    q = q / q.norm(dim=1, keepdim=True)
    # Ensure w > 0 for canonical form
    q[q[:, 0] < 0] *= -1
    return q


def make_ground_plane(n: int = 5000) -> dict:
    """Green/brown ground plane at y = -0.5."""
    means = torch.zeros(n, 3)
    means[:, 0] = torch.randn(n) * 2.0        # spread in X
    means[:, 1] = -0.5 + torch.randn(n) * 0.02  # thin layer at y = -0.5
    means[:, 2] = torch.randn(n) * 2.0        # spread in Z

    # Flat scales (wide in XZ, thin in Y)
    scales = torch.log(torch.tensor([[0.08, 0.002, 0.08]]).repeat(n, 1)
                       + torch.rand(n, 3) * 0.02)

    # Green-brown color
    colors = torch.zeros(n, 3)
    colors[:, 0] = 0.25 + torch.rand(n) * 0.15   # R
    colors[:, 1] = 0.45 + torch.rand(n) * 0.25   # G
    colors[:, 2] = 0.10 + torch.rand(n) * 0.10   # B

    return {
        "means": means,
        "scales": scales,
        "colors": colors,
        "quats": random_quaternions(n),
        "opacities": torch.ones(n) * 0.85,
    }


def make_cube(n: int = 3000, center=(0., 0.2, 0.), size=0.4) -> dict:
    """Red cube — Gaussians clustered on 6 faces."""
    means_list = []
    per_face = n // 6

    cx, cy, cz = center
    h = size / 2.0

    # 6 faces of the cube
    for axis in range(3):        # X, Y, Z
        for sign in [-1, 1]:
            pts = torch.randn(per_face, 3) * (size * 0.4)
            pts[:, axis] = sign * h + torch.randn(per_face) * 0.005
            pts[:, 0] += cx
            pts[:, 1] += cy
            pts[:, 2] += cz
            means_list.append(pts)

    means = torch.cat(means_list, dim=0)
    actual_n = means.shape[0]

    scales = torch.log(torch.ones(actual_n, 3) * 0.015
                       + torch.rand(actual_n, 3) * 0.005)

    # Bright red with slight variation
    colors = torch.zeros(actual_n, 3)
    colors[:, 0] = 0.85 + torch.rand(actual_n) * 0.15   # R
    colors[:, 1] = 0.05 + torch.rand(actual_n) * 0.10   # G
    colors[:, 2] = 0.05 + torch.rand(actual_n) * 0.10   # B

    return {
        "means": means,
        "scales": scales,
        "colors": colors,
        "quats": random_quaternions(actual_n),
        "opacities": torch.ones(actual_n) * 0.92,
    }


def make_sphere_cluster(
    n: int = 1500,
    center: tuple = (0., 0., 0.),
    radius: float = 0.3,
    color: tuple = (0.2, 0.4, 0.9),
) -> dict:
    """Spherical cluster of Gaussians."""
    # Sample points on/near a sphere surface
    phi = torch.rand(n) * 2 * np.pi
    cos_theta = torch.rand(n) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    r = radius * (0.85 + torch.rand(n) * 0.3)

    means = torch.zeros(n, 3)
    means[:, 0] = r * sin_theta * torch.cos(phi) + center[0]
    means[:, 1] = r * cos_theta + center[1]
    means[:, 2] = r * sin_theta * torch.sin(phi) + center[2]

    scales = torch.log(torch.ones(n, 3) * 0.012
                       + torch.rand(n, 3) * 0.008)

    colors = torch.zeros(n, 3)
    for i in range(3):
        colors[:, i] = color[i] + torch.rand(n) * 0.1

    return {
        "means": means,
        "scales": scales,
        "colors": colors,
        "quats": random_quaternions(n),
        "opacities": torch.ones(n) * 0.88,
    }


def merge_dicts(parts: list) -> dict:
    """Concatenate all Gaussian parameter tensors."""
    merged = {}
    for key in parts[0].keys():
        merged[key] = torch.cat([p[key] for p in parts], dim=0)
    return merged


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating synthetic gsplat test scene...")

    parts = [
        make_ground_plane(5000),
        make_cube(3000, center=(0.0, 0.2, 0.0), size=0.4),
        make_sphere_cluster(1500, center=(-1.0, 0.3, 0.8), radius=0.25,
                            color=(0.2, 0.4, 0.95)),    # Blue sphere
        make_sphere_cluster(1200, center=(1.0, 0.1, -0.6), radius=0.2,
                            color=(0.95, 0.85, 0.1)),   # Yellow sphere
        make_sphere_cluster(1000, center=(0.5, 0.5, 1.0), radius=0.18,
                            color=(0.1, 0.9, 0.85)),    # Cyan sphere
    ]

    checkpoint = merge_dicts(parts)

    # Ensure all tensors are float32
    for key in checkpoint:
        checkpoint[key] = checkpoint[key].float()

    total = checkpoint["means"].shape[0]
    print(f"  Total Gaussians: {total:,}")
    print(f"  means:     {list(checkpoint['means'].shape)}")
    print(f"  quats:     {list(checkpoint['quats'].shape)}")
    print(f"  scales:    {list(checkpoint['scales'].shape)}")
    print(f"  opacities: {list(checkpoint['opacities'].shape)}")
    print(f"  colors:    {list(checkpoint['colors'].shape)}")

    out_path = os.path.join(os.path.dirname(__file__), "data", "model.ckpt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(checkpoint, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved to: {out_path}  ({size_mb:.1f} MB)")
    print("Scene: ground plane + red cube + blue/yellow/cyan spheres")


if __name__ == "__main__":
    main()
