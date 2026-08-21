#!/usr/bin/env python3
"""
inspect_checkpoint.py

Prints the keys, shapes, dtypes, and value ranges of every tensor
in a gsplat .ckpt file so we can diagnose loading/colour issues.

Usage:
    python3 inspect_checkpoint.py [path/to/model.ckpt]
"""

import sys
import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "data/model.ckpt"
print(f"Loading checkpoint: {ckpt_path}\n")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if not isinstance(ckpt, dict):
    print(f"Checkpoint is not a dict — type: {type(ckpt)}")
    sys.exit(1)

print(f"{'Key':<30} {'Shape':<30} {'dtype':<12} {'min':>10} {'max':>10} {'mean':>10}")
print("─" * 100)
for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        print(
            f"{k:<30} {str(list(v.shape)):<30} {str(v.dtype):<12} "
            f"{v.float().min().item():>10.4f} {v.float().max().item():>10.4f} "
            f"{v.float().mean().item():>10.4f}"
        )
    else:
        print(f"{k:<30} (non-tensor: {type(v).__name__})")
