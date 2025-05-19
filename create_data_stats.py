#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-channel mean and standard deviation for the high-
 and low-resolution weather datasets.

Run:
    python compute_stats.py \
        --dataset_cerra /path/to/CERRA \
        --dataset_era5  /path/to/ERA5 \
        --batch_size 64
"""
from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import os
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── First-party ───────────────────────────────────────────────────────────────
from neural_lam.weather_dataset import ERA5tCERRAStats


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    """Utility to add --foo / --no-foo flags."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true")
    group.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


@torch.inference_mode()          # same as torch.no_grad(), but clearer intent
def compute_running_stats(loader: DataLoader, use_high: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Online (streaming) mean / std computation using two running sums:
        μ = Σ x / N
        σ = sqrt( Σ x² / N − μ² )
    Returns:
        mean, std – both shape (C,)
    """
    device_accum = torch.device("cpu")             # <- keep on CPU
    dtype_accum  = torch.float64                   # <- double for stability

    # running totals
    tot_count   = 0
    tot_sum     = None
    tot_sum_sq  = None

    for batch in tqdm(loader, desc="Accumulating statistics"):
        high, low = batch               # we only care about 'high' OR 'low', caller loops twice
        x = high if use_high else low

        # Bring to double on CPU
        x = x.to(device_accum, dtype=dtype_accum)

        # flatten everything except channel dim -> (B, C, -1)
        B, C, *spatial = x.shape
        x = x.view(B, C, -1)

        # accumulate
        tot_count  += B * x.shape[-1]
        batch_sum   = x.sum(dim=(0, 2))            # per-channel
        batch_sqsum = (x ** 2).sum(dim=(0, 2))

        if tot_sum is None:                        # first iteration
            tot_sum    = batch_sum
            tot_sum_sq = batch_sqsum
        else:
            tot_sum    += batch_sum
            tot_sum_sq += batch_sqsum

    mean = tot_sum / tot_count
    var  = tot_sum_sq / tot_count - mean ** 2
    std  = torch.sqrt(var.clamp(min=0.0))

    return mean.float(), std.float()               # down-cast back to fp32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_cerra",
        type=Path,
        default=Path("/aspire/CarloData/zz_UNETs/data/big_dataset/CERRA"),
    )
    parser.add_argument(
        "--dataset_era5",
        type=Path,
        default=Path("/aspire/CarloData/zz_UNETs/data/big_dataset/ERA5"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=4)
    add_bool_flag(parser, "pin_memory", default=False)  #set to False when using cpu only
    args = parser.parse_args()

    # ── Load datasets ────────────────────────────────────────────────────────
    ds = ERA5tCERRAStats(
        str(args.dataset_cerra),
        str(args.dataset_era5),
        split="train",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.n_workers > 0,
    )

    # ── Compute stats separately ────────────────────────────────────────────
    mean_high, std_high = compute_running_stats(loader, use_high=True)
    mean_low , std_low  = compute_running_stats(loader, use_high=False)

    # ── Save ────────────────────────────────────────────────────────────────
    for root, mean, std in [
        (args.dataset_cerra / "static", mean_high, std_high),
        (args.dataset_era5  / "static", mean_low , std_low ),
    ]:
        root.mkdir(parents=True, exist_ok=True)
        torch.save(mean, root / "parameter_mean.pt")
        torch.save(std , root / "parameter_std.pt")

    print("✓ Statistics saved.")


if __name__ == "__main__":
    main()
