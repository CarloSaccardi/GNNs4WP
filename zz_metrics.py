# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvtx
import torch
import tqdm
import os
from physicsnemo.utils.generative import StackedRandomGenerator
from tueplots import bundles, figsizes
from scipy.stats import ks_2samp  
from pprint import pprint

from typing import Callable, Optional

import torch
from torch import Tensor
from physicsnemo.utils.patching import GridPatching2D

import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_metrics(
    path_gt: str,
    path_pred: str,
    save_dir: str,
    *,
    var_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compare every .npy file that exists in BOTH `path_gt` and `path_pred`,
    assuming each file has shape (H, W, C) with the same C variables.

    Returns a nested dict: metrics[var_name][metric] = value
    and writes the same information to <save_dir>/metrics_new.txt.
    """

    path_gt, path_pred = Path(path_gt), Path(path_pred)
    gt_files   = {f.name: f for f in path_gt.glob("*.npy")}
    pred_files = {f.name: f for f in path_pred.glob("*.npy")}
    common     = sorted(gt_files.keys() & pred_files.keys())

    if not common:
        raise FileNotFoundError("No overlapping .npy filenames in the two folders.")

    # ----- discover channel count from the first file
    first = np.load(gt_files[common[0]])
    if first.ndim != 3:
        raise ValueError(
            f"Expected shape (H, W, C). Found {first.shape} in {common[0]!r}"
        )
    C = first.shape[-1]
    if var_names is None:
        var_names = [f"var{c}" for c in range(C)]
    if len(var_names) != C:
        raise ValueError("Length of var_names must equal number of channels (C).")

    # accumulators: metric_sums[var][metric] = running total
    metric_template = {
        "MAE": 0.0,
        "RMSE": 0.0,
        "SSIM": 0.0,
        "PSNR": 0.0,
        "Cramer": 0.0,
        "KS": 0.0,
        "Hill": 0.0,
    }
    metric_sums = {v: metric_template.copy() for v in var_names}

    # global metrics that combine u and v (channels 0 and 1)
    global_sums: dict[str, float] = {}
    if C >= 2:
        global_sums = {"Wind_RMSE": 0.0, "Vorticity_RMS": 0.0}

    n_files = 0

    # ── per‑file loop ─────────────────────────────────────────────────────────
    for fname in common:
        gt  = np.load(gt_files[fname]).astype(np.float32)
        prd = np.load(pred_files[fname]).astype(np.float32)

        gt  = ensure_channels_last(gt,  C)
        prd = ensure_channels_last(prd, C)

        if gt.shape != prd.shape:
            raise ValueError(f"Shape mismatch for {fname}: {gt.shape} vs {prd.shape}")
        if gt.shape[-1] != C:
            raise ValueError(f"Channel count changed in {fname}: got {gt.shape[-1]}, expected {C}")

        # ── per‑variable metrics ────────────────────────────────────────────
        for c, vname in enumerate(var_names):
            g = gt[..., c]
            p = prd[..., c]

            metric_sums[vname]["MAE"]    += compute_mae(p, g)
            metric_sums[vname]["RMSE"]   += compute_rmse(p, g)
            metric_sums[vname]["SSIM"]   += compute_ssim_metric(p, g)
            metric_sums[vname]["PSNR"]   += compute_psnr_metric(p, g)
            metric_sums[vname]["Cramer"] += compute_cramer(p, g)
            metric_sums[vname]["KS"]     += compute_ks_metric(p, g)
            metric_sums[vname]["Hill"]   += compute_hill_metric(p, g)

        # ── global wind metrics (need u & v) ────────────────────────────────
        if C >= 2:
            u_gt, v_gt   = gt[..., 0], gt[..., 1]
            u_prd, v_prd = prd[..., 0], prd[..., 1]

            global_sums["Wind_RMSE"]     += compute_wind_rmse(u_prd, v_prd, u_gt, v_gt)
            global_sums["Vorticity_RMS"] += compute_vorticity_rms(u_prd, v_prd, u_gt, v_gt)

        n_files += 1

    # ── average across files ─────────────────────────────────────────────────
    metrics = {
        v: {m: total / n_files for m, total in metric_sums[v].items()}
        for v in var_names
    }
    if global_sums:
        metrics["_GLOBAL_"] = {m: total / n_files for m, total in global_sums.items()}

    # ── save to disk ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / "metrics_new.txt"
    with open(out_path, "w") as fh:
        for v in list(metrics.keys()):
            fh.write(f"[{v}]\n")
            for m, val in metrics[v].items():
                fh.write(f"  {m}: {val:.6f}\n")
            fh.write("\n")

    return metrics


# ─────────────────────── basic metrics helpers ───────────────────────────────
def compute_mae(p: np.ndarray, g: np.ndarray) -> float:
    return np.abs(p - g).mean()


def compute_rmse(p: np.ndarray, g: np.ndarray) -> float:
    return np.sqrt(np.mean((p - g) ** 2))


def compute_ssim_metric(p: np.ndarray, g: np.ndarray) -> float:
    dr = g.max() - g.min()
    return float(ssim(g, p, data_range=dr))


def compute_psnr_metric(p: np.ndarray, g: np.ndarray) -> float:
    dr = g.max() - g.min()
    return float(psnr(g, p, data_range=dr))


def compute_cramer(p: np.ndarray, g: np.ndarray) -> float:
    gx = torch.from_numpy(g.flatten()).float().unsqueeze(1)
    px = torch.from_numpy(p.flatten()).float().unsqueeze(1)
    dxy = torch.cdist(gx, px).mean()
    dgg = torch.cdist(gx, gx).mean()
    dpp = torch.cdist(px, px).mean()
    return (2 * dxy - dgg - dpp).item()


# ───────────────────── physics‑oriented helpers ──────────────────────────────
def compute_wind_rmse(u_p: np.ndarray, v_p: np.ndarray,
                      u_g: np.ndarray, v_g: np.ndarray) -> float:
    """RMSE of wind‑speed magnitude."""
    speed_p = np.hypot(u_p, v_p)
    speed_g = np.hypot(u_g, v_g)
    return np.sqrt(np.mean((speed_p - speed_g) ** 2))


def compute_vorticity_rms(u_p: np.ndarray, v_p: np.ndarray,
                           u_g: np.ndarray, v_g: np.ndarray,
                           dx: float = 1.0, dy: float = 1.0) -> float:
    """
    RMS of vorticity error ζ = ∂v/∂x − ∂u/∂y using centred finite differences.
    """
    ζ_p = np.gradient(v_p, dx, axis=1) - np.gradient(u_p, dy, axis=0)
    ζ_g = np.gradient(v_g, dx, axis=1) - np.gradient(u_g, dy, axis=0)
    return np.sqrt(np.mean((ζ_p - ζ_g) ** 2))


def compute_ks_metric(p: np.ndarray, g: np.ndarray) -> float:
    """
    Two‑sample Kolmogorov–Smirnov statistic (two‑sided).
    """
    return ks_2samp(p.ravel(), g.ravel(), alternative="two-sided").statistic


def compute_hill_metric(p: np.ndarray, g: np.ndarray, k: int = 100) -> float:
    """
    Absolute difference of Hill tail indices – focuses on heavy‑tail behaviour.
    """
    def hill(x: np.ndarray, k_: int) -> float:
        x = np.abs(x.ravel()) + 1e-6     # ensure strictly positive
        x_sorted = np.sort(x)[::-1]      # descending
        k_ = min(k_, len(x_sorted) - 1)
        x_k = x_sorted[k_]
        return (1.0 / k_) * np.log(x_sorted[:k_] / x_k).sum()

    return abs(hill(p, k) - hill(g, k))


# ─────────────────────────── utility ─────────────────────────────────────────
def ensure_channels_last(arr: np.ndarray, C_expected: int) -> np.ndarray:
    """
    Convert array to (H, W, C) format if currently (C, H, W).
    Raises if channel axis cannot be found.
    """
    if arr.shape[-1] == C_expected:      # already (H, W, C)
        return arr
    if arr.shape[0] == C_expected:       # assume (C, H, W)
        return np.moveaxis(arr, 0, -1)   # -> (H, W, C)
    raise ValueError(f"Cannot locate channel axis in shape {arr.shape}")


# ──────────────────────────── CLI ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute metrics for two folders of (H,W,C) .npy files."
    )
    parser.add_argument("--path_gt",   required=True, help="Directory with ground‑truth .npy files")
    parser.add_argument("--path_pred", required=True, help="Directory with prediction  .npy files")
    parser.add_argument("--save_dir",  required=True, help="Where metrics_new.txt will be written")
    parser.add_argument(
        "--var_names",
        nargs="*",
        default=None,
        help=("Optional list of variable names (length must equal channel count), "
              "e.g. --var_names u10 v10 t2m sshf zust"),
    )
    args = parser.parse_args()

    metrics = compute_metrics(
        Path(args.path_gt).expanduser(),
        Path(args.path_pred).expanduser(),
        Path(args.save_dir).expanduser(),
        var_names=['u10', 'v10', 't2m', 'sshf', 'zust'],
    )

    pprint(metrics)