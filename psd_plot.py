#!/usr/bin/env python3
"""
plot_psd_allvars.py
===================

For every variable in VARS, compute the longitudinal power-spectral density (PSD)
of CERRA ground truth and of several prediction models, then save one log–log
plot per variable in plot_tests/<var>_psd.png.

Assumptions
-----------
* Every sample is stored as a .npy file whose last dimension holds the five
  variables in the exact order given by VARS.  If your order differs, adjust
  CHANNEL_MAP accordingly.
* Prediction files are (N, C, H, W) or (C, H, W).  The loader transposes them
  to (N, H, W, C).

Dependencies: numpy, matplotlib
"""

from __future__ import annotations
import pathlib
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# CONFIGURATION  (edit as needed)
# ──────────────────────────────────────────

COLOR_MAP = {
    "Full-CorrDiff":       "#0072B2",  # blue
    "Full-CorrDiff-PSD":   "#56B4E9",  # light blue
    "Regression-CorrDiff": "#009E73",  # green
    "Regression-CorrDiff-PSD": "#67C799",  # light green
    "CRPS-UNets":          "#D55E00",  # vermillion
    "CRPS-UNets-PSD":      "#E69F00",  # orange
}


VARS = ['u10', 'v10', 't2m', 'vorticity', 'divergence', 'k-energy'] #, 'sshf', 'zust', 'wind_speed']

CHANNEL_MAP = {var: i for i, var in enumerate(VARS)}  # adjust if order differs

CERRA_PATH = pathlib.Path(
    "/projects/0/prjs1154/Scandinavia/CERRA/samples/test"
)

ERA5_PATH = pathlib.Path(
    "/projects/0/prjs1154/Scandinavia/ERA5/samples/test"
)

MODEL_PATHS: Dict[str, pathlib.Path] = {
    "Full-CorrDiff"     : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/CorrDiffusion-0-Diffusion-06_24_17-8376/files"),
    # "Full-CorrDiff-PSD"     : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/CorrDiffusion-001-Diffusion-06_24_17-3916/files"),
    "Regression-CorrDiff"  : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/UNet-CNN-0-UNet-CNN-06_17_15-9228/files"),
    # "Regression-CorrDiff-Delf"  : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/UNet-CNN-Delft-UNet-CNN-07_10_00-3910/files"),
    "Regression-CorrDiff-PSD"  : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/UNet-CNN-Delft-weighted-UNet-CNN-07_10_00-9378/files"),
    # "Regression-CorrDiff-continue"  : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/UNet-CNN-flexContinue-UNet-CNN-07_02_11-0248/files"),
    "CRPS-UNets"            : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/CRPSresume-UNet-CNN-07_14_10-5016/files"),
    "CRPS-UNets-PSD"            : pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/CRPSwLoss-resum-UNet-CNN-08_04_20-5589/files"),
    # "CRPS-UNets"  : pathlib.Path("/projects/0/prjs1154/CentralEurope_2014_2020/preds_20142020_2dFFT/CRPSresume-UNet-CNN-07_14_10-5016/files"),
    #saved_models/UNet-CNN-Delft-UNet-CNN-07_10_00-3910
}

ERA5_DX_DEG = 25                       # longitude spacing of reference grid
N_BINS = 200                             # PDF histogram resolution
EPS = 1e-12                              # avoids log(0)
OUT_DIR = pathlib.Path("plot_tests")
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────
# FFT / PSD helper
# ──────────────────────────────────────────
def get_psd(data: np.ndarray, dx: float, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided PSD and wavenumber axis along `axis`."""
    data = np.moveaxis(data, axis, -1)
    n = data.shape[-1]
    fft = np.fft.rfft(data, axis=-1) / n
    psd = 2.0 * (np.abs(fft) ** 2)
    k = np.fft.rfftfreq(n, d=dx)
    return k, psd

# ──────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────
def load_stack(path: pathlib.Path) -> np.ndarray:
    """
    Load .npy stack → (N, lat, lon, C).
    Transposes channel-first files automatically.
    """
    def _to_chan_last(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:                          # C, H, W
            return arr.transpose(1, 2, 0)[None, ...]
        if arr.ndim == 4 and arr.shape[1] < 10:    # N, C, H, W
            return arr.transpose(0, 2, 3, 1)
        return arr

    if path.is_file():
        return _to_chan_last(np.load(path))

    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No *.npy under {path}")
    stack = np.array([np.load(f) for f in files])
    return _to_chan_last(stack)


def extract_var(stack: np.ndarray, var: str) -> np.ndarray:
    """(N, lat, lon) slice for `var`.  Wind-speed is √(u10²+v10²)."""
    if var == 'vorticity':
        u = stack[..., CHANNEL_MAP['u10']]
        v = stack[..., CHANNEL_MAP['v10']]
        # print(u,)
        return np.gradient(v, axis=2) - np.gradient(u, axis=1)
    elif var == 'divergence':
        u = stack[..., CHANNEL_MAP['u10']]
        v = stack[..., CHANNEL_MAP['v10']]
        return np.gradient(u, axis=2) + np.gradient(v, axis=1)
    elif var == 'k-energy':
        u = stack[..., CHANNEL_MAP['u10']]
        v = stack[..., CHANNEL_MAP['v10']]
        return 0.5 * (u**2 + v**2)
    else:
        return stack[..., CHANNEL_MAP[var]]

# ──────────────────────────────────────────
# PSD & PDF for one variable
# ──────────────────────────────────────────
def psd_for_var(stack: np.ndarray, var: str,
                dx_ref: float, n_ref_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean PSD over latitude & samples."""
    #n_lon = stack.shape[2]
    #dx = dx_ref * (n_ref_lon / n_lon)
    data = extract_var(stack, var)
    k, psd = get_psd(data, dx_ref, axis=2)
    return k, psd.mean(axis=1).mean(axis=0)

def pdf_for_var(stack: np.ndarray, var: str,
                bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return bin centres and PDF (density) for flattened variable values."""
    flat = extract_var(stack, var).ravel()
    pdf, edges = np.histogram(flat, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, pdf

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main() -> None:
    print("Loading CERRA ground truth …")
    cerra = load_stack(CERRA_PATH)
    era5 = load_stack(ERA5_PATH)
    n_ref_lon = cerra.shape[2]

    dx_ref = ERA5_DX_DEG * (era5.shape[2] / n_ref_lon)  # adjust dx for CERRA

    print("Loading model predictions …")
    model_stacks = {name: load_stack(path) for name, path in MODEL_PATHS.items()}

    # ---- holder for a single copy of the handles / labels we will turn into a legend
    legend_handles, legend_labels = None, None

    # ── loop over variables ──────────────────────────────────────────────────
    for var in VARS:
        print(f"\nVariable: {var}")

        # ----------   PSD   ----------
        k_ref, psd_ref = psd_for_var(cerra, var, dx_ref, n_ref_lon)
        k_models, psd_models = {}, {}
        for name, stack in model_stacks.items():
            k_m, psd_m = psd_for_var(stack, var, dx_ref, n_ref_lon)
            k_models[name], psd_models[name] = k_m, psd_m

        plt.figure(figsize=(7, 5))
        line_ref, = plt.loglog(k_ref, psd_ref, lw=3, c="k", label="CERRA")

        for name, k_m in k_models.items():
            psd_m = psd_models[name]
            linestyle = '--' if 'PSD' in name else '-'
            color = COLOR_MAP.get(name, None)
            plt.loglog(k_m, psd_m, label=name,
                       linestyle=linestyle, linewidth=2, color=color)

        plt.xlabel(r"Wavenumber $k$ (cycles deg$^{-1}$)", fontsize=23)
        plt.ylabel(r"PSD", fontsize=23)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.tight_layout()

        # ---- grab handles/labels only once (first variable) -----------------
        if legend_handles is None:
            legend_handles, legend_labels = plt.gca().get_legend_handles_labels()

        out_psd = OUT_DIR / f"{var}_psd.png"
        plt.savefig(out_psd, dpi=200)
        plt.close()

        # ----------   PDF (unchanged)  --------------------------------------
        # [...]  (your existing PDF code)
        # --------------------------------------------------------------------

    # ──────────────────────────────────────────
    # save legend as its *own* skinny figure
    # ──────────────────────────────────────────
    if legend_handles is not None:
        num_items = len(legend_labels)
        cols = int(np.ceil(num_items / 2))
        fig_leg = plt.figure(figsize=(12, 4.5))      # wide & short
        fig_leg.legend(
                    legend_handles, legend_labels,
                    loc='center',
                    ncol=cols,                # <- wraps into 2 rows
                    frameon=False,
                    fontsize=18,
                    handlelength=2.0,
                    columnspacing=1.2,
                    labelspacing=1.0,
                )
        fig_leg.set_constrained_layout(True)
        # fig_leg.tight_layout(pad=0.2)
        leg_path = OUT_DIR / "model_legend.png"
        fig_leg.savefig(leg_path, dpi=300, bbox_inches='tight',
                        transparent=False)            # transparent background
        plt.close(fig_leg)
        print(f"\nLegend strip saved as {leg_path.name}")

    print("\nAll figures saved in", OUT_DIR.resolve())



if __name__ == "__main__":
    main()
