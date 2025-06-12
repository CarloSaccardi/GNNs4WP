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
VARS = ['u10', 'v10', 't2m', 'sshf', 'zust', 'wind_speed']

CHANNEL_MAP = {var: i for i, var in enumerate(VARS)}  # adjust if order differs

CERRA_PATH = pathlib.Path(
    "/aspire/CarloData/zz_UNETs/data/big_dataset/CERRA/samples/test"
)

MODEL_PATHS: Dict[str, pathlib.Path] = {
    "SongUNetBase"     : pathlib.Path("/aspire/CarloData/zz_UNETs/data/big_dataset/preds/UNet-CNN-BigData-UNet-CNN-05_20_19-9299/files"),
    "SongUNet01"   : pathlib.Path("/aspire/CarloData/zz_UNETs/data/big_dataset/preds/UNet-CNN-01-UNet-CNN-06_09_11-4102/files"),
    "SongUNet001"  : pathlib.Path("/aspire/CarloData/zz_UNETs/data/big_dataset/preds/UNet-CNN-001-UNet-CNN-06_08_19-1434/files"),
    #"SongUNet0001" : pathlib.Path("/aspire/CarloData/zz_UNETs/data/big_datase/tpreds/UNet-CNN-0001-UNet-CNN-06_09_11-3299/files"),
    "GNNUNet"      : pathlib.Path("/aspire/CarloData/zz_UNETs/data/big_dataset/preds/UNet-GNN-BigData-graph_efm-4x64-05_22_10-8678/files"),
}

ERA5_DX_DEG = 0.25                       # longitude spacing of reference grid
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
    if var == 'wind_speed':
        u = stack[..., CHANNEL_MAP['u10']]
        v = stack[..., CHANNEL_MAP['v10']]
        return np.sqrt(u ** 2 + v ** 2)
    return stack[..., CHANNEL_MAP[var]]

# ──────────────────────────────────────────
# PSD & PDF for one variable
# ──────────────────────────────────────────
def psd_for_var(stack: np.ndarray, var: str,
                dx_ref: float, n_ref_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean PSD over latitude & samples."""
    n_lon = stack.shape[2]
    dx = dx_ref * (n_ref_lon / n_lon)
    data = extract_var(stack, var)
    k, psd = get_psd(data, dx, axis=2)
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
    n_ref_lon = cerra.shape[2]

    print("Loading model predictions …")
    model_stacks = {name: load_stack(path) for name, path in MODEL_PATHS.items()}

    # ── loop over variables ──────────────────────────────────────────────────
    for var in VARS:
        print(f"\nVariable: {var}")

        # ----------   PSD   ----------
        k_ref, psd_ref = psd_for_var(cerra, var, ERA5_DX_DEG, n_ref_lon)
        k_models, psd_models = {}, {}
        for name, stack in model_stacks.items():
            k_m, psd_m = psd_for_var(stack, var, ERA5_DX_DEG, n_ref_lon)
            k_models[name], psd_models[name] = k_m, psd_m

        plt.figure(figsize=(7, 5))
        plt.loglog(k_ref, psd_ref, lw=3, c="k", label="CERRA")
        for name, k_m in k_models.items():
            plt.loglog(k_m, psd_models[name], label=name)
        plt.xlabel(r"Wavenumber $k$ (cycles deg$^{-1}$)")
        plt.ylabel("PSD")
        plt.title(f"PSD of {var} (longitude)")
        plt.legend()
        plt.tight_layout()
        out_psd = OUT_DIR / f"{var}_psd.png"
        plt.savefig(out_psd, dpi=200)
        plt.close()

        # ----------   PDF (log-scale Y) ----------
        # Build one common bin grid that spans the full range of ALL datasets
        vals_min = min(extract_var(cerra, var).min(),
                    *(extract_var(s, var).min() for s in model_stacks.values()))
        vals_max = max(extract_var(cerra, var).max(),
                    *(extract_var(s, var).max() for s in model_stacks.values()))
        bins = np.linspace(vals_min, vals_max, N_BINS + 1)

        # CERRA histogram
        centres_ref, pdf_ref = pdf_for_var(cerra, var, bins)

        # Model histograms
        pdf_models = {name: pdf_for_var(stack, var, bins)[1]
                    for name, stack in model_stacks.items()}

        plt.figure(figsize=(7, 4))

        # CERRA reference curve
        plt.plot(centres_ref,
                np.log10(pdf_ref + EPS),   # <- Y-axis is log10(PDF)
                lw=3, c="k", label="CERRA")

        # All models
        for name, pdf in pdf_models.items():
            plt.plot(centres_ref, np.log10(pdf + EPS), label=name)

        plt.xlabel(var)                       # X-axis = actual variable values
        plt.ylabel(r"$\log_{10}$ PDF")        # Y-axis = log10(PDF)
        plt.title(f"log₁₀ PDF of {var}")
        plt.legend(ncol=2)
        plt.tight_layout()

        out_pdf = OUT_DIR / f"{var}_logpdf.png"
        plt.savefig(out_pdf, dpi=200)
        plt.close()


        print(f"  → saved {out_psd.name}, {out_pdf.name}")

    print("\nAll figures saved in", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
