#!/usr/bin/env python3
from __future__ import annotations
import pathlib
from typing import Dict, Tuple, List


import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, scale_x_log10, scale_y_log10,
    scale_color_manual, scale_linetype_manual,
    labs, theme, element_text, guides, guide_legend, theme_void, element_rect, element_line, element_blank
)
from plotnine import ggsave

# We need this for the automatic break calculation
from mizani.transforms import breaks_log

# Define our dynamic label formatting function using a lambda
power_of_10_labels = lambda breaks: [f"$10^{{{int(round(np.log10(b)))}}}$" for b in breaks]


# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
COLOR_MAP = {
    "Full-CorrDiff":       "#0072B2",
    "Full-CorrDiff-PSD":   "#56B4E9",
    "Regression-CorrDiff": "#009E73",
    "Regression-CorrDiff-PSD": "#67C799",
    "CRPS-UNets":          "#D55E00",
    "CRPS-UNets-PSD":      "#E69F00",
}

VARS = ['u10', 'v10', 't2m', 'vorticity', 'divergence', 'k-energy']

CHANNEL_MAP = {var: i for i, var in enumerate(VARS)}

CERRA_PATH = pathlib.Path("/projects/0/prjs1154/Scandinavia/CERRA/samples/test")
ERA5_PATH  = pathlib.Path("/projects/0/prjs1154/Scandinavia/ERA5/samples/test")

MODEL_PATHS: Dict[str, pathlib.Path] = {
    "Full-CorrDiff": pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/CorrDiffusion-0-Diffusion-06_24_17-8376/files"),
    "Regression-CorrDiff": pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_1dFFT/UNet-CNN-0-UNet-CNN-06_17_15-9228/files"),
    "Regression-CorrDiff-PSD": pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/UNet-CNN-Delft-weighted-UNet-CNN-07_10_00-9378/files"),
    "CRPS-UNets": pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/CRPSresume-UNet-CNN-07_14_10-5016/files"),
    "CRPS-UNets-PSD": pathlib.Path("/projects/0/prjs1154/Scandinavia/preds_20142020_2dFFT/CRPSwLoss-resum-UNet-CNN-08_04_20-5589/files"),
}

ERA5_DX_DEG = 25
OUT_DIR = pathlib.Path("plot_tests")
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────
# FFT / PSD helpers
# ──────────────────────────────────────────
def get_psd(data: np.ndarray, dx: float, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    data = np.moveaxis(data, axis, -1)
    n = data.shape[-1]
    fft = np.fft.rfft(data, axis=-1) / n
    psd = 2.0 * (np.abs(fft) ** 2)
    k = np.fft.rfftfreq(n, d=dx)
    return k, psd

def load_stack(path: pathlib.Path) -> np.ndarray:
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
    if var == 'vorticity':
        u = stack[..., CHANNEL_MAP['u10']]
        v = stack[..., CHANNEL_MAP['v10']]
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

def psd_for_var(stack: np.ndarray, var: str, dx_ref: float, n_ref_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    data = extract_var(stack, var)
    k, psd = get_psd(data, dx_ref, axis=2)
    return k, psd.mean(axis=1).mean(axis=0)

# ──────────────────────────────────────────
# Tidy data for plotnine
# ──────────────────────────────────────────
def make_psd_df(k_ref: np.ndarray, psd_ref: np.ndarray,
                k_models: Dict[str, np.ndarray],
                psd_models: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[dict] = []
    for k, p in zip(k_ref, psd_ref):
        rows.append(dict(k=k, psd=p, model="CERRA", color="#000000", linetype="solid"))
    for name, k_m in k_models.items():
        psd_m = psd_models[name]
        linetype = "dashed" if "PSD" in name else "solid"
        color = COLOR_MAP.get(name, None)
        for k, p in zip(k_m, psd_m):
            rows.append(dict(k=k, psd=p, model=name, color=color, linetype=linetype))
    df = pd.DataFrame(rows)
    levels = ["CERRA"] + list(k_models.keys())
    df["model"] = pd.Categorical(df["model"], categories=levels, ordered=True)
    return df

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main() -> None:
    print("Loading CERRA ground truth …")
    cerra = load_stack(CERRA_PATH)
    era5 = load_stack(ERA5_PATH)
    n_ref_lon = cerra.shape[2]
    dx_ref = ERA5_DX_DEG * (era5.shape[2] / n_ref_lon)

    print("Loading model predictions …")
    model_stacks = {name: load_stack(path) for name, path in MODEL_PATHS.items()}

    # ---- Separate legend strip as PDF (2 rows × 3 items) ----
    legend_labels    = ["CERRA"] + list(MODEL_PATHS.keys())   # 6 entries total
    legend_colors    = ["#000000"] + [COLOR_MAP.get(m) for m in MODEL_PATHS.keys()]
    legend_linetypes = ["solid"] + [("dashed" if "PSD" in m else "solid") for m in MODEL_PATHS.keys()]

    # Two points per model -> actual line segment (legend uses this glyph)
    df_leg = pd.DataFrame({
        "model": np.repeat(legend_labels, 2),
        "k":     np.tile([0.0, 1.0], len(legend_labels)),
        "psd":   0.0,
    })

    pleg = (
        ggplot(df_leg, aes("k", "psd", color="model", linetype="model", group="model"))
        + geom_line(size=2.4, show_legend=True)
        # Same name so color+linetype merge; we’ll also disable the extra guide explicitly
        + scale_color_manual(name="Model", values=legend_colors, limits=legend_labels)
        + scale_linetype_manual(name="Model", values=legend_linetypes, limits=legend_labels)
        + theme_void()
        + theme(
            panel_background=element_rect(fill='white'),
            panel_grid_major=element_line(color='lightgray', linetype='-'),
            panel_grid_minor=element_blank(),
            figure_size=(12, 3.6),
            # legend_position="center",
            legend_title=element_text(size=28),
            legend_text=element_text(size=28),
            # Make the legend swatch wider/taller for clearer dash pattern
            legend_key_width=18,   # points
            legend_key_height=8,   # points
        )
        # Control the *merged* legend and beef up the line in the key
        + guides(
            color=guide_legend(ncol=3, override_aes={"size": 3.2}),
            linetype=False  # avoid creating a second, duplicate guide
        )
    )

    leg_path = OUT_DIR / "model_legend.pdf"
    leg_path.parent.mkdir(exist_ok=True, parents=True)
    ggsave(pleg, filename=str(leg_path), dpi=700)
    print(f"Legend strip saved as {leg_path.name}")



    # ---- PSD plots (no legend) ----
    for var in VARS:
        print(f"\nVariable: {var}")
        k_ref, psd_ref = psd_for_var(cerra, var, dx_ref, n_ref_lon)
        k_models, psd_models = {}, {}
        for name, stack in model_stacks.items():
            k_m, psd_m = psd_for_var(stack, var, dx_ref, n_ref_lon)
            k_models[name], psd_models[name] = k_m, psd_m

        df = make_psd_df(k_ref, psd_ref, k_models, psd_models)
        # PSD figure without legend -> more room for curves
        p = (
            ggplot(df, aes("k", "psd", color="model", linetype="model"))
            + geom_line(size=1.2)
            + scale_x_log10(breaks=breaks_log(), labels=power_of_10_labels)
            + scale_y_log10(breaks=breaks_log(), labels=power_of_10_labels)
            + scale_color_manual(values=legend_colors, limits=legend_labels)
            + scale_linetype_manual(values=legend_linetypes, limits=legend_labels)
            + labs(x=r"Wavenumber $k$ (cycles deg$^{-1}$)", y="PSD", title="")
            + theme(
                panel_background=element_rect(fill='white'),
                # panel_grid_major=element_line(color='lightgray', linetype='-'),
                # panel_grid_minor=element_line(color='lightgray', linetype='-'),
                figure_size=(7, 5),
                axis_title_x=element_text(size=28),
                axis_title_y=element_text(size=28),
                axis_text_x=element_text(size=28),
                axis_text_y=element_text(size=28),
                axis_line_x=element_line(color='black', size=1.2),  # ← show x-axis
                axis_line_y=element_line(color='black', size=1.2),  # ← show y-axis
                legend_position='none'  # <-- remove legend from plot
            )
        )
        out_psd = OUT_DIR / f"{var}_psd.pdf"        # <-- PDF output
        ggsave(p, filename=str(out_psd), dpi=700)            # vector; no dpi needed
        print(f"Saved {out_psd}")

    # ---- Separate legend strip as PDF ----
    if MODEL_PATHS:
        df_leg = pd.DataFrame({"k": np.ones(len(legend_labels)),
                               "psd": np.ones(len(legend_labels)),
                               "model": legend_labels})
        pleg = (
            ggplot(df_leg, aes("k", "psd", color="model", linetype="model"))
            + geom_line(size=1.6)
            + scale_color_manual(values=legend_colors, limits=legend_labels)
            + scale_linetype_manual(values=legend_linetypes, limits=legend_labels)
            + theme_void()
            + theme(
                figure_size=(12, 3.8),          # wide & short legend-only canvas
                # legend_position="center",
                legend_title=element_text(size=20),
                legend_text=element_text(size=20),
            )
            + guides(
                color=guide_legend(ncol=int(np.ceil(len(legend_labels) / 2))),
                linetype=guide_legend(ncol=int(np.ceil(len(legend_labels) / 2))),
            )
        )
        leg_path = OUT_DIR / "model_legend.pdf"     # <-- PDF legend
        ggsave(pleg, filename=str(leg_path), dpi=700)
        print(f"Legend strip saved as {leg_path.name}")

    print("\nAll figures saved in", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
