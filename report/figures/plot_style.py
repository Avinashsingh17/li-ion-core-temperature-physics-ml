"""
plot_style.py — Shared matplotlib rcparams + fixed color map.

Imported by every figure script in report/figures/. Keeping fonts, sizes,
spines, grid, and the named color map consistent across the publication
figure set so the writeup reads as one document, not nine.

Colors:
  model              -- ML / model curves
  baseline_nominal   -- predict T_surface = nominal chamber setpoint
  baseline_tinf      -- predict T_surface = T_inf (cooling-tail asymptote)
  band_fill          -- shaded R_cs label-uncertainty band
  measured_surface   -- measured T_surface_C reference
  in_regime / inconclusive / out_of_regime
                     -- 25 °C / 10 °C / 0 °C severity coding for bars
"""
from __future__ import annotations

import matplotlib as mpl

COLORS: dict[str, str] = {
    "model":             "#1f77b4",   # blue
    "baseline_nominal":  "#7f7f7f",   # mid-gray
    "baseline_tinf":     "#ff7f0e",   # orange
    "band_fill":         "#a6cee3",   # light blue
    "band_edge":         "#6cb0d6",   # slightly darker for the band's outline
    "measured_surface":  "#000000",   # black
    "in_regime":         "#2ca02c",   # green  (25 °C: calibration regime)
    "inconclusive":      "#d4a017",   # amber  (10 °C: mild diff-ambient)
    "out_of_regime":     "#d62728",   # red    (0  °C: aggressive cold)
}


def apply_style() -> None:
    """Apply the publication-figure rcparams. Call once at script start."""
    mpl.rcParams.update({
        "figure.dpi":         200,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",

        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,

        "axes.spines.right":  False,
        "axes.spines.top":    False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     ":",
        "grid.linewidth":     0.7,

        "legend.fontsize":    9,
        "legend.frameon":     False,

        "lines.linewidth":    1.5,
    })
