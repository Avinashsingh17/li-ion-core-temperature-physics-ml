"""
make_figures.py — Render the publication figure set (Week-4 writeup).

READ-ONLY on labels / calibration / ML artifacts. Does NOT re-fit, re-simulate,
re-calibrate. All numbers come from the 2026-06-18b locked run.

Phase 0 prints an inventory + manifest. Then F1..F7 render; F8 is BLOCKED
(no labeled data for US06 @ 0 °C and no saved residual time-series).

Run:  python report/figures/make_figures.py  (from project root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local import: project_root/report/figures/plot_style.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_style import COLORS, apply_style

ROOT = HERE.parents[1]              # project root
FIGS = HERE                          # output dir
LABELED = ROOT / "data" / "labeled"
CALIB_JSON_P = ROOT / "data" / "calibration" / "calibration_results.json"
ML_DIR = ROOT / "data" / "ml"
OCV_P = ROOT / "data" / "processed" / "ocv_lg_25degC.parquet"


# ----- LOCKED 2026-06-18b ----------------------------------------------------
LOCKED = {
    "C_total": 43.20, "C_core": 40.10, "C_surf": 3.10,
    "R_cs_central": 3.394, "R_cs_low": 1.697, "R_cs_high": 8.486,
    "R_sa": 5.21, "T_inf_25C": 23.15,
    "T_inf_10C":  9.18, "T_inf_0C": -0.66,
    "T_core_band_peak_spread_C": 5.07,
}

# F3 LOCKED severity-stratified validation numbers (transcribed from
# user task spec — verified against `step6_validation_per_cycle` below).
F3_LOCKED = [
    {"label": "Mixed1 @ 25 °C", "regime": "in_regime",     "model": 0.383, "nominal": 0.918, "tinf": 1.160, "excursion": 2.72},
    {"label": "Mixed2 @ 25 °C", "regime": "in_regime",     "model": 0.380, "nominal": 0.826, "tinf": 1.232, "excursion": 3.03},
    {"label": "UDDS @ 10 °C",   "regime": "inconclusive",  "model": 0.468, "nominal": 0.759, "tinf": 0.158, "excursion": 0.92},
    {"label": "US06 @ 0 °C",    "regime": "out_of_regime", "model": 3.612, "nominal": 1.994, "tinf": 2.592, "excursion": 4.55},
    {"label": "LA92 @ 0 °C",    "regime": "out_of_regime", "model": 1.037, "nominal": 0.353, "tinf": 0.849, "excursion": 2.13},
]


# ============================================================================
# Phase 0 — Inventory + manifest scaffolding
# ============================================================================
manifest: list[dict] = []


def record(fig: str, status: str, source: str = "", note: str = "") -> None:
    manifest.append({"figure": fig, "status": status,
                     "source": source, "note": note})


def inventory_print() -> None:
    print("=" * 72)
    print("PHASE 0 — INVENTORY (LOCKED = 2026-06-18b)")
    print("=" * 72)

    # 1. Calibration JSON
    print("\n[1] calibration JSON")
    with open(CALIB_JSON_P) as f:
        calib = json.load(f)
    cp = calib["step4_central_fit"]
    print(f"  path: {CALIB_JSON_P.relative_to(ROOT)}")
    print(f"  step4_central_fit: C_core={cp['C_core_J_K']:.3f}, "
          f"C_surf={cp['C_surf_J_K']:.3f}, "
          f"R_cs={cp['R_cs_K_per_W']:.3f}, "
          f"R_sa={cp['R_sa_K_per_W']:.3f}")
    print(f"  T_inf @25C = {calib['step2_R_sa_result']['T_inf_by_ambient_C']['25']:.3f}")
    # Verify match to LOCKED.
    deviations = []
    if abs(cp["C_core_J_K"] - LOCKED["C_core"]) > 0.05:
        deviations.append(("C_core", LOCKED["C_core"], cp["C_core_J_K"]))
    if abs(cp["C_surf_J_K"] - LOCKED["C_surf"]) > 0.05:
        deviations.append(("C_surf", LOCKED["C_surf"], cp["C_surf_J_K"]))
    if abs(cp["R_cs_K_per_W"] - LOCKED["R_cs_central"]) > 0.05:
        deviations.append(("R_cs",   LOCKED["R_cs_central"], cp["R_cs_K_per_W"]))
    if abs(cp["R_sa_K_per_W"] - LOCKED["R_sa"]) > 0.05:
        deviations.append(("R_sa",   LOCKED["R_sa"], cp["R_sa_K_per_W"]))
    if deviations:
        print(f"  ⚠ DEVIATION from LOCKED 2026-06-18b: {deviations}")
        sys.exit("STOP — calibration JSON does not match LOCKED params.")
    print("  ✓ matches LOCKED 2026-06-18b")

    # 2. Labels index + sample
    print("\n[2] data/labeled/")
    idx_p = LABELED / "_labels_index.parquet"
    idx = pd.read_parquet(idx_p)
    print(f"  index path: {idx_p.relative_to(ROOT)}")
    print(f"  index shape: {idx.shape}; cycles: {sorted(idx['cycle'].unique())}")
    sample_p = next(p for p in sorted(LABELED.glob("*.parquet"))
                    if "_labels_index" not in p.name)
    sdf = pd.read_parquet(sample_p)
    need = ["time_s", "T_core_model_C", "T_core_model_lo_C",
            "T_core_model_hi_C", "T_surf_model_C", "T_surface_C"]
    missing = [c for c in need if c not in sdf.columns]
    print(f"  sample: {sample_p.name}  shape {sdf.shape}")
    print(f"  has columns needed: {[c for c in need if c in sdf.columns]}")
    if missing:
        print(f"  ⚠ MISSING in sample: {missing}")
        sys.exit("STOP — labeled parquets missing required columns.")

    # 3. ML LOCO files
    print("\n[3] data/ml/")
    ridge_loco_p = ML_DIR / "ridge_loco.parquet"
    hgbr_loco_p  = ML_DIR / "hgbr_loco.parquet"
    if not ridge_loco_p.exists() or not hgbr_loco_p.exists():
        print(f"  ⚠ missing one of {ridge_loco_p.name}, {hgbr_loco_p.name}")
        sys.exit("STOP — ML LOCO files missing.")
    rl = pd.read_parquet(ridge_loco_p)
    hl = pd.read_parquet(hgbr_loco_p)
    print(f"  ridge_loco: {rl.shape}; columns {list(rl.columns)}")
    print(f"  hgbr_loco:  {hl.shape}; columns {list(hl.columns)}")
    # Verify per-cycle ridge & HGBR RMSE saved.
    has_pc = ("ml_rmse" in rl.columns) and ("ml_rmse" in hl.columns) \
             and ("cycle" in rl.columns)
    print(f"  per-cycle ridge AND HGBR RMSE present? {has_pc}")
    rs = json.load(open(ML_DIR / "ridge_summary.json"))
    hs = json.load(open(ML_DIR / "hgbr_summary.json"))
    print(f"  ridge pooled LOCO RMSE = {rs['loco']['pooled']['rmse']:.4f}, "
          f"max = {rs['loco']['pooled']['max']:.4f}")
    print(f"  HGBR  pooled LOCO RMSE = {hs['loco']['pooled']['rmse']:.4f}, "
          f"max = {hs['loco']['pooled']['max']:.4f}")
    print(f"  ridge ablation Δ vs full (RMSE) = "
          f"{rs['ablation_drop_surface']['delta_rmse_vs_full']:+.4f}")
    print(f"  HGBR  ablation Δ vs full (RMSE) = "
          f"{hs['ablation_drop_surface']['delta_rmse_vs_full']:+.4f}")

    # 4. OCV table
    print("\n[4] OCV table")
    if OCV_P.exists():
        ocv = pd.read_parquet(OCV_P)
        print(f"  path: {OCV_P.relative_to(ROOT)}; shape {ocv.shape}; "
              f"columns {list(ocv.columns)}")
    else:
        print(f"  ⚠ missing: {OCV_P}")

    # 5. Severity-stratified surface-validation results
    print("\n[5] severity-stratified results")
    sv = calib.get("step6_validation_per_cycle", [])
    print(f"  step6_validation_per_cycle: {len(sv)} rows in calibration JSON")
    if sv:
        s_cols = sorted(sv[0].keys())
        print(f"  per-row keys: {s_cols[:6]} ... ({len(s_cols)} total)")
        # Verify the 5 cycles + cross-check with F3 LOCKED transcription.
        by_short = {row["file_id"].split("_")[2] + "@" + row["file_id"].split("_")[1]: row
                    for row in sv}
        # Map LOCKED F3 to JSON entries.
        # F3 LOCKED uses "Mixed1 @ 25 °C" etc.; JSON file_ids include ambient.
        f3_check_rows = []
        for row in sv:
            key = (row["file_id"].split("_25degC_")[0].split("_")[-1] if "25degC" in row["file_id"]
                   else row["file_id"].split("degC_")[0].split("_")[-1])
            f3_check_rows.append((row["file_id"], row["model_rmse_C"],
                                  row["baseline_nominal_rmse_C"],
                                  row["baseline_T_inf_rmse_C"],
                                  row["peak_excursion_T_s_from_T_inf_C"]))
        print(f"  step6 rows (file_id -> model RMSE / nominal RMSE / T_inf RMSE / excursion):")
        for fid, m, n, t, e in f3_check_rows:
            print(f"    {fid[:50]:50s}  model {m:.3f}  nom {n:.3f}  "
                  f"tinf {t:.3f}  excur {e:.3f}")

    # 6. F8 residual data
    print("\n[6] F8 residual time-series")
    print("  no parquet of residuals saved; only PNG of prior calibration")
    print("  diagnosis exists (data/eda_plots/residual_diag_LG_0degC_*.png).")
    print("  -> F8 BLOCKED per spec (no re-running the pipeline).")

    return calib, idx, rl, hl


# ============================================================================
# Helpers — labeled-cycle reader
# ============================================================================
def find_labeled(cycle_short: str) -> Path | None:
    for p in sorted(LABELED.glob(f"LG_25degC_{cycle_short}_*.parquet")):
        return p
    return None


# ============================================================================
# F1 — Two-state thermal schematic
# ============================================================================
def fig01_two_state(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5)
    ax.set_axis_off()

    # Core node (left)
    core = mp.FancyBboxPatch(
        (1.0, 2.3), 3.0, 2.0, boxstyle="round,pad=0.06",
        linewidth=2.0, edgecolor=COLORS["model"], facecolor="#dde7f5")
    ax.add_patch(core)
    ax.text(2.5, 3.6, "Core node",  ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(2.5, 3.0, f"C_core = {LOCKED['C_core']:.2f} J/K",
            ha="center", va="center", fontsize=10)
    ax.text(2.5, 2.6, "T_core (model-derived)", ha="center", va="center",
            fontsize=9, style="italic", color="#444")

    # Surface node (middle)
    surf = mp.FancyBboxPatch(
        (5.5, 2.3), 3.0, 2.0, boxstyle="round,pad=0.06",
        linewidth=2.0, edgecolor=COLORS["model"], facecolor="#dde7f5")
    ax.add_patch(surf)
    ax.text(7.0, 3.6, "Surface node", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(7.0, 3.0, f"C_surf = {LOCKED['C_surf']:.2f} J/K",
            ha="center", va="center", fontsize=10)
    ax.text(7.0, 2.6, "T_surf (measured)", ha="center", va="center",
            fontsize=9, style="italic", color="#444")

    # Ambient node (right)
    amb = mp.FancyBboxPatch(
        (10.0, 2.6), 1.8, 1.4, boxstyle="round,pad=0.04",
        linewidth=1.8, edgecolor="#555", facecolor="#f0f0f0")
    ax.add_patch(amb)
    ax.text(10.9, 3.5, "Ambient", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(10.9, 3.0, f"T_inf = {LOCKED['T_inf_25C']:.2f} °C",
            ha="center", va="center", fontsize=9)

    # R_cs arrow between core and surface
    ax.annotate(
        "", xy=(5.4, 3.3), xytext=(4.1, 3.3),
        arrowprops=dict(arrowstyle="<->", lw=2.0, color="#333"))
    ax.text(4.75, 3.7, f"R_cs = {LOCKED['R_cs_central']:.2f} K/W",
            ha="center", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    # Nudged clear of the core-node bottom edge (was overlapping at y=2.9).
    ax.text(4.75, 2.05, "(band: 1.70 – 8.49 K/W)",
            ha="center", va="top", fontsize=8, color="#666")

    # R_sa arrow between surface and ambient
    ax.annotate(
        "", xy=(9.9, 3.3), xytext=(8.6, 3.3),
        arrowprops=dict(arrowstyle="<->", lw=2.0, color="#333"))
    ax.text(9.25, 3.7, f"R_sa = {LOCKED['R_sa']:.2f} K/W",
            ha="center", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # Q input arrow
    ax.annotate(
        "", xy=(2.5, 2.25), xytext=(2.5, 0.6),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=COLORS["out_of_regime"]))
    ax.text(2.5, 0.4, "Q = I · (V − V_OCV)",
            ha="center", va="top", fontsize=11, color=COLORS["out_of_regime"],
            fontweight="bold")

    # Title + caption note
    ax.set_title("Two-state lumped thermal model (central params, 2026-06-18b)",
                 fontsize=12, pad=6)
    ax.text(6.0, 5.7,
            "Core temperature is model-derived — never measured.",
            ha="center", va="center", fontsize=10, style="italic")

    fig.savefig(out)
    plt.close(fig)
    record("F1 two_state_schematic.png", "GENERATED",
           "no data — matplotlib patches", "central params from calibration JSON")


# ============================================================================
# F2 — R_cs band sweep on US06 @ 25 °C
# ============================================================================
def fig02_band(out: Path) -> None:
    p = find_labeled("US06")
    if p is None:
        record("F2 rcs_band_sweep.png", "SKIPPED",
               "data/labeled/LG_25degC_US06_*.parquet",
               "labeled US06 @ 25 °C not found")
        return
    df = pd.read_parquet(p)
    t_min = df["time_s"].to_numpy() / 60.0
    tc = df["T_core_model_C"].to_numpy()
    tl = df["T_core_model_lo_C"].to_numpy()
    th = df["T_core_model_hi_C"].to_numpy()
    ts = df["T_surface_C"].to_numpy()

    band_spread = th - tl
    peak_spread = float(np.max(band_spread))
    peak_at_idx = int(np.argmax(band_spread))
    peak_t_min = t_min[peak_at_idx]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.fill_between(t_min, tl, th, color=COLORS["band_fill"], alpha=0.6,
                    label=f"R_cs band [{LOCKED['R_cs_low']:.2f}–"
                          f"{LOCKED['R_cs_high']:.2f}] K/W")
    ax.plot(t_min, tc, lw=1.7, color=COLORS["model"],
            label=f"model-derived T_core (R_cs={LOCKED['R_cs_central']:.2f} K/W, central)")
    ax.plot(t_min, ts, lw=1.1, ls="--", color=COLORS["measured_surface"],
            label="measured T_surface (reference)")

    # Annotate peak band spread — wording made explicit that this is the
    # pipeline-wide R_cs uncertainty, with US06 shown because the band is
    # widest there (NOT that 5.05 °C is the global figure). Annotation
    # placed in lower-mid empty area with arrow up to band peak; explicit
    # ylim ensures nothing is cropped.
    ax.set_ylim(min(tl.min(), ts.min()) - 0.5, th.max() + 1.5)
    ax.annotate(
        f"peak band spread = {peak_spread:.2f} °C  "
        f"(pipeline-wide R_cs uncertainty ≈ 5 °C; widest here on US06)",
        xy=(peak_t_min, th[peak_at_idx]),
        xytext=(peak_t_min - 32, th.max() - 1.5),
        fontsize=9, ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="#888", lw=0.6, pad=2.5),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#444"))

    ax.set_xlabel("time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("US06 @ 25 °C — model-derived core temperature with R_cs label-uncertainty band",
                 fontsize=12)
    # Legend at upper-left, per spec. The explicit ylim above gives 1.5 °C
    # of headroom so the legend sits above the band's early-cycle rise
    # without overlapping the curves.
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    record("F2 rcs_band_sweep.png", "GENERATED",
           "data/labeled/LG_25degC_US06_*.parquet",
           f"peak band spread {peak_spread:.2f} °C (label uncertainty from R_cs)")


# ============================================================================
# F3 — Severity-stratified validation bars
# ============================================================================
def fig03_severity(out: Path) -> None:
    rows = F3_LOCKED
    n_groups = len(rows)
    bar_w = 0.26
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    rects_m = ax.bar(x - bar_w, [r["model"]   for r in rows], bar_w,
                     label="thermal model", color=COLORS["model"],
                     edgecolor="black", linewidth=0.4)
    rects_n = ax.bar(x,         [r["nominal"] for r in rows], bar_w,
                     label="baseline: T_s = nominal setpoint",
                     color=COLORS["baseline_nominal"],
                     edgecolor="black", linewidth=0.4)
    rects_t = ax.bar(x + bar_w, [r["tinf"]    for r in rows], bar_w,
                     label="baseline: T_s = T_inf",
                     color=COLORS["baseline_tinf"],
                     edgecolor="black", linewidth=0.4)

    # x-tick labels color-coded by regime (the only place regime is encoded,
    # now that the separate "ambient regime" legend has been dropped).
    labels = [r["label"] for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick_label, r in zip(ax.get_xticklabels(), rows):
        tick_label.set_color(COLORS[r["regime"]])
        tick_label.set_fontweight("bold")

    # Annotate value on top of each bar.
    for rects in (rects_m, rects_n, rects_t):
        for rect in rects:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    v + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Per-group "|T_s - T_inf|_peak = ..." annotation — placed directly above
    # the centre of each group's own three bars, smaller font, regime-colored
    # to match the x-tick label.
    y_top_group = [max(r["model"], r["nominal"], r["tinf"]) for r in rows]
    for xi, r, ytop in zip(x, rows, y_top_group):
        ax.text(xi, ytop + 0.30,
                f"|T_s − T_inf|_peak = {r['excursion']:.2f} °C",
                ha="center", va="bottom", fontsize=8,
                color=COLORS[r["regime"]], fontweight="bold")

    # Raised y-limit per spec — ~4.6 to give headroom for the per-group
    # excursion annotations without crowding the top bars.
    ax.set_ylim(0, 4.6)
    ax.set_ylabel("surface-temperature RMSE (°C)")
    ax.set_title(
        "Severity-stratified surface-temperature validation — "
        "thermal model vs two trivial baselines",
        fontsize=12, pad=24)
    # One-line subtitle for the regime color code (replaces the dropped
    # "ambient regime" legend block). Placed inside the axes via the
    # ax.transAxes transform so the title's `pad=24` reliably puts it
    # ABOVE the subtitle.
    ax.text(0.5, 1.015,
            "x-axis color: green = in-regime 25 °C · "
            "amber = inconclusive 10 °C · "
            "red = out-of-regime 0 °C",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#444")
    # Only one legend now (model / baseline_nominal / baseline_tinf).
    ax.legend(loc="upper left")

    fig.savefig(out)
    plt.close(fig)
    record("F3 severity_validation_bars.png", "GENERATED",
           "transcribed LOCKED + cross-check vs calibration JSON step6",
           "5 cycles across 3 regimes; excursion annotated per group")


# ============================================================================
# F4 — LOCO ridge vs HGBR (per-cycle + pooled)
# ============================================================================
def fig04_loco(out: Path, ridge_loco: pd.DataFrame, hgbr_loco: pd.DataFrame,
               ridge_summary: dict, hgbr_summary: dict) -> None:
    # Cycle order: by excursion descending (US06 first, UDDS last).
    rl = ridge_loco.set_index("cycle")
    hl = hgbr_loco.set_index("cycle")
    cycles = list(hl.sort_values("peak_excursion_C", ascending=False).index)
    x = np.arange(len(cycles))
    bar_w = 0.30

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ridge_vals = rl.loc[cycles, "ml_rmse"].to_numpy()
    hgbr_vals  = hl.loc[cycles, "ml_rmse"].to_numpy()
    bt_vals    = hl.loc[cycles, "baseline_tinf_rmse"].to_numpy()

    r_bars = ax.bar(x - bar_w/2, ridge_vals, bar_w, color=COLORS["model"],
                    edgecolor="black", linewidth=0.4,
                    label="ridge (linear)")
    h_bars = ax.bar(x + bar_w/2, hgbr_vals, bar_w, color=COLORS["band_fill"],
                    edgecolor="black", linewidth=0.4,
                    label="HGBR (trees)")
    # Reference line: per-cycle T_inf baseline mean
    bt_mean = float(np.mean(bt_vals))
    ax.axhline(bt_mean, ls="--", lw=1.2, color=COLORS["baseline_tinf"],
               label=f"reference: mean baseline (T_core = T_inf) = {bt_mean:.2f} °C")

    pooled_r = ridge_summary["loco"]["pooled"]
    pooled_h = hgbr_summary["loco"]["pooled"]
    ax.set_xticks(x)
    ax.set_xticklabels(cycles, rotation=0)
    for rect, v in zip(r_bars, ridge_vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color="#234d70")
    for rect, v in zip(h_bars, hgbr_vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color="#3f7aa0")

    ax.set_ylabel("ML LOCO RMSE on T_core (°C, model-derived label)")
    ax.set_title(
        f"Leave-one-cycle-out: ridge vs HGBR  "
        f"(pooled RMSE: ridge {pooled_r['rmse']:.3f}, HGBR {pooled_h['rmse']:.3f}; "
        f"max: ridge {pooled_r['max']:.3f}, HGBR {pooled_h['max']:.3f})",
        fontsize=11, pad=8)
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)
    record("F4 loco_ridge_vs_hgbr.png", "GENERATED",
           "data/ml/ridge_loco.parquet + hgbr_loco.parquet + summaries",
           "near-tie; ridge better worst-case max")


# ============================================================================
# F5 — V_OCV(SOC)
# ============================================================================
def fig05_ocv(out: Path) -> None:
    if not OCV_P.exists():
        record("F5 ocv_curve.png", "SKIPPED", str(OCV_P), "OCV table not found")
        return
    df = pd.read_parquet(OCV_P)
    soc_col = "SOC" if "SOC" in df.columns else "soc"
    v_col   = "V_ocv" if "V_ocv" in df.columns else "v_ocv"
    soc = df[soc_col].to_numpy() * 100  # to percent for the x-axis
    v   = df[v_col].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(soc, v, lw=1.6, color=COLORS["model"])
    ax.set_xlabel("State of Charge (%)")
    ax.set_ylabel("V_ocv (V)")
    ax.set_title("Open-circuit voltage curve (LG HG2 @ 25 °C, from C/20 characterization)",
                 fontsize=11)
    ax.set_xlim(0, 100)
    fig.savefig(out)
    plt.close(fig)
    record("F5 ocv_curve.png", "GENERATED",
           "data/processed/ocv_lg_25degC.parquet", "")


# ============================================================================
# F6 — Surface fit (modeled vs measured) on Mixed1 @ 25 °C
# ============================================================================
def fig06_surface_fit(out: Path) -> None:
    p = find_labeled("Mixed1")
    if p is None:
        record("F6 calibration_surface_fit.png", "SKIPPED",
               "LG_25degC_Mixed1_*.parquet", "labeled Mixed1 @ 25 °C not found")
        return
    df = pd.read_parquet(p)
    t_min = df["time_s"].to_numpy() / 60.0
    ts_meas = df["T_surface_C"].to_numpy()
    ts_mod  = df["T_surf_model_C"].to_numpy()
    err = ts_mod - ts_meas
    rmse = float(np.sqrt(np.mean(err ** 2)))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(t_min, ts_meas, lw=1.1, color=COLORS["measured_surface"],
            label="measured T_surface")
    ax.plot(t_min, ts_mod, lw=1.4, color=COLORS["model"],
            label="modeled T_surface (central params)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Mixed1 @ 25 °C — modeled vs measured surface temperature "
                 f"(surface RMSE = {rmse:.3f} °C)", fontsize=11)
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    record("F6 calibration_surface_fit.png", "GENERATED",
           "data/labeled/LG_25degC_Mixed1_*.parquet",
           f"surface RMSE {rmse:.3f} °C")


# ============================================================================
# F7 — Ablation bars (full vs electrical-only; ridge vs HGBR)
# ============================================================================
def fig07_ablation(out: Path, ridge_summary: dict, hgbr_summary: dict) -> None:
    r_full   = ridge_summary["fixed_split"]["overall"]
    r_drop   = ridge_summary["ablation_drop_surface"]["overall"]
    h_full   = hgbr_summary["fixed_split"]["overall"]
    h_drop   = hgbr_summary["ablation_drop_surface"]["overall"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    labels = ["ridge\n(full)", "ridge\n(electrical-only)",
              "HGBR\n(full)",  "HGBR\n(electrical-only)"]
    cols = [COLORS["model"], COLORS["baseline_nominal"],
            COLORS["band_fill"], COLORS["baseline_tinf"]]

    rmse_vals = [r_full["rmse"], r_drop["rmse"], h_full["rmse"], h_drop["rmse"]]
    bars1 = ax1.bar(labels, rmse_vals, color=cols,
                    edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("fixed-split test RMSE (°C)")
    ax1.set_title("Surface-feature ablation — RMSE on TEST = Mixed1 + Mixed5",
                  fontsize=10)

    max_vals = [r_full["max"], r_drop["max"], h_full["max"], h_drop["max"]]
    bars2 = ax2.bar(labels, max_vals, color=cols,
                    edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars2, max_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.04, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("fixed-split test max-abs-err (°C)")
    ax2.set_title("Same split — max-abs-err  (ridge tighter WITH surface; HGBR tighter WITHOUT)",
                  fontsize=10)

    fig.suptitle(
        "Surface features help both models; ridge has the tighter tail "
        "WITH surface, HGBR edges ahead WITHOUT.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    record("F7 ablation_bars.png", "GENERATED",
           "data/ml/ridge_summary.json + hgbr_summary.json",
           "fixed-split RMSE + max-abs-err")


# ============================================================================
# Captions + manifest output
# ============================================================================
CAPTIONS: dict[str, str] = {
    "F1 two_state_schematic.png":
        "Two-state lumped thermal model used to generate the core-temperature "
        "labels. Core node (heat capacity C_core = 40.10 J/K, model-derived "
        "T_core) and surface node (C_surf = 3.10 J/K, measured T_surface) are "
        "coupled by R_cs = 3.39 K/W (R_cs band [1.70, 8.49] K/W carries the "
        "physical uncertainty); the surface dissipates to ambient (T_inf = "
        "23.15 °C at 25 °C) via R_sa = 5.21 K/W. Q = I·(V − V_OCV) injects "
        "irreversible heat at the core node. Core temperature is model-derived "
        "throughout this report — never measured. LOCKED 2026-06-18b.",
    "F2 rcs_band_sweep.png":
        "Model-derived core temperature on US06 @ 25 °C with the R_cs label-"
        "uncertainty band (R_cs ∈ [1.70, 8.49] K/W) shaded. Central line is the "
        "core trajectory at R_cs = 3.39 K/W (the geometric mid-band). The "
        "pipeline-wide R_cs uncertainty translates to a ~5 °C core-temperature "
        "spread; we show it here on US06 because that's where the band is "
        "widest (5.05 °C peak). It is the dominant physical uncertainty in the "
        "whole pipeline. Measured T_surface shown dashed for reference. US06 "
        "is the most aggressive 25 °C cycle and a thermal calibration cycle "
        "(see report-notes on US06 trustworthiness).",
    "F3 severity_validation_bars.png":
        "Surface-temperature RMSE of the calibrated thermal model on five "
        "held-out validation cycles, compared to two trivial baselines: "
        "(i) predict T_surface = nominal chamber setpoint; (ii) predict "
        "T_surface = T_inf (data-driven cooling-tail asymptote). Cycle labels "
        "are color-coded by ambient regime: green = in-regime (25 °C, where the "
        "model was calibrated), amber = inconclusive (10 °C, mild excursion), "
        "red = out-of-regime (0 °C, cold-ambient extrapolation where the model "
        "breaks down). Peak excursion |T_s − T_inf| is annotated per group — "
        "it is the dynamic signal that was available to win. Note the cold-"
        "ambient (0 °C) failure: US06 @ 0 °C produces a model RMSE of 3.61 °C "
        "vs. baseline 1.99 °C. LOCKED 2026-06-18b.",
    "F4 loco_ridge_vs_hgbr.png":
        "Leave-one-cycle-out ML RMSE on model-derived T_core labels for the 11 "
        "LG 25 °C cycles: ridge vs HGBR. Bars are per-cycle held-out RMSE; "
        "cycles sorted by peak excursion (US06 first). Pooled values "
        "annotated in the title (ridge 0.419 vs HGBR 0.412 °C; max 1.86 vs "
        "1.99 °C). Dashed reference line shows the mean per-cycle 'predict "
        "T_core = T_inf' baseline. The two ML models produce statistically "
        "indistinguishable held-out error — the surrogate map is effectively "
        "linear in this regime. Ridge has the better worst-case tail.",
    "F5 ocv_curve.png":
        "Open-circuit voltage V_OCV(SOC) for the LG HG2 cell, extracted from "
        "the C/20 characterization discharge at 25 °C. This lookup is used in "
        "the heat-generation term Q = I · (V − V_OCV) that drives the thermal "
        "model.",
    "F6 calibration_surface_fit.png":
        "Mixed1 @ 25 °C — modeled vs measured surface temperature using the "
        "locked 2026-06-18b central parameters (the same simulation that "
        "generated the core labels). Surface RMSE annotated. This is one of "
        "the two thermal-validation cycles whose pipeline-value claims are "
        "credible (see report-notes); modeled and measured track closely on "
        "in-regime data.",
    "F7 ablation_bars.png":
        "Surface-feature ablation on the fixed dev split (TEST = Mixed1 + "
        "Mixed5). Left: RMSE; right: max-abs-error. Each panel: ridge full, "
        "ridge electrical-only (no surface features), HGBR full, HGBR "
        "electrical-only. With full features ridge has the better worst-case "
        "tail (max-abs: ridge 1.35 vs HGBR 1.63 °C). Strip the surface features "
        "and HGBR becomes better on BOTH RMSE (0.696 vs ridge's 0.890 °C) AND "
        "max-abs (4.32 vs ridge's 4.39 °C) — surface signals are doing the work "
        "that keeps ridge's tail tighter; without them the tree model edges "
        "ahead. This is exactly the regime a sensorless (electrical-only) "
        "deployment target would land in, and is therefore a documented "
        "reopener for the model choice (see decisions/model-selection.md "
        "\"What would make us revisit\"). LOCKED 2026-06-18b.",
    "F8 cold_residual_drift.png":
        "BLOCKED — no parquet of (modeled − measured) surface residuals saved; "
        "the prior calibration plot exists in data/eda_plots/residual_diag_LG_0degC_US06_*.png "
        "but is not in the publication style and was generated against the "
        "pipeline at run-time. Per task spec we do not re-run the pipeline to "
        "regenerate. Cited in report-notes as the qualitative 'heat-accumulation' "
        "result.",
}


def write_captions(out: Path) -> None:
    lines = ["# Publication figure captions (Week-4)",
             "",
             "Each entry is the caption for the matching file in this directory.",
             "Captions feed the writeup verbatim. Numbers are 2026-06-18b LOCKED.",
             ""]
    for k in sorted(CAPTIONS.keys()):
        lines.append(f"## {k}")
        lines.append("")
        lines.append(CAPTIONS[k])
        lines.append("")
    out.write_text("\n".join(lines))


def print_manifest() -> None:
    print("\n" + "=" * 72)
    print("FINAL MANIFEST")
    print("=" * 72)
    for row in manifest:
        print(f"  {row['figure']:42s}  {row['status']}")
        print(f"      source: {row['source']}")
        if row["note"]:
            print(f"      note:   {row['note']}")


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)

    calib, idx, ridge_loco, hgbr_loco = inventory_print()
    ridge_summary = json.load(open(ML_DIR / "ridge_summary.json"))
    hgbr_summary  = json.load(open(ML_DIR / "hgbr_summary.json"))

    # Cross-check F3 LOCKED transcription against step6 (sanity).
    print("\n=== F3 cross-check (transcribed LOCKED vs step6 JSON) ===")
    step6 = {row["file_id"]: row for row in calib.get("step6_validation_per_cycle", [])}
    for r in F3_LOCKED:
        # Find matching JSON entry by short label
        short = r["label"].split(" @ ")[0]
        amb = r["label"].split(" @ ")[1].split(" ")[0]
        match = None
        for fid, row in step6.items():
            if (short in fid) and (f"_{amb}degC_" in fid or f"_n{amb}degC_" in fid):
                match = row
                break
        if not match:
            print(f"  ⚠ no step6 match for {r['label']}")
            continue
        ok_m  = abs(match["model_rmse_C"]            - r["model"])   < 0.01
        ok_n  = abs(match["baseline_nominal_rmse_C"] - r["nominal"]) < 0.01
        ok_t  = abs(match["baseline_T_inf_rmse_C"]   - r["tinf"])    < 0.01
        ok_x  = abs(match["peak_excursion_T_s_from_T_inf_C"] - r["excursion"]) < 0.02
        ok = ok_m and ok_n and ok_t and ok_x
        print(f"  {r['label']:18s}  match={ok}  "
              f"(model {match['model_rmse_C']:.3f}/{r['model']:.3f}; "
              f"nominal {match['baseline_nominal_rmse_C']:.3f}/{r['nominal']:.3f}; "
              f"tinf {match['baseline_T_inf_rmse_C']:.3f}/{r['tinf']:.3f}; "
              f"excur {match['peak_excursion_T_s_from_T_inf_C']:.3f}/{r['excursion']:.3f})")

    # Render
    print("\n=== Rendering figures ===")
    fig01_two_state(FIGS / "F1_two_state_schematic.png")
    fig02_band(FIGS / "F2_rcs_band_sweep.png")
    fig03_severity(FIGS / "F3_severity_validation_bars.png")
    fig04_loco(FIGS / "F4_loco_ridge_vs_hgbr.png",
               ridge_loco, hgbr_loco, ridge_summary, hgbr_summary)
    fig05_ocv(FIGS / "F5_ocv_curve.png")
    fig06_surface_fit(FIGS / "F6_calibration_surface_fit.png")
    fig07_ablation(FIGS / "F7_ablation_bars.png", ridge_summary, hgbr_summary)
    # F8 BLOCKED — recorded in manifest only
    record("F8 cold_residual_drift.png", "SKIPPED (BLOCKED)",
           "no parquet of residuals",
           "PNG exists in data/eda_plots/ from prior run but is not in "
           "publication style; spec forbids re-running the pipeline")

    write_captions(FIGS / "_captions.md")
    print_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
