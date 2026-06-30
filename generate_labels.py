"""
generate_labels.py — Week-3 label generation: T_core central + R_cs band.

Produces the modeled core-temperature labels (and the R_cs uncertainty band)
that the Week-3 ML stage will consume. Uses the LOCKED 2026-06-18b
calibration parameters; **this script does NO re-fitting** — the Step-5 band
splits are loaded from the saved calibration summary, and only regenerated
if the summary lacks them (and even then only the split-only Step-5 refit
runs, R_sa and T_inf are untouched).

Scope (locked): 11 LG 25 °C drive cycles — US06, LA92, UDDS, Mixed1..8.
The 25 °C HWFET file (dead — no current activity) and every Charge* file
are excluded. We deliberately stay inside the calibration regime per the
2026-06-18b "Implication for the Week-3 ML stage" note in
`knowledge/decisions/calibration-constrained-fit.md`.

Output (NEW dir; does NOT touch data/processed/):
    data/labeled/<file_id>.parquet     — one Parquet per cycle.
                                          Measured columns carried through,
                                          plus T_core_model_C (central — the
                                          label), T_core_model_lo_C and
                                          T_core_model_hi_C (R_cs band) and
                                          T_surf_model_C (sanity residual).
    data/labeled/_labels_index.parquet — per-cycle summary stats and flags.

Run:  python generate_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_model import ThermalParams
from calibrate import (
    precompute_cycle, simulate_T,
    CAL_RESULTS_JSON, C_TOTAL_J_K,
    _residuals_split_only,
)


# ============================================================================
# LOCKED 2026-06-18b calibration parameters — read-only here.
# ============================================================================
T_INF_25C_C            = 23.15   # cooling-tail asymptote at 25 °C ambient
R_SA_K_PER_W           = 5.21    # convective resistance, re-anchored
R_CS_CENTRAL_K_PER_W   = 3.394   # geometric central (k_radial = 0.5 W/(m·K))
R_CS_LOW_K_PER_W       = 1.697   # geometric high-k = 1.0 W/(m·K)
R_CS_HIGH_K_PER_W      = 8.486   # geometric low-k  = 0.2 W/(m·K)


# ============================================================================
# Paths and cycle selection
# ============================================================================
PROCESSED_DIR = Path("data/processed")
LABELED_DIR   = Path("data/labeled")
LABELS_INDEX  = LABELED_DIR / "_labels_index.parquet"

# 11 LG 25 °C drive cycles, ordered for predictable output.
CYCLE_GLOBS = [
    "LG_25degC_US06_*.parquet",
    "LG_25degC_LA92_*.parquet",
    "LG_25degC_UDDS_*.parquet",
    "LG_25degC_Mixed1_*.parquet",
    "LG_25degC_Mixed2_*.parquet",
    "LG_25degC_Mixed3_*.parquet",
    "LG_25degC_Mixed4_*.parquet",
    "LG_25degC_Mixed5_*.parquet",
    "LG_25degC_Mixed6_*.parquet",
    "LG_25degC_Mixed7_*.parquet",
    "LG_25degC_Mixed8_*.parquet",
]

# Carried through unchanged from the processed Parquet.
MEASURED_COLS = [
    "time_s", "voltage_V", "current_A", "power_W",
    "T_surface_C", "T_ambient_C", "Ah", "Wh",
    "dataset", "cycle", "ambient_setpoint_C", "file_id",
]

GAP_FLAG_THRESHOLD_C = 3.0   # central core-surface gap > this -> flag


# ============================================================================
# (R_cs, split) triples — loaded from saved calibration summary
# ============================================================================
def load_band_splits() -> dict[str, dict]:
    """Read the Step-5 (R_cs, fitted-split) triples from the calibrate.py JSON.

    Returns a dict keyed by 'low' / 'central' / 'high', each value
    {'R_cs': float, 'split': float, 'source': 'summary'|'regenerated'}.
    If the saved R_cs does not match the expected band point (or the entry
    is missing), the Step-5 split refit is regenerated AT THAT R_cs ONLY;
    R_sa and T_inf are untouched.
    """
    expected = {
        "low":     R_CS_LOW_K_PER_W,
        "central": R_CS_CENTRAL_K_PER_W,
        "high":    R_CS_HIGH_K_PER_W,
    }
    saved = {}
    if CAL_RESULTS_JSON.exists():
        with open(CAL_RESULTS_JSON) as f:
            saved = json.load(f).get("step5_band_per_point", {})

    out: dict[str, dict] = {}
    for name, R_cs_want in expected.items():
        rec = saved.get(name)
        if rec is not None and abs(float(rec["R_cs_K_per_W"]) - R_cs_want) < 0.1:
            out[name] = {
                "R_cs": float(rec["R_cs_K_per_W"]),
                "split": float(rec["split"]),
                "source": "summary",
            }
        else:
            # Fall back: regenerate ONLY the Step-5 split refit at the
            # expected R_cs. R_sa and T_inf stay locked.
            print(f"  band split for '{name}' not in saved summary at "
                  f"R_cs={R_cs_want:.3f} — regenerating split-only refit ...")
            split = regenerate_split_at_R_cs(R_cs_want)
            out[name] = {"R_cs": R_cs_want, "split": split, "source": "regenerated"}
    return out


def regenerate_split_at_R_cs(R_cs: float) -> float:
    """Step-5 split-only refit at a single R_cs. R_sa, T_inf, C_total fixed."""
    from scipy.optimize import least_squares
    from calibrate import CAL_CYCLE_GLOBS, expand_glob

    cal_paths = [expand_glob(g) for g in CAL_CYCLE_GLOBS]
    cal_caches = [precompute_cycle(p, T_inf_override=T_INF_25C_C) for p in cal_paths]
    res = least_squares(
        _residuals_split_only,
        np.array([0.93]),
        args=(C_TOTAL_J_K, R_SA_K_PER_W, R_cs, cal_caches, 10),
        bounds=([0.50], [0.99]),
        method="trf", max_nfev=80, verbose=0,
        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, diff_step=0.02,
    )
    return float(res.x[0])


def make_params(R_cs: float, split: float) -> ThermalParams:
    return ThermalParams(
        C_core=split * C_TOTAL_J_K,
        C_surf=(1.0 - split) * C_TOTAL_J_K,
        R_cs=R_cs,
        R_sa=R_SA_K_PER_W,
    )


# ============================================================================
# Per-cycle worker
# ============================================================================
def label_one_cycle(path: Path, splits: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """Simulate central + lo + hi band points; return (output_df, summary_row).

    Sanity asserts (will raise on violation):
        time_s monotone non-decreasing
        all modeled values finite, no NaN
        T_core_central >= T_surf_central (within numerical tolerance)
    """
    cache = precompute_cycle(path, soc_init=1.0, T_inf_override=T_INF_25C_C)
    file_id = path.stem

    sims: dict[str, dict] = {}
    for name in ("low", "central", "high"):
        params = make_params(splits[name]["R_cs"], splits[name]["split"])
        t, T_core, T_surf, _Q = simulate_T(
            cache, params, t_eval=cache["t_grid"],
            max_step=5, rtol=1e-6, atol=1e-8,
        )
        sims[name] = {"T_core": T_core, "T_surf": T_surf}

    # Build output frame: measured carried through, model columns appended.
    df_in = cache["df"]
    df_out = df_in[MEASURED_COLS].copy()
    df_out["T_core_model_C"]    = sims["central"]["T_core"]
    df_out["T_core_model_lo_C"] = sims["low"]["T_core"]
    df_out["T_core_model_hi_C"] = sims["high"]["T_core"]
    df_out["T_surf_model_C"]    = sims["central"]["T_surf"]
    df_out["T_inf_used_C"]      = T_INF_25C_C   # constant, explicit

    # Sanity asserts.
    t_arr = df_out["time_s"].to_numpy()
    assert (np.diff(t_arr) >= 0).all(), f"time not monotone: {file_id}"

    for col in ("T_core_model_C", "T_core_model_lo_C", "T_core_model_hi_C", "T_surf_model_C"):
        a = df_out[col].to_numpy()
        assert np.isfinite(a).all(), f"non-finite values in {col}: {file_id}"
        assert not pd.isna(df_out[col]).any(), f"NaN in {col}: {file_id}"

    diff = df_out["T_core_model_C"].to_numpy() - df_out["T_surf_model_C"].to_numpy()
    assert (diff >= -1e-3).all(), \
        f"T_core < T_surf in {file_id}: min(T_core-T_surf) = {diff.min():.4f}"

    # Per-cycle summary stats.
    T_s_meas = cache["T_surf_meas"]
    T_s_central = sims["central"]["T_surf"]
    surf_err = T_s_central - T_s_meas
    central_gap = sims["central"]["T_core"] - sims["central"]["T_surf"]
    band_spread = sims["high"]["T_core"] - sims["low"]["T_core"]
    excursion = float(np.max(np.abs(T_s_meas - T_INF_25C_C)))

    row = {
        "file_id":      file_id,
        "cycle":        str(df_in["cycle"].iloc[0]),
        "n_rows":       int(len(df_out)),
        "duration_min": float(df_out["time_s"].iloc[-1] / 60.0),
        "peak_T_core_model_C":             float(sims["central"]["T_core"].max()),
        "max_core_surf_gap_C":             float(central_gap.max()),
        "band_spread_max_C":               float(band_spread.max()),
        "peak_excursion_T_s_from_T_inf_C": excursion,
        "surface_rmse_C":                  float(np.sqrt(np.mean(surf_err ** 2))),
        "soc_clamp_low":                   int(cache["soc_clamp_low"]),
        "soc_clamp_high":                  int(cache["soc_clamp_high"]),
        "gap_flag":                        bool(central_gap.max() > GAP_FLAG_THRESHOLD_C),
        "out_file":                        path.name,
    }
    return df_out, row


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    print("=" * 72)
    print("Week-3 label generation — LG 25 °C drive cycles only")
    print("LOCKED 2026-06-18b params: NO re-fitting; splits loaded from summary.")
    print("=" * 72)

    paths: list[Path] = []
    for g in CYCLE_GLOBS:
        matches = sorted(PROCESSED_DIR.glob(g))
        if not matches:
            print(f"  ERROR: no match for {g}")
            continue
        if len(matches) > 1:
            print(f"  WARN: {len(matches)} matches for {g}; using {matches[0].name}")
        paths.append(matches[0])
    if len(paths) != 11:
        print(f"FATAL: expected 11 cycles, resolved {len(paths)}")
        return 2
    print(f"  resolved {len(paths)} cycles")

    print("\n  R_cs band (R_cs, split) triples:")
    splits = load_band_splits()
    for name, info in splits.items():
        C_c = info["split"] * C_TOTAL_J_K
        C_s = (1 - info["split"]) * C_TOTAL_J_K
        print(f"    {name:8s} R_cs = {info['R_cs']:.4f} K/W, "
              f"split = {info['split']:.4f}, "
              f"C_core = {C_c:5.2f}, C_surf = {C_s:4.2f} J/K  "
              f"[{info['source']}]")
    print(f"  fixed: C_total = {C_TOTAL_J_K:.2f} J/K, "
          f"R_sa = {R_SA_K_PER_W:.2f} K/W, T_inf(25 °C) = {T_INF_25C_C:.2f} °C")

    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print("\n  per-cycle:")
    for i, path in enumerate(paths, 1):
        df_out, row = label_one_cycle(path, splits)
        out_path = LABELED_DIR / path.name
        df_out.to_parquet(out_path, index=False)
        flag = "  [GAP > 3 °C]" if row["gap_flag"] else ""
        print(f"  ({i:>2}/{len(paths)}) {row['cycle']:7s}  "
              f"n={row['n_rows']:>5d}  "
              f"peak_T_core={row['peak_T_core_model_C']:5.2f} °C  "
              f"gap_max={row['max_core_surf_gap_C']:4.2f}  "
              f"band={row['band_spread_max_C']:4.2f}  "
              f"surf_RMSE={row['surface_rmse_C']:.3f}  "
              f"clamps low/high={row['soc_clamp_low']}/{row['soc_clamp_high']}{flag}")
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_parquet(LABELS_INDEX, index=False)

    cols = [
        "cycle", "duration_min",
        "peak_T_core_model_C", "max_core_surf_gap_C",
        "band_spread_max_C", "peak_excursion_T_s_from_T_inf_C",
        "surface_rmse_C", "soc_clamp_low", "soc_clamp_high", "gap_flag",
    ]
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 72)
    print("Summary table  (one row per cycle):")
    print("=" * 72)
    print(summary[cols].to_string(index=False))

    flagged = summary[summary["gap_flag"]]
    print(f"\n  cycles flagged for central gap > {GAP_FLAG_THRESHOLD_C} °C: "
          f"{len(flagged)}")
    if len(flagged):
        for _, r in flagged.iterrows():
            print(f"    {r['file_id']}: gap_max = {r['max_core_surf_gap_C']:.2f} °C")

    print(f"\n  outputs: {LABELED_DIR}/  "
          f"({len(paths)} cycle Parquets + _labels_index.parquet)")
    print("  sanity asserts (per cycle): time monotone, all finite, no NaN, "
          "T_core >= T_surf — ALL PASSED (this run would have raised otherwise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
