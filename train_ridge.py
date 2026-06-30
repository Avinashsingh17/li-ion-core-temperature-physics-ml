"""
train_ridge.py — Week-3 ridge baseline (first ML model).

Uses the labeled data in `data/labeled/` (produced by `generate_labels.py`)
and the `features_and_split` module. Trains a `StandardScaler + Ridge`
pipeline with alpha tuned by `GroupKFold` over the **train cycles only** —
the held-out cycles of the fixed split AND of each LOCO fold never see
alpha selection (no leakage; see
`knowledge/concepts/blocked-time-series-split.md`).

Reports:
  - FIXED SPLIT (TEST = Mixed1 + Mixed5; train = the other 9)
      overall RMSE/MAE/max + per-cycle table + baselines + margins
      top 10 features by |coefficient| on scaled inputs
  - ABLATION (drop_surface=True; electrical-only)
      same metrics; delta vs full
  - LOCO across all 11 cycles
      per-cycle table with thermal_role, peak |T_s−T_inf|, ML + baselines,
      pooled LOCO RMSE
  - Two honesty checks:
      (1) does ML beat baseline_surface? (the bar that matters)
      (2) does ML beat baseline_tinf, gated by excursion?

Stops after ridge — HGBR is the next step.

Run:  python train_ridge.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features_and_split import (
    COL_TCORE, COL_CYCLE, COL_TSURF, COL_TINF,
    load_labeled_data, build_features,
    tune_and_fit_ridge,
    metric_bundle, per_cycle_metrics,
    baseline_surface, baseline_tinf, peak_excursion,
)


# ============================================================================
# Locked split + thermal-role classification
# ============================================================================
TEST_CYCLE_SHORT_NAMES = ["Mixed1", "Mixed5"]   # FIXED dev split — held-out

# For LOCO reporting: which thermal role each cycle played upstream.
THERMAL_ROLE = {
    "US06":   "cal",   "LA92":   "cal",   "UDDS":   "cal",
    "Mixed1": "val",   "Mixed2": "val",
    "Mixed3": "unused","Mixed4": "unused","Mixed5": "unused",
    "Mixed6": "unused","Mixed7": "unused","Mixed8": "unused",
}

EXCUR_GATE_C = 1.5   # cycles below this excursion are uninformative vs T_inf

OUT_DIR = Path("data/ml")


# ============================================================================
# Helpers
# ============================================================================
def resolve_test_file_ids(df: pd.DataFrame, short_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in short_names:
        matches = df.loc[df["cycle"] == name, COL_CYCLE].unique()
        if len(matches) != 1:
            raise ValueError(f"expected 1 file_id for cycle {name!r}; got {matches}")
        out.append(matches[0])
    return out


def get_top_coefficients(pipeline, feature_names: list[str], k: int = 10
                         ) -> pd.DataFrame:
    """Top-k features by |coef| on scaled inputs (Ridge coefficients)."""
    ridge = pipeline.named_steps["ridge"]
    coefs = ridge.coef_
    out = pd.DataFrame({
        "feature":  feature_names,
        "coef":     coefs,
        "abs_coef": np.abs(coefs),
    })
    return out.sort_values("abs_coef", ascending=False).head(k).reset_index(drop=True)


def evaluate_held(df_held: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
                  ) -> tuple[dict, dict, dict, dict]:
    """Return (ml_metrics, baseline_surface_metrics, baseline_tinf_metrics, margins)."""
    bs = baseline_surface(df_held)
    bt = baseline_tinf(df_held)
    ml_m = metric_bundle(y_true, y_pred)
    bs_m = metric_bundle(y_true, bs)
    bt_m = metric_bundle(y_true, bt)
    margins = {
        "vs_surface_rmse": bs_m["rmse"] - ml_m["rmse"],
        "vs_tinf_rmse":    bt_m["rmse"] - ml_m["rmse"],
    }
    return ml_m, bs_m, bt_m, margins


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    print("=" * 78)
    print("Week-3 ridge baseline (StandardScaler + Ridge; alpha tuned by GroupKFold)")
    print("=" * 78)

    print("\nloading data/labeled/...")
    df_all = load_labeled_data()
    print(f"  loaded {len(df_all)} rows from {df_all[COL_CYCLE].nunique()} cycles")
    cycles_present = sorted(df_all["cycle"].unique())
    print(f"  cycles: {cycles_present}")
    print(f"  schema check: COL_TCORE={COL_TCORE!r}, COL_CYCLE={COL_CYCLE!r}  "
          f"-> both present? "
          f"{COL_TCORE in df_all.columns and COL_CYCLE in df_all.columns}")

    # ------------------------------------------------------------------
    # Build features (full and drop_surface).
    # ------------------------------------------------------------------
    print("\nbuilding features (full = surface included)...")
    feats_full, df_keep_full = build_features(df_all, drop_surface=False)
    n_full = len(feats_full)
    drop_pct_full = 100.0 * (1.0 - n_full / len(df_all))
    print(f"  {feats_full.shape[1]} features, "
          f"{n_full} / {len(df_all)} rows kept "
          f"(warm-up drop {drop_pct_full:.2f}%)")
    feature_names_full = list(feats_full.columns)

    print("\nbuilding features (ablation; drop_surface=True)...")
    feats_drop, df_keep_drop = build_features(df_all, drop_surface=True)
    n_drop = len(feats_drop)
    drop_pct_drop = 100.0 * (1.0 - n_drop / len(df_all))
    print(f"  {feats_drop.shape[1]} features, "
          f"{n_drop} / {len(df_all)} rows kept "
          f"(warm-up drop {drop_pct_drop:.2f}%)")
    feature_names_drop = list(feats_drop.columns)

    # ------------------------------------------------------------------
    # FIXED SPLIT
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FIXED SPLIT  (TEST = Mixed1 + Mixed5;  train = the other 9 cycles)")
    print("=" * 78)
    test_fids = resolve_test_file_ids(df_all, TEST_CYCLE_SHORT_NAMES)
    print("  TEST file_ids:")
    for fid in test_fids:
        print(f"    {fid}")

    def fixed_split_eval(feats: pd.DataFrame, df_keep: pd.DataFrame,
                        feat_names: list[str], label: str
                        ) -> tuple[float, dict, pd.DataFrame, pd.DataFrame]:
        X = feats.to_numpy()
        y = df_keep[COL_TCORE].to_numpy()
        test_mask = df_keep[COL_CYCLE].isin(test_fids).to_numpy()
        train_mask = ~test_mask
        groups_tr = df_keep.loc[train_mask, COL_CYCLE].to_numpy()
        n_train_cycles = int(np.unique(groups_tr).size)
        print(f"\n  [{label}] train rows = {train_mask.sum()} "
              f"({n_train_cycles} cycles); test rows = {test_mask.sum()} "
              f"({df_keep.loc[test_mask, COL_CYCLE].nunique()} cycles)")
        print(f"  [{label}] tuning alpha via GroupKFold over train cycles only ...")
        model, alpha = tune_and_fit_ridge(X[train_mask], y[train_mask], groups_tr)
        print(f"  [{label}] best alpha = {alpha:.4g}")

        y_pred = model.predict(X[test_mask])
        overall = metric_bundle(y[test_mask], y_pred)
        df_test = df_keep.loc[test_mask].reset_index(drop=True)
        per_cyc = per_cycle_metrics(df_test, y[test_mask], y_pred)

        # Add baselines + margins per cycle.
        extra = []
        for _, row in per_cyc.iterrows():
            cyc_mask = (df_test[COL_CYCLE] == row["file_id"]).to_numpy()
            yt = y[test_mask][cyc_mask]
            yh = y_pred[cyc_mask]
            ml_m, bs_m, bt_m, margins = evaluate_held(
                df_test.loc[cyc_mask].reset_index(drop=True), yt, yh)
            extra.append({
                "cycle":             row["cycle"],
                "file_id":           row["file_id"],
                "n":                 int(row["n"]),
                "ml_rmse":           ml_m["rmse"],
                "ml_mae":            ml_m["mae"],
                "ml_max":            ml_m["max"],
                "baseline_surface_rmse": bs_m["rmse"],
                "baseline_tinf_rmse":    bt_m["rmse"],
                "margin_vs_surface": margins["vs_surface_rmse"],
                "margin_vs_tinf":    margins["vs_tinf_rmse"],
            })
        per_cyc_full = pd.DataFrame(extra)
        return alpha, overall, per_cyc_full, model

    # Full features.
    alpha_full, overall_full, per_cyc_full, model_full = fixed_split_eval(
        feats_full, df_keep_full, feature_names_full, "FULL")
    print(f"\n  OVERALL test RMSE/MAE/max:  "
          f"{overall_full['rmse']:.4f} / {overall_full['mae']:.4f} / "
          f"{overall_full['max']:.4f}  °C")
    print("\n  per-cycle (TEST = held-out fixed split):")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(per_cyc_full.to_string(index=False))

    top_coefs = get_top_coefficients(model_full, feature_names_full, k=10)
    print("\n  TOP 10 features by |coef| (scaled inputs):")
    print(top_coefs.to_string(index=False))

    # ------------------------------------------------------------------
    # ABLATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ABLATION  (drop_surface=True;  electrical-only — no T_surface signals)")
    print("=" * 78)
    alpha_drop, overall_drop, per_cyc_drop, model_drop = fixed_split_eval(
        feats_drop, df_keep_drop, feature_names_drop, "DROP_SURFACE")
    print(f"\n  OVERALL ablation test RMSE/MAE/max:  "
          f"{overall_drop['rmse']:.4f} / {overall_drop['mae']:.4f} / "
          f"{overall_drop['max']:.4f}  °C")
    delta_rmse = overall_drop["rmse"] - overall_full["rmse"]
    print(f"  DELTA vs FULL test RMSE = {delta_rmse:+.4f} °C  "
          f"({'worse' if delta_rmse > 0 else 'better'} without surface signals)")
    print("\n  per-cycle (ablation):")
    print(per_cyc_drop.to_string(index=False))

    # ------------------------------------------------------------------
    # LOCO
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LEAVE-ONE-CYCLE-OUT (LOCO) — 11 folds, each tuning alpha on the other 10")
    print("=" * 78)
    excursions = peak_excursion(df_keep_full)
    X_full = feats_full.to_numpy()
    y_full = df_keep_full[COL_TCORE].to_numpy()
    fid_arr = df_keep_full[COL_CYCLE].to_numpy()

    pooled_pred = np.full(len(y_full), np.nan)
    loco_rows = []
    cycles_unique = df_keep_full[COL_CYCLE].unique()
    for i, held_fid in enumerate(cycles_unique, 1):
        held_mask = fid_arr == held_fid
        train_mask = ~held_mask
        Xt, yt = X_full[train_mask], y_full[train_mask]
        gt = fid_arr[train_mask]
        Xh, yh = X_full[held_mask], y_full[held_mask]
        df_held = df_keep_full.loc[held_mask].reset_index(drop=True)
        cyc_short = df_held["cycle"].iloc[0]

        model_i, alpha_i = tune_and_fit_ridge(Xt, yt, gt)
        yp = model_i.predict(Xh)
        pooled_pred[held_mask] = yp

        ml_m, bs_m, bt_m, margins = evaluate_held(df_held, yh, yp)
        role = THERMAL_ROLE.get(cyc_short, "?")
        excur = excursions[held_fid]
        loco_rows.append({
            "cycle":             cyc_short,
            "thermal_role":      role,
            "peak_excursion_C":  excur,
            "alpha":             alpha_i,
            "ml_rmse":           ml_m["rmse"],
            "ml_mae":            ml_m["mae"],
            "ml_max":            ml_m["max"],
            "baseline_surface_rmse": bs_m["rmse"],
            "baseline_tinf_rmse":    bt_m["rmse"],
            "margin_vs_surface": margins["vs_surface_rmse"],
            "margin_vs_tinf":    margins["vs_tinf_rmse"],
        })
        print(f"  ({i:>2}/11) {cyc_short:7s} role={role:6s} "
              f"excur={excur:4.2f}  alpha={alpha_i:.2g}  "
              f"ml_RMSE={ml_m['rmse']:.3f}  "
              f"vs_surf={margins['vs_surface_rmse']:+.3f}  "
              f"vs_tinf={margins['vs_tinf_rmse']:+.3f}")

    loco_df = pd.DataFrame(loco_rows)
    pooled_overall = metric_bundle(y_full, pooled_pred)
    print(f"\n  POOLED LOCO RMSE/MAE/max:  "
          f"{pooled_overall['rmse']:.4f} / {pooled_overall['mae']:.4f} / "
          f"{pooled_overall['max']:.4f}  °C")
    print("\n  LOCO table:")
    print(loco_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Honesty checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Honesty checks")
    print("=" * 78)

    print("\n  (1) Does ML beat baseline_surface? "
          "(predict T_core = T_surface_C)")
    print("      This is the bar that matters: surface is the trivial core proxy.")
    n_wins_surf = int((loco_df["margin_vs_surface"] > 0).sum())
    print(f"      LOCO: ML beats baseline_surface on {n_wins_surf} / 11 cycles.")
    for _, r in loco_df.iterrows():
        win = "WINS " if r["margin_vs_surface"] > 0 else "LOSES"
        print(f"        {r['cycle']:7s} ({r['thermal_role']:6s})  {win}  "
              f"margin = {r['margin_vs_surface']:+.3f} °C")

    print("\n  (2) Does ML beat baseline_tinf, GATED by excursion?")
    print("      Cycles with low peak |T_s − T_inf| have no dynamic signal for ML to win.")
    print(f"      Gating threshold: peak excursion >= {EXCUR_GATE_C} °C.")
    n_eligible = int((loco_df["peak_excursion_C"] >= EXCUR_GATE_C).sum())
    n_wins_tinf = int(((loco_df["peak_excursion_C"] >= EXCUR_GATE_C) &
                       (loco_df["margin_vs_tinf"] > 0)).sum())
    n_low_excur = int((loco_df["peak_excursion_C"] < EXCUR_GATE_C).sum())
    print(f"      eligible cycles: {n_eligible} / 11; "
          f"ML beats baseline_tinf on {n_wins_tinf} / {n_eligible}; "
          f"low-excursion (uninformative): {n_low_excur}")
    for _, r in loco_df.iterrows():
        if r["peak_excursion_C"] < EXCUR_GATE_C:
            verdict = (f"LOW EXCURSION ({r['peak_excursion_C']:.2f} °C) — "
                       f"uninformative")
        elif r["margin_vs_tinf"] > 0:
            verdict = f"WINS  margin = {r['margin_vs_tinf']:+.3f} °C"
        else:
            verdict = f"LOSES margin = {r['margin_vs_tinf']:+.3f} °C"
        print(f"        {r['cycle']:7s} ({r['thermal_role']:6s})  "
              f"excur={r['peak_excursion_C']:4.2f}  {verdict}")

    # Pipeline-value annotations.
    print("\n  Notes on what these LOCO numbers can support:")
    print("    - Pipeline-value claims are credible only on VAL cycles "
          "(Mixed1, Mixed2) —")
    print("      the cal cycles (US06, LA92, UDDS) trained the thermal model "
          "that made the labels.")
    print("    - US06 is FLAGGED (thermal-cal + widest band 5.05 + worst surface "
          "fit 1.046 °C).")
    print("      Treat its LOCO numbers as a stress test, not as headline "
          "generalization.")

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cyc_full.to_parquet(OUT_DIR / "ridge_fixed_per_cycle.parquet", index=False)
    per_cyc_drop.to_parquet(OUT_DIR / "ridge_ablation_per_cycle.parquet", index=False)
    loco_df.to_parquet(OUT_DIR / "ridge_loco.parquet", index=False)
    top_coefs.to_parquet(OUT_DIR / "ridge_top_coefficients.parquet", index=False)

    summary = {
        "fixed_split": {
            "test_file_ids":  test_fids,
            "n_features":     feats_full.shape[1],
            "warmup_drop_pct": drop_pct_full,
            "best_alpha":     alpha_full,
            "overall":        overall_full,
        },
        "ablation_drop_surface": {
            "n_features":     feats_drop.shape[1],
            "warmup_drop_pct": drop_pct_drop,
            "best_alpha":     alpha_drop,
            "overall":        overall_drop,
            "delta_rmse_vs_full": delta_rmse,
        },
        "loco": {
            "n_folds":          11,
            "pooled":           pooled_overall,
            "n_wins_vs_surface": n_wins_surf,
            "n_wins_vs_tinf_gated": n_wins_tinf,
            "n_low_excursion":  n_low_excur,
            "excur_gate_C":     EXCUR_GATE_C,
        },
    }
    (OUT_DIR / "ridge_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n  outputs saved to: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
