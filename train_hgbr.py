"""
train_hgbr.py — Week-3 HGBR (HistGradientBoostingRegressor) main ML model.

Head-to-head with the corrected ridge baseline (`train_ridge.py`, v2 24/17
features). Reuses `features_and_split.py` unchanged. HGBR alone — no scaler,
because trees are scale-invariant; no `StandardScaler` constant-feature
warnings either.

Hyperparameter selection is GROUP-HONEST. sklearn HGBR's `early_stopping=True`
uses `validation_fraction` to carve a random *within*-cycle val set — that
leaks across the cycle boundary. Two options were considered:
  (A) Carve early-stopping val from whole train cycles via GroupKFold and
      pass an explicit `eval_set`. HGBR has no `eval_set` parameter.
  (B) Treat `max_iter` as a hyperparameter inside the grid, with
      `early_stopping=False` everywhere, so iteration count is selected
      by the same group-honest GroupKFold that picks the other params.

We go with (B) — fully group-honest. `max_iter ∈ {300, 600}` covers the
range where rates of {0.03, 0.05, 0.1} plateau.

The selected config is then reused across all LOCO folds — mild
hyperparameter leak for folds whose held-out cycle was among the 9 fixed-
split train cycles. Acceptable for a baseline; per-fold retuning would be
the stricter alternative.

Reports:
  - FIXED SPLIT (TEST = Mixed1 + Mixed5): overall + per-cycle + baselines
    (`baseline_surface`, `baseline_tinf` at 23.15 °C, `peak_excursion`) +
    ablation delta.
  - LOCO (11 folds, reusing the tuned config).
  - RIDGE-vs-HGBR head-to-head (reads `data/ml/ridge_*.parquet`, no re-run).
  - Permutation importance on held-out fixed-split test rows (group-disjoint),
    top 10.

Run:  python train_hgbr.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from features_and_split import (
    COL_TCORE, COL_CYCLE, COL_TSURF, COL_TINF,
    load_labeled_data, build_features,
    grouped_cv_folds,
    metric_bundle, per_cycle_metrics,
    baseline_surface, baseline_tinf, peak_excursion,
)


# ============================================================================
# Locked split + thermal-role classification (same as ridge)
# ============================================================================
TEST_CYCLE_SHORT_NAMES = ["Mixed1", "Mixed5"]
THERMAL_ROLE = {
    "US06":   "cal", "LA92":   "cal", "UDDS":   "cal",
    "Mixed1": "val", "Mixed2": "val",
    "Mixed3": "unused", "Mixed4": "unused", "Mixed5": "unused",
    "Mixed6": "unused", "Mixed7": "unused", "Mixed8": "unused",
}
EXCUR_GATE_C = 1.5
HIGH_EXCURSION_C = 2.5    # for ridge-vs-HGBR "aggressive cycles" stratification

OUT_DIR = Path("data/ml")


# ============================================================================
# Helpers
# ============================================================================
def resolve_test_file_ids(df: pd.DataFrame, short_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in short_names:
        matches = df.loc[df["cycle"] == name, COL_CYCLE].unique()
        if len(matches) != 1:
            raise ValueError(f"expected 1 file_id for {name!r}, got {matches}")
        out.append(matches[0])
    return out


def tune_and_fit_hgbr(X: np.ndarray, y: np.ndarray, groups: np.ndarray
                      ) -> tuple[HistGradientBoostingRegressor, dict]:
    """HGBR with group-honest hyperparameter selection.

    `max_iter` is part of the grid (not selected by HGBR's random
    `validation_fraction` — that would leak across cycle boundaries).
    `early_stopping=False` everywhere; `l2_regularization = 1.0` fixed.
    """
    base = HistGradientBoostingRegressor(
        early_stopping=False,
        l2_regularization=1.0,
        random_state=0,
    )
    grid = {
        "learning_rate":    [0.03, 0.05, 0.1],
        "max_leaf_nodes":   [15, 31],
        "min_samples_leaf": [100, 300],
        "max_iter":         [300, 600],
    }
    cv = grouped_cv_folds(groups)
    gs = GridSearchCV(
        base, grid, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, refit=True,
    )
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, dict(gs.best_params_)


def evaluate_held(df_held, y_true, y_pred):
    bs = baseline_surface(df_held)
    bt = baseline_tinf(df_held)
    ml_m = metric_bundle(y_true, y_pred)
    bs_m = metric_bundle(y_true, bs)
    bt_m = metric_bundle(y_true, bt)
    return ml_m, bs_m, bt_m, {
        "vs_surface_rmse": bs_m["rmse"] - ml_m["rmse"],
        "vs_tinf_rmse":    bt_m["rmse"] - ml_m["rmse"],
    }


def fixed_split_eval(feats, df_keep, test_fids, label):
    X = feats.to_numpy()
    y = df_keep[COL_TCORE].to_numpy()
    test_mask = df_keep[COL_CYCLE].isin(test_fids).to_numpy()
    train_mask = ~test_mask
    groups_tr = df_keep.loc[train_mask, COL_CYCLE].to_numpy()

    print(f"\n  [{label}] train rows = {train_mask.sum()} "
          f"({np.unique(groups_tr).size} cycles); test rows = {test_mask.sum()}")
    print(f"  [{label}] tuning HGBR via GroupKFold on TRAIN cycles only ...")
    model, best_params = tune_and_fit_hgbr(X[train_mask], y[train_mask], groups_tr)
    print(f"  [{label}] best params: {best_params}")

    y_pred = model.predict(X[test_mask])
    overall = metric_bundle(y[test_mask], y_pred)
    df_test = df_keep.loc[test_mask].reset_index(drop=True)
    per_cyc = per_cycle_metrics(df_test, y[test_mask], y_pred)

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
    return best_params, overall, pd.DataFrame(extra), model, test_mask, y_pred


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    print("=" * 78)
    print("Week-3 HGBR model — head-to-head with corrected ridge baseline")
    print("HGBR alone, no scaler; max_iter inside the grid (group-honest).")
    print("=" * 78)

    df_all = load_labeled_data()
    print(f"\n  loaded {len(df_all)} rows from {df_all[COL_CYCLE].nunique()} cycles")

    feats_full, df_keep_full = build_features(df_all, drop_surface=False)
    feats_drop, df_keep_drop = build_features(df_all, drop_surface=True)
    feature_names_full = list(feats_full.columns)
    print(f"  features: full = {len(feature_names_full)}; "
          f"drop_surface = {feats_drop.shape[1]}")

    test_fids = resolve_test_file_ids(df_all, TEST_CYCLE_SHORT_NAMES)

    # --------------------------------------------------------------
    # Fixed split — full
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FIXED SPLIT  (TEST = Mixed1 + Mixed5)")
    print("=" * 78)
    best_full, overall_full, per_cyc_full, model_full, test_mask_full, y_pred_full = \
        fixed_split_eval(feats_full, df_keep_full, test_fids, "FULL")
    print(f"\n  OVERALL test RMSE/MAE/max: "
          f"{overall_full['rmse']:.4f} / {overall_full['mae']:.4f} / "
          f"{overall_full['max']:.4f}  °C")
    print("\n  per-cycle:")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(per_cyc_full.to_string(index=False))

    # --------------------------------------------------------------
    # Ablation
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ABLATION  (drop_surface=True; electrical-only)")
    print("=" * 78)
    best_drop, overall_drop, per_cyc_drop, model_drop, _, _ = \
        fixed_split_eval(feats_drop, df_keep_drop, test_fids, "DROP_SURFACE")
    print(f"\n  OVERALL ablation test RMSE/MAE/max: "
          f"{overall_drop['rmse']:.4f} / {overall_drop['mae']:.4f} / "
          f"{overall_drop['max']:.4f}  °C")
    hgbr_ablation_delta = overall_drop["rmse"] - overall_full["rmse"]
    print(f"  DELTA vs FULL test RMSE = {hgbr_ablation_delta:+.4f} °C "
          f"({'worse' if hgbr_ablation_delta > 0 else 'better'} without surface)")
    print("\n  per-cycle (ablation):")
    print(per_cyc_drop.to_string(index=False))

    # --------------------------------------------------------------
    # LOCO — reuse the FULL config
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LOCO (11 folds, reusing the FULL tuned config)")
    print("=" * 78)
    print(f"  config: {best_full}")
    print("  caveat: config was selected on the fixed-split 9 train cycles,")
    print("  so for LOCO folds whose held-out cycle was among those 9, there's")
    print("  a mild hyperparameter leak. Per-fold retuning is the stricter")
    print("  alternative.")

    excursions = peak_excursion(df_keep_full)
    X_full = feats_full.to_numpy()
    y_full = df_keep_full[COL_TCORE].to_numpy()
    fid_arr = df_keep_full[COL_CYCLE].to_numpy()

    pooled_pred = np.full(len(y_full), np.nan)
    loco_rows: list[dict] = []
    cycles_unique = df_keep_full[COL_CYCLE].unique()
    for i, held_fid in enumerate(cycles_unique, 1):
        held_mask = fid_arr == held_fid
        train_mask = ~held_mask
        Xt, yt = X_full[train_mask], y_full[train_mask]
        Xh, yh = X_full[held_mask], y_full[held_mask]
        df_held = df_keep_full.loc[held_mask].reset_index(drop=True)
        cyc_short = df_held["cycle"].iloc[0]

        model_i = HistGradientBoostingRegressor(
            **best_full,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        model_i.fit(Xt, yt)
        yp = model_i.predict(Xh)
        pooled_pred[held_mask] = yp

        ml_m, bs_m, bt_m, margins = evaluate_held(df_held, yh, yp)
        role = THERMAL_ROLE.get(cyc_short, "?")
        excur = excursions[held_fid]
        loco_rows.append({
            "cycle":             cyc_short,
            "thermal_role":      role,
            "peak_excursion_C":  excur,
            "ml_rmse":           ml_m["rmse"],
            "ml_mae":            ml_m["mae"],
            "ml_max":            ml_m["max"],
            "baseline_surface_rmse": bs_m["rmse"],
            "baseline_tinf_rmse":    bt_m["rmse"],
            "margin_vs_surface": margins["vs_surface_rmse"],
            "margin_vs_tinf":    margins["vs_tinf_rmse"],
        })
        print(f"  ({i:>2}/11) {cyc_short:7s} role={role:6s} "
              f"excur={excur:4.2f}  "
              f"ml_RMSE={ml_m['rmse']:.3f}  "
              f"vs_surf={margins['vs_surface_rmse']:+.3f}  "
              f"vs_tinf={margins['vs_tinf_rmse']:+.3f}")

    loco_df = pd.DataFrame(loco_rows)
    pooled_overall = metric_bundle(y_full, pooled_pred)
    print(f"\n  POOLED LOCO RMSE/MAE/max: "
          f"{pooled_overall['rmse']:.4f} / {pooled_overall['mae']:.4f} / "
          f"{pooled_overall['max']:.4f}  °C")
    print("\n  LOCO table:")
    print(loco_df.to_string(index=False))

    # --------------------------------------------------------------
    # Ridge-vs-HGBR head-to-head
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RIDGE-vs-HGBR head-to-head  (ridge numbers read from data/ml/, no re-run)")
    print("=" * 78)
    ridge_loco = pd.read_parquet(OUT_DIR / "ridge_loco.parquet")
    ridge_summary = json.loads((OUT_DIR / "ridge_summary.json").read_text())
    ridge_pooled = ridge_summary["loco"]["pooled"]
    ridge_ablation_delta = ridge_summary["ablation_drop_surface"]["delta_rmse_vs_full"]

    cmp_pooled = pd.DataFrame([
        {"model": "ridge (24 feats v2)",
         "pooled_rmse": ridge_pooled["rmse"],
         "pooled_mae":  ridge_pooled["mae"],
         "pooled_max":  ridge_pooled["max"]},
        {"model": "HGBR",
         "pooled_rmse": pooled_overall["rmse"],
         "pooled_mae":  pooled_overall["mae"],
         "pooled_max":  pooled_overall["max"]},
    ])
    print("\n  Pooled LOCO:")
    print(cmp_pooled.to_string(index=False))

    print(f"\n  Fixed-split ablation delta (full − electrical-only, °C):")
    print(f"    ridge:  {ridge_ablation_delta:+.4f}")
    print(f"    HGBR:   {hgbr_ablation_delta:+.4f}")
    delta_of_delta = hgbr_ablation_delta - ridge_ablation_delta
    direction = "shrinks" if abs(hgbr_ablation_delta) < abs(ridge_ablation_delta) else "widens"
    print(f"    change: {delta_of_delta:+.4f}  ({direction} with HGBR)")

    ridge_by_cycle = ridge_loco.set_index("cycle")["ml_rmse"].rename("ridge_rmse")
    hgbr_by_cycle = loco_df.set_index("cycle")["ml_rmse"].rename("hgbr_rmse")
    side = pd.concat([ridge_by_cycle, hgbr_by_cycle], axis=1)
    side["delta_hgbr_minus_ridge"] = side["hgbr_rmse"] - side["ridge_rmse"]
    side["excursion_C"] = loco_df.set_index("cycle")["peak_excursion_C"]
    side["role"] = loco_df.set_index("cycle")["thermal_role"]
    side = side.reset_index().sort_values("excursion_C", ascending=False)
    print("\n  Per-cycle LOCO ML RMSE — ridge vs HGBR (sorted by excursion):")
    print(side.to_string(index=False))

    # Head-to-head answers
    print("\n  Head-to-head answers:")
    high = side[side["excursion_C"] >= HIGH_EXCURSION_C]
    n_better_high = int((high["delta_hgbr_minus_ridge"] < 0).sum())
    print(f"  (1) High-excursion cycles (peak >= {HIGH_EXCURSION_C} °C; "
          f"{len(high)} cycles incl. US06 stress):")
    print(f"      HGBR beats ridge on {n_better_high}/{len(high)}; "
          f"mean ridge={high['ridge_rmse'].mean():.4f}, "
          f"mean HGBR={high['hgbr_rmse'].mean():.4f}, "
          f"mean delta={high['delta_hgbr_minus_ridge'].mean():+.4f} °C.")
    print(f"  (2) Worst-case pooled LOCO max:")
    print(f"      ridge {ridge_pooled['max']:.4f} -> HGBR {pooled_overall['max']:.4f}  "
          f"(change {pooled_overall['max'] - ridge_pooled['max']:+.4f})")
    low_cycles = ["UDDS", "LA92"]
    print(f"  (3) Low-excursion ties (UDDS, LA92):")
    for cyc in low_cycles:
        row = side[side["cycle"] == cyc].iloc[0]
        print(f"      {cyc}: ridge={row['ridge_rmse']:.3f}  "
              f"HGBR={row['hgbr_rmse']:.3f}  "
              f"delta={row['delta_hgbr_minus_ridge']:+.3f}")
    print(f"  (4) Electrical-only ablation gap (full vs drop_surface, fixed-split RMSE):")
    print(f"      ridge {ridge_ablation_delta:+.4f}  ->  "
          f"HGBR {hgbr_ablation_delta:+.4f}  "
          f"(change {delta_of_delta:+.4f}; {direction})")

    # --------------------------------------------------------------
    # Permutation importance (held-out fixed-split test rows; group-disjoint)
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Permutation importance (held-out fixed-split test rows, group-disjoint)")
    print("=" * 78)
    print("  n_repeats=5, scoring=neg_root_mean_squared_error...")
    X_test = X_full[test_mask_full]
    y_test = y_full[test_mask_full]
    perm = permutation_importance(
        model_full, X_test, y_test,
        n_repeats=5, random_state=0, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    imp = pd.DataFrame({
        "feature":         feature_names_full,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=False).head(10).reset_index(drop=True)
    print("\n  Top 10 features by permutation importance (held-out test):")
    print(imp.to_string(index=False))

    # --------------------------------------------------------------
    # Honesty checks (parallel to ridge)
    # --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Honesty checks")
    print("=" * 78)
    print("\n  (1) ML vs baseline_surface (LOCO):")
    n_wins_surf = int((loco_df["margin_vs_surface"] > 0).sum())
    print(f"      HGBR beats baseline_surface on {n_wins_surf}/11.")
    for _, r in loco_df.iterrows():
        win = "WINS " if r["margin_vs_surface"] > 0 else "LOSES"
        print(f"        {r['cycle']:7s} ({r['thermal_role']:6s})  "
              f"{win}  margin = {r['margin_vs_surface']:+.3f} °C")
    print(f"\n  (2) ML vs baseline_tinf, gated at excursion >= {EXCUR_GATE_C}:")
    n_eligible = int((loco_df["peak_excursion_C"] >= EXCUR_GATE_C).sum())
    n_wins = int(((loco_df["peak_excursion_C"] >= EXCUR_GATE_C) &
                  (loco_df["margin_vs_tinf"] > 0)).sum())
    n_low = int((loco_df["peak_excursion_C"] < EXCUR_GATE_C).sum())
    print(f"      eligible: {n_eligible}/11;  HGBR beats baseline_tinf on "
          f"{n_wins}/{n_eligible};  low-excursion: {n_low}")
    for _, r in loco_df.iterrows():
        if r["peak_excursion_C"] < EXCUR_GATE_C:
            verdict = f"LOW EXCURSION ({r['peak_excursion_C']:.2f}) — uninformative"
        elif r["margin_vs_tinf"] > 0:
            verdict = f"WINS  margin = {r['margin_vs_tinf']:+.3f}"
        else:
            verdict = f"LOSES margin = {r['margin_vs_tinf']:+.3f}"
        print(f"        {r['cycle']:7s} ({r['thermal_role']:6s})  "
              f"excur={r['peak_excursion_C']:4.2f}  {verdict}")
    print("\n  Standing caveats: pipeline-value claims credible only on VAL "
          "cycles (Mixed1, Mixed2);")
    print("  US06 flagged (thermal-cal + widest band 5.05 + worst surface fit "
          "1.046 °C);")
    print("  R_cs label band ~5 °C T_core spread accompanies every number.")

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cyc_full.to_parquet(OUT_DIR / "hgbr_fixed_per_cycle.parquet", index=False)
    per_cyc_drop.to_parquet(OUT_DIR / "hgbr_ablation_per_cycle.parquet", index=False)
    loco_df.to_parquet(OUT_DIR / "hgbr_loco.parquet", index=False)
    imp.to_parquet(OUT_DIR / "hgbr_perm_importance.parquet", index=False)
    side.to_parquet(OUT_DIR / "ridge_vs_hgbr_loco.parquet", index=False)
    summary = {
        "fixed_split": {
            "test_file_ids": test_fids,
            "best_params":   best_full,
            "overall":       overall_full,
        },
        "ablation_drop_surface": {
            "best_params": best_drop,
            "overall":     overall_drop,
            "delta_rmse_vs_full": hgbr_ablation_delta,
        },
        "loco": {
            "n_folds": 11,
            "pooled":  pooled_overall,
            "config_caveat": "config selected on fixed-split 9 train cycles; "
                              "mild leak for LOCO folds whose held-out cycle "
                              "was among those 9",
        },
        "ridge_vs_hgbr": {
            "ridge_pooled":              ridge_pooled,
            "hgbr_pooled":               pooled_overall,
            "ridge_ablation_delta_rmse": ridge_ablation_delta,
            "hgbr_ablation_delta_rmse":  hgbr_ablation_delta,
        },
    }
    (OUT_DIR / "hgbr_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n  outputs saved to: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
