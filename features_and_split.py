"""
features_and_split.py — Feature engineering + CV utilities for the ML stage.

Schema constants pinned to the real columns in `data/labeled/*.parquet`:
  COL_TCORE  = "T_core_model_C"    -- the label
  COL_CYCLE  = "file_id"           -- the GROUP key (whole-cycle blocking)
  COL_TSURF  = "T_surface_C"       -- measured surface (trivial baseline)
  COL_TINF   = "T_inf_used_C"      -- constant T_inf (level-correct baseline)

Public API (used by `train_ridge.py` and downstream ML scripts):
  load_labeled_data()            -> DataFrame with all 11 LG 25 °C cycles.
  build_features(df, drop_surface=False)
                                 -> (X_df, df_kept). 23 features (full) or
                                    20 features (electrical-only ablation).
                                    Causal features only. Drops warm-up rows
                                    where any rolling/lag column is NaN.
  grouped_cv_folds(groups, k=5)  -> GroupKFold splitter for GridSearchCV.
  tune_and_fit_ridge(X, y, g)    -> (best_pipeline, best_alpha). alpha is
                                    tuned ONLY on (X, y, g); never sees the
                                    held-out cycle.
  metric_bundle(y_true, y_pred)  -> {"rmse", "mae", "max"}
  per_cycle_metrics(df, y, y_hat)-> per-file_id RMSE/MAE/max table.
  baseline_surface(df)           -> predict T_core = T_surface_C.
  baseline_tinf(df)              -> predict T_core = T_inf_used_C (≈ 23.15).
  peak_excursion(df)             -> {file_id: max |T_surface − T_inf|}.

Locked decisions used here:
  - never random-split time series (whole-cycle blocking, see
    knowledge/concepts/blocked-time-series-split.md).
  - alpha tuning only sees train cycles (no leakage from held-out folds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Schema constants — pinned to data/labeled/ column names.
# ----------------------------------------------------------------------------
COL_TCORE = "T_core_model_C"
COL_CYCLE = "file_id"
COL_TSURF = "T_surface_C"
COL_TINF  = "T_inf_used_C"

LABELED_DIR = Path("data/labeled")

# Feature-engineering windows (seconds; matches the 1 Hz grid).
# 2026-06-19 locked spec: lags at 1/5/30/60 s; rolling means of I² and I·V
# at 5/30/60 s; long-window (60 s) smoothed surface temp + abs-current.
LAG_WINDOWS_S       = (1, 5, 30, 60)
ROLL_PHYS_WINDOWS_S = (5, 30, 60)
ROLL_LONG_S         = 60


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_labeled_data(labeled_dir: Path = LABELED_DIR) -> pd.DataFrame:
    """Load all cycle Parquets from `data/labeled/` into one DataFrame.

    Excludes `_labels_index.parquet`. Each cycle's rows are concatenated; the
    `file_id` column identifies them.
    """
    files = sorted(p for p in labeled_dir.glob("*.parquet")
                   if p.name != "_labels_index.parquet")
    if not files:
        raise FileNotFoundError(f"no cycle Parquets in {labeled_dir}")
    parts = [pd.read_parquet(p) for p in files]
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def _features_for_cycle(g: pd.DataFrame, drop_surface: bool) -> pd.DataFrame:
    """Build features for ONE cycle (locked spec, restored 2026-06-19).

    Full set: **24 features**. `drop_surface=True` strips all **7**
    surface-derived features (instantaneous T_surface, its 4 lags, its rolling
    mean, and dT_surface/dt), leaving **17 electrical-only features**.

    Locked spec:
      - Instantaneous I, V (and T_surface unless drop_surface).
      - Lags of I, V, (T_surface) at 1 / 5 / 30 / 60 s.
      - Rolling means of I² and I·V at 5 / 30 / 60 s.
      - Smoothed T_surface_roll_60s (physical; dropped in ablation).
      - dT_surface/dt (dropped in ablation).
      - abs(I)_roll_60s (physical).

    Explicitly NOT in the feature set (removed in the 2026-06-19 correction):
      - cycle-position / cumulative quantities: time_in_cycle_s, Ah, Wh,
        cum_abs_charge — not physical thermal drivers, they extrapolate
        out-of-range on long cycles (UDDS), and they give the electrical-only
        ablation a non-electrical shortcut.
      - constant features at single-ambient: T_ambient_C, T_inf_used_C.

    All rolling / lag operations are CAUSAL. The earliest `ROLL_LONG_S`
    (60) rows per cycle will have NaN in at least one feature; those rows
    are dropped at the cycle boundary by `build_features`.
    """
    I  = g["current_A"]
    V  = g["voltage_V"]
    Ts = g["T_surface_C"]

    feats = pd.DataFrame(index=g.index)

    # --- Electrical features (always present) ---
    # Instantaneous (2).
    feats["current_A"] = I
    feats["voltage_V"] = V

    # Lags of I, V at 1/5/30/60 s (8). shift(+k) returns value k seconds prior.
    for n in LAG_WINDOWS_S:
        feats[f"current_lag_{n}s"] = I.shift(n)
        feats[f"voltage_lag_{n}s"] = V.shift(n)

    # Rolling means of I² (Joule-heating proxy) at 5/30/60 s (3).
    I_sq = I ** 2
    for n in ROLL_PHYS_WINDOWS_S:
        feats[f"I_squared_roll_{n}s"] = I_sq.rolling(n).mean()

    # Rolling means of I·V (signed instantaneous power) at 5/30/60 s (3).
    IV = I * V
    for n in ROLL_PHYS_WINDOWS_S:
        feats[f"IV_roll_{n}s"] = IV.rolling(n).mean()

    # abs(I) rolling mean — physical (1).
    feats["abs_current_roll_60s"] = I.abs().rolling(ROLL_LONG_S).mean()

    # --- Surface-derived features (dropped by drop_surface=True) ---
    if not drop_surface:
        # Instantaneous (1).
        feats["T_surface_C"] = Ts
        # Lags at 1/5/30/60 s (4).
        for n in LAG_WINDOWS_S:
            feats[f"T_surface_lag_{n}s"] = Ts.shift(n)
        # Time derivative (1).
        feats["dT_surf_dt"] = Ts.diff()
        # Smoothed rolling mean — physical (1).
        feats["T_surface_roll_60s"] = Ts.rolling(ROLL_LONG_S).mean()

    return feats


def build_features(df: pd.DataFrame, drop_surface: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features per-cycle, then concatenate.

    Returns `(feats, df_kept)` — both aligned and indexed [0..N_kept-1].
    `df_kept` is the rows of the input df that survived the warm-up drop
    (rows where any feature column was NaN). `groups` for CV come from
    `df_kept[COL_CYCLE]`.

    Feature count: 24 (full) or 17 (drop_surface=True, electrical-only).
    Warm-up drop is `ROLL_LONG_S = 60` rows per cycle ≈ 0.7 % of typical
    cycle length.
    """
    df_sorted = df.sort_values([COL_CYCLE, "time_s"]).reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for _, g in df_sorted.groupby(COL_CYCLE, sort=True):
        parts.append(_features_for_cycle(g, drop_surface))
    feats = pd.concat(parts, axis=0)
    feats = feats.dropna()
    df_kept = df_sorted.loc[feats.index].reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    return feats, df_kept


# ----------------------------------------------------------------------------
# Cross-validation utilities
# ----------------------------------------------------------------------------
def grouped_cv_folds(groups: np.ndarray, k: int | None = None) -> GroupKFold:
    """`GroupKFold` splitter sized to the number of unique groups.

    Pass the returned object as `cv=...` to `GridSearchCV`, then call
    `gs.fit(X, y, groups=groups)`. Each fold's test set contains all rows
    from one or more whole cycles — never partial cycles.
    """
    n_groups = int(np.unique(groups).size)
    if k is None:
        k = min(5, n_groups)
    k = max(2, min(k, n_groups))
    return GroupKFold(n_splits=k)


def tune_and_fit_ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       alphas: np.ndarray | None = None
                       ) -> tuple[Pipeline, float]:
    """`StandardScaler` + `Ridge`, alpha-tuned by GroupKFold over `groups`.

    alpha is selected using ONLY the data passed in here. The held-out
    cycle of the outer split / LOCO loop never participates — that's the
    no-leakage rule for blocked time-series CV. Returns the best-refit
    pipeline and the chosen alpha.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 13)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    cv = grouped_cv_folds(groups)
    gs = GridSearchCV(
        pipe,
        {"ridge__alpha": alphas},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae":  float(np.mean(np.abs(err))),
        "max":  float(np.max(np.abs(err))),
    }


def per_cycle_metrics(df: pd.DataFrame, y_true: np.ndarray,
                      y_pred: np.ndarray) -> pd.DataFrame:
    """Per-`file_id` RMSE/MAE/max table."""
    work = df.copy().reset_index(drop=True)
    work["_y_true"] = np.asarray(y_true)
    work["_y_pred"] = np.asarray(y_pred)
    rows = []
    for fid, g in work.groupby(COL_CYCLE, sort=True):
        m = metric_bundle(g["_y_true"], g["_y_pred"])
        rows.append({
            "file_id": fid,
            "cycle":   g["cycle"].iloc[0],
            "n":       len(g),
            "rmse":    m["rmse"],
            "mae":     m["mae"],
            "max":     m["max"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Baselines (per the level-bias-vs-dynamics concept)
# ----------------------------------------------------------------------------
def baseline_surface(df: pd.DataFrame) -> np.ndarray:
    """Predict T_core = T_surface_C — the trivial 'core ≈ surface' proxy.

    This is the bar that matters in this project. If ML doesn't beat
    surface-as-core, the dynamics it claims to model aren't earning their
    keep relative to "just use the thermometer you already have."
    """
    return df[COL_TSURF].to_numpy()


def baseline_tinf(df: pd.DataFrame) -> np.ndarray:
    """Predict T_core = T_inf_used_C — the level-correct constant baseline."""
    return df[COL_TINF].to_numpy()


def peak_excursion(df: pd.DataFrame) -> dict[str, float]:
    """Per-cycle peak |T_surface_C − T_inf_used_C|.

    A small excursion means T_inf already covers most of the y-range and
    the dynamics have nothing to win — use this to gate honesty claims
    against `baseline_tinf` per the concept-map "level bias vs dynamics"
    gotcha (always pair RMSE with excursion).
    """
    out: dict[str, float] = {}
    for fid, g in df.groupby(COL_CYCLE, sort=True):
        out[fid] = float(np.max(np.abs(g[COL_TSURF] - g[COL_TINF])))
    return out
