# Blocked time-series split (GroupKFold + LOCO)

## Plain explanation
In a time-series dataset, adjacent rows aren't independent — the row at `t=100s` is essentially the same physical state as `t=99s`. A random train/test split puts neighbors of a test row in the train set, so the test "generalization" RMSE is really memorization. The fix is to split by an **outer block** that's guaranteed to be a whole, contiguous chunk of contextually-similar rows — for us, an entire drive cycle (`file_id`). `GroupKFold` (with `groups=file_id`) holds out whole cycles per fold; **leave-one-cycle-out (LOCO)** does the same with `n_folds = n_cycles`. Combined with within-train hyperparameter tuning, this gives an honest measurement of how the model performs on a cycle it has never seen.

## Role here
The ridge baseline (and every ML stage after it) must report metrics from a split where the test cycle was wholly invisible during both training **and** hyperparameter selection. The fixed split holds out Mixed1 + Mixed5 entirely; LOCO holds each of the 11 cycles out once. Inside each split, `tune_and_fit_ridge` runs its own GroupKFold over the **train cycles only** to pick `alpha` — so the held-out cycle of the outer split never participates in selecting `alpha` either.

## Where used
- [`features_and_split.py`](../../features_and_split.py) — `grouped_cv_folds`, `tune_and_fit_ridge` (the nested-CV wrapper that this concept exists to justify).
- [`train_ridge.py`](../../train_ridge.py) — fixed dev split + 11-fold LOCO, both built on `tune_and_fit_ridge` so neither outer test ever leaks into alpha selection.
- [`decisions/validation-surface-only`](../decisions/validation-surface-only.md) — the "hold out whole cycles, never rows" rule this concept formalises.

## Gotcha
**Tune your hyperparameters inside the train folds only.** The most common version of this mistake: someone splits cycles into train/test, then calls `GridSearchCV(..., cv=KFold(5))` on the train data — but doesn't pass `groups` to the CV, so the CV is row-wise random. The inner-fold "validation" set contains rows from the same cycles as the inner-fold "train" set, so the CV scoring is inflated and the chosen `alpha` is optimistic. The fix is `GroupKFold(...)` with `groups=file_id` everywhere — both the outer split and the inner alpha-selection CV. Equivalently: ask yourself "could any row from the outer-test cycle be reached by the inner CV?" — if yes, your alpha selection is leaking.
