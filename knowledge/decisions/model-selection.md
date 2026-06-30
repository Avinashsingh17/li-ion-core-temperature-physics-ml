# Model selection: ridge as the reported core-temperature regressor

## The question
Which ML model do we report as the project's core-temperature regressor — ridge or HGBR?

## Options considered
- Ridge regression baseline.
- HistGradientBoostingRegressor (HGBR).
- Report both as co-equal.

## The call
**Report ridge as the single primary model.** HGBR is retained in the codebase and documented as a checked alternative that did not separate from ridge.

## Why
- **Pooled LOCO error is near-identical**: ridge RMSE 0.419 °C vs HGBR 0.412 °C — a ~0.007 °C (<2 %) gap, within the noise of fold composition and hyperparameter selection.
- **Ridge has the better worst-case tail with surface features**: pooled LOCO max 1.86 °C vs HGBR 1.99 °C; fixed-split max with full features 1.35 °C vs HGBR 1.63 °C. For a safety-relevant estimate, worst-case matters. Note: in the electrical-only ablation HGBR pulls ahead on max-abs (HGBR 4.32 °C vs ridge 4.39 °C, both fixed-split) — surface signals are doing the work that keeps ridge's tail tighter; without them, the tree model edges ahead (see "What would make us revisit" below).
- **Ridge is simpler**: faster, interpretable coefficients, no early-stopping group-leak subtlety.
- **The near-tie is itself the finding**: the engineered-feature → T_core map is effectively linear over the 25 °C regime; nonlinearity buys nothing here.
- **Model class is not the ceiling**: choice of model moves the prediction ~0.01 °C, while the R_cs label band is ~5 °C of T_core spread. The dominant uncertainty is physical (unobservable R_cs), not the ML.

## What would make us revisit
- More data or more diverse cycles (multi-ambient, a second cell) where nonlinear structure might emerge.
- A target regime with stronger I/V ↔ T_core nonlinearity.
- A sensorless (electrical-only) deployment target — the ablation gap differs between models, so re-examine there.

## Caveats carried
Pipeline-value claims credible only on thermal-validation cycles (Mixed1, Mixed2); US06 flagged (thermal-cal, widest R_cs band, worst surface fit); the R_cs label band accompanies every reported number.

## Related
- [report-notes](../report-notes.md) — the ridge≈HGBR linearity finding and the average-vs-tail tradeoff.
- [Blocked time-series split](../concepts/blocked-time-series-split.md).
- [calibration-constrained-fit](calibration-constrained-fit.md) — origin of the R_cs label band that dominates the uncertainty.
