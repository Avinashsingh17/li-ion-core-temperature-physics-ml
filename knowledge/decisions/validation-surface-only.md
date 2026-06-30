# Validation: surface only, on held-out cycles

## The question
How do we validate the thermal model when the target we ultimately care about — **core temperature** — is **not measured** in either dataset?

## Options considered
- Validate only against the surface-temperature trace.
- Validate against a literature core-temperature dataset for a similar cell.
- Skip validation, report calibration error only.

## The call
- Validate the thermal model against **surface temperature** on **held-out cycles** — cycles that were never used during calibration.
- Hold-out is by **whole cycle and/or whole ambient temperature**, never random per-row splits ([blocked time-series split](../concepts/blocked-time-series-split.md) — concept note todo).
- Report **RMSE, MAE, and max error** on `T_surface_C` for the held-out cycles, plus a **naive "predict T_surface = T_ambient"** baseline (and the **no-T_surface ablation** for the downstream ML model, when that exists).
- **Core-temperature labels are model-derived and cannot be directly validated.** Every writeup states this explicitly.
- **Plausibility check on core**: compare core−surface gap to literature values for 18650 cells at similar C-rates — single-°C for normal driving, growing with C-rate. If the calibrated model predicts unreasonable gaps, that's a red flag even though it's not a formal validation.

## Why
- We have **no ground truth** for core temperature.
- A held-out-cycle split is the appropriate generalization test for time-series; per-row random splits **leak** future information into training and inflate accuracy ([leakage](../concepts/data-leakage.md) — concept note todo).
- Stating the core labels are unvalidated avoids overclaiming on a portfolio project and matches the actual epistemic state.

## What would make us revisit
- A core-temperature measurement (e.g. thermocouple insertion, fiber-optic sensor, distributed temperature sensing) becomes available for the LG HG2 or a near-equivalent.
- A peer-reviewed core-temperature dataset for a similar cylindrical cell is published.

## Related
- [Calibration: constrained fit](calibration-constrained-fit.md) — produces the parameters being validated.
- [Two-state thermal ODE](thermal-ode-two-state.md) — produces the temperatures being validated.
- Concept (todo): `blocked-time-series-split.md`.
- Concept (todo): `data-leakage.md`.
- Concept (todo): `naive-baseline.md` — why every regression report needs one.
