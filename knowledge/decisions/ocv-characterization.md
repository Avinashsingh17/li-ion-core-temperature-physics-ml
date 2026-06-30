# OCV characterization: from a low-rate (C/20) discharge

## The question
How do we build `V_ocv(SOC)`, the open-circuit-voltage curve used by the [heat-generation formula](heat-generation-irreversible-only.md)?

## Options considered
- **Pseudo-OCV from a low-rate (C/20) discharge sweep** — single sweep, near-quasistatic.
- **Pulse–rest OCV from HPPC** — measure voltage after relaxation at each SOC.
- **Datasheet curve** — use the published LG HG2 OCV table.

## The call
Build `V_ocv(SOC)` from the **LG C/20 characterization discharge file** at 25 °C (file: `549_C20DisCh_25degC_LGHG2.mat` / `.csv`). SOC is computed by **coulomb counting** the standardized `Ah` column over that file:

$$\text{SOC}(t) = 1 + \frac{Ah(t) - Ah(0)}{Ah_\text{nominal}}$$

with `Ah_nominal = 3.0 Ah` (LG HG2 rated capacity). Sign convention: discharge accumulates negative `Ah`, so `SOC` decreases from 1 → 0 as the cell discharges.

## Why
- A clean, monotonic, near-equilibrium sweep is the lowest-error path to `V_ocv(SOC)` and produces a smooth curve suitable for interpolation in the heat-gen formula.
- HPPC reconstruction is messier — requires segmenting rest periods, picking relaxation times, and trusting that the rest is long enough.
- Datasheet curves don't reflect this specific cell or operating conditions.
- HPPC remains a fallback / cross-check.
- We assume `V_ocv(SOC)` is temperature-independent for now (this is consistent with omitting the entropic term — see [heat generation](heat-generation-irreversible-only.md)).

## What would make us revisit
- The C/20 file turns out to be contaminated (logger drop, partial discharge, non-monotonic capacity).
- Residual error from the calibrated thermal model concentrates near SOC endpoints (where the C/20 curve is steepest and small SOC errors → large `V_ocv` errors).
- We later want temperature-dependent OCV and re-run characterization at multiple ambients.

## Related
- [Heat generation: irreversible only](heat-generation-irreversible-only.md) — consumes this curve.
- [Calibration approach](calibration-constrained-fit.md) — sensitive to OCV errors.
- Concept (todo): `coulomb-counting.md`.
- Concept (todo): `ocv-soc-curve.md`.
- Concept (todo): `c-rate.md` — what "C/20" means, why it matters for pseudo-OCV.
