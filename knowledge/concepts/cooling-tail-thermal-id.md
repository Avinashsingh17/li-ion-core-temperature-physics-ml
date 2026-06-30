# Cooling-tail thermal-resistance identification

## Plain explanation
When current to a cell stops, heat generation stops, and the cell's surface temperature decays exponentially toward ambient with a characteristic time constant `τ`. For a single-node lumped thermal model, `τ = C·R_conv` — capacity times convective resistance. So if you can measure `τ` from a clean cooling tail (and you know `C` from physics), you get `R_conv` directly without fitting it jointly with anything else. This is the standard way to anchor the convection resistance in a constrained-fit calibration: get one parameter from a regime where it's the only one that matters.

For a two-state model (core + surface), the decay is technically bi-exponential — a fast surface↔core equilibration and a slow whole-cell cooling toward ambient. After the fast mode dies, the surface decays at the slow time constant `τ_slow ≈ (C_core + C_surf)·R_sa = C_total · R_sa`. Fit a single exponential to the post-fast portion and you get `R_sa = τ / C_total`.

## Role here
The [calibration decision](../decisions/calibration-constrained-fit.md) uses cooling tails to **anchor R_sa first**, then holds it fixed during the rest of the fit. If R_sa floated, it would trade off against R_cs and destroy the one clean anchor we have. Anchoring R_sa from independent data (rest segments) and fitting the rest is exactly what makes the otherwise ill-posed calibration tractable.

## Where used
- [`calibrate.py`](../../calibrate.py) — `find_cooling_tails`, `fit_exponential_decay`, `step2_anchor_R_sa`.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — Step 2 of the calibration pipeline.

## Gotcha
**A "cooling tail" must be genuinely at rest.** A long CV-charge segment at low current still generates heat (charge transfer over polarization is dissipative); fitting that as a cooling tail gives a wrong τ. Filter strictly on `|I| < ε` (e.g. 0.05 A) sustained for the full duration. Also: the tail must start with the cell genuinely warmer than ambient — if `T_s ≈ T_amb` already, there's no decay to fit. In our LG data the cooling tails undershoot ambient by ~1 °C after a few minutes, suggesting a chamber-air vs cell-thermocouple calibration offset — the slow-mode τ from the early portion is still meaningful, but downstream R_sa carries this caveat.
