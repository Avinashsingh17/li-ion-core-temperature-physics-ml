# Lin et al. 2014 — Lumped-parameter electro-thermal model for cylindrical batteries

## Citation

Lin, X., Perez, H. E., Mohan, S., Siegel, J. B., Stefanopoulou, A. G., Ding, Y., Castanier, M. P. (2014). *A lumped-parameter electro-thermal model for cylindrical batteries*. **Journal of Power Sources** 257, 1–11. DOI: [10.1016/j.jpowsour.2014.01.097](https://doi.org/10.1016/j.jpowsour.2014.01.097).

- PDF: `knowledge/A lumped-parameter electro-thermal model for cylindrical batteries.pdf` (11 pp).
- Text extracted with **pymupdf 1.26.7** → `knowledge/A lumped-parameter electro-thermal model for cylindrical batteries.txt`.

## Core idea (in our words)

A 5-state, control-oriented model that pairs an **equivalent-circuit electrical model** (OCV + ohmic R + two RC pairs) with the **two-state lumped thermal model** (core + surface), coupled bidirectionally via heat generation `Q = I·(V_ocv − V_T)` and via temperature-dependence of the electrical parameters. Applied to a 2.3 Ah A123 26650 **LiFePO4** cell. The key methodological contribution is a **decoupled parameterization**: electrical parameters are identified under isothermal / SOC-invariant pulse-relaxation tests; thermal parameters are then identified from a single drive-cycle by Laplace-domain least-squares on the surface temperature. `C_s` is pre-calculated from casing geometry and `c_p` rather than fitted, removing one degree of freedom in the thermal identification. Validated on two drive cycles (22C peak, 5–38 °C) — voltage RMSE 20–45 mV, surface/core temp RMSE under 1 °C.

## Key equations

**Two-state thermal model** (paper eq. 4):

$$C_c \frac{dT_c}{dt} = Q + \frac{T_s - T_c}{R_c}$$

$$C_s \frac{dT_s}{dt} = \frac{T_f - T_s}{R_u} - \frac{T_s - T_c}{R_c}$$

**Heat generation** (paper eq. 5):

$$Q = I \cdot (V_\text{ocv} - V_T)$$

Symbol glossary (with paper's sign convention):

| Symbol | Meaning | Unit |
|---|---|---|
| `T_c`, `T_s` | core and surface (skin) temperatures | °C |
| `T_f` | coolant / ambient flow temperature | °C |
| `Q` | volumetric heat-generation rate | W |
| `I` | terminal current — **paper defines I > 0 for discharge** | A |
| `V_T` | measured terminal voltage | V |
| `V_ocv` | open-circuit voltage, function of SOC | V |
| `C_c`, `C_s` | core and surface heat capacities | J/K |
| `R_c` | core ↔ surface conduction resistance | K/W |
| `R_u` | surface ↔ ambient convection resistance | K/W |

**Sign-convention reconciliation** (important — we use opposite sign):
- Paper: `I > 0` (discharge) and `V_T < V_ocv` ⇒ `Q = I·(V_ocv − V_T) > 0`.
- Ours: `I < 0` (discharge) and `V < V_ocv` ⇒ `Q = I·(V − V_ocv) > 0`.

The two expressions are algebraically identical because both `I` and the voltage difference flip sign together. **No contradiction.**

**Parametric identification** (paper eq. 11): they reduce the unknown 4-parameter system to 3 lumped parameters `(α, β, γ)` and identify them via least-squares; recover `R_u, R_c, C_c` from `(α, β, γ)` plus the pinned `C_s` (paper eq. 20). 95 % confidence intervals from the LS covariance matrix `cov(θ̂) = (FᵀF)⁻¹·σ²` (paper eqs. 17–19).

## Parameter values reported (Table 1)

For an A123 26650 LiFePO4 cell (2.3 Ah, ~70 g class):

| Param | Value | 95 % CI |
|---|---|---|
| `C_s` | 4.5 J/K | (pinned, no CI) |
| `C_c` | 62.7 J/K | 59.1 – 66.4 |
| `R_c` | 1.94 K/W | 1.86 – 2.01 |
| `R_u` | 3.19 K/W | 3.19 – 3.20 |

**Absolute values do NOT transfer.** Lin's cell is A123 26650 LFP (~70 g, LFP chemistry). Ours are LG HG2 / Panasonic PF 18650 NMC (~47 g). Different chemistry, different format, different mass, different cycler/fan setup. The numbers above are **not** usable as starting parameters for our calibration.

**Only the qualitative ratios carry as loose priors** for sanity-checking *our own* identified values:
- `C_c ≫ C_s` — the jellyroll holds essentially all the thermal mass; the casing is thin.
- `R_c < R_u` — internal conduction faster than external convection at typical fan settings.
- Total `C_c + C_s ≈ m·c_p` — within ~7 % for Lin's cell, supporting our pin-`C_total = m·c_p` strategy in [calibration-constrained-fit](../decisions/calibration-constrained-fit.md). For our 47 g 18650 with `c_p ≈ 800–1000 J/(kg·K)`, that's `~38–47 J/K` total — about ~60 % of Lin's `C_total = 67 J/K`, mass-scaled.

**`R_u` deserves a special caveat**: it's a property of the *coolant / fan / chamber* around the cell, not the cell itself. Lin's `R_u = 3.19 K/W` reflects his flow chamber, not anything intrinsic to a cell. Our `R_u` must come from cooling tails on our own data — there is no portable prior for it from this paper.

**Validation RMSEs** (paper Table 2; reference for our own held-out reporting):
- Charge-sustaining cycle (peak 22C, 25–38 °C): V_T 20.3 mV, T_s 0.65 °C, T_c 0.92 °C.
- Charge-depleting cycle (peak 10C, 5–23 °C):  V_T 45.4 mV, T_s 0.20 °C, T_c 0.56 °C.

**Electrical parameters** (only relevant if we add a voltage model later): R_s ≈ 8–19 mΩ (5–45 °C), R_1 ≈ 7–35 mΩ, R_2 ≈ 5–45 mΩ; C_1 ≈ 500–3500 F, C_2 ≈ tens of kF.

## What's directly reusable

- **The two-state thermal ODE form is exactly the model we adopted.** Identical equations modulo sign convention. Lock-in for [thermal-ode-two-state](../decisions/thermal-ode-two-state.md).
- **Pinning `C_s` from geometry/`c_p`, then fitting the rest** is the same pattern as our constrained calibration. Lock-in for [calibration-constrained-fit](../decisions/calibration-constrained-fit.md).
- **95 % CIs from the LS covariance matrix** are exactly what we plan to report with `scipy.optimize.least_squares`. (Note: `scipy.optimize.curve_fit` returns the same `pcov`; with `least_squares`, build it from `J^T J · σ²`.)
- **Parameter ratios as priors / sanity bounds**: `C_c / C_s ~ 10–20` and `R_u / R_c ~ 1–2` are reasonable starting expectations for a different cylindrical Li-ion cell.
- **Validation framing**: report RMSE on both V (if applicable) and T_s on held-out drive cycles; compare to ~0.5–1 °C target.

## What is NOT directly reusable

- **Electrical equivalent-circuit RC model and its 2-RC parameterization** — we are not modelling voltage dynamics. Our project uses raw measured `V` directly in the heat-gen formula, not a reconstructed `V_T`. We skip the entire electrical-identification half of the paper.
- **Numerical parameter values** — A123 26650 LiFePO4 vs. our LG HG2 / Panasonic PF 18650 NMC differ in chemistry, mass (~70 g vs. ~47 g) and geometry. Ratios may transfer; numbers won't.
- **Laplace-domain parametric LS / filtered regressors** — Lin handles the unmeasured `T_c` by Laplace-transform manipulation of eq. 4 to eliminate it from the regression. Our planned approach is **forward-integrate the ODE with `solve_ivp`** and fit parameters against `T_s` residuals in time domain (simpler in scipy, no signal differentiation, no filter-frequency tuning). Same goal, different mechanics.
- **LiFePO4-specific OCV findings** (charge/discharge hysteresis) — LG HG2 (NMC) hysteresis is much smaller; our discharge-only OCV from [the C/20 file](../decisions/ocv-characterization.md) is adequate.
- **Core thermocouple validation** — Lin physically inserted a thermocouple to measure `T_c`. We have no equivalent measurement; our core estimates remain unvalidated per [validation-surface-only](../decisions/validation-surface-only.md).

## Contradictions with our locked decisions

**None.** The paper's methodology is the principal reference for what we already locked in; sign-convention difference in `Q = I·(V_ocv − V_T)` vs. our `Q = I·(V − V_ocv)` is algebraically equivalent, not a conflict.

One **important context difference** to flag in writeups: Lin reports core-temperature RMSE because they instrumented the core; we will report only surface RMSE and explicitly state that core values are model-derived.

## Related notes

- [decisions/thermal-ode-two-state](../decisions/thermal-ode-two-state.md) — the model we'll integrate.
- [decisions/heat-generation-irreversible-only](../decisions/heat-generation-irreversible-only.md) — `Q = I·(V − V_ocv)`.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — same pin-then-fit approach.
- [decisions/ocv-characterization](../decisions/ocv-characterization.md) — Lin uses the same C/20 strategy.
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — where we diverge: we have no core measurement.
- [concepts/ocv-soc-curve](../concepts/ocv-soc-curve.md), [concepts/coulomb-counting](../concepts/coulomb-counting.md).
