# Lin et al. 2013 — Online parameterization & identifiability of the two-state thermal model

## Citation

Lin, X., Perez, H. E., Siegel, J. B., Stefanopoulou, A. G., Li, Y., Anderson, R. D., Ding, Y., Castanier, M. P. (2013). *Online parameterization of lumped thermal dynamics in cylindrical lithium ion batteries for core temperature estimation and health monitoring*. **IEEE Trans. Control Systems Technology** 21(5), 1745–1755. DOI: [10.1109/TCST.2012.2217143](https://doi.org/10.1109/TCST.2012.2217143).

- PDF: `knowledge/Online_Parameterization_of_Lumped_Thermal_Dynamics_in_Cylindrical_Lithium_Ion_Batteries_for_Core_Temperature_Estimation_and_Health_Monitoring.pdf` (11 pp).
- Text extracted with **pymupdf 1.26.7** → sibling `.txt`.

## Core idea (in our words)

The same two-state thermal model as Lin 2014, but here treated as an **online identification problem** for a vehicle BMS. Heat generation is the simpler **Joule-only** form `Q = I²·R_e`, which lets the unknown internal resistance `R_e` be lumped in as one of the parameters to identify. Recursive least squares (RLS) is applied in the Laplace domain after eliminating the unmeasured core temperature `T_c`. **The central identifiability finding**: the model has 5 physical parameters but the regressor structure exposes only **3 lumped parameters** `(α, β, γ)` — so **2 of the 5 must be presumed**, and the authors argue that the **heat capacities `C_c, C_s` are the right ones to pin** (they don't drift with aging, they only set transient timescales, and they can be approximated from geometry). The paper also derives a quantitative **persistent-excitation (PE) test** for whether a drive cycle has enough thermal excitation to make the LS estimator converge, then extends RLS with non-uniform forgetting factors to track a time-varying `R_e` (Arrhenius in temperature) and to detect long-term SOH degradation via `R_e` growth.

## Key equations

**Two-state thermal model** (paper eq. 1) — same as Lin 2014, but with Joule heating:

$$C_c \dot{T_c} = I^2 R_e + \frac{T_s - T_c}{R_c}$$

$$C_s \dot{T_s} = \frac{T_f - T_s}{R_u} - \frac{T_s - T_c}{R_c}$$

**Parametric form after Laplace-eliminating `T_c`** (paper eq. 8): with `T_f` taken as steady,

$$s^2 T_s - s T_{s,0} = \alpha\, I^2 + \beta\,(T_f - T_s) + \gamma\,(s T_s - T_{s,0})$$

The three identifiable lumped parameters (paper eq. 11):

$$\alpha = \frac{R_e}{C_c C_s R_c}, \quad \beta = \frac{1}{C_c C_s R_c R_u}, \quad \gamma = -\left(\frac{C_c+C_s}{C_c C_s R_c} + \frac{1}{C_s R_u}\right)$$

Once `α, β, γ` are identified and `C_c, C_s` are presumed, recover the rest (paper eq. 12):

$$\beta(C_c+C_s)C_s R_u^2 + \gamma C_s R_u + 1 = 0 \;\;\Rightarrow\;\; R_u,\quad R_c = \frac{1}{\beta C_s C_c R_u},\quad R_e = \alpha C_c C_s R_c$$

**Persistent-excitation condition** (paper eq. 5) — a drive cycle's regressor `φ(t) = [I², T_f−T_s, sT_s]ᵀ` is **persistently exciting** if there exist `T_0 > 0`, `α_1 ≥ α_0 > 0` such that

$$\alpha_1 I_M \;\geq\; \frac{1}{T_0}\int_t^{t+T_0} \phi(\tau)\phi^T(\tau)\,d\tau \;\geq\; \alpha_0 I_M \quad \forall t \geq 0$$

In practice: compute the regressor autocorrelation matrix over each cycle period; if its smallest eigenvalue stays comfortably above zero, the cycle has enough thermal excitation. `α_0` bounds the convergence time as `τ ≤ 2/α_0` for gradient methods.

Symbol glossary:

| Symbol | Meaning | Unit |
|---|---|---|
| `T_c`, `T_s`, `T_f` | core, surface, ambient flow temps | °C |
| `I` | current — paper uses **negative for discharge** (Fig. 4 caption) | A |
| `R_e` | lumped internal resistance (Joule heating term) | Ω |
| `R_c`, `R_u` | conduction (core↔surface), convection (surface↔ambient) | K/W |
| `C_c`, `C_s` | core, surface heat capacities | J/K |
| `α, β, γ` | identifiable lumped parameters per eq. 11 | (mixed) |
| `φ` | regressor vector for RLS | (mixed) |

## Parameter values reported

Same A123 26650 LiFePO4 cell as Lin 2014. Compared in paper Table I and Table II:

| Param | Lin 2013 value | Lin 2014 value | Notes |
|---|---|---|---|
| `C_c` (J/K) | **67** (assumed) | 62.7 (fitted) | within ~7 % of each other |
| `C_s` (J/K) | **4.5** (assumed, from casing) | **4.5** (pinned, same source) | identical |
| `R_c` (K/W) | 1.83 (identified) | 1.94 (fitted) | within ~6 % |
| `R_u` (K/W) | 3.03 (identified) | 3.19 (fitted) | within ~5 % |
| `R_e` (mΩ) | 11.4 (identified) | (not directly comparable) | between Rs and Rs+R1+R2 of Lin 2014 |

Persistent-excitation diagnostics for the urban-assault drive cycle:
- Smallest regressor eigenvalue `α_0 ≈ 2.4×10⁻⁴ s⁻¹`.
- Largest `α_1 ≈ 0.086 s⁻¹`.
- Conservative parameter convergence time `τ ≤ 8333 s` (90% settling ≤ 19186 s ≈ 5.3 h).

## What's directly reusable

- **The identifiability result is the single most important takeaway for us.** It formally justifies the "constrained, not free 4-param" choice in [calibration-constrained-fit](../decisions/calibration-constrained-fit.md). At minimum, **two** of the four thermal parameters must be pinned externally to avoid an ill-posed fit; we go further and pin/constrain three (`m·c_p` total split into `C_c, C_s`, plus `R_u` from cooling tails).
- **`C_c` and `C_s` are the right things to pin** — slow-drifting, only affect transients, externally calculable. The paper makes this argument explicitly (page 3, last paragraph).
- **The quadratic in `R_u` (eq. 12a)** gives an analytical closed form once `α, β, γ` are known. Useful as a cross-check for our cooling-tail-derived `R_u` — if a parametric LS estimator ever fits the lumped triplet, eq. 12a recovers `R_u` and we can compare.
- **Persistent-excitation as a cycle-suitability test**: not every drive cycle excites the thermal dynamics enough to identify parameters. For our held-out / calibration split, we should prefer cycles with non-trivial `α_0` (low-eigenvalue of the regressor covariance) — this can flag cycles unsuitable for thermal calibration before we run them. **This deserves a concept stub**: `persistent-excitation.md`.
- **Parameter values for cross-comparison** (same Lin 2014 caveat: 26650 LFP vs our 18650 NMC — ratios may carry, absolute numbers won't).

## What is NOT directly reusable

- **Heat generation `Q = I²·R_e`**. This is the older / simpler formulation; Lin 2014 itself upgrades to `Q = I·(V_ocv − V_T)` and notes the latter resolves a meaningful discrepancy where `R_e` becomes cycle-dependent. We follow Lin 2014's heat-gen form, not Lin 2013's. See [heat-generation-irreversible-only](../decisions/heat-generation-irreversible-only.md).
- **Online recursive LS with forgetting factors**. We're doing **offline batch calibration** on labeled drive cycles, not real-time identification on a vehicle. Forgetting factors are unnecessary for our portfolio scope.
- **The Luenberger / Kalman adaptive observer** (§VI). We are not building an online state estimator; we forward-integrate the ODE with `solve_ivp` using fixed identified parameters and compare against measured `T_s`.
- **SOH monitoring via `R_e` growth** (§VIII). Out of scope — single fresh cell.
- **Laplace-domain elimination of `T_c`** to derive the regressor. Our planned path is time-domain forward integration; we never need to eliminate `T_c` because `scipy.integrate.solve_ivp` evolves both states from initial conditions. Lin's Laplace trick exists because he wanted to use a linear-regression identifier; our nonlinear-least-squares approach (`scipy.optimize.least_squares` on `T_s` residual) avoids that requirement.
- **Core thermocouple validation** — they drilled a hole, inserted a thermocouple, and validated `T_c` directly (Fig. 2). We have no core measurement; this is exactly why [validation-surface-only](../decisions/validation-surface-only.md) declares core labels unvalidated.

## Contradictions with our locked decisions

**None — but two important context points worth flagging:**

1. The paper's `Q = I²·R_e` form **disagrees with our heat-gen formula**, but this is not a contradiction with us: it's the formulation Lin himself supersedes in the 2014 paper. Our decision aligns with the *later* Lin paper.
2. The paper presumes **only** `C_c, C_s` (2 of 5 params) — we go further and additionally constrain `R_u` from cooling tails. **Our calibration is *more* constrained than Lin 2013's**, not less. That's safer for an unobserved core node, but it does mean we're relying on the cooling-tail extraction being accurate (a real risk: if the cell isn't actually at thermal steady state during the "cooling tail," the inferred `R_u` is biased). Worth flagging in any writeup.

## Concepts this paper introduces / sharpens for us

- **Identifiability** (count of estimable parameters vs count of physical parameters; rank of the regressor structure).
- **Persistent excitation** (signal-richness condition needed for parameter convergence under LS).
- **Forgetting factors in RLS** (only relevant if we ever go online — currently not).

Stubs to add to the index: [persistent-excitation](../concepts/persistent-excitation.md), [identifiability](../concepts/identifiability.md).

## Related notes

- [Lin 2014 source note](lin-2014-electro-thermal.md) — supersedes the Joule-only heat-gen used here.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — directly motivated by this paper's identifiability result.
- [decisions/thermal-ode-two-state](../decisions/thermal-ode-two-state.md) — the model we share with Lin.
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — explains why we can't replicate Lin's direct `T_c` validation.
