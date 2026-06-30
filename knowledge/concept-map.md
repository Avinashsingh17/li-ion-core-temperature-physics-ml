# Concept map

A learner-facing glossary of the physics, ML, and data concepts in active use in this project. Each entry is a short pointer — when a concept has its own deep-dive note in `concepts/`, the entry links to it instead of duplicating.

**Format**:

```
### <name>
- **Plain explanation**: 3–5 sentences, intuitive, no unexplained jargon
- **Role in this project**: 1–2 sentences, why we need it here specifically
- **Where used**: module(s) + function name(s) + week; links to decisions/concepts
- **Gotcha** (optional): the easy mistake to make
```

Append new entries as each implementation step introduces them.

---

## Physics & thermal modeling

### C-rate
- **Plain explanation**: A normalized current. "1C" is the current that would fully discharge the cell in one hour, so a 3 Ah cell at 1C draws 3 A. "C/20" is the rate that takes 20 hours. C-rate lets you compare currents across cells of different capacities, and it's the natural language for "how hard is the cell being driven?"
- **Role in this project**: Determines how much polarization shows up in the measured voltage — relevant for distinguishing OCV characterization (low rate) from drive cycles (high rate).
- **Where used**: `build_ocv.py` (effective-C-rate sanity check during inspection); drive-cycle classification context throughout. Week 2.
- **Gotcha**: A cell's nominal capacity drifts with temperature and age. "1C" today may not be "1C" next year.

### State of charge (SOC)
- **Plain explanation**: The fraction of the cell's usable charge that's still inside it, between 0 (empty) and 1 (full). Not a measured quantity — inferred from how much current has flowed in or out since a known reference point. It's the input variable for most cell-level lookup tables (especially `V_ocv(SOC)`).
- **Role in this project**: SOC is the x-axis of the OCV curve we feed into the heat-generation formula. Every time the thermal model needs `V_ocv`, it computes SOC first.
- **Where used**: `build_ocv.py` (`compute_soc`); will be used in thermal model heat-gen evaluation. Week 2.
- **Linked deep dive**: [coulomb-counting](concepts/coulomb-counting.md).

### Open-circuit voltage (OCV)
- **Plain explanation**: The cell's terminal voltage when no current is flowing and it has fully relaxed. It depends almost only on SOC and chemistry; for a Li-ion 18650 it's roughly 2.7 V at empty and 4.2 V at full. It's the "thermodynamic" voltage — what the cell *wants* to read, separate from any polarization-driven offset under load.
- **Role in this project**: Feeds the irreversible heat-generation formula `Q = I·(V − V_ocv)`. Knowing `V_ocv(SOC)` is the prerequisite for computing heat from `I` and `V`.
- **Where used**: `build_ocv.py` (`load_ocv` returns a callable); referenced by [decisions/heat-generation-irreversible-only](decisions/heat-generation-irreversible-only.md). Week 2.
- **Linked deep dive**: [ocv-soc-curve](concepts/ocv-soc-curve.md).

### Coulomb counting
- **Plain explanation**: Tracking SOC by integrating current over time, treating the cell as a bucket of charge. If you know the bucket's size and you've been watching the spout, you know how full it is — assuming you started from a known state.
- **Role in this project**: How we turn the cumulative `Ah` column into an SOC axis in `build_ocv.py`, and how SOC will be tracked across drive cycles when the thermal model runs.
- **Where used**: `build_ocv.py` (`compute_soc`). Week 2.
- **Linked deep dive**: [coulomb-counting](concepts/coulomb-counting.md).

### Pseudo-OCV / near-equilibrium approximation
- **Plain explanation**: True OCV requires zero current, but a real cycler can't measure voltage at literally zero current for hours. So instead we run a very low-current discharge (e.g. C/20) where polarization is small but non-zero, and treat the measured voltage as "approximately OCV." It's an engineering compromise: not exactly thermodynamic OCV, but much cheaper than waiting for full relaxation.
- **Role in this project**: How we build the OCV table from the LG C/20 file. Cleaner and faster than reconstructing OCV from HPPC rest periods.
- **Where used**: `build_ocv.py` (`load_c20`, `segment_discharge`); rationale in [decisions/ocv-characterization](decisions/ocv-characterization.md). Week 2.
- **Linked deep dive**: [ocv-soc-curve](concepts/ocv-soc-curve.md).
- **Gotcha**: At very low and very high SOC, the OCV curve is steep — small SOC errors translate into large `V_ocv` errors, and the C/20 file may not cover the full 0–100 % range.

### Overpotential / polarization
- **Plain explanation**: When current flows through a cell, the terminal voltage diverges from the OCV because of internal resistance and electrochemical kinetics. That divergence is the "overpotential": `V − V_ocv`. It contains both ohmic drop (instant, proportional to current) and slower charge-transfer / diffusion components. The energy associated with overpotential dissipates as heat.
- **Role in this project**: The irreversible heat-generation formula `Q = I·(V − V_ocv)` is literally "current times overpotential" — overpotential is the engine of heating in our model.
- **Where used**: Referenced by [decisions/heat-generation-irreversible-only](decisions/heat-generation-irreversible-only.md). Computed implicitly inside the heat-gen call. Week 2 → Week 3.

### Heat generation `Q = I·(V − V_ocv)`
- **Plain explanation**: The electrical-to-thermal coupling for our model. Whatever electrical energy doesn't go into changing the cell's stored chemical energy ends up as heat; that surplus is `I` times the overpotential `(V − V_ocv)`. The product is non-negative in both charge and discharge, as physics requires.
- **Role in this project**: The single source term driving the thermal ODE. Without it the thermal model has no input.
- **Where used**: Will be in `thermal_model.py` (Week 3). Decided in [decisions/heat-generation-irreversible-only](decisions/heat-generation-irreversible-only.md). Sign correctness covered in [decisions/sign-convention](decisions/sign-convention.md).
- **Gotcha**: With our raw current convention (negative = discharge), the *formula* looks sign-flipped vs. Lin 2014. It isn't — the algebra is identical. Don't "fix" it.

### Irreversible vs. entropic heating
- **Plain explanation**: A battery has two sources of heat. **Irreversible heat** comes from overpotential — friction-like losses that always produce heat regardless of direction. **Entropic (reversible) heat** comes from the temperature dependence of the reaction itself: as charge moves, the entropy of the cell changes, releasing or absorbing heat. Entropic heat can be positive or negative; irreversible heat is always positive.
- **Role in this project**: We model **only the irreversible term**, because computing the entropic term requires `∂U_ocv/∂T` (temperature-resolved OCV), which neither dataset provides.
- **Where used**: [decisions/heat-generation-irreversible-only](decisions/heat-generation-irreversible-only.md). Week 2.
- **Gotcha**: At low C-rates the entropic term can be a significant fraction of the irreversible term. Our model will under- or over-predict heating in those regimes, with the sign depending on SOC.

### I²R (Joule) heating
- **Plain explanation**: The simplest heat-generation model: `Q = I² · R_internal`. It's always positive (squared current) but it treats the cell as a single resistor and can't distinguish charge from discharge dynamics. Lin 2013 used it; Lin 2014 upgraded to `Q = I·(V − V_ocv)`.
- **Role in this project**: Sign-immune cross-check. If a model run produces `Q < 0` somewhere from the `I·(V − V_ocv)` formula, recomputing with `I²·R` (which is structurally positive) is a fast diagnostic for a sign bug.
- **Where used**: Not in the main pipeline; referenced as a cross-check in [decisions/heat-generation-irreversible-only](decisions/heat-generation-irreversible-only.md). Week 3.

### Heat capacity / specific heat
- **Plain explanation**: How much energy it takes to raise an object's temperature by one degree. Specific heat (`c_p`, J/(kg·K)) is a per-mass property of a material; heat capacity (`C`, J/K) is the same property for a specific object (mass × c_p). For Li-ion cells, `c_p ≈ 800–1000 J/(kg·K)`. Our 47 g 18650 has total heat capacity ~ 40–47 J/K.
- **Role in this project**: Determines how fast the cell heats up under a given heat-generation rate. Sets the thermal time constants. We pin `C_total = m·c_p` from physical properties rather than fitting it.
- **Where used**: [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md); will be in `thermal_model.py` (Week 3).
- **Gotcha**: `C_core` and `C_surf` individually are not as well known as their sum — splitting them requires a literature ratio or a structural assumption.

### Thermal resistance
- **Plain explanation**: How hard it is for heat to flow from one place to another, in K/W. A large `R` means a small temperature gradient drives only a small heat flow — heat is "trapped." `R_int` (core ↔ surface) measures conduction through the jellyroll and contact; `R_conv` (surface ↔ ambient) measures convection through the surrounding air or coolant.
- **Role in this project**: The two `R` parameters in the thermal ODE control the steady-state gradients (`T_core − T_surf = Q·R_int`) and the cooling rate (`τ_cool = (C_core+C_surf)·R_conv`).
- **Where used**: [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md); [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md). Week 3.
- **Gotcha**: `R_int` is the **least identifiable** parameter from surface-only data — see [concepts/identifiability](concepts/identifiability.md). It's also the multiplier on the very gap we're trying to estimate.

### Convective cooling (cooling tails)
- **Plain explanation**: When the cell is at rest (no current), heat generation stops and the surface temperature decays exponentially toward ambient — a "cooling tail." The decay time constant is set by `(C_core + C_surf) · R_conv`. By fitting an exponential to a cooling tail, you can back out `R_conv` independently of any modeling of the heat-generation phase.
- **Role in this project**: Our calibration uses cooling tails to anchor `R_sa` before fitting the remaining thermal parameters. This collapses one degree of identification freedom without requiring a free-parameter fit.
- **Where used**: [`calibrate.py`](../calibrate.py) — `find_cooling_tails`, `fit_exponential_decay`, `step2_anchor_R_sa`. Week 2. Run produced R_sa = 2.35 K/W from 2 tails.
- **Linked deep dive**: [cooling-tail-thermal-id](concepts/cooling-tail-thermal-id.md).
- **Gotcha**: A "cooling tail" must actually be at rest. A long charge segment at low current still has heat generation and is not a true cooling tail; filtering on `|I| < ε` for some threshold matters.

### Thermal time constant
- **Plain explanation**: For a first-order thermal system, the characteristic time it takes the temperature to relax to within ~37 % of its final value after a step change. Mathematically: `τ = R·C`. A small τ means the cell responds quickly; a large τ means it lags. For 18650 cells with passive cooling, τ is typically tens of seconds to a few minutes.
- **Role in this project**: Sets how long a "cooling tail" needs to be to fit `R_conv` reliably. Also tells us how much temporal resolution we need in the resampled data — 1 Hz is plenty for thermal dynamics with τ on the order of a minute or more.
- **Where used**: Implicit in [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) cooling-tail strategy; conceptual basis for choosing 1 Hz sampling in `data_loading.py` (`resample_1hz`). Week 1 → Week 3.

### Lumped-capacitance assumption
- **Plain explanation**: Treating an object as a single uniform "lump" with one temperature, ignoring internal spatial gradients. Valid when the object's internal conduction is much faster than the surface heat exchange (small Biot number). For a Li-ion cell that's *only partly* true: the radial gradient is significant at high C-rates, but the longitudinal (axial) gradient is small. We take the longitudinal gradient as zero and resolve the radial as two nodes.
- **Role in this project**: Justifies modeling the cell with two states (core and surface) rather than a full radial PDE. Tradeoff: simpler model, faster to fit, but cannot capture detailed internal distributions.
- **Where used**: [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md). Week 3.
- **Gotcha**: At very high C-rates the radial gradient grows and the two-node approximation degrades. Our LG/Panasonic data tops out around 10–15C, comfortably in the regime where two-node is reasonable.

### Two-state (core/surface) model
- **Plain explanation**: A thermal model with two lumped temperatures: a core node (where heat is generated) and a surface node (where heat exchanges with ambient). They're coupled by internal conduction `R_int`. Surface temperature is measured; core temperature is the quantity we ultimately want to estimate.
- **Role in this project**: Our thermal model structure. The whole point of using two states instead of one is that surface temperature alone underestimates internal heating at high C-rates — there's a real core/surface gap, and the project goal is to predict it.
- **Where used**: [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md). Week 3.

### Coupled first-order ODEs
- **Plain explanation**: A system of differential equations where each equation contains the derivative of *one* state variable, but the right-hand side mixes multiple states. "Coupled" means the equations can't be solved one at a time — they evolve together. "First-order" means only first derivatives appear (no `d²T/dt²` directly).
- **Role in this project**: Both thermal ODEs (one for core, one for surface) are first-order, and each contains both temperatures on the right side. We integrate them jointly with `solve_ivp`.
- **Where used**: [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md). Week 3.

### ODE integration with `solve_ivp`
- **Plain explanation**: `scipy.integrate.solve_ivp` takes a function that returns `dy/dt` given the current state and time, plus an initial condition, and produces the time-evolved state. Default method is an adaptive Runge-Kutta. It chooses step sizes automatically to balance accuracy and cost; you don't pre-discretize the time axis.
- **Role in this project**: How we forward-evolve the two-state thermal ODE on each drive cycle, given the heat-gen and ambient-temp time series as inputs.
- **Where used**: Planned in `thermal_model.py` (Week 3); design committed in [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md).
- **Gotcha**: The input signals (`I`, `V`, `T_amb`) are sampled at 1 Hz, but `solve_ivp` will ask for the input at arbitrary intermediate times. Wrap each input as a `scipy.interpolate.interp1d` callable before passing to the RHS function — don't pull from the DataFrame inside the RHS.

---

## ML, data & evaluation

### Time-series data
- **Plain explanation**: Data where each row is indexed by time and adjacent rows are not independent. Today's temperature depends on yesterday's. Standard ML assumes independent samples; time series violate that, which breaks naive train/test splits and breaks naive cross-validation. Treating time-series data the way you'd treat tabular data is one of the most common ML mistakes.
- **Role in this project**: Every record in `data/processed/` is a time series — drive cycles where I, V, T_surface, etc. are sampled at 1 Hz.
- **Where used**: `data_loading.py` end-to-end; will be the substrate for all of Week 3's thermal model and Week 4's ML stage.

### Resampling to uniform 1 Hz
- **Plain explanation**: The raw cycler data is logged at variable rates (10 Hz during drive cycles, slower during pauses). For both ODE integration and ML feature engineering, a uniform time grid is much easier to work with. We use linear interpolation to put every record onto a 1 Hz grid that starts at time 0.
- **Role in this project**: Makes `solve_ivp` inputs trivially interpolatable, makes feature engineering uniform across records, and reduces file sizes substantially without losing thermally meaningful information (thermal time constants are seconds to minutes, well above 1 Hz Nyquist).
- **Where used**: `data_loading.py` (`resample_1hz`). Week 1.
- **Gotcha**: Linear interpolation between sparse pause samples is fine for slowly-varying signals but smears out sharp transitions if you have them. The cumulative columns (`Ah`, `Wh`) interpolate cleanly because they're smooth by construction.

### Why you never random-split a time series
- **Plain explanation**: A row from the middle of a UDDS cycle is almost identical to the rows immediately before and after — they share the same drive-cycle context, same ambient, same battery state. If you randomly split rows into train and test, the test rows are statistically nearly equal to nearby training rows, and your "test RMSE" measures memorization, not generalization. The fix is to hold out entire cycles (or entire ambient temperatures, or entire cells) — chunks large enough that test data was never seen.
- **Role in this project**: Drives every validation decision we'll make. Held-out **whole cycles** is the minimum bar; held-out ambient temps is stronger.
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md); [`features_and_split.py`](../features_and_split.py) implements the GroupKFold/LOCO blocking. Week 3+.
- **Linked deep dive**: [blocked-time-series-split](concepts/blocked-time-series-split.md).
- **Gotcha**: It's tempting to use `sklearn.model_selection.train_test_split` because it's one line. Resist. Use `GroupKFold` with a cycle ID, or a manual cycle-level split. **And tune hyperparameters inside the train folds only** — the inner CV must also be GroupKFold, or alpha selection silently leaks the held-out cycle.

### Data leakage
- **Plain explanation**: When information that wouldn't be available at prediction time sneaks into training. The classic examples: scaling using statistics computed on the full dataset (test stats leak into train), feature engineering across a future window, or random splitting time series. The result is inflated validation scores that don't survive contact with reality.
- **Role in this project**: Two specific leakage risks here. (1) Random splits — covered above. (2) Future information in rolling-mean / lag features: a window that extends *forward* in time leaks future data into "now."
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md). Week 3+.
- **Gotcha**: `pandas.DataFrame.rolling(window=10).mean()` defaults to centered windows. Use `.rolling(window=10, center=False).mean()` to keep features causal.

### Generalization / held-out validation
- **Plain explanation**: A model "generalizes" if its performance on unseen data matches its performance on training data. Held-out validation — keeping a chunk of data the model never saw during training — is how we measure that. For time series, "unseen" must mean unseen *cycles* or *conditions*, not unseen *rows*. A model that nails training data but fails on held-out is overfit.
- **Role in this project**: Held-out cycles drive every evaluation metric we report — both for the thermal-model surface fit (Week 3) and the ML stage (Week 4).
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md). Week 3+.
- **Gotcha**: Held-out **surface** RMSE does *not* certify the core predictions — see [report-notes.md](report-notes.md).

### Naive baseline
- **Plain explanation**: The simplest possible predictor that a real model has to beat in order to be worth shipping. For temperature estimation, the naive baselines are "predict T_ambient" (assume cell is at ambient) and "predict T_surface" (use surface as a proxy for core). Reporting a baseline alongside the model's RMSE forces honesty: a 1 °C-RMSE model sounds good until you find out the naive baseline got 1.5 °C.
- **Role in this project**: Required reporting for both the thermal model's surface fit and the ML stage's core estimates.
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md); [`calibrate.py`](../calibrate.py) `naive_baseline_metrics` reports both `nominal_setpoint` and `T_inf` baselines.
- **Gotcha**: Pick baselines that an actual user could compute without our model. "Predict T_amb" is one; "predict last seen T_surface" is another. Don't pick a baseline so weak that beating it is meaningless — but also, see the next entry: don't pick one so weak that *winning* against it tells you nothing about whether your model's *dynamics* are right.

### Level bias vs dynamics — why a model can lose to a trivial baseline
- **Plain explanation**: A surface-temperature prediction error has two distinct parts. **Level**: a constant or slowly-varying offset — wrong starting point, wrong asymptote, wrong calibration zero. **Dynamics**: the shape of the response to load — how T_s rises during a discharge, decays during a rest. A model can have correct dynamics but the wrong level (predict the right shape, in the wrong place on the y-axis) — and lose to a constant-only baseline that has no dynamics but happens to be at the right level. Conversely, a model can have the right level but bad dynamics. The two failure modes look identical in headline RMSE but mean different things.
- **Role in this project**: The 2026-06-16 calibration lost to `T_s = T_amb_nominal` by ~1 °C per cycle. Diagnosis: the ODE was integrating around the wrong steady-state asymptote (nominal setpoint instead of the cooling-tail T_inf), producing a uniform ~1 °C overshoot. Pure level error. After switching the cooling sink to T_inf (2026-06-17), the model beat the nominal baseline — but that comparison conflated the level fix with any dynamics contribution. The 2026-06-18 integrity check added a second baseline `T_s = T_inf` (already at the correct level) to isolate the dynamics contribution. Result: model loses to the T_inf baseline on every held-out cycle, meaning on these mild cycles the model's dynamics are not currently adding value beyond knowing the rest temperature.
- **Where used**: [`calibrate.py`](../calibrate.py) `naive_baseline_metrics`; both baselines reported in Step 6 of [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md); the 2026-06-18 entry in [report-notes.md](report-notes.md).
- **Gotcha**: Always include a constant baseline that **already has the correct level** (e.g., the cooling-tail asymptote, the long-time average, the last-rest value). Otherwise you cannot tell whether your model is winning because its dynamics are right or because its mean is. The level-correct baseline isolates the dynamics contribution; a model that ties or loses to it tells you the dynamics aren't earning their parameters on this evaluation. **And gate by excursion**: a small `|T_observed − constant_baseline|` peak means the level-correct baseline already covers most of the y-range and the dynamics have nothing to win. Report the peak excursion next to RMSE so a "loss to T_inf baseline" on a 0.9 °C-excursion cycle isn't conflated with one on a 4.5 °C-excursion cycle.

### RMSE / MAE / max error
- **Plain explanation**: Three common error metrics. **RMSE** (root mean squared error) penalizes large errors quadratically — heavy on outliers. **MAE** (mean absolute error) penalizes all errors linearly — more robust. **Max error** is the single worst miss — relevant for safety margins. They tell different stories; report all three.
- **Role in this project**: Every model performance number — thermal-model surface RMSE, ML core RMSE, baseline RMSE — comes with all three. Reporting only RMSE invites optimizing for it and hiding tail behavior.
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md). Weeks 3–4.
- **Gotcha**: RMSE and MAE differ when errors are non-normally distributed. A model with low RMSE but high MAE-to-RMSE ratio has rare big misses; the other way around suggests systematic small bias.

### Feature engineering (lags, rolling means, derivatives)
- **Plain explanation**: For time-series ML, raw current and voltage at the current instant rarely contain enough information; the model often needs **lags** (`I(t−1)`, `I(t−5)`), **rolling statistics** (mean current over the last 10 seconds), and **derivatives** (`dI/dt`) to capture dynamics. These are computed as new columns from the raw signals and become the ML model's actual inputs.
- **Role in this project**: Planned for the Week 4 ML stage. Will turn the per-row `(I, V, T_surface, T_ambient)` into something a non-recurrent model (like gradient boosting) can use to capture temporal context.
- **Where used**: Planned in Week 4 ML module — not yet written.
- **Gotcha**: Every rolling window must be one-sided (past only). Two-sided windows leak future information into training features and inflate validation scores.

### Ridge regression / regularization
- **Plain explanation**: Linear regression that penalizes large coefficient magnitudes by adding `λ·Σβ²` to the loss. This shrinks coefficients toward zero, trading a little fit on training data for a lot of stability — especially valuable when input features are correlated (multicollinearity). The hyperparameter `λ` (or `alpha` in sklearn) controls how aggressive the shrinkage is.
- **Role in this project**: Likely first baseline ML model in Week 4 — interpretable, fast, lets us see which engineered features actually matter.
- **Where used**: Planned for Week 4 (`sklearn.linear_model.Ridge`). Not yet written.
- **Gotcha**: Ridge needs scaled features; the penalty `Σβ²` treats all coefficients on equal footing, so different-scale features distort the fit. Use `StandardScaler` (fit on train only, to avoid [data leakage](#data-leakage)).

### Gradient boosting
- **Plain explanation**: An ensemble method that builds many small decision trees, each one trained to correct the residuals of the ensemble so far. The result is a flexible nonlinear regressor that captures interactions automatically without manual feature crosses. Sklearn's `HistGradientBoostingRegressor` is the fast modern implementation.
- **Role in this project**: The main ML model for the core-temp regressor, head-to-head against the ridge baseline.
- **Where used**: [`train_hgbr.py`](../train_hgbr.py); compared to ridge in `data/ml/ridge_vs_hgbr_loco.parquet` and `data/ml/hgbr_summary.json`. Week 3.
- **Contrast with ridge**: Trees are **collinearity-robust** — they don't care that voltage_V and voltage_lag_1s are nearly identical, they'll just pick whichever splits cleanest at each node. Ridge spreads weight across collinear copies and ends up with a confused top-10 coefficient ordering. The empirical contrast on this project: ridge's top-10 was dominated by **voltage** features (voltage_lag_60s leading) because they correlate strongly with the polarization signal a linear model needs to reason about; HGBR's top-10 by permutation importance is dominated by the **smoothed surface temperature** (T_surface_roll_60s, T_surface_lag_60s, T_surface_C) because trees can extract the surface→core relationship directly, with `I²_roll_60s` fourth as the Joule-heating signal. Different models surface different signals; both are correct readings of what each model leans on.
- **Gotcha**: Boosting handles many features gracefully, but it'll still overfit if you let it. The number of trees and tree depth are the key knobs; tune them with held-out cycles, not random row splits. **And** sklearn HGBR's `early_stopping=True` carves the val set via a random *within-cycle* split (`validation_fraction`) — that leaks across cycle boundaries. Either treat `max_iter` as a hyperparameter inside the GroupKFold grid (what `train_hgbr.py` does), or carve the val set from whole train cycles yourself.

### Ablation study
- **Plain explanation**: Systematically removing one component of the model (a feature, a sub-model, an input) and re-evaluating, to find out how much that component actually contributed. If removing it doesn't hurt much, it wasn't doing much; if it tanks performance, that component is doing real work.
- **Role in this project**: The planned **no-T_surface ablation** — does the ML model still beat the naive baseline when we deny it the surface-temperature input? If yes, the ML is genuinely learning the cell's thermal dynamics from `I` and `V`. If no, it was just regressing surface onto core via a small offset.
- **Where used**: [decisions/validation-surface-only](decisions/validation-surface-only.md). Planned for Week 4.

### Physics-informed / hybrid modeling
- **Plain explanation**: An ML model that incorporates known physics — either as a soft loss term, as a hard structural constraint, or by using the physics model to *generate* training data the ML then learns from. The motivation is to get a good model with much less labeled data than pure ML would need, and to make the model behave sensibly outside the training distribution.
- **Role in this project**: The whole project shape. The thermal model produces core-temp labels from `I, V, T_s`; the ML stage learns the surrogate mapping. Closely related to [sim-to-real](concepts/sim2real-gap.md) — our "synthetic" labels come from our calibrated thermal model, which itself was calibrated on real surface data.
- **Where used**: Project framing — see [sources/zheng-2025-sim2real](sources/zheng-2025-sim2real.md). Weeks 3–4.
- **Gotcha**: "Physics-informed" doesn't guarantee correctness. If the physics model is biased (e.g. wrong `R_int`), the ML inherits that bias and confidently amplifies it. See [report-notes.md](report-notes.md).

### Least-squares Jacobian confidence intervals
- **Plain explanation**: After `scipy.optimize.least_squares` converges, the Jacobian at the solution lets you compute approximate 95 % confidence intervals: `cov ≈ σ²·(JᵀJ)⁻¹`, then `±1.96·√diag(cov)`. Standard and fast for reporting parameter uncertainty.
- **Role in this project**: How we quantify whether each fitted thermal parameter is well-constrained by the surface-temperature data. Reported alongside every fit in `calibrate.py`.
- **Where used**: [`calibrate.py`](../calibrate.py) — `_jacobian_ci`, used in `step3_diagnostic_fit` and `step4_central_fit`. Week 2.
- **Linked deep dive**: [least-squares-jacobian-confidence](concepts/least-squares-jacobian-confidence.md).
- **Gotcha**: A tight Jacobian-based CI describes *local* curvature only. If the cost has a flat ridge, the optimizer converges to some point on the ridge with steep perpendicular curvature, returning a confidently tight CI on a parameter that's actually unidentified. Pair it with a band sweep when you suspect identifiability problems.

### Identifiability
- See [concepts/identifiability](concepts/identifiability.md).
- **Where used**: [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md); [decisions/validation-surface-only](decisions/validation-surface-only.md); confirmed empirically by Step 5 of `calibrate.py` (R_cs band sweep). Weeks 2–3.

### Persistent excitation
- See [concepts/persistent-excitation](concepts/persistent-excitation.md).
- **Where used**: Cycle selection for thermal calibration — [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md). Helper to compute α₀ per cycle planned for Week 3.

### Sim-to-real gap
- See [concepts/sim2real-gap](concepts/sim2real-gap.md).
- **Where used**: The framing of why label quality caps ML quality — [report-notes.md](report-notes.md); [decisions/validation-surface-only](decisions/validation-surface-only.md). Weeks 3–4.
