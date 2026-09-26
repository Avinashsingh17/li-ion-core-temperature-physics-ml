# Calibration: constrained fit (not free 4-param)

## The question
How do we calibrate the lumped two-state thermal-model parameters when the core node is **unobserved**?

The model has roughly four parameters:
- `C_core` — core heat capacity (J/K).
- `C_surf` — surface (case) heat capacity (J/K).
- `R_int` — internal thermal resistance, core ↔ surface (K/W).
- `R_conv` — convection resistance, surface ↔ ambient (K/W).

But we only measure surface temperature.

## Options considered
- **Free 4-parameter fit** against surface temperature.
- **Constrained fit** with physically-derived priors / pinned values.
- **Two-stage**: identify cooling parameters from rest segments, then thermal capacitance from heating segments.

## The call
**Constrained fit, not free 4-param.** Specifically:

1. **Pin total heat capacity** from physical properties: `C_total = m · c_p`, with `m` ≈ 47 g (LG HG2) and `c_p ≈ 800–1000 J/(kg·K)`. Split between `C_core` and `C_surf` per a literature ratio rather than freeing both.
2. **Constrain `R_conv`** from **cooling-tail** segments — periods with `I ≈ 0` where surface temperature decays exponentially toward ambient; the decay time constant gives `(C_core + C_surf) · R_conv`, anchoring `R_conv`.
3. **Fit only the remaining poorly-pinned parameter(s)** (primarily `R_int`) using `scipy.optimize.least_squares` on the surface-temperature residual.
4. **Report parameter confidence intervals** from the `least_squares` Jacobian (cov ≈ `(JᵀJ)⁻¹ · σ²`).
5. **Document which parameters are well-constrained and which aren't.** The writeup states this explicitly.

## Why
- The core node is unobserved → a naive 4-param least-squares is **ill-posed** (an identifiability problem): the optimizer can trade off `C_core` vs `C_surf` and `R_int` vs `R_conv` to fit the surface trace while producing nonsensical core dynamics.
- Anchoring physically-meaningful parameters from first principles (`m·c_p`) and from a clean signal regime (cooling tails) collapses the degeneracy and leaves a small, well-conditioned residual problem.
- This makes parameter uncertainty quantifiable instead of unbounded.

## Risk: `R_int` is the least-identifiable parameter — and it sets the core/surface gap

Added 2026-06-12 after ingesting [Lin 2013](../sources/lin-2013-identifiability.md) and [Zheng 2025](../sources/zheng-2025-sim2real.md).

The decision above leaves `R_int` (≡ `R_c`, core↔surface conduction) as the principal free parameter. **That is exactly the parameter that's least identifiable from surface-temperature data**:

- At quasi-steady state, the core/surface gap is `T_core − T_surf ≈ Q · R_int`. So `R_int` is the multiplier on the very quantity we ultimately want to estimate.
- Lin 2013 derives formally that of the 5 physical thermal parameters, only 3 lumped combinations are identifiable from `T_s` alone — `R_int` sits in the weakest direction.
- Zheng 2025 reaches the same conclusion empirically: their thermal conductivity `k_t` (the analog of `R_int`) is the only thermal parameter that ML cannot recover from when only surface signals are available — because surface temperature is structurally insensitive to it.

**Consequence for our pipeline.** The thermal model produces the core-temperature **labels** that the ML stage trains against. An `R_int` error biases those labels by roughly a fixed offset (≈ `Q · Δ R_int`). The ML model has **no ground truth on core temperature**, so it cannot detect or correct that bias — it will faithfully reproduce biased labels.

**Consequence for validation.** A good held-out **surface**-temperature fit does **NOT** certify the core labels. A model with the wrong `R_int` can match `T_s` essentially perfectly while predicting `T_core` badly. This failure mode is invisible to the headline `T_s` RMSE metric. See [validation-surface-only](validation-surface-only.md) and the headline caveat in [report-notes](../report-notes.md).

**LOCKED 2026-06-16: Option A** — anchor `R_int` from jellyroll geometry, then sweep across its plausible physical range and report the resulting `T_core` spread as a label band.

The two candidates considered:

- **Option A (LOCKED)** — anchor `R_int` from first principles (radial-conduction `R = ln(r_o/r_i)/(2π·k·L)` using 18650 dimensions and `k_radial` ∈ [0.2, 1.0] W/(m·K)). Re-fit the heat-capacity split at each of `R_cs_low`, `R_cs_central`, `R_cs_high`; the spread of the resulting `T_core` trajectories is the label band that propagates into the ML stage.
- Option B (rejected) — fit `R_int` and propagate the local Jacobian CI only. Rejected because the diagnostic-fit CI turned out to be *locally* tight but globally meaningless (see results below).

## Label generation provenance (2026-06-19, `generate_labels.py`)

Produced the modeled core-temperature labels (`T_core_model_C`) plus the
R_cs uncertainty band (`T_core_model_lo_C` / `T_core_model_hi_C`) that the
Week-3 ML stage will consume. **No re-fitting** — Step-5 splits were loaded
directly from `data/calibration/calibration_results.json`.

| input | value |
|---|---|
| C_total | 43.20 J/K (pinned from m·c_p, locked) |
| T_inf @ 25 °C | 23.15 °C (cooling-tail asymptote, single US06 tail, LOW CONFIDENCE) |
| R_sa | 5.21 K/W (median of 29 free-asymptote τ values) |
| R_cs band | low 1.697 / central 3.394 / high 8.486 K/W (geometric) |
| splits (Step 5) | low 0.9500 / central 0.9283 / high 0.9300 |
| C_core / C_surf (central) | 40.10 / 3.10 J/K |
| soc_init | 1.0 (LG CCCV before every drive cycle) |

**Cycles** (11 LG 25 °C drive cycles, locked): US06, LA92, UDDS, Mixed1..8.
HWFET @ 25 °C is excluded (dead — no current activity). All Charge* files
excluded. Cold ambients (0 °C / 10 °C / 40 °C / negatives) excluded per the
regime restriction above — the cold-ambient model errors documented in the
2026-06-18b validation make those cycles unsafe as label sources.

**Output**: `data/labeled/<file_id>.parquet` (one per cycle) plus
`data/labeled/_labels_index.parquet`. `data/processed/` is untouched.

**Sanity asserts** (per cycle, would have raised on violation): time_s
monotone non-decreasing; all modeled values finite; no NaN in T_core / T_surf
columns; `T_core_central ≥ T_surf_central` everywhere (numerical tolerance
1e-3 °C). All 11 cycles passed.

### Per-cycle summary (`_labels_index.parquet`)

| cycle | dur (min) | peak T_core (°C) | max gap (°C) | band spread max (°C) | peak \|T_s−T_inf\| (°C) | surface RMSE (°C) | clamps low/high | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| US06   |  66.9 | **31.28** | **3.36** | **5.05** | 3.67 | 1.046 | 0 / 0  | **gap > 3 °C** |
| LA92   | 168.0 | 25.63 | 1.02 | 1.41 | 1.88 | 0.329 | 0 / 0  | |
| UDDS   | 266.1 | 24.92 | 0.76 | 0.75 | 1.56 | 0.498 | 0 / 0  | |
| Mixed1 | 128.7 | 29.24 | 2.47 | 3.09 | 2.72 | 0.384 | 0 / 0  | |
| Mixed2 | 131.9 | 29.19 | 2.53 | 2.71 | 3.03 | 0.379 | 0 / 17 | |
| Mixed3 | 123.0 | 29.00 | 2.37 | 2.93 | 2.40 | 0.384 | 0 / 3  | |
| Mixed4 | 134.7 | 28.46 | 2.15 | 2.57 | 2.40 | 0.333 | 0 / 0  | |
| Mixed5 | 120.3 | 29.87 | 2.82 | 3.48 | 3.14 | 0.400 | 0 / 0  | |
| Mixed6 | 130.2 | 28.82 | 2.29 | 2.90 | 2.30 | 0.436 | 0 / 0  | |
| Mixed7 | 122.0 | 28.15 | 2.03 | 2.39 | 2.09 | 0.360 | 0 / 0  | |
| Mixed8 | 143.2 | 28.03 | 1.97 | 2.58 | 2.09 | 0.369 | 0 / 0  | |

The Mixed1/Mixed2 surface RMSEs (0.384 / 0.379 °C) match the 2026-06-18b
validation values (0.383 / 0.380 °C) within numerical noise, confirming
the label run used the same parameters and produced the same trajectories.

**1 cycle flagged for central gap > 3 °C**: US06 @ 25 °C (gap_max = 3.36 °C,
band spread 5.05 °C). US06 is the most aggressive 25 °C cycle (peak |I| ≈
17 A); the larger gap is physical, not a sanity failure. The wider band
spread there means R_cs uncertainty translates to wider T_core label
uncertainty on this specific cycle — the ML stage should weight US06 by
that band, not treat it as a clean point estimate.

**Clamp counts**: Mixed2 (17 high) and Mixed3 (3 high) hit the upper SOC
clamp briefly. Small, brief overshoots above SOC = 1.0 from regen bursts;
no low-clamp hits anywhere. Inconsequential for labels.

## Calibration result (run 2026-06-18b, `calibrate.py` — T_inf rule fix + aggressive validation)

The 2026-06-18a result below was corrected with two refinements:

1. **T_inf rule fix.** The cross-ambient mean offset (−0.85 °C) was being applied at warm ambients with too-few tails, biasing 25 °C T_inf high. The rule is now: **any ambient with ≥1 non-excluded tail uses the median of its own tails**, with a `LOW CONFIDENCE` flag for n=1. Only ambients with **zero** tails use the cross-ambient mean offset. This restores 25 °C T_inf to 23.15 °C (single US06 tail) instead of the biased 24.16 °C fallback. The offset-vs-ambient trend is now surfaced so the reader can see why a global mean is the wrong extrapolation to warm ambients.
2. **Aggressive cold validation cycles added.** `US06 @ 0 °C` and `LA92 @ 0 °C` — both leakage-clean (added to the tail-pool exclusion list; 0 °C tail pool drops from 10 to 8, still strong), both at an ambient whose T_inf is well-anchored (median of 8 tails).

### Offset-vs-ambient trend (29 tails after leakage + aggressive-val exclusion)

| ambient | median offset | n tails | note |
|---|---|---|---|
|  0 °C | −0.66 °C |  8 | well-anchored |
| 10 °C | −0.82 °C | 19 | well-anchored |
| 25 °C | **−1.85 °C** |  **1** | **LOW CONFIDENCE — only US06 retained** |
| 40 °C | −1.62 °C |  1 | LOW CONFIDENCE (already was) |
| cross-ambient mean: −0.85 °C | (used only for 0-tail ambients: −20, −10 °C) | | |

The offset **grows in magnitude with temperature**. Using a single cross-ambient mean to fill in a warm ambient is therefore the wrong fallback; per-ambient medians (even from a single tail) are closer to the truth than the mean.

### Cleaned diagnostic + central fit

| stage | result |
|---|---|
| Step 3 diagnostic fit | `split = 0.903 ± 0.006`, `R_cs = 1.62 ± 0.009 K/W` (±0.5 % local CI)†. Cost dropped 10 % from initial — the optimizer moved this time, in contrast to the 2026-06-18a run where the biased T_inf had it stuck. |
| Step 4 central params | `C_core = 40.10 J/K, C_surf = 3.10 J/K, R_cs = 3.39 K/W, R_sa = 5.21 K/W`. Cal RMSE = 0.540 °C — close to the 2026-06-17 run's 0.474 °C, confirming the original numbers were essentially right; the 2026-06-18a regression was the fallback bias. |
| Step 5 band sweep | splits stable 0.93 → 0.95 across R_cs ∈ [1.70, 8.49] K/W; cal RMSE varies 0.52 → 0.57 °C; **T_core peak spreads 5.07 °C** — R_cs unidentifiability still holds. |

† Integrator-noise artifact of the loosely integrated fitting residual; see report/part2_phase_a.md §7.4 and report-notes [2026-09-23].

### SEVERITY-STRATIFIED held-out validation (5 cycles)

| cycle | regime | peak `|T_s − T_inf|` | clamps | model RMSE | baseline (nominal) | baseline (T_inf) | margin vs nominal | margin vs T_inf | core−surf gap (max) |
|---|---|---|---|---|---|---|---|---|---|
| Mixed1 @ 25 °C | mild same-amb | 2.72 °C | 0 / 0 | **0.383** | 0.918 | 1.160 | **+0.535** (wins) | **+0.777** (wins) | 2.46 °C |
| Mixed2 @ 25 °C | mild same-amb | 3.03 °C | 0 / 17 | **0.380** | 0.826 | 1.232 | **+0.447** (wins) | **+0.852** (wins) | 2.53 °C |
| UDDS  @ 10 °C  | mild diff-amb | 0.92 °C | 0 / 0 | 0.468 | 0.759 | 0.158 | +0.292 (wins) | **−0.309 (loses)** | 1.05 °C |
| **US06 @ 0 °C** | **aggressive cold** | **4.55 °C** | 0 / 0 | **3.612** | 1.994 | 2.592 | **−1.618 (loses)** | **−1.020 (loses)** | **6.69 °C** |
| **LA92 @ 0 °C** | **aggressive cold** | **2.13 °C** | 0 / 0 | **1.037** | 0.353 | 0.849 | **−0.684 (loses)** | **−0.188 (loses)** | **2.79 °C** |

**Excursion column reads as "the dynamic signal that was there for the model to win"**: a small excursion (e.g. UDDS @ 10 °C at 0.92 °C) means the constant T_inf baseline already covers most of the y-range; the model's dynamics have little room to add value. A large excursion (Mixed1/2 at 25 °C; US06 @ 0 °C) is a real dynamics test.

### Where the model adds value — the honest characterization

- **In the calibration regime (25 °C mild drive cycles)** the model now **beats both baselines on both held-out cycles by clear margins** (0.45–0.85 °C). This is the dynamics contribution working as intended.
- **At a different ambient where T_s barely moves (UDDS @ 10 °C, excursion 0.92 °C)** the model beats the nominal baseline but loses the dynamics-only test against T_inf. The error is too small to demonstrate dynamics value either way; **the model isn't broken there, the test isn't strong enough.**
- **At aggressive cold ambient (0 °C drive cycles)** the model **loses badly to both baselines**: surface RMSE 1.04–3.61 °C, core−surface gap 2.79–6.69 °C (vs literature ~1–2 °C normal). The model is over-predicting heat at 0 °C — extrapolation from 25 °C-calibrated parameters to a cold-ambient regime where internal resistance, OCV characteristic, and convective behaviour are likely all different.

**Residual decomposition on the aggressive cycles** (`plot_residual_diagnosis` output): error roughly balanced between rest periods (US06: rest RMSE 3.32, load RMSE 3.69; LA92: rest 0.85, load 1.08) — neither purely a level bias nor purely a dynamics issue. Visually the error CLIMBS through the cycle rather than holding steady, indicating heat accumulation: the model produces more heat per cycle than the cell actually generates at 0 °C, and never dissipates it.

### Central locked params (2026-06-18b, supersedes 2026-06-18a)

| param | value | basis |
|---|---|---|
| `C_core` | 40.10 J/K | from split = 0.928 |
| `C_surf` | 3.10 J/K  | C_total − C_core |
| `R_cs`   | 3.39 K/W  | geometric central (R_cs band [1.70, 8.49]) |
| `R_sa`   | 5.21 K/W  | median of 29 free-asymptote τ values |

R_cs label band T_core peak spread: **5.07 °C** (unidentifiability conclusion unchanged).

### Implication for the Week-3 ML stage

1. **Train and evaluate at 25 °C first.** The model demonstrably adds value in its calibration regime — ML labels there carry the dynamics contribution.
2. **Excursion gating.** The ML stage should report peak `|T_s − T_inf|` alongside RMSE/MAE/max, like the table above. A small excursion means an unambitious test and a small error margin to win. Don't claim ML dynamics value on a cycle the baseline already nails.
3. **Cold-ambient cycles are NOT safe for label generation as-is.** The 0 °C model error (1–3.6 °C surface RMSE) tells us the thermal model isn't currently a good label source there. Options for Week 3: (a) restrict the ML stage to 25 °C cycles only, or (b) add temperature-dependent corrections to the thermal model (OCV(T), R_internal(T)) before label generation. The latter is out of the locked 4-week scope; (a) is the pragmatic choice.
4. **R_cs label band remains the right T_core uncertainty** for the ML output, but only on cycles where the central trajectory is itself credible (i.e. not the 0 °C cycles in their current state).

## Calibration result (run 2026-06-18a, `calibrate.py` — with leakage filter + 2nd baseline)

> ⚠ The 2026-06-18a result below used `MIN_TAILS_PER_AMBIENT = 2`, which forced 25 °C T_inf to the cross-ambient-mean fallback (24.16 °C) — biased high relative to the single US06 tail observation (23.15 °C). The 2026-06-18b result above corrects the rule to `MIN_TAILS_PER_AMBIENT = 1`. Numbers below retained for change-tracking; **use 2026-06-18b**.


The 2026-06-17 run was redone with two evaluation-integrity checks:

1. **Leakage filter on the cooling-tail pool.** Tail search now explicitly excludes every validation file (Mixed1 @ 25 °C, Mixed2 @ 25 °C, UDDS @ 10 °C) before any T_inf or R_sa is derived. The 2026-06-17 25 °C T_inf used Mixed2's tail — that was leakage.
2. **Second baseline `T_surface = T_inf`** alongside the existing `T_surface = nominal_setpoint`. The model-vs-T_inf margin measures the dynamics contribution only — the level is already correct.

### Cleaned T_inf per ambient (run 2026-06-18)

|  ambient | tails used | T_inf used | source |
|---|---|---|---|
| 0 °C  | 10 | −0.67 °C | median of 10 tails (unchanged from prior) |
| 10 °C | 19 |  9.18 °C | median of 19 tails (unchanged — UDDS @ 10 °C was already absent from the pool) |
| **25 °C** | **1** | **24.16 °C** | **FALLBACK: setpoint + cross-ambient mean offset (−0.84 °C). Only US06 remained after excluding Mixed2.** |
| 40 °C | 1 | 39.16 °C | fallback (only US06 tail; already a fallback before) |

**Caveat on 25 °C**: the single non-validation US06 tail at 25 °C had `T_inf = 23.15 °C` (offset −1.85 °C), but per the user's fallback rule (n_tails < 2 → use cross-ambient mean) we use **24.16 °C**. The cross-ambient mean offset is dominated by colder ambients where the offset is smaller (−0.67 at 0 °C, −0.82 at 10 °C). The fallback therefore over-estimates 25 °C T_inf relative to the single observation we do have, and the 25 °C model fit suffers as a result. This is the price of the integrity check.

**R_sa re-anchored from 31 free-asymptote τ values**: median **5.21 K/W** (range [4.54, 8.69]). Essentially unchanged from the 32-tail estimate (5.22) — R_sa is well-anchored by the abundant 10 °C and 0 °C tails and was not sensitive to dropping Mixed2.

### Cleaned diagnostic + central fit

| stage | result |
|---|---|
| Step 3 diagnostic fit | `split = 0.930 ± 0.012`, `R_cs = 1.50 ± 0.021 K/W` (±1.4 % local CI)†. **Cost did NOT drop from initial** — optimizer terminated by xtol without moving. With the 25 °C T_inf biased high, the residual is dominated by a level offset that split/R_cs cannot fix. |
| Step 4 central params | `C_core = 41.29 J/K, C_surf = 1.91 J/K, R_cs = 3.39 K/W, R_sa = 5.21 K/W` (refit converged from same starting point with R_cs frozen). Cal RMSE = 0.889 °C. |
| Step 5 band sweep | split stable 0.92 → 0.96 across R_cs ∈ [1.70, 8.49] K/W. Cal RMSE varies 0.82 → 0.92 °C across the 5× R_cs spread. **T_core peak spreads 5.09 °C** — R_cs unidentifiability still holds. |

† This CI comes from the superseded 2026-06-18a run of the same diagnostic fit, computed with the same Jacobian-CI method. §7.4 demonstrated the artifact on the locked 2026-06-18b value, not on this one; see report-notes [2026-09-23].

### Cleaned held-out validation — TWO baselines

| cycle | duration | clamps low/high | model RMSE | baseline (nominal) | baseline (T_inf) | margin vs nominal | margin vs T_inf |
|---|---|---|---|---|---|---|---|
| Mixed1 @ 25 °C | 128.7 min | 0 / 0  | **1.060 °C** | 0.918 °C | 0.472 °C | **−0.141 (loses)** | **−0.587 (loses)** |
| Mixed2 @ 25 °C | 131.9 min | 0 / 17 | **0.896 °C** | 0.826 °C | 0.463 °C | **−0.070 (loses)** | **−0.433 (loses)** |
| UDDS  @ 10 °C  | 248.2 min | 0 / 0  | **0.468 °C** | 0.759 °C | 0.158 °C | **+0.292 (wins)**  | **−0.309 (loses)** |

**The model loses to the T_inf baseline on every held-out cycle.** Even at 10 °C — where T_inf has 19 supporting tails and is reliable — the model adds 0.31 °C of dynamics noise vs just predicting `T_surface = 9.18 °C` constant. **On the two 25 °C cycles the model also loses to the nominal-setpoint baseline**, but that result is contaminated by the 25 °C T_inf fallback bias and shouldn't be treated as decisive on its own.

**Reframing the model's value.** The previous "model beats baseline" headline was based on the leakage-contaminated 25 °C T_inf. After the integrity check, the honest summary is:
- Model **beats nominal setpoint** baseline only at 10 °C (1 of 3 cycles), and only by 0.29 °C.
- Model **loses to T_inf baseline** on all 3 cycles by 0.31–0.59 °C.
- The dynamics contribution (vs T_inf baseline) is **negative everywhere**: on these mild drive cycles, the model's overshoot adds error rather than capturing dynamics the constant baseline misses.

**Core−surface gap plausibility**: max 1.02–2.46 °C across validation cycles, still within the "single °C in normal drive, growing with C-rate" range. Core labels still cannot be validated directly (per [validation-surface-only](validation-surface-only.md)) — and the surface failure does NOT necessarily mean core predictions are wrong; it means that on cycles where T_s barely deviates from T_inf, the surface signal is too small to demonstrate dynamics value.

**Central locked params** (2026-06-18, supersedes 2026-06-17): `C_core = 41.29 J/K`, `C_surf = 1.91 J/K`, `R_cs = 3.39 K/W`, `R_sa = 5.21 K/W`. The label band on T_core peak remains 5.09 °C across R_cs ∈ [1.70, 8.49] K/W — the unidentifiability conclusion is unchanged.

### Implication for the Week-3 ML stage

The honest finding is that on these LG drive cycles at moderate ambients, the two-state thermal model does not currently demonstrate that its dynamics add value beyond "predict T_inf." The ML stage will train on **model-derived T_core labels** that propagate this caveat. The Week-3 plan must include a sanity check against the analogous core baseline: ML T_core prediction RMSE vs "predict T_core = T_inf" — if the ML stage also fails that bar, the model+ML pipeline is contributing no measurable value beyond a constant. The R_cs label band is still the right uncertainty to propagate; the question is whether the central trajectory is better than constant.

## Calibration result (run 2026-06-17, `calibrate.py`)

> ⚠ The 2026-06-17 result below contains a leakage on the 25 °C cooling-tail pool (Mixed2 @ 25 °C tail was used while Mixed2 is also a validation cycle). Numbers retained for historical comparison only; **use the 2026-06-18 result above**.


**Pipeline** (locked, six stages):

1. **Pin `C_total = m·c_p`**: cell mass 48 g (LG HG2 spec / consensus), `c_p = 900 J/(kg·K)` (Li-ion literature mid-range) → `C_total = 43.2 J/K`.

2. **Cooling-tail fits with FREE asymptote → R_sa AND data-driven T_inf**: the previous run forced the exponential asymptote to the nominal chamber setpoint, which both (a) biased τ low → R_sa low, and (b) hid the fact that LG provides no measured chamber temperature — the cell relaxes to a value below the folder-encoded setpoint. The fix: fit `T_s(t) = T_inf + ΔT₀·exp(−t/τ)` with **`T_inf` free** and report it per ambient group.

   Result on 107 candidate files across 4 ambient groups → 46 rest-periods found → 32 usable tails:

   | nominal °C | n tails | median T_inf | T_inf − nominal |
   |---|---|---|---|
   |  0 °C | 10 | −0.67 °C | −0.67 °C |
   | 10 °C | 19 |  9.18 °C | −0.82 °C |
   | 25 °C |  2 | 23.41 °C | −1.59 °C |
   | 40 °C |  1 | 38.38 °C | −1.62 °C |
   | mean offset across all tails: −0.86 °C | | | |

   The cell consistently relaxes ~0.6–1.6 °C below the nominal setpoint (offset grows with temperature). For ambients without their own tails, the mean offset is applied. The thermal ODE now uses `T_inf` as its cooling sink, not the folder setpoint. **Caveat**: LG provides no measured chamber temperature so we cannot disentangle "chamber-air actually differs from setpoint" vs "thermocouple offset" — the data-driven `T_inf` accounts for whichever it is.

   `R_sa` is re-anchored from the **free-asymptote τ values**: median **R_sa = 5.22 K/W** (range [4.54, 8.69]). The setpoint-forced fits in the previous run gave R_sa = 2.35 K/W — biased >2× low because forcing the asymptote up made the exponential look faster.

3. **Diagnostic fit `(split, R_cs)`** with `C_total`, `R_sa` fixed; calibration cycles US06 / LA92 / UDDS at 25 °C. `split = 0.882 ± 0.003`, **`R_cs = 1.59 ± 0.007 K/W (±0.4 %)` local CI**. Cost dropped 5 % from initial — meaningful optimization (previously 0.6 % with the bad ambient reference). The local CI is still much tighter than the global flatness — see Step 5.

   This CI comes from the 2026-06-17 run of the same diagnostic fit, computed with the same Jacobian-CI method. §7.4 demonstrated the artifact on the locked 2026-06-18b value, not on this one; see report-notes [2026-09-23].

4. **Anchor `R_cs` from geometry**: same recipe as before. With the larger R_sa, Lin 2014's `R_c/R_u ≈ 0.6` cross-check now gives `R_cs ≈ 3.13 K/W`, which lies **inside** the geometric range `[1.70, 8.49]` K/W — no band widening required. **R_cs ∈ [1.70, 8.49] K/W**, geometric central 3.39 K/W.

5. **Central fit + label band**: re-fit split at each R_cs:

   | R_cs (K/W) | split | cal surface RMSE |
   |---|---|---|
   | 1.70 | 0.929 | 0.518 °C |
   | 3.39 | 0.930 | 0.474 °C |
   | 8.49 | 0.929 | 0.450 °C |

   The splits are now stable (~0.93 across the band — previously they scattered 0.87 → 0.99 because the fit was compensating for the biased target). Cal surface RMSE varies by 0.068 °C across a 5× R_cs spread, and **T_core peak spreads 5.06 °C** — confirming the R_cs unidentifiability empirically, now from a calibrated rather than biased fit.

6. **Held-out validation** against measured surface temp (3 cycles, no overlap with cal):

   | cycle | duration | model RMSE | baseline (T_s = nominal) | clamp low/high | core−surf gap (max / mean) |
   |---|---|---|---|---|---|
   | Mixed1 @ 25 °C   | 128.7 min | **0.467 °C** | 0.918 °C | 0 / 0  | 2.46 / 0.68 °C |
   | Mixed2 @ 25 °C   | 131.9 min | **0.363 °C** | 0.826 °C | 0 / 17 | 2.52 / 0.64 °C |
   | UDDS  @ 10 °C    | 248.2 min | **0.469 °C** | 0.759 °C | 0 / 0  | 1.05 / 0.34 °C |

   **The model now beats the naive baseline on every validation cycle**, including the held-out ambient at 10 °C — RMSE improvements 38–56 %. This confirms the ambient-reference diagnosis: the previous "model worse than baseline" result was caused by feeding the ODE the wrong cooling sink, not a structural model error.

   **Core−surface gap plausibility**: max 1.05–2.52 °C across validation cycles, comfortably within the literature "single °C in normal drive, growing with C-rate" range. Core labels remain model-derived and unvalidated per [validation-surface-only](validation-surface-only.md).

**Central locked params** (2026-06-17, supersedes prior): `C_core = 40.16 J/K`, `C_surf = 3.04 J/K`, `R_cs = 3.39 K/W`, `R_sa = 5.22 K/W`. The old central `C_surf = 0.58 J/K` and scattered band splits were artifacts of fitting to a biased ambient target — they should not be cited.

**Label-band on T_core**: peak spread ~5.06 °C across the R_cs band (down marginally from 5.7 °C). This remains the headline uncertainty that propagates into the Week-3 ML stage.

## Note: Lin's absolute parameter values are loose priors at best

The numerical values in [Lin 2014 Table 1](../sources/lin-2014-electro-thermal.md) (`C_c = 62.7 J/K`, `C_s = 4.5 J/K`, `R_c = 1.94 K/W`, `R_u = 3.19 K/W`) come from an **A123 26650 LFP** cell — different chemistry (LFP vs. our NMC), different format (26650 vs. our 18650), different mass (~70 g vs ~47 g), different cycler / fan setup. **Only the qualitative ratios carry as starting expectations**:

- `C_c ≫ C_s` — the jellyroll holds essentially all the thermal mass, the casing is thin.
- `R_c < R_u` — internal conduction is faster than external convection at typical fan settings (though see next bullet).

Even the qualitative `R_c < R_u` is **especially test-setup dependent**: `R_u` is a property of the *coolant / fan / chamber geometry* around the cell, not the cell itself. Our cycler ran the LG/Panasonic cells inside a different chamber than Lin's flow box; their `R_u` is essentially meaningless as a number for us. Treat `R_u` as needing to come from cooling tails on *our* data only.

## What would make us revisit

- A formal [identifiability](../concepts/identifiability.md) analysis on our specific data suggests an additional free parameter is justified — or reveals that even with the constraints we're applying, `R_int` is still ill-conditioned.
- A direct or quasi-direct core-temperature measurement becomes available (then `R_int` and the `C` split become identifiable from data).
- Cooling-tail-derived `R_conv` varies systematically with ambient temperature (forced vs natural convection regime change) — would need ambient-dependent calibration.
- We pick option A (anchored `R_int`) and the surface-temperature residual gets noticeably worse — telling us the anchored value is wrong, and we need a different prior.

## Related

- [Two-state thermal ODE](thermal-ode-two-state.md) — defines what the parameters mean.
- [Validation: surface only](validation-surface-only.md) — what we can and cannot check after fitting.
- [Sign convention reconciliation](sign-convention.md) — ensures `Q` has the right sign before it ever enters this fit.
- [Heat generation: irreversible only](heat-generation-irreversible-only.md) — defines `Q`.
- [report-notes](../report-notes.md) — `R_int` identifiability is the headline caveat for the final writeup.
- [Lin 2014 source note](../sources/lin-2014-electro-thermal.md) — same constrained-fit methodology (pin `C_s` from geometry, fit the rest with Jacobian CIs).
- [Lin 2013 source note](../sources/lin-2013-identifiability.md) — the identifiability result that motivates this whole decision.
- [Zheng 2025 source note](../sources/zheng-2025-sim2real.md) — empirical confirmation of the `R_int` / `k_t` blind spot.
- [Concept: identifiability](../concepts/identifiability.md).
- Concept (todo): `least-squares-jacobian-confidence.md`.
- Concept (todo): `cooling-tail-thermal-id.md`.
