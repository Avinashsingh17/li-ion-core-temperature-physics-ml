# Report notes

Append-only. Items here are **headline caveats, limitations, and findings that must end up in the final writeup** — not footnotes, not buried, not softened.

Format: one section per item; date-stamped; phrase the headline as it should read in the report.

---

## [2026-06-12] `R_int` is the least identifiable parameter — core-temp labels inherit an irreducible bias

**Headline for the report.**
> The internal thermal resistance `R_int` (core ↔ surface) is the least identifiable parameter of the lumped thermal model when only surface temperature is measured. Because `R_int` directly sets the core/surface temperature gap, the model-derived core-temperature labels used to train the ML stage inherit a bias that cannot be detected or corrected from surface-only data. A good held-out **surface**-temperature fit is therefore **not** evidence that the core-temperature predictions are correct.

**Why this is a headline, not a footnote.**

- **The physics**: at quasi-steady state, `T_core − T_surf ≈ Q · R_int`. The very quantity we are trying to predict scales linearly with the very parameter we cannot pin down.
- **The math**: [Lin et al. 2013](sources/lin-2013-identifiability.md) shows formally that only 3 of the 5 thermal parameters are identifiable from `T_s` alone, and `R_int` (≡ `R_c`) sits in the weakest direction.
- **The data**: [Zheng et al. 2025](sources/zheng-2025-sim2real.md) confirms empirically — perturbations to thermal conductivity `k_t` (the analog of `R_int`) are the only thermal-parameter perturbations their ML model cannot recover from when only surface signals are available. Two papers, very different methodologies, identical conclusion.
- **Our pipeline specifically**: the thermal model generates the core-temperature labels for the ML stage. Any `R_int` error biases the labels by ≈ `Q · ΔR_int`. The ML cannot see ground-truth core temperature (no measurement exists), so it cannot detect or correct that bias. It will faithfully learn biased labels and report a confident, low-RMSE prediction of those biased labels.
- **The reporting failure mode**: quoting only the headline surface-temperature RMSE on held-out cycles hides this. A model with the wrong `R_int` can match `T_s` essentially perfectly while predicting `T_core` badly.

**Mitigation TBD — not locked at the time of writing.** Two candidates:

- **Option A**: anchor `R_int` from datasheet thermal conductivity × jellyroll geometry; report a residual *label band* on `T_core`; ML produces central estimate + uncertainty.
- **Option B**: leave `R_int` as the principal fit parameter; report the Jacobian-derived CI on `R_int` and propagate it to a `T_core` uncertainty band; do not produce a separate label band.

Decision lives in [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md). Whichever option is taken, the writeup must state which one and report the corresponding band.

**Where in the writeup**: this belongs **both** in the limitations section and explicitly in the headline of the results/discussion. Not just a sentence in "future work."

**Cross-refs**:
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md)
- [decisions/validation-surface-only](decisions/validation-surface-only.md)
- [sources/lin-2013-identifiability](sources/lin-2013-identifiability.md)
- [sources/zheng-2025-sim2real](sources/zheng-2025-sim2real.md)
- [concepts/identifiability](concepts/identifiability.md)
- [concepts/sim2real-gap](concepts/sim2real-gap.md)

---

## [2026-06-12] SOC convention choices have small biases the writeup must declare

**Headline for the report.**
> The state-of-charge axis used throughout the thermal-model pipeline is built from a single 25 °C C/20 characterization file. Three explicit conventions — what `SOC = 1.0` means, which capacity normalizes SOC, and how each drive cycle's initial SOC is chosen — together introduce a small, documented bias in the heat-generation term that the writeup must declare rather than gloss over.

**Why this is a writeup-level caveat, not a footnote.**

- `SOC = 1.0` is defined as the C/20 file's starting state, where measured `V = 4.176 V`. The cell at true full charge (post-CCCV, 4.20 V) maps to roughly `SOC ≈ 1.01–1.02` in the *table's own units*, but the table is capped at 1.0. So the top of every drive cycle uses `V_ocv = 4.176 V` instead of the true ~4.20 V — a ~24 mV under-estimation that decays as the cell discharges.
- `SOC` is normalized by the **datasheet nominal 3.0 Ah**, not the measured 2.778 Ah at 25 °C (~92.6 % of nominal). Both `build_ocv.py` and `thermal_model.py` use the same value (`build_ocv.AH_NOMINAL`), so the model is internally consistent — but the "100 %" SOC in our pipeline corresponds to ~92.6 % of the cell's actual usable capacity.
- Each drive cycle's `soc_init` defaults to 1.0, justified by the LG protocol (CCCV before every test). For non-LG cells (Panasonic) or contiguous multi-cycle files, this assumption is wrong and must be overridden — currently a manual call-site choice.

**Mitigations (TBD, not blocking the next step).**

- Future refinement: switch both `build_ocv.py` and `thermal_model.py` to the measured 25 °C capacity (2.778 Ah). Requires rebuilding the OCV table; OCV-table SOC range becomes [0 %, 100 %] instead of [7.4 %, 100 %].
- Future refinement: derive `soc_init` per cycle from the first sample's terminal voltage inverted through the OCV table, rather than assuming 1.0.

**Where in the writeup**: methods section (data preparation / SOC convention) AND assumptions list. One sentence each is enough; the point is to declare it.

**Cross-refs**:
- [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md) — "SOC tracking conventions" section.
- [decisions/ocv-characterization](decisions/ocv-characterization.md) — the C/20 file used to build the table.

---

## [2026-06-12] OCV table is discharge-only; charge phases will be slightly biased

**Headline for the report.**
> The `V_ocv(SOC)` lookup is built from the **discharge** half of the LG C/20 file only. Li-ion cells exhibit small (~10–50 mV) voltage hysteresis between charge and discharge; our heat-generation formula uses the discharge OCV in both directions, so heat predicted during regen / charge segments has a small built-in bias.

**Why declare it.**

- The drive cycles we'll calibrate against (UDDS, HWFET, LA92, US06, Cycle 1–4) all include regen current — positive current driving the cell toward higher SOC for short stretches.
- During those stretches, `V > V_ocv_discharge(SOC) + hysteresis`, so `(V − V_ocv)` is over-estimated by the hysteresis magnitude, and the heat-gen formula slightly over-predicts heating during regen.
- For LG HG2 NMC the hysteresis is small (≪ LFP); for our scope it's tolerable, but the bias direction is asymmetric (positive on regen, neutral on discharge) and a careful reader will ask.

**Mitigation (future refinement).**

- Average the charge and discharge halves of the C/20 file to produce a hysteresis-midpoint OCV. Cleaner; requires a small change to `build_ocv.py`.

**Where in the writeup**: limitations section. Not headline-level (effect is small for NMC), but worth a sentence.

**Cross-refs**:
- [decisions/ocv-characterization](decisions/ocv-characterization.md)
- [concepts/ocv-soc-curve](concepts/ocv-soc-curve.md) — the hysteresis caveat is already noted there.

---

## [2026-06-12] SOC clamp counts are a per-cycle monitor; report them in the cycle table

**Headline for the report.**
> Cold drive cycles are expected to discharge the cell below the OCV table's 7.4 % lower bound. `thermal_model.simulate_cycle` clamps coulomb-counted SOC to the table range and **counts the clamp hits per cycle**. Any cycle with non-zero low-clamp hits used SOC = 7.4 % (and `V_ocv = 2.80 V`) for portions of its trajectory, biasing `Q` upward for those samples. The clamp count for each calibration / validation cycle must appear in the per-cycle results table — silent clamps would hide a real cold-cycle problem.

**Where in the writeup**: per-cycle results table (one column: `SOC clamp samples`).

**Cross-refs**:
- [decisions/thermal-ode-two-state](decisions/thermal-ode-two-state.md) — clamping rule.
- [decisions/ocv-characterization](decisions/ocv-characterization.md) — origin of the 7.4 % lower bound.

---

## [2026-06-17] R_cs unidentifiability — empirically confirmed (band spread 5.1 °C T_core vs 0.07 °C surface RMSE)

**Headline for the report.**
> The two-state thermal model is **structurally unable to identify the core↔surface conduction resistance** `R_cs` from surface-temperature measurements alone, *on our own data*. Sweeping `R_cs` across its physically plausible range [1.70, 8.49] K/W — a factor of 5 — changes the calibration surface-temperature RMSE by only **0.068 °C** while the predicted peak core temperature on a representative US06 cycle moves by **5.06 °C**. The locally tight Jacobian-based 95 % CI on `R_cs` from the diagnostic fit (±0.4 %) is therefore misleading and must always be paired with the band-sweep result; quoting the local CI alone would understate the actual uncertainty by orders of magnitude.

**This result is a sensitivity / identifiability statement, NOT a validation of the model or the core labels.** The band sweep tells us how much `T_core` would move under our R_cs uncertainty *assuming the model is otherwise correct*. The actual validation of model correctness is the held-out surface fit (separate entry below).

**Why this is the project's headline result.**

- Two independent papers ([Lin 2013](sources/lin-2013-identifiability.md), [Zheng 2025](sources/zheng-2025-sim2real.md)) predicted this from theory. We confirmed it empirically on LG HG2 data.
- It is a **calibrated** measurement of our method's blind spot, not a guess.
- The label band (T_core peak spread = 5.06 °C) is the quantified uncertainty that propagates into the ML-stage core-temperature labels.

**Where in the writeup**: methods (calibration approach), results (the band sweep table), AND limitations / conclusion. This is the "we found our method's weakest point and quantified it" framing from [presentation-notes.md](presentation-notes.md) Week 2.

**Cross-refs**:
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — calibration result section with the full numbers (run 2026-06-17).
- [concepts/least-squares-jacobian-confidence](concepts/least-squares-jacobian-confidence.md) — gotcha: local Jacobian CI ≠ global identification.
- [concepts/identifiability](concepts/identifiability.md).

---

## [2026-06-17] Data-driven ambient reference: LG nominal setpoint is NOT the cooling sink

**Headline for the report.**
> The LG dataset reports ambient temperature as the folder-encoded chamber **setpoint** — not a measured chamber temperature, because none was recorded. Free-asymptote cooling-tail fits across 32 rest periods at 4 ambients show the cell consistently relaxes to a temperature **below the nominal setpoint**: median offsets of −0.67 °C at 0 °C, −0.82 °C at 10 °C, −1.59 °C at 25 °C, −1.62 °C at 40 °C (mean offset across tails: −0.86 °C). We use this data-driven `T_inf` as the ODE cooling sink throughout calibration and validation. The original cause — actual chamber air differing from setpoint, vs cell thermocouple offset — cannot be disentangled from this dataset alone (no independent thermometer), but is unimportant for the model because the cooling-tail asymptote correctly captures whichever it is.

**Why this matters for the report.**

- A reader needs to know that `T_amb` in our methods section is **not** the column `T_ambient_C` from the source dataset — it's our per-ambient `T_inf` estimate.
- The offset varies with temperature (smaller at 0 °C, larger at 25 °C and 40 °C), so we use per-ambient values, not a global constant.
- For ambients where we don't have cooling tails, the mean offset (−0.86 °C) is applied. Flag this explicitly.
- The first calibration run anchored R_sa from setpoint-forced fits and got R_sa = 2.35 K/W. Re-anchoring from free-asymptote fits gives **R_sa = 5.22 K/W** — more than 2× higher, which is the difference between an over-confined exponential and one that captures the true decay. The earlier R_sa is wrong and should not be quoted.

**Where in the writeup**: methods (data-driven ambient definition), and a one-sentence acknowledgement in limitations that the original cause is undetermined.

**Cross-refs**:
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — Step 2 of the calibration pipeline.
- [concepts/cooling-tail-thermal-id](concepts/cooling-tail-thermal-id.md) — the free-asymptote 3-parameter fit recipe.

---

## [2026-06-18b] Held-out evaluation, severity-stratified — where the model adds value vs thermal severity

**Headline for the report.**
> Two integrity refinements applied to the calibration: (a) the cooling-tail leakage filter from 2026-06-18a is retained; (b) the single-tail T_inf rule is **fixed** so that any ambient with at least one non-excluded tail uses its own median rather than the cross-ambient mean offset — necessary because the rest-temperature offset grows in magnitude with ambient (0 °C: −0.66 °C, 10 °C: −0.82 °C, 25 °C single-tail: −1.85 °C). Two **aggressive cold validation cycles** (US06 @ 0 °C, LA92 @ 0 °C) were added at an ambient anchored by 8 leakage-free tails. The five-cycle severity-stratified evaluation tells a much sharper story than the 2026-06-18a binary "loses to T_inf everywhere":
>
> - **In the calibration regime (25 °C mild drive cycles, ~3 °C excursion)**: model BEATS both baselines on both held-out cycles — +0.45 to +0.85 °C margin vs T_inf. The dynamics contribution is real here.
> - **At a different ambient with tiny excursion (UDDS @ 10 °C, 0.92 °C excursion)**: model beats nominal but loses to T_inf by 0.31 °C. The cycle is too gentle to demonstrate dynamics value either way; the test isn't strong, not that the model is broken there.
> - **At aggressive cold (0 °C drive cycles)**: model LOSES badly to both baselines. US06 @ 0 °C surface RMSE 3.61 vs T_inf baseline 2.59; LA92 @ 0 °C RMSE 1.04 vs 0.85. Core−surface gaps blow out to 6.7 °C and 2.8 °C respectively — implausibly large vs literature.
>
> The honest characterization: **the calibrated two-state thermal model adds demonstrable value in its calibration regime (25 °C mild cycles) and breaks down at cold ambient where parameters that we held constant (internal resistance, OCV characteristic, convection) likely have temperature dependence the model does not represent**. The 0 °C residuals are roughly balanced between rest and load periods but climb steadily through the cycle — heat accumulation: the model produces more heat per cycle than the cell actually generates at 0 °C, and the model never dissipates it.

**Why this is the writeup-level result, not the 2026-06-18a verdict.**

- The 2026-06-18a single-tail rule (`MIN_TAILS_PER_AMBIENT = 2`) forced 25 °C T_inf to the cross-ambient-mean fallback (24.16 °C). That was wrong because the offset varies systematically with ambient — applying a colder-anchored mean to a warm ambient under-corrects.
- With the corrected per-ambient median (25 °C T_inf = 23.15 °C from the single US06 tail, flagged LOW CONFIDENCE), the diagnostic fit converged (cost dropped 10 % from initial) and the 25 °C held-out evaluation cleanly demonstrates dynamics value.
- The aggressive cold cycles are the strongest validation signal in the project so far: leakage-clean, T_inf well-anchored, excursion large enough to actually test dynamics — and they expose a real limitation. **Supersedes the 2026-06-18a "loses to T_inf everywhere" framing**, which was inflated by the warm-ambient T_inf bias.

**Mitigations / open work for Week 3+.**

- Restrict ML label generation to 25 °C cycles. The thermal model is a trustworthy label source there and would be the safest path within scope.
- Add temperature-dependent corrections (OCV(T), R_internal(T)) to the thermal model. Likely out of scope for the locked 4-week window but is the right fix for cold-ambient labels.
- Get more 25 °C cooling tails. Single-tail is documented as LOW CONFIDENCE and the central params would tighten with more data.

**Where in the writeup**: results section (the severity-stratified table is the main result), limitations (cold-ambient breakdown), and methods (per-ambient T_inf with low-confidence flag).

**Cross-refs**:
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — full Step 6 table with severity stratification (2026-06-18b section).
- [concept-map → level bias vs dynamics](concept-map.md) — and the excursion-gating idea added there.
- [decisions/validation-surface-only](decisions/validation-surface-only.md).

**Supersedes** the 2026-06-18a "model LOSES to T_inf baseline on every cycle" entry (the 25 °C losses there were largely an artifact of the T_inf fallback bias; this entry replaces it with the calibration-regime / mild diff-ambient / aggressive-cold characterization).

---

## [2026-06-18a (superseded by 2026-06-18b)] Held-out evaluation, integrity-cleaned — model LOSES to the T_inf baseline on every cycle

**Headline for the report.**
> After two integrity checks on the 2026-06-17 calibration — (a) excluding all validation files from the cooling-tail pool so T_inf and R_sa never see held-out data, and (b) adding a second baseline `T_surface = T_inf` (the cooling-tail asymptote — already has the correct level) — the calibrated two-state thermal model loses to the `T_inf` baseline on every held-out cycle: Mixed1 @ 25 °C (model 1.060 vs T_inf 0.472, −0.587 °C margin), Mixed2 @ 25 °C (0.896 vs 0.463, −0.433), UDDS @ 10 °C (0.468 vs 0.158, −0.309). The model also loses to the **nominal-setpoint** baseline at 25 °C (Mixed1 −0.141, Mixed2 −0.070) and only beats it at 10 °C UDDS (+0.292). On these LG drive cycles at moderate ambients, the model's dynamics are **not currently adding value** beyond knowing the rest temperature.

**Why this is the headline and not the 2026-06-17 win.**

- The 2026-06-17 25 °C T_inf was 23.41 °C, derived as the median of two tails — **one of which was Mixed2 @ 25 °C, a validation cycle**. That tail informed the cooling sink used in evaluating Mixed1 and Mixed2 themselves. Classic leakage; the held-out evaluation was not actually held-out.
- After excluding the three validation files (Mixed1, Mixed2, UDDS @ 10 °C), the 25 °C tail pool drops from 2 to **1** (just US06). Per the documented fallback rule (n_tails < 2 → use cross-ambient mean offset), the 25 °C T_inf becomes 24.16 °C — biased ~1 °C above the single observation (US06 had T_inf = 23.15 °C) because the cross-ambient mean is dominated by the small offsets at 0 °C and 10 °C. The 25 °C fallback bias accounts for some but **not all** of the 25 °C regression.
- The 10 °C result is the **cleanest test** of the model: T_inf comes from 19 non-validation tails (high confidence), and the model still loses to the T_inf baseline by 0.31 °C. **The dynamics contribution is negative on the leakage-clean cycle too.** This is what the integrity check was for: separating "level wrong" from "dynamics wrong," and showing they're both contributing here.

**Mitigations / future work.**

- Get more 25 °C tails: scan the 40 °C dataset more carefully; widen the tail-search thresholds (shorter min duration) at 25 °C to see if more rest periods qualify.
- Pick more demanding drive cycles for evaluation. The current val set (Mixed1/Mixed2 at 25 °C, UDDS at 10 °C) is mild — `T_s − T_inf` swings 1–2 °C peak — so the model has little dynamic signal to add. A more aggressive cycle at colder ambient (e.g., US06 @ −10 °C) would have larger thermal excursions where the dynamics matter more.
- The model's failure is on SURFACE temperature on these particular cycles. **It does NOT mean the core-temperature predictions are wrong** — only that we cannot validate them through the surface metric on mild cycles. The R_cs label band remains the honest uncertainty.

**Where in the writeup**: this is now the dominant validation result. Headline of results, prominently in limitations, and the comparison table must show BOTH baselines (nominal AND T_inf) for every cycle.

**Cross-refs**:
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — full Step 6 table with both baselines and the 25 °C fallback caveat.
- [concept-map → level bias vs dynamics](concept-map.md) — the concept that motivated the second baseline.
- [decisions/validation-surface-only](decisions/validation-surface-only.md) — the held-out-cycle reporting rule.

**Supersedes** the 2026-06-17 "model BEATS baseline" entry, which was based on the leakage-contaminated 25 °C T_inf.

## [2026-06-19] ML evaluation at 25 °C: no held-out ambient, few cycles → LOCO is the headline

**Headline for the report.**
> The ML stage trains and evaluates on 25 °C cycles only (cold-ambient is out of
> regime, per the thermal-stage finding). At 25 °C there are 11 usable drive
> cycles (US06, LA92, UDDS, Mixed1–8); the 25 °C HWFET file is excluded as a
> pause-only file with no current activity. Because all 11 share one ambient,
> there is NO held-out-ambient generalization test — cross-temperature
> generalization is untested by construction. And 8 of the 11 are near-duplicate
> Mixed cycles, so any single 2-cycle holdout is an n=2 across-cycle estimate.
> The headline ML evaluation is therefore leave-one-cycle-out across all 11
> cycles, not a single fixed split.

**Where in the writeup**: methods (evaluation protocol) + limitations.

## [2026-06-19] ML labels are a deterministic surrogate of the central thermal model

**Headline for the report.**
> The core-temperature labels are a deterministic function of (I, V, T_amb)
> through the calibrated central thermal model. The ML model learns a surrogate
> of that function. A low held-out ML RMSE therefore certifies that the surrogate
> GENERALIZES across unseen cycles — it does NOT certify that the core
> predictions are physically correct. Physical correctness is carried entirely
> by the R_cs label band (~5 °C T_core peak spread) and is unchanged by anything
> the ML stage does. Both numbers must be reported together.

**Where in the writeup**: results headline + limitations.

## [2026-06-19] Pipeline-value claims must rest on thermal-validated cycles, not calibration cycles

**Headline for the report.**
> US06, LA92, and UDDS at 25 °C are the thermal model's CALIBRATION cycles, and
> the 25 °C T_inf (23.15 °C) was derived from US06's cooling tail. Any claim that
> the pipeline beats a trivial constant (T_core = T_inf) is therefore circular on
> those cycles. Such claims are made only on the thermal-VALIDATION cycles
> (Mixed1, Mixed2), where the surface fit was checked out-of-sample and T_inf was
> not derived from the cycle. Calibration cycles are still used for ML training
> and for surrogate-generalization measurement under leave-one-cycle-out, but are
> annotated as thermal-in-sample wherever their numbers appear.

**Where in the writeup**: results (baseline comparison) + methods.

## [2026-06-19] Excursion gating: report the T_core = T_inf baseline per cycle, don't claim value where the constant wins

**Headline for the report.**
> The thermal model beats the constant T_inf baseline only on mild 25 °C drive
> cycles with a meaningful excursion. The ML stage reports peak |T_s − T_inf|
> alongside RMSE/MAE/max for every cycle, and reports the ML-vs-(T_core = T_inf)
> margin per cycle. On a low-excursion cycle the constant already covers most of
> the temperature range, so the model has little room to add value — a small
> error there is not evidence of dynamics value. Value is claimed only where the
> excursion is large enough to make the test meaningful.

**Where in the writeup**: results (per-cycle table) + methods (evaluation protocol).

## [2026-06-19] US06 is the least-trustworthy label yet the highest-stakes cycle

**Headline for the report.**
> US06 @ 25 °C is simultaneously the cycle where the core–surface gap is largest
> (3.36 °C, the safety-relevant regime) and the cycle where the model-derived
> label is least trustworthy: it has the worst thermal-model surface fit
> (RMSE 1.046 °C vs ~0.33–0.50 elsewhere) and the widest R_cs uncertainty band
> (5.05 °C T_core spread). The method is therefore least certain exactly where
> the stakes are highest. US06's ML errors must always be read together with its
> band, never as a clean point estimate; its apparent baseline "wins" partly
> reflect both trivial baselines also failing on this aggressive cycle, not model
> skill alone. (US06 is also a thermal-calibration cycle — see the pipeline-value
> caveat — so it cannot anchor a generalization claim.)

**Where in the writeup**: limitations + results discussion (and flag on any US06 number).

## [2026-06-21] Ridge ≈ HGBR: the surrogate map is near-linear; label uncertainty dwarfs model choice

**Headline for the report.**
> Across the 25 °C cycles, a regularized linear model (ridge) and a gradient-boosted
> tree ensemble (HGBR) produce statistically indistinguishable held-out error
> (pooled leave-one-cycle-out RMSE 0.419 vs 0.412 °C — a ~0.01 °C, <2 % gap, within
> the noise of fold composition and hyperparameter selection). The engineered-feature
> → core-temperature map is therefore effectively linear in this regime, and the
> choice of model class moves the prediction by hundredths of a degree. By contrast,
> the R_cs label-uncertainty band is ~5 °C of core-temperature spread. The dominant
> source of uncertainty in the whole pipeline is the unobservable internal thermal
> resistance, not the ML model — the ML was never the bottleneck.

**Where in the writeup**: results headline + discussion. Ties the ML stage back to the R_cs identifiability thread.

## [2026-06-21] HGBR trades a slightly better average for a slightly worse tail

**Headline for the report.**
> HGBR achieves marginally lower average error than ridge but a worse worst case:
> pooled LOCO max error 1.99 °C vs ridge's 1.86 °C, and in the electrical-only
> (no-surface) ablation the tree model's max error reaches 4.32 °C vs ridge's
> 3.19 °C. This is characteristic gradient-boosting behaviour — a tighter fit to the
> bulk of the data at the cost of poorer extrapolation on the tails. Whether ridge or
> HGBR is "better" therefore depends on whether typical-case or worst-case error is
> the operative metric; for a safety-relevant core-temperature estimate, the worst
> case argues for ridge.

**Where in the writeup**: results (model comparison) + limitations.

---

## [2026-09-23] Part 2 Phase A: corrections to Part 1's thermal-parameter uncertainty claims

**Headline for the report.**
> A Fisher-information / Cramér–Rao audit of the locked thermal model finds that `R_cs` is weakly identified in a specific trade-off with the surface heat capacity `C_surf`, not on its own, and that the core/surface heat-capacity `split` is not identified at all. The Jacobian-derived parameter confidence intervals reported in Part 1 were wrong by roughly a factor of 50: both fitting residuals integrate the ODE at `rtol = 1e-3`, and their finite-difference Jacobian measured integrator noise rather than the derivative. The corrected 95 % bound on `R_cs` is ±29.5 %, not ±0.55 %, and is itself a lower bound. The headline ~5 °C `R_cs` band, the core-temperature labels, and every downstream ML result survive unchanged.

**The six corrections** (§7.6, in order). Status legend, following §7.8's own definition: `JSON-backed` = value present in the locked `calibration_results.json` or a JSON in `part2/results/`; `console-only` = reported from scratch-diagnostic console output that was not persisted.

1. **Parameter confidence intervals.** Part 1: `step3_diagnostic_fit.R_cs_ci95_pct` (0.55 %), `split_ci95` (0.67 %) and `step4_central_fit.split_ci95`. → Corrected: these are artifacts of a noise-dominated finite difference and should not be quoted; the defensible figure for `R_cs` is ±29.5 % (95 %). They were understated ~52× for `R_cs` and ~17× for `split` (§7.4); §7.4 also gives a factor of 54 for `R_cs`, which §7.1 rounds to "roughly 50". Bases: 52× / 17× = the two-parameter (`split`, `R_cs`) CRB at Part 1's diagnostic-fit point (`split` 0.9034, `R_cs` 1.6147), on the three calibration cycles at stride 10 (n = 3008), with Part 1's residual RMS σ = 0.5544 K and no autocorrelation correction, ÷ Part 1's reported CI; 54× = ±29.5 % ÷ 0.55 %, where ±29.5 % is the same two-parameter, stride-10, uncorrected basis with σ set instead to the tight-tolerance residual RMS of 0.5763 K (§7.4). Source: §7.6 item 1, §7.4. Status: **JSON-backed** for the Part 1 values and both ratios (`phase_a_noise_diagnostic.json` → `part1_week2_crb.stride10.crb_rel_ci95_pct` ÷ `part1_week2_crb.operating_point.reported_ci95_pct_*`: 28.40 / 0.5497 = 51.7 for `R_cs`, 11.33 / 0.6673 = 17.0 for `split`); **console-only** for the ±29.5 % figure itself, which is in no JSON and is *not* among the items §7.8 discloses as console-only. The nearest JSON-backed value is 28.40 % at σ = 0.5544 K.
2. **How the local CI relates to the band.** Part 1: the tight local CI was "misleading, and must always be paired with the band sweep." → Replaced: once the Jacobian is computed correctly, the local CI (±29.5 %) and the band sweep (a factor of five in `R_cs`) point the same direction; the two measures agree. Source: §7.6 item 2. Status: the factor of five is **JSON-backed** (`phase_a_split_refit_diagnostic.json` → `locked_band_per_point.{low,high}.R_cs_K_per_W`, 1.697 → 8.486 K/W); ±29.5 % is **console-only**, as in item 1.
3. **`split` is a prior, not a fit.** Part 1: the band is built by "sweep `R_cs`, refit `split` at each point." → Corrected: the refit is uninformative, and at the high band point it did not occur at all — the locked `split` there is `0.9300000000000000`, identical to the initial guess `p0 = 0.93`. At tight tolerance all three refits run to the upper bound 0.99 (`C_surf` = 0.432 J/K). Source: §7.6 item 3, §7.5. Status: **JSON-backed** (`phase_a_split_refit_diagnostic.json` → `fits[*].split`, `locked_band_per_point.high.split`, `optimizer.p0`); the supporting objective profile (tight cost monotone across the feasible range) is **console-only (§7.8)**.
4. **Band spread at tight tolerance.** Part 1: 5.0686 °C. → 5.0658 °C at tight tolerance (0.055 %), below the precision at which it is quoted; not material. Source: §7.6 item 4. Status: **JSON-backed** (`phase_a_split_refit_diagnostic.json` → `band.locked_json.spread_max`, `band.locked_splits_tight.spread_max`).
5. **Solver-tolerance portability.** Part 1 practice: traces and labels integrated at `rtol = 1e-6`, `atol = 1e-8`. → `rtol = 1e-6` is deterministic within an environment but not portable across them; the stored labels carry roughly 3e-3 °C rms of their own integration error. Solver tolerances should be pinned explicitly in any future specification. Source: §7.6 item 5. Status: **console-only (§7.8)**.
6. **Rounded constants in label generation.** Part 1 practice: `generate_labels.py` uses `T_inf = 23.15` and `R_sa = 5.21` against the locked full-precision values. → This contributes about 2e-3 °C. Source: §7.6 item 6. Status: **console-only (§7.8)** for the 2e-3 °C magnitude.

**What survives** (§7.6, as stated there).

- **The headline `R_cs` band.** Peak modelled core-temperature spread: 5.0686 °C (locked artifact: loose fits, 1e-6 traces), 5.0658 °C (tight traces, locked splits), 5.0177 °C (tight traces, tight bound splits). Moving `split` from 0.928 to the boundary at 0.99 changes the headline by 0.05 °C, about one percent — the band is robust to a co-parameter being completely unidentified. **JSON-backed** (`phase_a_split_refit_diagnostic.json` → `band.*.spread_max`).
- **The labels and the ML stage.** The loose integrator is confined to the two fitting-residual functions; every stored trace, label and reported metric was integrated at `rtol = 1e-6`, `atol = 1e-8`. Labels were generated at the locked splits, which remain a defensible prior. Nothing downstream requires regeneration; the LOCO results, the ridge-versus-HGBR comparison and the no-surface ablation stand unchanged.
- **Tolerance bound on stored traces and metrics.** ≤ 2e-2 °C on any trace and ≤ 3.4e-3 °C on any validation metric (Mixed1 surface RMSE 0.382905 → 0.383299 at tight tolerance). **Console-only (§7.8)**, except 0.382905, which is the locked Step 6 value (`calibration_results.json` → `step6_validation_per_cycle`).

**What this supersedes** (pointers only — none of these entries is edited).

- **[2026-06-17] R_cs unidentifiability — empirically confirmed (band spread 5.1 °C T_core vs 0.07 °C surface RMSE)** — its statement that the local Jacobian CI (±0.4 %) "is therefore misleading and must always be paired with the band-sweep result" is replaced by corrections 1–2. The ±0.4 % it quotes comes from the run that entry cites (2026-06-17); §7 analysed the locked 2026-06-18b values. Its band-sweep headline survives.
- **[2026-06-12] `R_int` is the least identifiable parameter — core-temp labels inherit an irreducible bias** — qualified: §7.3 finds `R_cs` weak in a trade-off with `C_surf`, which has the slightly larger CRB (25.47 % vs 23.49 %, 1σ), and §7.5 finds `split` not identified at all. Its Option B proposed reporting a Jacobian-derived CI on `R_int`; per correction 1, the pipeline's Jacobian-derived CIs should not be quoted.
- **[2026-06-18b] Held-out evaluation, severity-stratified — where the model adds value vs thermal severity** — qualified on one point, "the diagnostic fit converged (cost dropped 10 % from initial)": §7.4 finds the locked Step 3 fit does not reproduce (a re-run lands at `split = 0.9367`, `R_cs = 1.4722` against the locked 0.9034, 1.6147; console-only). Its severity-stratified validation results stand, within the §7.6 bound.
- Outside report-notes, also carrying the Part 1 CIs (not edited): [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md), Step 3 result rows.

**Open reproducibility gap.** The claims marked console-only above will become JSON-backed via planned subcommands in `part2/identifiability.py`; until then they rest on console output.

**Where in the writeup**: [report/part2_phase_a.md](../report/part2_phase_a.md) §7, linked from the closing "Part 2" section of `report/writeup.md`. `writeup.md` Section 5 was left as published; no edits to it are proposed here.

**Cross-refs**:
- [report/part2_phase_a.md](../report/part2_phase_a.md) — §7, the written record (§7.6 corrections, §7.8 provenance).
- [part2/identifiability.py](../part2/identifiability.py) — FIM/CRB module; `--noise-diagnostic` for §7.7.
- [part2/results/phase_a_identifiability.json](../part2/results/phase_a_identifiability.json) — §7.3 CRB and eigenstructure.
- [part2/results/phase_a_noise_diagnostic.json](../part2/results/phase_a_noise_diagnostic.json) — §7.4 ratios, §7.7 autocorrelation.
- [part2/results/phase_a_split_refit_diagnostic.json](../part2/results/phase_a_split_refit_diagnostic.json) — §7.5 refits, §7.6 band spreads.
- [concepts/least-squares-jacobian-confidence](concepts/least-squares-jacobian-confidence.md) — the local-CI gotcha that correction 2 revisits.
- [concepts/identifiability](concepts/identifiability.md)
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — Part 1 calibration steps and the CIs corrected here.

---

## [2026-09-25] Part 2 item 2: §7 made reproducible; revision filed

**Headline for the report.**
> Every data-derived figure in `report/part2_phase_a.md` §7 is now regenerated from code into JSON by `part2/identifiability.py`, which calls Part 1's own functions and compares each regenerated value with the figure as first published (commit f77b4c4). Regeneration corrected three figures and clarified several statements in §7.4, §7.6 and §7.7. No conclusion changed.

**The three corrected figures** (§7, revision note under the status block).

1. **§7.7, offset share of the residual.** Roughly 81 % of the mean-square residual is a constant offset and 19 % is scatter, not 82 % / 18 %. Source: `phase_a_noise_diagnostic.json` → `part2_autocorrelation.residual_mean_C` and `residual_rms_C` (mean² / rms² = 81.47 %); regenerated in `phase_a_derived.json` (`acceptance`, §7.7 rows).
2. **§7.6, integration-error bound on stored traces.** ≤ 2.2e-2 °C, not 2e-2 °C, over all four stored channels of all 11 labelled cycles; the maximum, 0.021958 °C, is US06's upper band trace (`T_core_model_hi_C`). Source: `phase_a_tolerance_bounds.json` → `stages.labels.maxima.tight_vs_stored.max_abs_K`.
3. **§7.6 item 5, the stored labels' own integration error.** Up to 3.7e-3 °C rms on the central traces (LA92), not roughly 3e-3 °C; the band traces up to 7.8e-3 °C rms (US06 upper band). Source: `phase_a_tolerance_bounds.json` → `stages.labels.per_cycle["LG_25degC_LA92_10-29-18_03.53_551_LA92_25degC_LGHG2.parquet"].T_core_model_C.tight_vs_stored.rms_K` (3.683e-3) and `stages.labels.maxima.tight_vs_stored.rms_K` (7.829e-3).

**What this closes** (pointers only; the earlier entry is not edited). In **[2026-09-23] Part 2 Phase A: corrections to Part 1's thermal-parameter uncertainty claims**:

- its **Open reproducibility gap**: the claims it marked console-only now trace to JSON in `part2/results/` (`phase_a_jacobian_diagnostic.json`, `phase_a_split_profile.json`, `phase_a_tolerance_bounds.json`, `phase_a_derived.json`);
- its flag that ±29.5 % is in no JSON: the figure is now `phase_a_jacobian_diagnostic.json` → `stages.jacobian.ci_route.tight["0.02"].R_cs_pct` (29.529 %, computed by `calibrate._jacobian_ci`), and the tight-tolerance residual RMS behind it, 0.5763 K, is `stages.jacobian.residuals.tight_rms_K`.

**What remains outside the JSONs** (§7.8): the external physical figures in §7.5 (steel specific heat and can mass); code constants quoted from `calibrate.py` and `generate_labels.py`; the environment described in §7.8.

**Recorded failed check** (§7.8): the tight-tolerance Jacobian at the `x_scale=1.0` refit point is not step-invariant at the 2 % step (`split` column, 3.7e-3 against a 1e-3 gate); the report-only diagnostics point to truncation error, not noise. No §7 figure depends on it.

**Where in the writeup**: [report/part2_phase_a.md](../report/part2_phase_a.md) §7, revision note under the status block and §7.8 rewritten. `writeup.md` is unchanged.

**Cross-refs**:
- [report/part2_phase_a.md](../report/part2_phase_a.md) — §7 as revised.
- [part2/identifiability.py](../part2/identifiability.py) — the reproducibility subcommands (§7.8).
- [part2/results/phase_a_derived.json](../part2/results/phase_a_derived.json), [phase_a_jacobian_diagnostic.json](../part2/results/phase_a_jacobian_diagnostic.json), [phase_a_split_profile.json](../part2/results/phase_a_split_profile.json), [phase_a_tolerance_bounds.json](../part2/results/phase_a_tolerance_bounds.json).
- [decisions/calibration-constrained-fit](decisions/calibration-constrained-fit.md) — Step 3 CIs annotated.
- [concepts/least-squares-jacobian-confidence](concepts/least-squares-jacobian-confidence.md) — project note on step invariance.

---

## [2026-10-02] Part 2 Phase B0: the split is identifiable in the model, not from this data

**Headline for the report.**
> A synthetic-data check (§8) finds that the core/surface heat-capacity `split` is identifiable in the model but not from this data. Fitted to noise-free synthetic data generated from the locked central point, it is recovered exactly from every starting point; with white noise at the real noise level, the fits scatter as the Cramér–Rao bound predicts. The real residual is strongly autocorrelated, and its per-cycle offsets alone, and its shape alone, each drive the split to its 0.99 bound. A better-designed test input would not fix this: the limit is the structure of the model error, not a lack of excitation.

**Setup** (§8.2). Synthetic data: the model at the locked central point (`split = 0.9283`, `R_cs = 3.394 K/W`) with the real heat input, cooling sink and initial conditions of the three calibration cycles, re-fitted with the same bounds [0.50, 0.99], start point (0.93) and stride (10) as §7.5's split-only fit, at tighter optimiser tolerances; G4 confirms that the two fits give the same result on the real data. Simulator: exact modal integration, within 8e-8 K of a tight `solve_ivp` reference. Four gates passed (reference agreement, bitwise determinism, step invariance, reproduction of §7.5's real-data fit at 1.3e-6 relative cost). The decision rules were written to the output JSON before any fit ran.

**The three findings** (§8.1, §8.3).

1. **Identifiable in the model.** Noise-free (T0): 0.9283 from all 5 starts, error ≤ 2.2e-14. White noise at σ = 0.559 K, the real residual's RMS (T1, 200 draws): mean 0.9216, sd 0.042 against a Cramér–Rao sd of 0.038; 7 % of fits at a bound, all at 0.99, against 5.4 % predicted. Fitting `split` and `R_cs` jointly gives the same picture: exact recovery from clean data and, under white noise, sd 0.043 for split and 0.31 K/W for `R_cs` (bound 0.32 K/W), with no `R_cs` fit at a bound.
2. **Not identifiable from this data.** Autocorrelated noise of the same σ, integrated autocorrelation time 292 s as measured in §7.7 (T1b, 200 draws): mean 0.8501, sd 0.174; 36 % of fits at a bound (20.5 % at 0.99, 15.5 % at 0.50).
3. **The real residual drives the split to the bound.** Offset only (T2), shape only (T3) and the full residual (T4, which reproduces the real data) each land at 0.99. The per-cycle means (+0.94 K US06, −0.28 K LA92, −0.48 K UDDS) carry 86 % of the residual's mean square; the remaining shape has an RMS of 0.21 K. (§7.7's 81 % is a single mean over US06 alone.)

**Caveat** (§8.4). This does not, on its own, show that the real residual pulls the split upward *systematically*: autocorrelated noise of the same size sent 20.5 % of fits to the 0.99 bound by chance, so one dataset landing there is not decisive. What tips the balance is physical. The bound corresponds to a surface heat capacity of 0.43 J/K, far below any plausible steel can (§7.5), and a residual dominated by per-cycle offsets and a slow 292 s mode (§7.7) is the signature of model error, not measurement noise.

**Consequence for Phase B** (§8.5). Phase B, designing a test input that would identify the split or `R_cs`, is not pursued: published input-design work (Mendoza et al. 2017; Doosthosseini & Fathy 2020) sharpens the Fisher information of a model assumed exact, and here the binding constraint is model error. Recorded for future work, untested: if the residual were stationary noise with the measured autocorrelation, a generalised-least-squares fit would bound the split at sd 0.013 against the white-noise 0.038; any such estimate would remain model-conditional, with no core measurement to check it against. The labels and every ML result are unaffected: the `R_cs` band moves by about 1 % when the split is moved to its bound (§7.6).

**Where in the writeup**: [report/part2_phase_b0.md](../report/part2_phase_b0.md) §8, linked from §7.5 of [report/part2_phase_a.md](../report/part2_phase_a.md) and from the closing "Part 2" section of `report/writeup.md`.

**Cross-refs**:
- [report/part2_phase_b0.md](../report/part2_phase_b0.md) — §8, the written record.
- [part2/synthetic_identifiability.py](../part2/synthetic_identifiability.py) — the check (§8.6).
- [part2/results/phase_b0_synthetic_identifiability.json](../part2/results/phase_b0_synthetic_identifiability.json) — every §8 number, the gates and the pre-registered rules.
- [report/part2_phase_a.md](../report/part2_phase_a.md) — §7.5 (split bound-pinned), §7.7 (residual autocorrelation).
- [sources/lin-2012-adaptive-observer](sources/lin-2012-adaptive-observer.md) — the same failure mode (Figs. 6–7).
- [concepts/identifiability](concepts/identifiability.md)
