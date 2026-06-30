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
