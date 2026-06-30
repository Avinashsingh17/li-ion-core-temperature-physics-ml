# Zheng et al. 2025 — Merging physics-based synthetic data and ML for thermal monitoring

## Citation

Zheng, Y., Liu, W., Che, Y., Grimm, F., Zhao, J., Hu, X., Onori, S., Teodorescu, R., Offer, G. J. (2025). *Merging physics-based synthetic data and machine learning for thermal monitoring of lithium-ion batteries: The role of data fidelity*. **arXiv preprint 2509.10380**.

- PDF: `knowledge/Merging Physics-Based Synthetic Data and Machine Learning for Thermal Monitoring of Lithium-ion Batteries The Role of Data Fidelity.pdf` (27 pp).
- Text extracted with **pymupdf 1.26.7** → sibling `.txt`.

## Core idea (in our words)

A two-stage framework for estimating battery internal temperature **without ever needing real internal-temperature labels**. Stage 1: generate a synthetic training set by sweeping inputs and parameters through a physics-based electro-thermal model (an equivalent-circuit electrical sub-model + a polynomial-approximation radial thermal sub-model), then pre-train an LSTM-RNN to map measured signals `(I, V_t, T_s, T_f)` → internal temperatures `(T_s, T_c)`. Stage 2: bridge the inevitable sim-to-real gap using **unsupervised domain adaptation** — freeze the LSTM layer, generate pseudo-labels from the physics model on real input data, and fine-tune the top FC layer by minimizing feature-distribution discrepancy (MMD and CORAL metrics) between source (synthetic) and target (real) high-level features. Validated on three cylindrical cells under drive cycles down to −15 °C, reaching RMSE 0.5 °C with only prior-knowledge thermal parameters and <0.1 °C with identified thermal parameters. The headline scientific finding is that **synthetic-data fidelity matters more than quantity**, with a counter-intuitive twist: a *worse* physics model can still train a *good* ML model **as long as the right thermal parameters are roughly right**.

## Key equations

The paper's actual model details live in supplementary notes that aren't in the main-text PDF, so I can only transcribe what's stated explicitly. The mismatch-injection form (eq. 1):

$$\tilde{\theta} = \theta^* (1 + \varepsilon)$$

with `θ*` the identified benchmark and `ε ∈ [−0.45, +0.45]` the perturbation coefficient used to manufacture controlled sim-to-real gaps.

The thermal model is referred to as the **polynomial approximation (PA) model** of the radial heat equation in a cylindrical cell. Mathematically this is a different beast from our two-state lumped ODE — it's a low-order spectral approximation of the radial PDE. Heat generation in their setup comes from the ECM's resistive elements (`R_0`, `R_1`), not from `I·(V − V_ocv)`.

Symbol glossary (as used in the paper):

| Symbol | Meaning | Unit |
|---|---|---|
| `h` | convective heat-transfer coefficient at cell surface | W/(m²·K) |
| `c_p` | specific heat capacity of cell | J/(kg·K) |
| `k_t` | thermal conductivity of cell | W/(m·K) |
| `R_0`, `R_1`, `C_1` | first-order ECM parameters | Ω, Ω, F |
| `T_s`, `T_c`, `T_f` | surface, core, ambient/flow temps | °C |
| `ε` | parameter perturbation coefficient | dimensionless |

## Parameter values reported

Numerical parameter tables (S1, S2, S5) live in the supplementary information not present in the main PDF. The main text reports only **performance metrics and sensitivities**:

| Result | Value |
|---|---|
| Best LSTM-DA RMSE (matched params) | **< 0.1 °C** |
| Best LSTM-DA RMSE (prior-knowledge params) | **0.5 °C** |
| Oxford HEV1/HEV2 (demanding, ~7 °C internal gradient) RMSE | 0.50 / 0.61 °C (Scenario 2) |
| PA model RMSE reduction by ML | up to **91 %** |
| Tolerable `c_p` drift before performance drops | **< 20 %** (per their cited meta-analysis) |
| Tolerable `k_t` drift before performance drops | **< 23 %** (per their cited meta-analysis) |
| Tolerable `h` mismatch (recoverable by TL) | up to ±45 % |
| Tolerable `R_0`, `R_1` mismatch (recoverable by TL) | up to ±50 % |

Cells used (paper Table 1):
- Cell #1: A123 ANR26650 M1B LFP 2.5 Ah (FUDS / HWFET / HPPC at −15, 5, 25 °C, with embedded core thermocouple).
- Cell #2: LG INR21700 M50T NMC 5 Ah (same protocol). **Closest chemistry analog to our LG HG2.**
- Cell #3 (Oxford Richardson 2015): A123 ANR26650 M1A LFP 2.3 Ah, Artemis HEV cycle, 30 A peak, forced convection, ~7 °C internal gradient.

## What's directly reusable

- **The overall framework is the closest paper to this project's stated goal** (physics + ML hybrid for internal temp). Concretely transferable ideas:
  - **Pre-train on physics-simulated data → fine-tune on real data.** Even without their LSTM-DA machinery, the high-level shape applies: use the thermal model to *generate* a training set with `T_c` labels, then train a regressor on it. Without TL, this still gives a "thermal-model surrogate" we can deploy on real signals.
  - **`T_s` measurement is the right closure signal** for any closed-loop estimator. Their LSTM input set `(I, V, T_s, T_f)` is exactly what our Parquet schema already provides.
- **The "data fidelity > data quantity" finding** changes our default ML-stage instinct. Implication for our project: don't waste effort on generating huge synthetic sets across implausible parameter ranges. Concentrate on getting `c_p` and `k_t` (or our `C_c, C_s, R_c` equivalents) defensibly close to truth; sweep `h` (our `R_u`) widely without worrying.
- **Parameter-sensitivity ranking** (paper §4.2, Fig. 5–6). For internal-temperature estimation specifically:
  - `h` (≈ our `R_u`) **mismatch is tolerable** — the trained ML model can disambiguate `h` from inputs and recover.
  - `c_p` and `k_t` (≈ our heat capacities and `R_c`) **need to be roughly right** — they affect intrinsic thermal dynamics that surface signals can't fully reveal.
  - **`k_t` mismatch is the worst case**: it perturbs the core temperature *without* perturbing the surface temperature, so the surface-only input set fundamentally cannot recognize the gap. This is **the same identifiability point Lin 2013 makes** — our cooling-tail trick anchors `R_u` (= `h`-equivalent), but `R_c` (= `k_t`-equivalent) remains the slippery one. Worth carrying forward as a calibration risk.
- **Cell #2 is an NMC 21700 (similar chemistry to our LG HG2 18650)**. Their Supplementary Table S2 reportedly contains identified `c_p` and `k_t` values for it. **If the supplementary becomes accessible, those numbers are reasonable priors** for our `C_total` and `R_c` constraints. (Their Cell #2 is 5 Ah / 21700 vs our 3 Ah / 18650 — same chemistry family, different geometry; `c_p` should transfer roughly, `R_c` will be geometry-dependent.)
- **Pseudo-labeling** as a way to bootstrap supervision when the target has no ground truth. Even sklearn-only, we can implement a simpler analog: use the calibrated thermal-model `T_c` outputs as "labels" on the real drive cycles, then train an sklearn regressor on real `(I, V, T_s, T_amb)` → `T_c`. That regressor is then validated on **held-out cycles** (per our existing decision) against `T_s`, with `T_c` predictions remaining model-derived.
- **Richardson/Howey Oxford dataset** is cited as a reference public dataset with **actual measured core temperature** for an LFP 26650. If we ever need a sanity-check on whether our model-predicted core gap is plausible, this is a public reference.

## What is NOT directly reusable

- **LSTM-RNN architecture**. Our project preference is **sklearn-only by default**. The architectural details (LSTM depth, FC layers, hidden sizes) and the deep-learning training infrastructure are out of scope. We can use the *framing* with a different model class.
- **Unsupervised domain adaptation (MMD + CORAL)**. This is a deep-learning technique requiring autograd through the feature alignment loss. Implementing it inside sklearn would be a significant lift. For a 4-week scope, **skip the DA layer entirely** and rely on having an adequately-calibrated thermal model on real cycles directly. **Possible scope creep alert**: don't be tempted to add UDA — this would more than double the project.
- **Polynomial-approximation radial thermal sub-model**. Mathematically more detailed than our two-state lumped ODE (it resolves internal radial gradients). Switching to PA is a separate design choice we already considered and rejected in [thermal-ode-two-state](../decisions/thermal-ode-two-state.md). Their findings about `k_t` sensitivity still transfer in spirit (the thermal-resistance parameter governing core↔surface heat transport is harder to identify than `h`-equivalents), but with different numerical thresholds.
- **Heat generation via ECM resistive losses**. They simulate `Q` from `I, R_0, R_1, C_1`. We compute `Q = I·(V − V_ocv)` directly from **measured** `V`. Our approach skips the ECM entirely and uses richer signal information. (Their finding that ML is robust to ECM-parameter mismatch is a strength of *their* sim-data approach, less relevant to ours.)
- **Their RMSE numbers** (0.1–0.6 °C) are reference points, but **not directly comparable** to our held-out reporting:
  - They have measured `T_c` ground truth (embedded thermocouples) — we don't.
  - They use LSTMs — we use simpler regressors.
  - Their best results require either matched thermal params or transfer-learning fine-tuning on real data; our scope is roughly "scenario 2" (prior-knowledge params, no TL). 0.5 °C surface-temp RMSE is a reasonable aspirational target for our ML stage.

## Contradictions with our locked decisions

**None outright, but two important context flags** for our writeup:

1. **Sensitivity ranking for our calibration**: the paper says `h` (~ our `R_u`) is recoverable from ML, but `k_t` (~ our `R_c`) is not. Our [calibration decision](../decisions/calibration-constrained-fit.md) constrains `R_u` from cooling tails and treats `R_c` as the principal free parameter — **this puts the most weight on the *hardest-to-identify* parameter**, exactly the case the paper warns about. Worth re-examining: is there *any* way to anchor `R_c` (e.g., from datasheet thermal conductivity × geometry) so that we don't end up with the same identifiability problem the paper flags for `k_t`?
2. **Scope creep risk**: the paper's "full" methodology (LSTMs + UDA + pseudo-labels) is a magnet for ambition. If we attempt the LSTM portion mid-project, we lose the 4-week timeline. Reference Scenario 2 (prior-knowledge thermal params, no LSTM-DA) is the closest analog to what we can realistically build.

## Concepts this paper introduces / sharpens for us

- **Sim-to-real gap** — distributional mismatch between physics-simulated data and real measurements.
- **Synthetic data fidelity vs quantity** — a small accurate dataset beats a large noisy one.
- **Pseudo-labels** — using a (possibly inaccurate) model's outputs as supervision when truth is unavailable.
- **Domain adaptation** (MMD, CORAL) — narrow alignment in feature space rather than full re-training. *Reference only*; out of scope to implement.
- **Parameter sensitivity decomposition** — which model parameters can be inferred from available measurements and which cannot — closely related to Lin 2013 [identifiability](../sources/lin-2013-identifiability.md).

Concept stubs to add: `sim2real-gap.md`, `pseudo-labeling.md`, `synthetic-data-fidelity.md`.

## Related notes

- [Lin 2014 source note](lin-2014-electro-thermal.md) — the electro-thermal model directly cited by Zheng (ref [20]) as the lineage of the two-state thermal sub-model concept.
- [Lin 2013 source note](lin-2013-identifiability.md) — the identifiability/parameterization paper Zheng cites as ref [25]; the `k_t`-equivalent unidentifiability finding is the same point.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — the paper's `k_t`-sensitivity finding adds urgency to anchoring `R_c` too, not just `R_u`.
- [decisions/thermal-ode-two-state](../decisions/thermal-ode-two-state.md) — our two-state model is simpler than Zheng's PA model.
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — Zheng has core-thermocouple ground truth; we don't.
