# 7. Part 2, Phase A — an identifiability audit of the thermal model

> **Status**: Phase A of Part 2. Diagnostic only. The locked Part 1 run
> `2026-06-18b` was not re-run, re-fitted, or re-locked; no labels were
> regenerated. Every number below comes from read-only analysis of the locked
> artifacts or from clearly-marked diagnostic re-fits written to `part2/`.
>
> **Reminder carried forward from Part 1**: the core temperatures discussed
> here are model-derived outputs of the two-state thermal ODE. No measurement
> of internal cell temperature exists in this dataset or in any public dataset
> we are aware of for these cells.

> **Revised 2026-09-25.** Every data-derived figure in this section is now
> regenerated from code into JSON (§7.8). Doing so corrected three figures: the
> offset share of the residual in §7.7 (81 %, not 82 %), the integration-error
> bound on stored traces in §7.6 (2.2e-2 °C, not 2e-2 °C), and the stored
> labels' own integration error in §7.6 item 5 (up to 3.7e-3 °C rms, not
> roughly 3e-3 °C). It also clarified several statements in §7.4, §7.6 and
> §7.7. No conclusion changed.

## 7.1 What this section asks, and the short answer

Part 1 ended on a limitation: the core↔surface conduction resistance `R_cs`
could not be pinned down from surface-temperature data, and the resulting
core-temperature labels carried a ~5 °C band. Phase A asks the obvious next
question — *how much information about each thermal parameter does a surface
thermocouple actually carry?* — using local sensitivity analysis and the
Cramér–Rao lower bound.

Three things came out of it, one of them unwelcome:

1. **`R_cs` is not weakly identified on its own.** It is weakly identified in a
   specific trade-off with the surface heat capacity `C_surf`. The two can
   compensate for each other almost exactly while leaving the modelled surface
   temperature unchanged.
2. **The parameter confidence intervals reported in Part 1 are wrong**, by a
   factor of roughly 50. They were computed from a finite-difference Jacobian
   of a loosely-integrated ODE, and were measuring integrator noise rather than
   the derivative. The corrected figure for `R_cs` is ±29.5 % (95 %), not
   ±0.55 %.
3. **The core/surface heat-capacity split is not identified at all.** With the
   integrator tightened, the fit objective is monotone across the entire
   feasible range and runs to its upper bound, at a physically impossible
   `C_surf` of 0.43 J/K.

What survives: the ~5 °C `R_cs` band, the core-temperature labels, and every
downstream ML result in Part 1. Section 7.6 states the corrections precisely
and 7.7 states what Phase A itself cannot tell you.

## 7.2 Method

For a parameter vector θ and a measurement `T_surf(t)` with assumed
independent Gaussian noise of standard deviation σ, the Fisher information
matrix is `FIM = SᵀS / σ²`, where `S` is the matrix of sensitivities
`∂T_surf/∂θ` evaluated along a drive cycle. The diagonal of `inv(FIM)` gives
the Cramér–Rao lower bound on the variance of any unbiased estimate of θ. In
plain terms: if perturbing a parameter barely moves the modelled surface
temperature relative to the noise floor, no estimator can recover that
parameter well, and the CRB quantifies how badly.

Sensitivities were computed by central finite difference in *relative*
perturbation, so every column of `S` carries units of K per fractional change
and the columns are directly comparable in magnitude. Two numerical
precautions proved essential and are the reason this analysis found what it
found:

- **Determinism.** The ODE was integrated at `rtol = 1e-10`, `atol = 1e-12` on
  a fixed 1 Hz evaluation grid. Two runs at identical parameters were verified
  bitwise identical (max difference 0.000e+00 K) before any derivative was
  taken.
- **Step invariance.** Sensitivities were computed at relative step sizes
  0.001, 0.005, 0.01 and 0.05, and the column norms agreed across the whole
  50× range to 6.8e-4 relative. A genuine derivative does not depend on the
  step size; noise divided by a smaller step does.

Analysis at the locked central parameters uses US06 @ 25 °C (n = 4016, 1 Hz).
Analysis matched to Part 1's diagnostic fit uses all three calibration cycles
(US06, LA92, UDDS @ 25 °C) on the stride-10 grid Part 1 itself used
(n = 3008), with the cooling sink held at the locked `T_inf = 23.150 °C`.

## 7.3 Finding 1 — the weak direction is `C_surf` ↔ `R_cs`

At the locked central parameters, the CRB relative standard errors are:

| parameter | nominal | unit | CRB 1σ | CRB 95 % |
|---|---|---|---|---|
| `C_surf` | 3.0987 | J/K | 25.47 % | 49.93 % |
| `R_cs` | 3.3944 | K/W | 23.49 % | 46.05 % |
| `C_core` | 40.1013 | J/K | 9.26 % | 18.15 % |
| `R_sa` | 5.2062 | K/W | 0.29 % | 0.56 % |

The FIM condition number is 1.31e4 — poorly conditioned but not singular, so
an exact inverse was used. The eigenvector of the smallest eigenvalue
(λ = 9.94, against a largest of 1.30e5) is:

```
C_surf  +0.714
R_cs    -0.658
C_core  +0.239
R_sa    -0.001
```

This is a physically interpretable compensation. Raising `R_cs` impedes heat
flow from core to surface; lowering `C_surf` means the surface needs less
energy to reach the same temperature. The two effects cancel at the surface
node, which is the only node we measure. The correlation matrix says the same
thing twice over: `corr(C_core, R_cs) = −0.98` and
`corr(C_surf, R_cs) = −0.58`.

The practical consequence is that **the `C_surf` direction carries much of
the `R_cs` uncertainty.** Constraining the heat capacities to the
`C_core + C_surf = C_total` relation Part 1 used — the pipeline's actual
parameterisation — gives an `R_cs` 95 % bound of 29.5 %, against 46.0 % for
the unconstrained four-parameter case. The two figures are not a controlled
comparison: the constrained result also uses three calibration cycles rather
than one and the measured residual RMS (0.576 K) rather than an assumed
0.5 K sensor noise. The direction is clear; the magnitude attributable to
the constraint alone is not isolated here.

A second, incidental result: `R_sa` has a CRB of 0.29 % and essentially zero
participation in the weak direction. The convective resistance is sharply
determined and orthogonal to everything else. Part 1's decision to anchor it
independently from cooling tails is retroactively validated — it cost nothing,
and fitting it jointly would also have been safe.

## 7.4 Finding 2 — Part 1's reported confidence intervals are wrong

Part 1's `calibrate.py` reports parameter confidence intervals as
`s²·inv(JᵀJ)` with `s² = Σr²/(n−p)`, using the Jacobian returned by
`scipy.optimize.least_squares`. That is the same object as a CRB with σ set to
the residual RMS, so the two computations should agree. They did not: Part 1
reported ±0.55 % on `R_cs` where the CRB gives ±29.5 %, a factor of 54.

The cause is the solver tolerance inside the residual function. Both fitting
residuals (`_residuals_split_Rcs`, `_residuals_split_only`) integrate the ODE
at `rtol = 1e-3`, `atol = 1e-5`, `max_step = 20 s`. The finite-difference
Jacobian of that function measures integrator noise rather than the derivative.
A 2×2 factorial at the locked Step-3 point isolates the cause. Tightening
rtol/atol alone brings the noise-to-signal ratio below 2e-3 in both columns.
Tightening max_step alone leaves the `split` column noise-dominated
(noise-to-signal 2.4 at the 2 % step, 2.6 at the 0.5 % step). A long max_step
amplifies the problem, since reducing it alone cuts the `R_cs` noise about
660-fold at the 2 % step, but it does not cause it.

The step-invariance test identifies this unambiguously, using the column norm
of the Jacobian with respect to `R_cs`:

| relative FD step | ‖J_Rcs‖, tight (1e-10) | ‖J_Rcs‖, loose (1e-3) |
|---|---|---|
| 0.02 | 2.3761 | 82.5 |
| 0.005 | 2.3767 | 272.1 |
| 0.001 | 2.3763 | — |

Tight tolerances give the same derivative across a 20× range of step sizes.
Loose tolerances quadruple when the step is quartered — the 1/step signature of
a difference dominated by noise. Measured directly, the finite-difference
numerator for `R_cs` contains 0.153 K of signal against 5.33 K of noise at step
0.02, and 0.038 K against 4.39 K at step 0.005: noise-to-signal ratios of 35×
and 114×. At a single point, the loose and tight residuals differ by up to
0.276 K (0.056 K rms) against a residual whose own RMS is 0.576 K.

An inflated Jacobian makes `JᵀJ` too large, which makes `s²·inv(JᵀJ)` too
small, which makes the confidence interval too tight. Because the inflation is
per-column and noise-dependent, it affects each parameter differently — which
is why `split` was understated by 17× and `R_cs` by 52×, rather than by a
common factor.

Two independent confirmations that ±29.5 % is the right number:

- Rebuilding the Jacobian at tight tolerance using `calibrate.py`'s own
  residual function and its own CI routine gives 29.53 % for `R_cs` and
  11.78 % for `split`.
- The CRB computed through entirely separate code gives 28.40 % at σ = 0.5544;
  rescaling to the tight-tolerance residual RMS of 0.5763 gives 29.52 %. The
  two routes agree to within 0.01 percentage points (0.008); most of that gap
  is the n versus n − 2 normalisation of the variance estimate.

A related symptom, noted for completeness: **the locked Step 3 fit does not
reproduce.** Re-running it with identical code and settings lands at
`split = 0.9367`, `R_cs = 1.4722`, against the locked `0.9034`, `1.6147`. Both
terminate on `xtol` with the same message recorded in the locked artifact, at
essentially equal cost. The two `R_cs` values are 8.8 % apart (the `split`
values 3.7 %), comfortably inside a genuine ±29.5 % interval — they are not
inconsistent, merely imprecise in a way the reported ±0.55 % concealed
entirely.

A hypothesis that the `x_scale="jac"` option was corrupting the returned
Jacobian was tested and refuted, both by reading the SciPy source (`trf_bounds`
returns the raw Jacobian; the scaling is applied only to a separate operator
used for the trust-region subproblem) and empirically: a refit with
`x_scale=1.0` still returns Jacobian columns 15.8× (`split`) and 45.0× (`R_cs`)
the size of the tight-tolerance ones. The refit's Jacobian is evaluated at its
own solution rather than at the locked Step-3 point, so this comparison spans
the shift described above.

## 7.5 Finding 3 — the heat-capacity split is not identified

Part 1 parameterises the heat capacities as `C_core = split · C_total` and
`C_surf = (1 − split) · C_total`, with `C_total = 43.2 J/K` pinned from mass
and specific heat, and fits `split` at each of the three `R_cs` band points.

Re-running those three fits at tight tolerance — identical in every other
respect, same `p0 = 0.93`, same bounds `[0.50, 0.99]`, same optimiser settings —
gives:

| band point | `R_cs` (K/W) | locked `split` | tight `split` | Δ |
|---|---|---|---|---|
| low | 1.6972 | 0.9500 | 0.9900 | +4.21 % |
| central | 3.3944 | 0.9283 | 0.9900 | +6.65 % |
| high | 8.4860 | 0.9300 | 0.9900 | +6.45 % |

All three land on 0.99, the upper bound. That is not an interior optimum, so
before reporting it the objective was profiled directly at the central `R_cs`
across nine values of `split` at both tolerances:

| `split` | tight cost | loose cost | |
|---|---|---|---|
| 0.80 | 477.165 | 529.774 | |
| 0.85 | 474.393 | 524.165 | |
| 0.90 | 471.847 | 536.690 | |
| 0.9283 | 470.537 | 460.352 | ← locked |
| 0.93 | 470.461 | 455.164 | loose minimum (= `p0`) |
| 0.95 | 469.607 | 482.227 | |
| 0.97 | 468.805 | 464.950 | |
| 0.98 | 468.402 | 462.056 | |
| 0.99 | 467.946 | 466.466 | tight minimum (= bound) |

The tight objective is **monotone decreasing across the entire feasible
range**, with a total variation of 9.22 cost units — 1.97 % of its minimum. The
loose objective is **not monotone**, varies by 81.53 units (17.9 %), and has a
spurious minimum at 0.93. The integrator noise is roughly nine times larger
than the entire signal the optimiser is trying to descend, and the locked value
sits next to a noise artifact at exactly the initial guess.

The physics confirms the reading. The tight optimum wants `C_surf = 0.432 J/K`.
A cylindrical cell's steel can is a few grams of steel at roughly
500 J/(kg·K) — order 1–3 J/K. The optimiser is not discovering the can's heat
capacity; it is discovering that shrinking the surface node lets the surface
track the ambient input faster, which absorbs residual left over from the
model's structural error. An objective that rewards an unphysical value is the
signature of a parameter the data cannot constrain.

**The locked `split` is noise-pinned; the tight `split` is bound-pinned.
Neither is an interior optimum, and `split` should not be described as a fitted
quantity.** The locked values remain a defensible *prior* — a 93/7 split of a
43.2 J/K total is reasonable from cell construction — but they are a prior the
data neither confirmed nor refuted.

One further detail worth recording: at the high band point the locked `split`
is `0.9300000000000000`, identical to the initial guess to sixteen decimal
places. The optimiser never moved. The locked artifact nonetheless records a
confidence interval of ±0.0034 for it.

## 7.6 What survives, and what must be corrected

**Survives — the headline `R_cs` band.** Sweeping `R_cs` across its plausible
range moves the peak modelled core temperature by:

| variant | band spread (°C) |
|---|---|
| locked artifact (loose fits, 1e-6 traces) | 5.0686 |
| tight traces, locked splits | 5.0658 |
| tight traces, tight (bound) splits | 5.0177 |

Moving `split` from 0.928 to the boundary at 0.99 — collapsing `C_surf`
sevenfold — changes the headline by 0.05 °C, about one percent. **The `R_cs`
band is robust to a co-parameter being completely unidentified.** That is not
luck; it is a substantive result about which parameter carries the
core-temperature uncertainty.

**Survives — the labels and the ML stage.** The loose integrator is confined to
the two fitting-residual functions. Every stored trace, every label, and every
reported metric was integrated at `rtol = 1e-6`, `atol = 1e-8`. Integration
error in anything stored or reported is bounded at ≤ 2.2e-2 °C on any stored
trace (all four stored channels of all 11 labelled cycles; the maximum is
US06's upper band trace) and ≤ 3.4e-3 °C on any validation metric (Mixed1
surface RMSE moves from 0.382905 to 0.383299 at tight tolerance). Labels were
generated at the locked splits, which remain a defensible prior. Nothing
downstream requires regeneration, and the LOCO results, the ridge-versus-HGBR
comparison, and the no-surface ablation all stand unchanged.

**Corrections to Part 1:**

1. The Jacobian-derived confidence intervals in the locked artifact —
   `step3_diagnostic_fit.R_cs_ci95_pct` (0.55 %), `split_ci95` (0.67 %), and
   `step4_central_fit.split_ci95` — are artifacts of a noise-dominated finite
   difference and should not be quoted. The defensible figure for `R_cs` is
   ±29.5 % (95 %).
2. Part 1 framed the tight local CI as "misleading, and must always be paired
   with the band sweep." That framing is replaced: once the Jacobian is
   computed correctly, the local CI (±29.5 %) and the band sweep (a factor of
   five in `R_cs`) point the same direction: both say `R_cs` is poorly
   determined by surface data. They measure different things, a local lower
   bound under an optimistic noise model and a sweep across a physically
   plausible range, so they are not expected to agree in magnitude. They are no
   longer in conflict.
3. `split` is a prior, not a fit. Methods text describing the band as "sweep
   `R_cs`, refit `split` at each point" must note that the refit is
   uninformative, and that at the high band point it did not occur at all.
4. Band spread 5.0686 → 5.0658 at tight tolerance (0.055 %), below the
   precision at which it is quoted. Not material; recorded for completeness.
5. Both solver settings are deterministic within an environment (bitwise across
   processes, §7.8) but neither is portable across environments. With every
   input identical, re-running at `rtol = 1e-6` gives a Mixed1 surface RMSE of
   0.382919 against the locked 0.382905, and Part 1's `rtol = 1e-3` fitting
   residual at the locked Step-3 point gives an RMS of 0.5841 K against the
   recorded 0.5544 K. The environment of the June run was not recorded. The
   stored central labels carry up to 3.7e-3 °C rms of their own integration
   error (LA92), and the band traces up to 7.8e-3 °C rms (US06 upper band).
   Solver tolerances should be pinned explicitly in any future specification.
6. `generate_labels.py` uses rounded constants (`T_inf = 23.15`,
   `R_sa = 5.21`) against the locked full-precision values, contributing about
   2e-3 °C.

## 7.7 Limitations of this analysis

**The CRB is optimistic, because its own assumption is violated.** It assumes
independent Gaussian measurement noise. The residual is not that. Measured on
US06 at the locked parameters, the residual autocorrelation is ρ(60 s) = 0.82,
ρ(300 s) = 0.47, ρ(600 s) = 0.31, with an integrated autocorrelation time of
292 s. That is within striking distance of the system's own thermal time
constant `R_sa · C_total = 225 s` — the residual is the cell's slow thermal
mode, mismatched, not a noise process. Its mean is +0.94 °C against a standard
deviation of 0.45 °C, so roughly 81 % of the mean-square residual is a constant
offset and 19 % is scatter. Treating it as correlated noise would give an
effective sample size of about 7 out of 4016 on US06 (about 51 across the three
calibration cycles, reusing US06's τ_int), and would widen the `R_cs` 95 %
bound to about ±217 %, wider than the parameter itself. That figure is not
adopted either: treating structural model error as a stochastic process is not
defensible in the first place. The honest statement is that **±29.5 % is a
lower bound on the uncertainty in `R_cs`, and the true figure is larger** —
consistent in direction with the band sweep's factor of five.

**The analysis is local and model-conditional.** The CRB linearises around one
operating point and answers "if the cell obeyed this two-state ODE, how much
information about its parameters is in this measurement?" Structural error —
radial temperature gradients, the omitted entropic heat term, the absence of
temperature-dependent parameters — makes the real situation worse, not better.

**Scope.** The four-parameter analysis uses a single cycle (US06 @ 25 °C); the
two-parameter analysis uses the three 25 °C calibration cycles. Nothing here
extends to the cold-ambient regime, which Part 1 already documented as outside
the model's reliable range.

**Not addressed.** Whether any drive-cycle or excitation protocol *would* make
`split` or `R_cs` identifiable is the question Phase B asks. The monotone
profile in §7.5 suggests the answer for `split` may be a clean no, but that is
a hypothesis, not a result.

## 7.8 Reproduction

```
python part2/identifiability.py                                                # §7.3
python part2/identifiability.py --noise-diagnostic                             # §7.7 autocorrelation
python part2/identifiability.py --derived                                      # arithmetic on existing JSON
python part2/identifiability.py --jacobian-diagnostic --stage jacobian         # §7.4
python part2/identifiability.py --jacobian-diagnostic --stage factorial        # §7.4 cause
python part2/identifiability.py --jacobian-diagnostic --stage refit            # §7.4 refits
python part2/identifiability.py --jacobian-diagnostic --stage refit-jacobian   # informational
python part2/identifiability.py --split-profile                                # §7.5
python part2/identifiability.py --tolerance-bounds --stage labels              # §7.6
python part2/identifiability.py --tolerance-bounds --stage metrics             # §7.6
```

Stages run in the order listed, and each writes JSON to `part2/results/`. The
commands from `--derived` onward call Part 1's own functions: the fitting
residuals, the CI routine, the Step 6 metric function, and the label generator.
They change solver settings only by wrapping `simulate_T` in worker processes,
and they log the tolerances each call actually used. Each run requires a clean,
committed tree. Before any derivative is taken, it checks for bitwise
determinism across processes. It records the git commit, the package versions,
and the md5 of the locked calibration file.

**Provenance.** Every number in this section traces to the locked
`calibration_results.json` or to a JSON in `part2/results/`. The exceptions are
the external physical figures in §7.5 (steel specific heat and can mass), code
constants quoted from `calibrate.py` and `generate_labels.py`, and the
environment below. Each reproducibility JSON carries an acceptance table that
compares every regenerated value with the figure as first published in commit
f77b4c4. Every mismatch it records is either a figure corrected in this
revision or the portability result in §7.6 item 5. One stage records a failed
check. The tight-tolerance Jacobian at the `x_scale=1.0` refit point is not
step-invariant at the 2 % step (`split` column, 3.7e-3 against a 1e-3 gate);
the report-only diagnostics point to truncation error, not noise. No figure in
this section depends on it.

Environment: Python 3.12.4 (Anaconda), numpy 1.26.4, scipy 1.13.1, pandas
2.2.2, run with `PYTHONNOUSERSITE=1`. On the machine used, user site-packages
hold numpy 2.4.1 and pandas 3.0.0, which the interpreter otherwise picks up.
Solver tolerances are pinned in the module constants and must not be relaxed;
§7.4 shows what happens when they are.
