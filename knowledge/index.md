# Index

Catalog of every note in this knowledge base. One line per note, with a short hook and a link.

See also:
- [report-notes.md](report-notes.md) — headline caveats and findings reserved for the final writeup.
- [concept-map.md](concept-map.md) — learner-facing glossary of physics, ML, and data concepts in active use, cross-linked to where each appears in code & decisions.
- [presentation-notes.md](presentation-notes.md) — plain-language, audience-facing weekly recaps for explaining the project. User-authored, filed verbatim.
- [report/part2_phase_a.md](../report/part2_phase_a.md) — Part 2, Phase A identifiability audit (§7): corrects Part 1's thermal-parameter confidence intervals; `split` not identified; the `R_cs` band survives.

## Sources

- [Lin et al. 2014 — Lumped-parameter electro-thermal model](sources/lin-2014-electro-thermal.md) — 5-state ECM + two-state thermal; decoupled parameterization; A123 26650 LFP reference numbers for `C_c, C_s, R_c, R_u`. Our thermal model is theirs.
- [Lin et al. 2013 — Online parameterization & identifiability](sources/lin-2013-identifiability.md) — only 3 of 5 physical params identifiable from `T_s` alone; argues for pinning `C_c, C_s`; persistent-excitation test for cycle suitability.
- [Zheng et al. 2025 — Physics-based synthetic data + ML](sources/zheng-2025-sim2real.md) — LSTM + UDA framework; data-fidelity-over-quantity; warns `k_t` (~ our `R_c`) is the hardest parameter to identify from surface signals.

## Concepts

- [Coulomb counting](concepts/coulomb-counting.md) — integrate `I` over time to track SOC; what fails (drift, capacity fade, self-discharge).
- [V_ocv(SOC) curve](concepts/ocv-soc-curve.md) — open-circuit-voltage lookup vs SOC; pseudo-OCV from C/20; hysteresis & temperature caveats.
- [Identifiability](concepts/identifiability.md) — what makes a parameter recoverable from data; why a confident-looking fit can still be ill-posed.
- [Persistent excitation](concepts/persistent-excitation.md) — the input-richness condition that lets LS estimators converge; cycle-suitability test.
- [Sim-to-real gap](concepts/sim2real-gap.md) — distributional mismatch between simulator and reality; in our pipeline, the labels are themselves a simulator output.
- [Least-squares Jacobian confidence intervals](concepts/least-squares-jacobian-confidence.md) — local CI recipe; gotcha: a tight local CI does NOT confirm global identification (band-sweep paired test).
- [Cooling-tail thermal-resistance identification](concepts/cooling-tail-thermal-id.md) — fit exponential decay during rest to anchor R_sa independently before the main fit; the tail must be genuinely at rest.
- [Blocked time-series split (GroupKFold + LOCO)](concepts/blocked-time-series-split.md) — hold out whole cycles, not rows; nested GroupKFold for alpha selection inside train folds only.

Stubs referenced but not yet written:
- `entropic-heat.md`
- `c-rate.md`
- `lumped-thermal-model.md`
- `data-leakage.md`
- `naive-baseline.md`
- `polarization-overpotential.md`
- `pseudo-labeling.md` (Zheng 2025)
- `synthetic-data-fidelity.md` (Zheng 2025)

## Decisions

- [Heat generation: irreversible only](decisions/heat-generation-irreversible-only.md) — `Q = I·(V − V_ocv)` with raw current; entropic term omitted; I²R as cross-check.
- [Sign convention reconciliation](decisions/sign-convention.md) — our `Q = I·(V − V_ocv)` (I<0 discharge) is algebraically identical to Lin's `Q = I·(V_ocv − V_T)` (I>0 discharge); don't "fix" the apparent mismatch.
- [OCV characterization](decisions/ocv-characterization.md) — build `V_ocv(SOC)` from the LG C/20 discharge file at 25 °C via coulomb counting.
- [Calibration: constrained fit](decisions/calibration-constrained-fit.md) — pin `m·c_p`, fix `R_conv` from cooling tails, fit only the remainder with `least_squares` and report Jacobian-based CIs. Includes `R_int` identifiability risk and mitigation TODO.
- [Two-state thermal ODE](decisions/thermal-ode-two-state.md) — two coupled first-order ODEs (core, surface), `solve_ivp`, inputs on the 1 Hz grid.
- [Validation: surface only](decisions/validation-surface-only.md) — RMSE/MAE/max-error on held-out cycles against surface temp + naive baseline; core labels stated as model-derived & unvalidated.
- [Model selection](decisions/model-selection.md) — ridge reported as the primary regressor; HGBR a checked alternative that didn't separate from it; the near-tie shows the surrogate map is near-linear and R_cs dominates uncertainty.
