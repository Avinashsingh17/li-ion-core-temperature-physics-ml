# Identifiability

## Plain explanation
A model parameter is *identifiable* if the data you've measured can pin it down to a unique value. If two different values of a parameter would produce indistinguishable model outputs (given your measurements), the parameter is **unidentifiable** from that data — no matter how clean the data, how big the dataset, or how good the optimizer. The optimizer will happily return *some* value, but it will be picking arbitrarily inside a manifold of equally good fits, and the confidence interval will (or should) blow up.

Formally: parameter `θ` is identifiable if `y(t; θ_1) = y(t; θ_2)` for all `t` implies `θ_1 = θ_2`. Two unobservably-different parameter values violate this.

## Role here
Our lumped thermal model has 5 physical parameters (`C_core, C_surf, R_int, R_conv`, plus the heat-gen term) but we measure only **one** state: surface temperature. [Lin 2013](../sources/lin-2013-identifiability.md) derives that of those 5, **only 3 lumped combinations** are identifiable from `T_s` alone — and `R_int` (which sets the core/surface gradient) sits in the weakest of those directions. This is the entire reason for the [constrained-fit calibration decision](../decisions/calibration-constrained-fit.md): we must pin or anchor at least 2 of the 5 from physics first, or the LS estimator is solving an ill-posed problem.

## Where used
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — justifies pinning `C_total = m·c_p` and constraining `R_conv` from cooling tails.
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — explains why good `T_s` RMSE doesn't certify the core estimate.
- [report-notes.md](../report-notes.md) — the headline caveat for the final writeup is fundamentally an identifiability statement.
- [sources/lin-2013-identifiability](../sources/lin-2013-identifiability.md) — formal derivation.
- [sources/zheng-2025-sim2real](../sources/zheng-2025-sim2real.md) — empirical confirmation: ML can't fix what surface measurements can't see.

## Gotcha
Identifiability is a property of **(model + measurements + input excitation)** together — not of the model alone. The same model can be identifiable from one drive cycle and not another (see [persistent-excitation](persistent-excitation.md)). And the optimizer doesn't refuse to converge on an unidentifiable problem — it picks a value and reports it with a tight-looking residual. The truth shows up in the **confidence interval**: a CI that spans orders of magnitude, or that the optimizer fails to report at all, is the warning sign. "The fit looks great" and "the parameter is correct" are not the same statement.
