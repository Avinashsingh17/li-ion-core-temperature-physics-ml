# Least-squares Jacobian confidence intervals

## Plain explanation
After `scipy.optimize.least_squares` converges, you can construct an approximate confidence interval for each fitted parameter from the Jacobian at the solution. The standard recipe: residual variance `σ² = Σr²/(n−p)`; covariance matrix `cov = σ²·(JᵀJ)⁻¹`; standard error per parameter = `√diag(cov)`; 95 % CI ≈ `±1.96·std_err`. This treats the parameter uncertainty as the local quadratic shape of the cost surface at the optimum. Cheap, fast, and the standard way to report uncertainty when you don't want to bootstrap.

## Role here
The [calibration decision](../decisions/calibration-constrained-fit.md) commits to reporting Jacobian-derived 95 % CIs on whichever parameters end up fitted (the heat-capacity split, `R_cs` in the diagnostic step). These CIs let us say which parameters are "well-constrained" versus "weakly constrained" — without claiming statistical exactness.

## Where used
- [`calibrate.py`](../../calibrate.py) `_jacobian_ci` — computes the 95 % CI from `result.jac` returned by `least_squares`.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — reports the diagnostic-fit CI on `R_cs`.
- [sources/lin-2014-electro-thermal](../sources/lin-2014-electro-thermal.md) — same recipe (their eq. 17–19).

## Gotcha
**A tight Jacobian-based CI does NOT mean the parameter is identified globally** — it only describes the *local* curvature of the cost at the converged point. If the cost has a flat valley along one direction (the textbook unidentifiability symptom), the optimizer may converge to *some* point in the valley with steep local curvature in the perpendicular directions, returning a tight CI that completely misses the valley itself. The remedy is a band sweep: vary the suspected weak parameter over its physical range, re-fit the others, and check whether the residual changes meaningfully. If it doesn't, the local CI lied — and that's the actual identifiability evidence. **This is exactly what happened in the diagnostic fit here**: local CI on `R_cs` came out ±2.3 %, but a manual sweep showed `R_cs` could move by a factor of 6 with the surface RMSE changing by only 0.014 °C. See [report-notes](../report-notes.md).
