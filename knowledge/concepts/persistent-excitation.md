# Persistent excitation

## Plain explanation
For a least-squares estimator to recover the right model parameters, the **input signals must vary enough** to make every dynamic mode of the model show up in the output. A constant current tells you nothing about a thermal time constant. A single brief pulse barely does. A drive cycle that hammers the cell across many frequencies and amplitudes is information-rich, and the estimator can disentangle the parameters from it.

Formally: the rolling-window time-average of the regressor outer product `(1/T)∫ φ(τ)φ(τ)ᵀ dτ` must stay sandwiched between two positive-definite bounds `α_0 · I ≤ ... ≤ α_1 · I` for all time windows. `α_0` (the smallest eigenvalue, lower bound) controls convergence speed — bigger is better, and zero means at least one parameter is unidentifiable from this input.

## Role here
Not every drive cycle in our processed dataset is a useful **calibration** cycle for the thermal model. A short, gentle cycle won't drive enough surface-temperature variation to identify the parameters even in principle. We can compute the smallest eigenvalue of the regressor covariance per cycle and use it to **rank**: keep the well-excited cycles for thermal fitting, demote the rest to validation-only. This is also a defense against silently fitting on a cycle that doesn't actually contain the information we're claiming to extract.

## Where used
- [sources/lin-2013-identifiability](../sources/lin-2013-identifiability.md) — defines the PE test formally; computes `α_0 = 2.4 × 10⁻⁴ s⁻¹` on the scaled urban-assault cycle, giving a worst-case convergence time of ~5 hours.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — implicitly relies on the calibration cycle being well-excited; this concept is the formal test of that assumption.
- Future code: a small helper that loads a Parquet record, builds the regressor `[I², T_amb − T_surf, ṪᎤs]`, and reports `α_0`. Not yet written.

## Gotcha
A drive cycle can have rich **current** variation but poor **thermal** variation if the ambient is mild and the cycle is short — the cell electrically gets thrashed without the surface temperature ever moving much. PE here is about the **regressor in the parametric thermal model**, which includes `(T_amb − T_surf)` and the time derivative of `T_s`, not just `I²`. A 10-minute aggressive cycle at 25 °C ambient can fail PE for `R_conv` even if it crushes the cell electrically. **Always check PE on the actual thermal regressor**, not on the current signal as a proxy.
