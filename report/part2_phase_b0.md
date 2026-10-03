# 8. Part 2, Phase B0 — is the split unidentifiable, or is it the data?

> **Status**: Phase B0 of Part 2. Diagnostic only. The locked Part 1 run
> `2026-06-18b` was not re-run, re-fitted or re-locked, and no labels were
> regenerated. Every new number below comes from
> `part2/results/phase_b0_synthetic_identifiability.json`; numbers carried over
> from §7 cite their section.
>
> **Reminder carried forward from Part 1**: the core temperatures discussed
> here are model-derived outputs of the two-state thermal ODE, not
> measurements.

## 8.1 The question, and the short answer

§7.5 found that fitting the core/surface heat-capacity split to the real
surface data drives it to its upper bound (0.99), at a physically impossible
surface heat capacity. That leaves two explanations. Either the split cannot be
recovered from surface temperature even in principle, or it can, and something
about the real data prevents it.

The answer is the second:

1. **The split is identifiable in the model.** Fitted to noise-free synthetic
   data, it is recovered exactly from every starting point. With white noise at
   the real noise level, the fits scatter as the Cramér–Rao bound predicts.
2. **It is not identifiable from this data.** The real residual is strongly
   autocorrelated. Noise with the same size and autocorrelation spreads the
   split estimates over most of the feasible range, with about a third of fits
   ending at a bound.
3. **The real residual pulls the split to the upper bound.** Its per-cycle
   offsets alone do so, and so does its shape alone.

The practical consequence: a better-designed test input (the original Phase B)
would not fix this, because the limit is the structure of the model error, not
a lack of excitation.

## 8.2 Method

**Structural check.** The thermal model is linear and time-invariant, with two
inputs: the heat generation `Q(t)` and the cooling sink `T_inf`. Its transfer
functions from each input to `T_surf` were derived symbolically, and the rank
of the coefficient Jacobian was computed. With all four parameters free, each
input channel alone is rank-deficient by one (rank 3 of 4), and the two
together are full rank. With Part 1's parameterisation (`C_total` and `R_sa`
fixed, `split` and `R_cs` free), the rank is 2 for every channel, dropping only
at `split = 0`, outside the feasible range. Nothing in the model's structure
prevents identifying the split. In the LG data `T_inf` is constant, so its
channel is excited only by the cells' initial offset above the sink,
0.62–0.72 K.

**Synthetic data.** Surface-temperature data were generated from the locked
central point (`split = 0.9283`, `R_cs = 3.394 K/W`), using the real heat
input, the real cooling sink and the real initial conditions of the three
calibration cycles. They were then re-fitted with the same bounds [0.50, 0.99],
start point (0.93) and stride (10) as §7.5's split-only fit, at tighter
optimiser tolerances; G4 below confirms that the two fits give the same result
on the real data. There are six variants:

| tier | synthetic data = model at truth plus… |
|---|---|
| T0 | nothing (five starting points) |
| T1 | white Gaussian noise, σ = 0.559 K, the real residual's RMS (200 draws) |
| T1b | autocorrelated noise, same σ, integrated autocorrelation time 292 s as measured in §7.7 (200 draws) |
| T2 | the real residual's per-cycle mean (offset only) |
| T3 | the real residual minus its per-cycle mean (shape only) |
| T4 | the full real residual, which reproduces the real data exactly |

**Simulator.** Because the model is linear, each one-second interval was
integrated exactly in modal coordinates, using the true sub-second heat input.
With 16 sub-points per second it agrees with a very tight `solve_ivp` reference
(rtol 1e-12, atol 1e-14, max_step 1 s) to within 8e-8 K, which made the
hundreds of fits below practical.

**Safeguards.** Four gates had to pass before any result counted:
- (G1) agreement with the reference integrator (≤ 1e-6 K);
- (G2) bitwise determinism across processes;
- (G3) step invariance of the split sensitivity (≤ 1e-3);
- (G4) reproduction of §7.5's real-data fit: split at 0.99, with cost within
  0.01 % (achieved: 1.3e-6 relative).

All four passed. The decision rules for reading the results were written into
the output file before any fit ran.

## 8.3 Results

| tier | mean split | sd | fits at a bound |
|---|---|---|---|
| truth | 0.9283 | — | — |
| T0 clean | 0.9283 from all 5 starts (error ≤ 2.2e-14) | — | 0 of 5 |
| T1 white noise | 0.9216 | 0.042 | 7 %, all at 0.99 (the bound predicts 5.4 %) |
| T1b autocorrelated noise | 0.8501 | 0.174 | 36 % (20.5 % at 0.99, 15.5 % at 0.50) |
| T2 offset only | 0.99 | — | at bound |
| T3 shape only | 0.99 | — | at bound |
| T4 full residual (real data) | 0.99 | — | at bound |

The Cramér–Rao bound for white noise at this point is sd 0.038. The T1 spread,
0.042, is within the pre-registered tolerance. Fitting `split` and `R_cs`
jointly gives the same picture: exact recovery from clean data and, under white
noise, sd 0.043 for split and 0.31 K/W for `R_cs` (bound 0.32 K/W), with no
`R_cs` fit at a bound.

The real residual at the truth point has an RMS of 0.559 K. Its per-cycle means
(+0.94 K US06, −0.28 K LA92, −0.48 K UDDS) carry 86 % of its mean square, and
the remaining shape has an RMS of 0.21 K. This differs from §7.7's 81 % because
§7.7 took a single mean over US06 alone.

## 8.4 What this does and does not show

It shows that the split's failure in §7.5 is a property of the data, not of the
model. The model can resolve the split from perfect data and from white noise
at the real noise level. It cannot do so from errors as autocorrelated as the
real ones.

It does not, on its own, show that the real residual pulls the split upward
*systematically*. Autocorrelated noise of the same size sent 20.5 % of fits to
the 0.99 bound by chance, so one dataset landing there is not decisive. What
tips the balance is physical. The bound corresponds to a surface heat capacity
of 0.43 J/K, far below any plausible steel can (§7.5). A residual dominated by
per-cycle offsets and a slow 292 s mode (§7.7) is the signature of model error,
not measurement noise. The defensible statement is: **the split is identifiable
in principle, is not identifiable from this data, and the real residual drives
it to the bound.**

This fits the literature. The one paper that identifies core and surface heat
capacities separately does so in noise-free simulation, with a model identical
to the plant. It also shows that a small model mismatch biases both capacities
while the surface fit stays good (Lin et al. 2012, Figs. 6–7;
`knowledge/sources/lin-2012-adaptive-observer.md`).

## 8.5 Consequence for Part 2

Phase B, designing a test input that would identify the split or `R_cs`, is not
pursued. Published input-design work (Mendoza et al. 2017; Doosthosseini &
Fathy 2020) sharpens the Fisher information of a model that is assumed exact.
Here the binding constraint is model error, which a better input would not
remove.

One direction is worth recording for future work. If the residual were
stationary noise with the measured autocorrelation, a fit that accounts for
that autocorrelation (generalised least squares) would bound the split far more
tightly than the white-noise bound: sd 0.013 against 0.038. That is because the
split's signature sits at higher frequencies than the slow error. Whether a
pre-whitened or high-pass fit recovers a physical split from the real data is
untested. Any such estimate would also remain model-conditional, because there
is no core measurement to check it against.

The labels and every ML result are unaffected: the `R_cs` band moves by about
1 % when the split is moved to its bound (§7.6).

## 8.6 Reproduction

```
python part2/synthetic_identifiability.py
```

This writes `part2/results/phase_b0_synthetic_identifiability.json`. It takes
about five minutes, must run on a clean committed tree, and records its
decision rules before any computation. The environment is as in §7.8.
