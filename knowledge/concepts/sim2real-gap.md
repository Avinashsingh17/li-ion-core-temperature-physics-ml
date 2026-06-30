# Sim-to-real gap

## Plain explanation
An ML model trained on data from a simulator tends to do well on more simulated data and worse on real-world data, because the simulator is never perfect. The "gap" is the distributional mismatch between simulation outputs and real measurements: slightly different means, variances, spectra, edge cases. A model that latched onto a quirk of the simulator gets surprised when the quirk isn't there in reality. Domain-adaptation techniques like the MMD/CORAL alignment used in [Zheng 2025](../sources/zheng-2025-sim2real.md) exist specifically to narrow this gap by aligning feature distributions between source (synthetic) and target (real) data.

## Role here
We are **not** building a synthetic-pretraining + transfer-learning pipeline, so the gap doesn't bite us in its classical form. But the analogous failure mode lives in our project anyway: the **thermal-model-derived core-temperature labels are themselves a simulator output**. The ML stage trains against those labels and inherits whatever bias the calibrated thermal model has.

Surface-temperature ground truth caps how badly we can mis-calibrate `R_conv` and `C_surf` — because we can compare against `T_s` measurements. But there is no ground truth for core temperature, so the corresponding bias from a wrong `R_int` is **invisible to the data**. In our pipeline, **label quality caps ML quality**, and we cannot detect the cap empirically.

This is the same identifiability bottleneck that [identifiability](identifiability.md) describes, viewed from the ML-pipeline side instead of the optimization side.

## Where used
- [report-notes.md](../report-notes.md) — the `R_int` identifiability entry is the concrete way this gap manifests in our pipeline.
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — explicitly states what we can validate (the surface side) and what we cannot (the core side).
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — the constraints on parameter fitting are partly motivated by limiting the size of the gap on the core side.
- [sources/zheng-2025-sim2real](../sources/zheng-2025-sim2real.md) — the framework that explicitly tries to bridge the gap with transfer learning; we deliberately don't adopt that machinery (scope), which means we're more exposed to it, not less.

## Gotcha
"My ML model has 0.3 °C RMSE on held-out cycles" is a statement about **surface temperature** — not about core. If the underlying labels are biased, the ML will faithfully reproduce biased labels, and the held-out surface RMSE is silent on that bias. Reporting only the headline RMSE on the surface side is **the** failure mode for projects like this; if our writeup doesn't explicitly say "core predictions inherit a bias we cannot measure from data," we will be overclaiming.
