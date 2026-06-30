# Publication figure captions (Week-4)

Each entry is the caption for the matching file in this directory.
Captions feed the writeup verbatim. Numbers are 2026-06-18b LOCKED.

## F1 two_state_schematic.png

Two-state lumped thermal model used to generate the core-temperature labels. Core node (heat capacity C_core = 40.10 J/K, model-derived T_core) and surface node (C_surf = 3.10 J/K, measured T_surface) are coupled by R_cs = 3.39 K/W (R_cs band [1.70, 8.49] K/W carries the physical uncertainty); the surface dissipates to ambient (T_inf = 23.15 °C at 25 °C) via R_sa = 5.21 K/W. Q = I·(V − V_OCV) injects irreversible heat at the core node. Core temperature is model-derived throughout this report — never measured. LOCKED 2026-06-18b.

## F2 rcs_band_sweep.png

Model-derived core temperature on US06 @ 25 °C with the R_cs label-uncertainty band (R_cs ∈ [1.70, 8.49] K/W) shaded. Central line is the core trajectory at R_cs = 3.39 K/W (the geometric mid-band). The pipeline-wide R_cs uncertainty translates to a ~5 °C core-temperature spread; we show it here on US06 because that's where the band is widest (5.05 °C peak). It is the dominant physical uncertainty in the whole pipeline. Measured T_surface shown dashed for reference. US06 is the most aggressive 25 °C cycle and a thermal calibration cycle (see report-notes on US06 trustworthiness).

## F3 severity_validation_bars.png

Surface-temperature RMSE of the calibrated thermal model on five held-out validation cycles, compared to two trivial baselines: (i) predict T_surface = nominal chamber setpoint; (ii) predict T_surface = T_inf (data-driven cooling-tail asymptote). Cycle labels are color-coded by ambient regime: green = in-regime (25 °C, where the model was calibrated), amber = inconclusive (10 °C, mild excursion), red = out-of-regime (0 °C, cold-ambient extrapolation where the model breaks down). Peak excursion |T_s − T_inf| is annotated per group — it is the dynamic signal that was available to win. Note the cold-ambient (0 °C) failure: US06 @ 0 °C produces a model RMSE of 3.61 °C vs. baseline 1.99 °C. LOCKED 2026-06-18b.

## F4 loco_ridge_vs_hgbr.png

Leave-one-cycle-out ML RMSE on model-derived T_core labels for the 11 LG 25 °C cycles: ridge vs HGBR. Bars are per-cycle held-out RMSE; cycles sorted by peak excursion (US06 first). Pooled values annotated in the title (ridge 0.419 vs HGBR 0.412 °C; max 1.86 vs 1.99 °C). Dashed reference line shows the mean per-cycle 'predict T_core = T_inf' baseline. The two ML models produce statistically indistinguishable held-out error — the surrogate map is effectively linear in this regime. Ridge has the better worst-case tail.

## F5 ocv_curve.png

Open-circuit voltage V_OCV(SOC) for the LG HG2 cell, extracted from the C/20 characterization discharge at 25 °C. This lookup is used in the heat-generation term Q = I · (V − V_OCV) that drives the thermal model.

## F6 calibration_surface_fit.png

Mixed1 @ 25 °C — modeled vs measured surface temperature using the locked 2026-06-18b central parameters (the same simulation that generated the core labels). Surface RMSE annotated. This is one of the two thermal-validation cycles whose pipeline-value claims are credible (see report-notes); modeled and measured track closely on in-regime data.

## F7 ablation_bars.png

Surface-feature ablation on the fixed dev split (TEST = Mixed1 + Mixed5). Left: RMSE; right: max-abs-error. Each panel: ridge full, ridge electrical-only (no surface features), HGBR full, HGBR electrical-only. With full features ridge has the better worst-case tail (max-abs: ridge 1.35 vs HGBR 1.63 °C). Strip the surface features and HGBR becomes better on BOTH RMSE (0.696 vs ridge's 0.890 °C) AND max-abs (4.32 vs ridge's 4.39 °C) — surface signals are doing the work that keeps ridge's tail tighter; without them the tree model edges ahead. This is exactly the regime a sensorless (electrical-only) deployment target would land in, and is therefore a documented reopener for the model choice (see decisions/model-selection.md "What would make us revisit"). LOCKED 2026-06-18b.

## F8 cold_residual_drift.png

BLOCKED — no parquet of (modeled − measured) surface residuals saved; the prior calibration plot exists in data/eda_plots/residual_diag_LG_0degC_US06_*.png but is not in the publication style and was generated against the pipeline at run-time. Per task spec we do not re-run the pipeline to regenerate. Cited in report-notes as the qualitative 'heat-accumulation' result.
