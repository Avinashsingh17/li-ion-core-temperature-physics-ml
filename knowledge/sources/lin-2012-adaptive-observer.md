# Lin et al. 2012 — Quadruple adaptive observer of the core temperature in cylindrical Li-ion batteries

## Citation

Lin, X., Stefanopoulou, A. G., Perez, H. E., Siegel, J. B., Li, Y., Anderson, R. D. (2012). *Quadruple adaptive observer of the core temperature in cylindrical Li-ion batteries and their health monitoring*. **Proc. American Control Conference (ACC) 2012**, 578–583. DOI: [10.1109/ACC.2012.6315386](https://doi.org/10.1109/ACC.2012.6315386).

- PDF: `part2/literature/Quadruple_adaptive_observer_of_the_core_temperature_in_cylindrical_Li-ion_batteries_and_their_health_monitoring.pdf` (6 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Model** (§II, p. 1–2). The two-state lumped thermal model: `C_c` is the heat capacity of the jelly roll; `C_s` "is related to the heat capacity of the battery casing". `R_c` is a lumped resistance that includes both conduction and contact resistance. `R_u` is a convection resistance computed from the coolant flow velocity `V` through a Nusselt correlation (Eq. 3). Heat generation is `I²R_e` (Eq. 1).
- **Parameterisation** (§III–IV, p. 2). A Laplace-domain parametric model `z = θᵀφ` (Eqs. 4, 7–8) with `θ = [α β γ µ]ᵀ`. Eq. 9 maps the four lumped parameters back to `C_c, C_s, R_e, R_c`; `R_u(V)` is treated as known. `µ = 1/C_s` multiplies the coolant regressor `s·(T_f − T_s)/R_u(V)`. Identification is by recursive least squares with normalisation `m² = 1 + φᵀφ` (Eq. 5), after a second-order filter that makes the signals proper (Eq. 10). An adaptive observer (Eq. 12) uses the identified parameters by certainty equivalence (p. 3).
- **Simulation only** (§V, p. 3). The plant is a thermal model of an A123 32157 LiFePO4/graphite cell. Its "parameters are assumed by scaling up values from [8] and [18]" (Table I). The input is the Urban Assault Cycle (UAC) with a time-varying coolant velocity (Fig. 3) and the air temperature fixed at 25 °C. All four parameters converge to their nominal values from initial values "quite away from" them (Fig. 4).
- **Temperature-dependent `R_e`** (§VI, p. 4). When `R_e` varies with `T_c`, plain RLS gives biased `C_c`, `C_s` and `R_c` (Fig. 6). The biased set corrupts the `T_c` estimate "without causing large errors in the estimated surface temperature `T_s`" (Fig. 7).
- **Fix** (§VII, p. 5). A non-uniform forgetting factor `η = diag(η₁, 0, 0, 0)`, applied to `α` only with `η₁ = 0.35`, removes the bias (Figs. 8–9).
- **SOH** (p. 5). `R_e` growth is tracked over 500 repeated UAC cycles with growth of 0.17 %/cycle (Fig. 10).

## Key equations

**Parametric model** (paper Eqs. 7–8; `T_c,0 = T_s,0` assumed):

$$s^2T_s - sT_{s,0} = \alpha\,I^2 + \beta\,\frac{T_f - T_s}{R_u(V)} + \gamma\,(sT_s - T_{s,0}) + \mu\, s\,\frac{T_f - T_s}{R_u(V)}$$

$$\alpha = \frac{R_e}{C_cC_sR_c},\quad \beta = \frac{1}{C_cC_sR_c},\quad \gamma = -\frac{C_c + C_s}{C_cC_sR_c},\quad \mu = \frac{1}{C_s}$$

The `γ` term above uses the Eq. 8 regressor, `sT_s − T_s,0` (the Laplace transform of `dT_s/dt`). Eq. 7 as printed has `s(T_s − T_s,0)` for the same term, so the paper is inconsistent with itself here.

**Inverse map** (paper Eq. 9):

$$C_c = -\frac{\gamma}{\beta} - \frac{1}{\mu},\quad C_s = \frac{1}{\mu},\quad R_e = \frac{\alpha}{\beta},\quad R_c = -\frac{\mu^2}{\gamma\mu + \beta}$$

## Parameter values reported (Table I)

Simulated A123 32157 LFP plant (assumed, not measured):

| Param | Nominal | Initial guess |
|---|---|---|
| `C_c` | 268 J/K | 100 J/K |
| `C_s` | 18.8 J/K | 50 J/K |
| `R_e` | 3.5 mΩ | 1 mΩ |
| `R_c` | 1.266 K/W | 0.5 K/W |

Implied split `C_c/(C_c + C_s)` = 268/286.8 = 0.934.

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings ("What's directly reusable", "What is NOT directly reusable", "Contradictions with our locked decisions").*

- This is the only Tier-1 paper that identifies `C_c` and `C_s` separately. It does so under noise-free simulation, with a model identical to the plant, and with a known, time-varying `R_u`.
- Figs. 6–7 show the mechanism §7.5 infers: biased heat capacities and `R_c` that still fit `T_s` while corrupting `T_c`.
- The implied split is 0.934.
- The LG data has a constant chamber and no coolant-velocity channel, so the paper's route to `C_s` is unavailable to us. Phase B0 (§8) found that pinning `C_total` makes the split structurally identifiable: it is recovered exactly from noise-free synthetic data. It is not identifiable from the real data, whose strongly autocorrelated residual drives it to its bound. This is the same failure mode as this paper's Figs. 6–7.

## Related notes

- [sources/lin-2013-identifiability](lin-2013-identifiability.md) — the same group's 2013 journal paper on online parameterisation: only three lumped parameters are identifiable from `T_s` alone.
- [sources/lin-2014-electro-thermal](lin-2014-electro-thermal.md) — the same two-state model, with `C_s` pinned.
- [decisions/calibration-constrained-fit](../decisions/calibration-constrained-fit.md) — our pinned-`C_total` fit.
- [concepts/identifiability](../concepts/identifiability.md).
- [report/part2_phase_a.md](../../report/part2_phase_a.md) §7.5 — the split is not identified.
