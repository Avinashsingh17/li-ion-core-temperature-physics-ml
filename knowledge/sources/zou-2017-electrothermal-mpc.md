# Zou et al. 2017 — Electrothermal dynamics-conscious charging via state-monitored predictive control

## Citation

Zou, C., Hu, X., Wei, Z., Tang, X. (2017). *Electrothermal dynamics-conscious lithium-ion battery cell-level charging management via state-monitored predictive control*. **Energy** 141, 250–259. DOI: [10.1016/j.energy.2017.09.048](https://doi.org/10.1016/j.energy.2017.09.048).

- PDF: `part2/literature/1-s2.0-S0360544217315712-main.pdf` (10 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Model** (§2, p. 2–3). The two-state thermal model (Eq. 2) is coupled to a second-order ECM (Eq. 1). Heat is `Q = I·(V₁ + V₂ + R₀·I)` (Eq. 3).
- **States** (§2, p. 3). `T_a = (T_s + T_c)/2` (Eq. 4) is modelled directly in place of `T_c`, so that `T_a`-dependent electrical parameters are easier to handle (Eq. 5).
- **Observability** (§3, p. 3–4). Local observability is checked numerically along the trajectory with a matrix the paper calls the "Observability Gramian" (Eq. 10). Its rank equals `n` at every step for a UDDS and a constant-current test (Fig. 2). The states are then estimated with an adaptive EKF.
- **Control** (§4–5). State-monitored, linear-time-varying MPC for charging, in simulation.
- **Parameters** (Table 1, p. 3). Constant values for an A123 26650 LFP cell, cited to ref. [24]. Ref. [24] is Lin et al. 2014, *J. Power Sources* 257 (p. 10).

## Key equations

**Thermal model** (paper Eq. 2; `T_f` = ambient):

$$\frac{dT_s}{dt} = \frac{T_f - T_s}{R_uC_s} - \frac{T_s - T_c}{R_cC_s},\qquad \frac{dT_c}{dt} = \frac{T_s - T_c}{R_cC_c} + \frac{Q}{C_c}$$

**Heat generation** (paper Eq. 3):

$$Q = I\,(V_1 + V_2 + R_0 I)$$

## Parameter values reported (Table 1)

| `C_n` | `C_c` | `C_s` | `R_u` | `R_c` | `T_f` |
|---|---|---|---|---|---|
| 2.3 Ah | 62.7 J/K | 4.5 J/K | 15 K/W | 1.94 K/W | 25 °C |

`C_c`, `C_s` and `R_c` equal the Lin 2014 Table 1 values. `R_u` = 15 K/W does not: Lin 2014 reports 3.19 K/W. Implied split `C_c/(C_c + C_s)` = 62.7/67.2 = 0.933.

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings.*

- The parameters are pinned from literature. The observability check concerns states, not parameter identifiability.
- The implied split is 0.933.
- Relevant to Phase C/D.
- The controller sections have not been reviewed yet.

## Related notes

- [sources/lin-2014-electro-thermal](lin-2014-electro-thermal.md) — source of the Table 1 values.
- [decisions/thermal-ode-two-state](../decisions/thermal-ode-two-state.md) — the same thermal ODE.
- [concepts/identifiability](../concepts/identifiability.md).
