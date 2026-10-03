# Doosthosseini & Fathy 2020 — On the structure of the optimal input for maximizing Li-ion battery thermal parameter identifiability

## Citation

Doosthosseini, M., Fathy, H. K. (2020). *On the structure of the optimal input for maximizing lithium-ion battery thermal parameter identifiability*. **Proc. American Control Conference (ACC) 2020**, Denver, CO, 379–385. IEEE Xplore document 9147802 (the document number and DOI are not printed in the PDF and were not verified).

- PDF: `part2/literature/On_the_Structure_of_the_Optimal_Input_for_Maximizing_Lithium-Ion_Battery_Thermal_Parameter_Identifiability.pdf` (7 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Model** (§II, p. 2). A first-order lumped model after Bernardi et al. (Eq. 1). It is reduced to a single unknown, `θ = hA/mC_p`, the reciprocal of the thermal time constant (Eq. 2).
- **Experiment** (§II, p. 2). Zero-current thermal cycling. The input `u` is the rate of change of chamber temperature; the output is the cell temperature.
- **Problem** (§II, p. 3). A Pareto-weighted objective trades Fisher information (the integral of `s²`) against the L2 norm of `u`, with bounds on the cell temperature (Eqs. 7, 9).
- **Pontryagin analysis** (§III, p. 3–6). The optimal trajectory switches between arcs where the cell-temperature bound is inactive and arcs where it is active. Both kinds of arc have unstable dynamics. On a bound-active arc the sensitivity decays as `x₃(0)e^{−θt}` (Eq. 18), so it adds only a finite amount of information however long the test lingers there.
- **Dynamic programming** (§IV, p. 6). The DP solution oscillates between the cell-temperature bounds (Figs. 1–3). The paper says this is "consistent with" the Pontryagin analysis (§IV) and "supports these insights" (abstract).
- **Noise and model** (§II, p. 2). Zero-mean white Gaussian measurement noise; the model is assumed to represent the battery exactly.

## Key equations

**Model and sensitivity** (paper Eqs. 2, 8):

$$\dot x_1 = u,\qquad \dot x_2 = \theta\,(x_1 - x_2),\qquad y = x_2,\qquad \dot s = x_1 - x_2 - \theta s$$

**Fisher information** (paper Eqs. 4, 6):

$$F = \frac{1}{\sigma^2}\sum_{k=1}^{N} s(k\delta t)^2 \approx \frac{1}{\sigma^2\,\delta t}\int_0^{N\delta t} s(\tau)^2\,d\tau$$

## Parameter values reported (Table I)

`A` = 0.004 m², `m` = 50e-3 kg, `I` = 0 A, `R` = 2.21e-2 Ω. These are nominal values for a 26650 LFP cell, taken from Mendoza et al. 2016. The DP settings (§IV, p. 6) are: cell temperature in [−10, 50] °C, `u` in ±3/60 °C/s, and a 60 s step, which the paper says is 24 % of the thermal time constant.

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings.*

- There is no split. The target corresponds to `R_sa·C_total`, which is sharp.
- The LG cycles do not exercise chamber-temperature excitation.

## Related notes

- [sources/mendoza-2017-combined-identifiability](mendoza-2017-combined-identifiability.md) — the same group's current-input design for the combined model.
- [concepts/identifiability](../concepts/identifiability.md), [concepts/persistent-excitation](../concepts/persistent-excitation.md).
- [concepts/cooling-tail-thermal-id](../concepts/cooling-tail-thermal-id.md) — our `R_sa` route.
