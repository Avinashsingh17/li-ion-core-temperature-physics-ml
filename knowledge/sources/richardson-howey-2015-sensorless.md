# Richardson & Howey 2015 — Sensorless internal temperature estimation with a Kalman filter and impedance

## Citation

Richardson, R. R., Howey, D. A. (2015). *Sensorless battery internal temperature estimation using a Kalman filter with impedance measurement*. **IEEE Transactions on Sustainable Energy** 6(4), 1190–1199. DOI: [10.1109/TSTE.2015.2420375](https://doi.org/10.1109/TSTE.2015.2420375). arXiv:1501.06160 (the arXiv identifier is not printed in the PDF and was not verified).

- PDF: `part2/literature/Sensorless_Battery_Internal_Temperature_Estimation_Using_a_Kalman_Filter_With_Impedance_Measurement.pdf` (10 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Thermal model** (§III-A–B, p. 2–3). 1-D radial heat conduction in a cylinder with uniform `ρ`, `c_p` and `k_t` (Eq. 2a). It is reduced to two states, the volume-averaged temperature and the temperature gradient, by a polynomial approximation (PA; Eqs. 4–9). The inputs are `Q` and `T_∞`.
- **Measurement** (§II, p. 2; §VII, p. 6). The real part of the impedance at 215 Hz is the measurement in an EKF (Eq. 15). A dual EKF (DEKF) also estimates the convection coefficient `h` (§VII–VIII).
- **Parameterisation** (§VI, p. 5). `ρ` comes from the measured cell mass divided by its volume. `k_t`, `c_p` and `h` are fitted offline with MATLAB `fminsearch` against the measured core and surface temperatures.
- **Cell and setup** (§V, p. 4). A123 ANR26650m1-A (2.3 Ah, LiFePO4/graphite). The core thermocouple is inserted through a hole drilled in the positive-electrode end.
- **Fit quality** (§VI, p. 5). Parameterisation RMSE 0.18 °C core and 0.19 °C surface; validation RMSE 0.21 °C core and 0.16 °C surface.
- **Why estimate `h` online** (§VII, p. 5). The paper cites Kim et al. [6]: changes in `h` affect the predicted surface and core temperatures more than changes in the other thermal parameters.

## Key equations

**Thermal PDE and boundary condition** (paper Eqs. 2a–2b):

$$\rho c_p\frac{\partial T}{\partial t} = k_t\frac{\partial^2 T}{\partial r^2} + \frac{k_t}{r}\frac{\partial T}{\partial r} + \frac{Q(t)}{V_b},\qquad \left.\frac{\partial T}{\partial r}\right|_{r_o} = -\frac{h}{k_t}\left(T(r_o) - T_\infty\right)$$

**Heat generation** (paper Eq. 3; the entropic term is neglected and `U_OCV` is held at its 50 % SOC value):

$$Q = I\,(V - U_\text{OCV}) + I\,T\,\frac{\partial U_\text{OCV}}{\partial T}$$

## Parameter values reported

Table II (p. 4), read from a rendered image of the page:

| Param | Units | Reference (refs) | Initial | Identified |
|---|---|---|---|---|
| `ρ` | kg m⁻³ | 2047–2118 ([4], [25], [26]) | – | 2107 |
| `c_p` | J kg⁻¹ K⁻¹ | 1004.9–1109.2 ([3], [7], [4]) | 1050 | 1171.6 |
| `k_t` | W m⁻¹ K⁻¹ | 0.488–0.690 ([22], [25], [4]) | 0.55 | 0.404 |
| `h` | W m⁻² (as printed) | – | 20 | 39.3 |

`ρ` is measured (mass/volume), not fitted. The fitted `c_p` is above its reference range and `k_t` is below it. The paper prints the units of `h` as W m⁻²; a convection coefficient is W m⁻² K⁻¹.

- RMSEs are as listed above (§VI, p. 5).
- Experiments: two 3500 s profiles built from the Artemis HEV cycle, with currents from −23 to +30 A. The chamber was at 8 °C. A 215 Hz impedance measurement was taken every 24 s (§V, p. 4).

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings.*

- There is no split. Uniform `ρc_p` fixes it by geometry.
- The paper uses a measured core temperature, which we lack.
- The dominance of `h` fits our sharp `R_sa` (CRB 0.56 %).
- Impedance is a second channel and a candidate for identifying `R_cs`. It needs EIS hardware.

## Related notes

- [sources/lin-2014-electro-thermal](lin-2014-electro-thermal.md) — the lumped alternative to the PA model; it is the paper's ref. [7].
- [decisions/validation-surface-only](../decisions/validation-surface-only.md) — we have no core measurement.
- [concepts/identifiability](../concepts/identifiability.md).
- [report/part2_phase_a.md](../../report/part2_phase_a.md) §7.3 — the `R_sa` CRB.
