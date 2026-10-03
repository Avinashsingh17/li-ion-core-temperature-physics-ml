# Mendoza et al. 2017 — Maximizing parameter identifiability of a combined thermal and electrochemical battery model

## Citation

Mendoza, S., Rothenberger, M., Liu, J., Fathy, H. K. (2017). *Maximizing parameter identifiability of a combined thermal and electrochemical battery model via periodic current input optimization*. **IFAC-PapersOnLine** 50(1), 7314–7320. DOI: [10.1016/j.ifacol.2017.08.1468](https://doi.org/10.1016/j.ifacol.2017.08.1468).

- PDF: `part2/literature/1-s2.0-S240589631732044X-main.pdf` (7 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Model** (§2, p. 2–4). A second-order ECM (Eq. 1) plus a first-order lumped thermal model after Bernardi et al. with reversible and irreversible heat (Eq. 2). The combined model (Eq. 4) has eleven parameters (§3–4, Eq. 10): `1/Q`, the ECM eigenvalue, `1/C₁` and `R₂`; `hA/mC_p` and `1/mC_p`; and a fourth-order entropy-coefficient polynomial (five coefficients).
- **Thermal nominal values** (Table 2, p. 4). Taken from Forgez et al. (2010): `mC_p` = 48.52 J/K and a thermal eigenvalue of −1.7e-3 s⁻¹.
- **Input design** (§3, p. 4). `F = SᵀS/σ²` (Eq. 7), with sensitivities from 0.5 % parameter perturbations. `det F` is maximised over a two-sine multisine with four variables, `A₁, A₂, ω₁, ω₂` (Eqs. 8–9). The solver is Nelder–Mead with 200 random starts.
- **Validation** (§4, p. 5–6). Four sets of 500 Monte Carlo runs at test lengths of 6, 8, 10 and 12 h, simulated on a 26650 LFP cell. The abstract and contributions (p. 1–2) conclude that 12 h is enough: "both the mean and variance of the identified parameters have converged".
- **`mC_p` is estimated too** (Table 3, p. 5; added in verification, not in the brief). Mean estimates were 60.128, 60.079 and 52.171 J/K at N = 21,600, 36,000 and 43,200, against a nominal 48.52 J/K.
- **Literature cited** (p. 2). Forgez et al. (2010) obtain a heat capacity and two thermal resistances from current pulses. Perez et al. (2012) first identify the ECM, then compute the two-state thermal parameters from the calculated heat generation.

## Key equations

**Thermal model** (paper Eq. 2):

$$\frac{dT}{dt} = -\frac{hA}{mC_P}(T - T_\text{amb}) + \frac{C(\text{SOC})}{mC_P}\,I\,T + \frac{R_2}{mC_P}\,I^2$$

**Fisher information** (paper Eq. 7):

$$F = \frac{1}{\sigma^2}S^\mathsf{T}S$$

Notation slip in the paper: the text defines `λ_T ≡ mC_p/hA` (p. 4). Its tabulated value, −1.7e-3 s⁻¹, has the sign and units of `−hA/mC_p`. Fig. 6 plots the thermal time constant at about 585–592 s, which is 1/1.7e-3.

## Parameter values reported (Tables 2–3)

| Quantity | Nominal (Table 2) | 12 h MC mean (Table 3, N = 43,200) |
|---|---|---|
| `mC_p` | 48.52 J/K | 52.171 J/K |
| `λ_T` | −1.7e-3 s⁻¹ | −1.701e-3 s⁻¹ |

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings.*

- The model has a single thermal node, so there is no split. The study is simulation only.
- It estimates two thermal quantities: `hA/mC_p`, the inverse thermal time constant (our `1/(R_sa·C_total)`, which is already sharp), and `mC_p` itself (our `C_total`, which we pin rather than fit).
- Even in simulation with a correct model, the 12 h mean estimate of `mC_p` is 52.17 J/K against a nominal 48.52 J/K (7.5 % high; Table 3), while `λ_T` is recovered within 0.1 %. Heat capacity is the hard thermal quantity even for a single node.
- Forgez 2010 is the next read.

## Related notes

- [sources/doosthosseini-2020-optimal-thermal-input](doosthosseini-2020-optimal-thermal-input.md) — the same group's single-parameter thermal follow-up.
- [sources/liu-2016-identifiability-to-control](liu-2016-identifiability-to-control.md) — the same FIM/CRB machinery, carried through to control.
- [concepts/identifiability](../concepts/identifiability.md), [concepts/persistent-excitation](../concepts/persistent-excitation.md).
- [report/part2_phase_a.md](../../report/part2_phase_a.md) §7.2 — our FIM method.
