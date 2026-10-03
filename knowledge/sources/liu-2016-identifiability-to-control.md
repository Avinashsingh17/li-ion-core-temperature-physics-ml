# Liu et al. 2016 — Can an identifiability-optimizing test protocol improve the robustness of subsequent health-conscious battery control?

## Citation

Liu, J., Rothenberger, M., Mendoza, S., Mishra, P., Jung, Y.-S., Fathy, H. K. (2016). *Can an identifiability-optimizing test protocol improve the robustness of subsequent health-conscious lithium-ion battery control? An illustrative case study*. **Proc. American Control Conference (ACC) 2016**, Boston, MA, 6320–6325.

- PDF: `part2/literature/Can_an_identifiability-optimizing_test_protocol_improve_the_robustness_of_subsequent_health-conscious_lithium-ion_battery_control_an_illustrative_case_study.pdf` (6 pp; git-ignored).
- Verified against text extracted with **PyMuPDF 1.27.1** (scratch only, not filed).

## Core idea (in our words)

What the paper claims:

- **Scope** (§II, p. 2–3). Electrochemical only: a second-order ECM and a single-particle model (SPM). The cell is a commercial 18650 LFP/graphite cell.
- **Test design and identification** (§III–IV, p. 3–4). The test cycle is optimised with the ECM by maximising `det F` over a two-sine input, solved with a genetic algorithm (Eq. 13). It is then run on a commercial cycler. Four SPM health parameters `[ε_n, ε_p, R_cell, D_n]` are estimated by least squares (Eq. 14, Table II). The whole process is repeated for a benchmark cycle (FTP-75 three times, plus a charge).
- **Control** (§V, p. 4–5). The SPM drives pseudospectral, flatness-based fast-charge optimisation under a lithium-plating constraint `η_sr ≥ δ` on the side-reaction overpotential (Eqs. 15–16). Several margins `δ` are used.
- **Monte Carlo** (§VI, p. 5). Parameter distributions come from the CRB, with sensitivities from 0.5 % perturbations, `σ` = 0.005 V, and the Table II estimates as the mean. 500 parameter sets are drawn. The likelihood of violating the constraint is 38.02–43.35 % with the benchmark cycle and 0 % with the optimised cycle, for `δ` = 0–2 mV (Table IV).
- **Discards** (§VI, p. 5). Benchmark-cycle samples with `ε_p > 1` are unphysical and are not used.

## Key equations

**Fisher information and CRB** (paper Eqs. 10, 12):

$$F = \frac{1}{\sigma^2}S^\mathsf{T}S,\qquad \operatorname{cov}(\hat\theta) \ge F^{-1}$$

**Plating constraint** (paper Eq. 15):

$$\eta_{sr} = \phi_{1,n} - \phi_{2,n} \ge \delta$$

## Parameter values reported (Tables III–IV)

| `δ` | Benchmark cycle | Optimised cycle |
|---|---|---|
| 0 mV | 43.35 % | 0 % |
| 0.5 mV | 42.59 % | 0 % |
| 1 mV | 41.06 % | 0 % |
| 1.5 mV | 40.30 % | 0 % |
| 2 mV | 38.02 % | 0 % |

Normalised CRB standard deviations (Table III), benchmark → optimised: `ε_n` 0.0105 → 0.0022, `ε_p` 1.0188 → 0.1362, `R_cell` 0.7532 → 0.0242, `D_n` 0.0379 → 0.0085.

## Our reading

*Interpretation, not the paper's claims. This heading replaces the template's three relevance headings.*

- No thermal content.
- It is a template for Phase C: band or CRB → samples → Monte Carlo through a constrained policy → violation likelihood.
- Its CRB assumes a correct model. §7.7 says ours is only a lower bound, so Phase C should sample from the `R_cs` band, not only from the CRB.

## Related notes

- [sources/mendoza-2017-combined-identifiability](mendoza-2017-combined-identifiability.md) — the same group and the same `F = SᵀS/σ²` design.
- [concepts/identifiability](../concepts/identifiability.md).
- [report/part2_phase_a.md](../../report/part2_phase_a.md) §7.6–7.7 — the surviving `R_cs` band and why the CRB is only a lower bound.
