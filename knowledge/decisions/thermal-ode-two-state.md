# Thermal model: two-state lumped ODE

## The question
What form does the thermal model take?

## Options considered
- **Single-state lumped** — one node, surface temperature only.
- **Two-state lumped** — separate core and surface nodes coupled by internal thermal resistance.
- **PDE discretization** (1D radial, or full 3D) — spatially resolved.

## The call
**Two coupled first-order ODEs**, integrated with `scipy.integrate.solve_ivp`:

$$C_\text{core} \frac{dT_\text{core}}{dt} = Q(t) - \frac{T_\text{core} - T_\text{surf}}{R_\text{int}}$$

$$C_\text{surf} \frac{dT_\text{surf}}{dt} = \frac{T_\text{core} - T_\text{surf}}{R_\text{int}} - \frac{T_\text{surf} - T_\text{amb}(t)}{R_\text{conv}}$$

Symbols & units:
- `T_core`, `T_surf` — core and surface (case) temperatures, °C.
- `T_amb(t)` — ambient temperature input, °C (from `T_ambient_C`, interpolated on the 1 Hz grid).
- `Q(t)` — heat source, W, from [heat generation](heat-generation-irreversible-only.md).
- `C_core`, `C_surf` — node heat capacities, J/K (see [calibration](calibration-constrained-fit.md)).
- `R_int` — internal core↔surface thermal resistance, K/W.
- `R_conv` — convection resistance surface↔ambient, K/W.

`Q`, `T_amb`, and any other time-varying inputs are interpolated on the 1 Hz grid produced by `data_loading.py`. Inputs to `solve_ivp` are passed as callables (closures over the Parquet record) so the integrator can sample inputs at its own internal time steps.

## SOC tracking conventions

Three explicit choices, all in `thermal_model.py`. They barely move `Q` in the flat mid-SOC plateau but matter at the OCV-table endpoints:

(a) **`SOC = 1.0` reference state** = the C/20 characterization file's starting state, where `V = 4.176 V` (not the cell's true full-charge 4.20 V after CCCV). Drive-cycle starts (assumed fully charged per LG protocol) are treated as `SOC = 1.0`, which accepts a ~24 mV under-estimation of `V_ocv` at the very top that decays as the cell discharges.

(b) **Capacity for normalization**: `AH_NOMINAL = 3.0 Ah` (LG HG2 datasheet), imported from `build_ocv.AH_NOMINAL` so the thermal model and the OCV table are guaranteed to use the same value. The measured 25 °C capacity from the C/20 file was ~2.778 Ah (92.6 % of nominal); switching both modules to the measured value is a future refinement, flagged in [report-notes](../report-notes.md).

(c) **Initial SOC per drive cycle**: `1.0` by default, justified by the LG protocol (CCCV to 4.2 V before every drive cycle). Overridable via the `soc_init` kwarg of `simulate_cycle` for non-LG or contiguous-multi-cycle files.

## SOC clamping rule

`SOC` is clamped to the OCV-table range `[0.074, 1.000]` **before** looking up `V_ocv`.

- We clamp on **coulomb-counted SOC**, not on terminal voltage. A hard discharge pulse drops `V` below 2.8 V via IR drop while the true cell state is still mid-SOC; clamping on `V` would corrupt the lookup.
- `build_ocv.load_ocv` returns an `interp1d` with `bounds_error=True` — without the clamp the lookup raises.
- Silent clamps could hide a real problem (e.g. cold cycles where the cell genuinely discharges below 7.4 %), so we **count and log clamp hits per cycle** in the simulation result.

## Default unfitted parameters

Order-of-magnitude only — plausibility for the sanity-run step in `thermal_model.py`. NOT calibrated. The calibration step in [calibration-constrained-fit](calibration-constrained-fit.md) replaces them.

| Param | Default | Basis |
|---|---|---|
| `C_core` | 40 J/K | most of `C_total ≈ m·c_p ≈ 0.048 × 900 ≈ 43 J/K`; ~14:1 split per Lin 2014 |
| `C_surf` | 3 J/K  | thin 18650 can; remainder of `C_total` |
| `R_cs`   | 1.5 K/W | mid-range of Lin 2014's identified `R_c = 1.94 K/W` on a 26650 |
| `R_sa`   | 3.0 K/W | passive convection in still air; cooling tails will refine |

Lin's absolute values are **not directly portable** to our cell — different chemistry, format, mass, and test setup. Only the qualitative ratios `C_core ≫ C_surf` and `R_cs < R_sa` carry as priors. See [sources/lin-2014-electro-thermal](../sources/lin-2014-electro-thermal.md).

## Sanity-run result (2026-06-12)

First simulation on `LG_25degC_UDDS_551`, unfitted defaults, 4.4 h cycle, 15,966 samples:

- `T_core ≥ T_surf ≥ T_amb` held in **100 %** of samples — model wired correctly.
- `Q ≥ 0` in 15,965 of 15,966 samples; one near-zero negative reading (−0.06 W, just past the 0.05 W noise threshold) consistent with numerical noise at a current zero-crossing, not a sign bug.
- 3,040 samples with `Q > 0.1 W` — cycle has substantial heating activity to learn from.
- SOC clamp hits: **0 low / 0 high** — UDDS at 25 °C stayed comfortably inside the OCV-table range. Cold-ambient cycles are expected to hit the low clamp.
- Modeled `T_surf` over-shoots measurement by ~1.2 °C; expected with unfitted parameters, will be closed by calibration.

## Why
- A single-state model cannot distinguish internal vs surface dynamics — but that distinction **is the whole point** of the project (predicting internal core temperature).
- A PDE discretization is overkill for an 18650 cell at the resolution and timescales we care about, and would explode the calibration cost on a 4-week portfolio scope.
- The two-state lumped form is the **standard cylindrical-cell thermal model** (cf. Lin et al. 2014). It captures the core/surface gradient with two ODEs and four physical parameters.

## What would make us revisit
- Surface-temperature residuals after calibration show systematic structure (oscillation, lag) that a two-state model structurally cannot capture.
- Large axial gradients become a concern (e.g. tab heating during fast cycling).
- The application requires resolving radial temperature distribution, not just core vs surface.

## Related
- [Heat generation: irreversible only](heat-generation-irreversible-only.md) — supplies `Q(t)`.
- [Calibration: constrained fit](calibration-constrained-fit.md) — identifies the four parameters.
- [Validation: surface only](validation-surface-only.md) — what we check after the model is wired up.
- [Sign convention reconciliation](sign-convention.md) — why `Q = I·(V − V_ocv)` with `I < 0` discharge is identical to Lin's form.
- [Lin 2014 source note](../sources/lin-2014-electro-thermal.md) — the paper this ODE form comes from.
- Concept (todo): `lumped-thermal-model.md` — the physical assumptions behind RC-equivalent thermal models.
