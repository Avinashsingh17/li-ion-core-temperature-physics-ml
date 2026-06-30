# Heat generation: irreversible only (no entropic term)

## The question
Which heat-generation formulation does the thermal model use to convert the electrical signals (I, V) into a heat source `Q` for the lumped thermal ODE?

## Options considered
- **Full Bernardi**: `Q = I·(V − V_ocv) − I·T·(∂U_ocv/∂T)` — irreversible + entropic.
- **Irreversible only**: `Q = I·(V − V_ocv)`.
- **Ohmic proxy**: `Q = I²·R` — simple but sign-immune and doesn't capture polarization.

## The call
**Irreversible only**, using the **raw current sign**:

$$Q(t) = I(t) \cdot \bigl(V(t) - V_\text{ocv}(\text{SOC}(t))\bigr)$$

- `I` is the **raw** signed current (A); negative = discharge, positive = charge/regen.
- `V` is the measured terminal voltage (V).
- `V_ocv(SOC)` is the open-circuit voltage curve (V), built per [OCV characterization](ocv-characterization.md).
- Result `Q` is in watts.

**Sign convention is critical**: write `(V − V_ocv)`, **not** `(V_ocv − V)`. With the raw current sign, this yields `Q ≥ 0` in both directions:
- Discharge: `I < 0` and `V < V_ocv` → product is positive.
- Charge: `I > 0` and `V > V_ocv` → product is positive.

**I²R is retained as a sign-immune cross-check**, not as the primary source.

## Why
- The entropic term needs `∂U_ocv/∂T`, and neither LG nor Panasonic provides temperature-resolved OCV.
- The irreversible term dominates at the C-rates and ambient temperatures in this project; entropic contribution is typically smaller in magnitude and SOC-dependent in sign.
- The polarity rule `(V − V_ocv)` with raw `I` is the right one to keep `Q ≥ 0`; the swapped form would flip the sign every direction change and silently produce cooling during charge.

## What would make us revisit
- Acquiring `∂U_ocv/∂T` data from a temperature-resolved OCV measurement.
- Surface-temp residuals after [calibration](calibration-constrained-fit.md) showing systematic SOC-dependence at constant C-rate — a telltale of a missing entropic contribution.
- Operation at higher C-rates where entropic effects become non-negligible.

## Related
- [OCV characterization](ocv-characterization.md) — supplies `V_ocv(SOC)`.
- [Two-state thermal ODE](thermal-ode-two-state.md) — consumes `Q(t)` as the heat source.
- Concept (todo): `entropic-heat.md` — what the omitted term physically represents.
- Concept (todo): `coulomb-counting.md` — how SOC is built from `Ah`.
