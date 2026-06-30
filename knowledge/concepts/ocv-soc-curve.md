# V_ocv(SOC) curve

## What it is
A lookup function that returns the cell's **open-circuit voltage** — the voltage you'd measure at the terminals if no current flowed and the cell had fully relaxed — as a function of **state of charge**. For a Li-ion 18650 it's a monotonic curve, ~2.7 V at empty up to ~4.2 V at full, with a roughly-flat plateau in the middle and steeper drops near the endpoints.

## Intuition
Voltage during driving has two parts:
- **Thermodynamic** — what the cell *wants* to read at this SOC if nothing was happening. That's `V_ocv(SOC)`.
- **Polarization / IR drop** — extra voltage offset because current is flowing through internal resistance and electrochemical kinetics. That's `V − V_ocv`.

The first part is reversible and depends only on chemistry and SOC. The second part is where heat comes from. Separating them is what `V_ocv(SOC)` enables.

## Concrete example in this project
The heat generation formula in [decisions/heat-generation-irreversible-only](../decisions/heat-generation-irreversible-only.md) is

$$Q = I \cdot (V - V_\text{ocv})$$

You can't compute it without knowing `V_ocv` at each instant. Since SOC changes throughout a drive cycle (via [coulomb counting](coulomb-counting.md)), we need a `V_ocv(SOC)` lookup that we can evaluate at every time step.

We build it once from the LG C/20 characterization file at 25 °C — a near-equilibrium sweep where polarization is small enough that the measured terminal voltage ≈ `V_ocv`. That gives a single static curve, and we (currently) assume it doesn't depend on temperature.

## Pseudo-OCV vs true OCV
At C/20, current is small but not zero, so there's still some polarization. The curve we extract is technically *pseudo-OCV*, not the true thermodynamic OCV. For our purposes:
- The error from polarization at C/20 is much smaller than the dynamic-cycle polarization we're trying to extract, so it's tolerable.
- The cleanest alternative would be a pulse-rest sequence (HPPC) with long rests, but those have their own segmentation noise and a fully-relaxed rest takes hours.

This is why the [OCV characterization decision](../decisions/ocv-characterization.md) calls for C/20.

## Failure modes it has
- **Hysteresis**: the V_ocv reached from above (during discharge) and from below (during charge) often differ by 10–50 mV. We're using only the discharge half. If predicted heat shows a discharge/charge asymmetry, hysteresis is a likely cause.
- **Temperature dependence**: the *true* OCV shifts slightly with temperature (the entropic term, ~mV/K). We're omitting this and assuming `V_ocv(SOC, T) ≈ V_ocv(SOC, 25°C)`. Fine at moderate C-rates; less fine at extreme temperatures.
- **SOC coverage gaps**: our C/20 file starts at V ≈ 4.18 V, not the full 4.20 V, so the top sliver of the SOC range is missing. Code must either clip or warn when downstream SOC exceeds the table.

## When it matters in this project
- Building heat generation `Q(t)` from `I, V, SOC` for every cycle.
- Calibration: any error in `V_ocv` propagates directly into `Q`, which propagates into the fitted thermal parameters. Endpoints (high and low SOC) are especially sensitive because the curve is steep there — small SOC errors produce big `V_ocv` errors.

## Related
- [coulomb-counting](coulomb-counting.md) — how SOC is computed.
- [decisions/ocv-characterization](../decisions/ocv-characterization.md) — the choice of C/20 over alternatives.
- [decisions/heat-generation-irreversible-only](../decisions/heat-generation-irreversible-only.md) — what consumes this curve.
- Concept (todo): `entropic-heat.md`.
- Concept (todo): `polarization-overpotential.md`.
