# Coulomb counting

## What it is
A way of tracking **state of charge (SOC)** by integrating current over time. If you know how much charge a fully-charged cell holds (its capacity, in amp-hours), and you measure how much charge has flowed in or out since then, you can compute what fraction remains:

$$\text{SOC}(t) = \text{SOC}(t_0) + \frac{1}{Q_\text{nominal}} \int_{t_0}^{t} I(\tau)\,d\tau$$

In code, with the standardized `Ah` column (which is already the integral of current — the cycler did the integration in hardware):

```python
SOC = 1.0 + (Ah - Ah[0]) / Q_nominal
```

where `SOC(t_0) = 1` if we start from full charge.

## Intuition
A battery is a bucket of charge. Current is the rate at which charge enters or leaves. If you know the bucket's size and you've been watching the spout, you know how full it is right now — assuming you started knowing how full it was, and the bucket isn't leaking.

## Concrete example in this project
For the LG C/20 characterization file at 25 °C:
- The cell starts full (V ≈ 4.18 V).
- A near-constant 150 mA discharge runs for ~20 hours.
- `Ah` at the end of the discharge half ≈ −2.78 Ah.
- With `Q_nominal = 3.0 Ah`: `SOC = 1 + (−2.78 / 3.0) ≈ 0.073` — the cell is at about 7% SOC.

That `SOC` is then paired with the measured voltage to produce the `V_ocv(SOC)` lookup used in heat generation. See [decisions/ocv-characterization](../decisions/ocv-characterization.md).

## Failure modes it has (= what you have to be careful about)
- **Drift**: small current-measurement bias integrates to a large SOC error over hours.
- **Wrong starting SOC**: if you don't actually start from a known full (or empty) state, every SOC value is offset.
- **Capacity fade**: real cells lose capacity with age and cycles. `Q_nominal = 3.0 Ah` is a datasheet number; if the cell now holds 2.7 Ah, every SOC reading is too high near empty.
- **No self-discharge correction**: real cells leak charge when sitting; coulomb counting on the spout misses what evaporates from the bucket.

For a fresh cell on a single C/20 sweep, none of these are big issues, which is why C/20 is a clean choice for OCV characterization.

## When it matters in this project
- Building `V_ocv(SOC)` for the heat-generation formula — every entry on the OCV table is `(SOC_via_coulomb_counting, measured_V_at_rest)`.
- Later, when running the thermal model on a drive cycle, we'll need SOC on that cycle too (to look up `V_ocv`). Coulomb counting from the start of each cycle is what gets used; resetting properly between files is essential to avoid drift accumulating across the whole dataset.

## Related
- [V_ocv(SOC) curve](ocv-soc-curve.md).
- [decisions/ocv-characterization](../decisions/ocv-characterization.md).
- [decisions/heat-generation-irreversible-only](../decisions/heat-generation-irreversible-only.md).
