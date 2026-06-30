# Sign convention reconciliation — ours vs. Lin 2014

## The question
Lin 2014 writes the heat generation as `Q = I · (V_ocv − V_T)` with `I > 0` for discharge. We write it as `Q = I · (V − V_ocv)` with `I < 0` for discharge (raw signed current from the cycler logs). The two formulas **look like sign-flipped versions of each other**. Which is right? Should we "fix" ours to match the paper?

## The call
**Both formulas are algebraically identical.** Keep ours as written. Do **NOT** edit either factor to make it look like Lin's.

## Algebra

Let `I_L` be Lin's signed current (positive = discharge) and `I` be our raw current (negative = discharge). They differ only in convention:

$$I = -I_L$$

The terminal voltage `V` is the same physical signal in both papers (`V ≡ V_T`). Lin's heat generation:

$$Q_L = I_L \cdot (V_\text{ocv} - V_T)$$

Ours:

$$Q = I \cdot (V - V_\text{ocv}) = (-I_L) \cdot (V_T - V_\text{ocv}) = (-I_L) \cdot (-(V_\text{ocv} - V_T)) = I_L \cdot (V_\text{ocv} - V_T) = Q_L$$

So `Q ≡ Q_L`. Both reduce to `+|I| · |ΔV|` in either regime:

| regime | sign of `I` (Lin) | sign of `V − V_ocv` | sign of `V_ocv − V_T` | `Q_L`            | `Q` (ours)     |
|---|---|---|---|---|---|
| discharge | `I_L > 0`  | `V − V_ocv < 0`  | `V_ocv − V_T > 0` | `(+) · (+) = +|I||ΔV|` | `(−) · (−) = +|I||ΔV|` |
| charge    | `I_L < 0`  | `V − V_ocv > 0`  | `V_ocv − V_T < 0` | `(−) · (−) = +|I||ΔV|` | `(+) · (+) = +|I||ΔV|` |

`Q > 0` in both directions, as physics requires (the cell dissipates heat whether it's being charged or discharged).

## Why this looks different from the paper on purpose

- The LG and Panasonic raw data ship with **`I < 0` for discharge** as the native convention. Re-signing every record on load would put a second sign convention inside our pipeline — one for the loader, one for the model — and that's exactly the kind of bug that hides for weeks.
- We picked the formula that reads off raw signals natively. The math is Lin's; the surface appearance is ours.

**Future readers** (including future-you, comparing to the paper):

- Do **NOT** "fix" `(V − V_ocv)` to `(V_ocv − V)`. That would silently flip the sign of `Q`.
- Do **NOT** flip the loader to make `I > 0` for discharge. That would also silently flip the sign of `Q`, and add a redundant convention layer.
- The formulas are different on the page **and identical in physics**.

## Implementation guard

The heat-gen code computes `Q = I_raw * (V - V_ocv_lookup(SOC))` directly. A unit test should assert `Q > 0` on at least one mid-discharge sample where neither factor is near zero (e.g., a UDDS sample with `|I| > 1 A` and SOC near 0.5). The same test should be repeated on a charge sample. TODO: add when the thermal-model code lands.

## Revisit if

- We add an equivalent-circuit electrical model that exposes `V_T` as a reconstructed quantity. Then `V_T` and the raw measured `V` become two distinct symbols and the sign accounting has to be re-checked.
- We add the entropic heat term `−I · T · (∂U_ocv/∂T)`. The sign of `∂U_ocv/∂T` interacts with the current sign the same way; redo the algebra explicitly when that term is introduced.
- We start working with charge files (`Charge*.parquet`) for thermal calibration. The decision section above already covers charge correctness, but actual charge calibration would be a good place to verify the guard test passes both ways.

## Related

- [Heat generation: irreversible only](heat-generation-irreversible-only.md) — the formula whose sign this note reconciles.
- [Lin 2014 source note](../sources/lin-2014-electro-thermal.md) — paper using the opposite convention.
- [decisions/ocv-characterization](ocv-characterization.md) — produces `V_ocv` used here.
