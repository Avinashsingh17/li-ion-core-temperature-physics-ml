# Presentation notes

Audience-facing, plain-language recaps of each week's work. For use when explaining or presenting the project to a non-technical audience.

This file is distinct from:
- [`report-notes.md`](report-notes.md) — honest caveats and limitations for the final writeup.
- [`concept-map.md`](concept-map.md) — learner-facing glossary of physics & ML concepts.

Cross-link to those files where they touch (especially the surface-vs-core caveat from `report-notes.md`); don't duplicate the content here.

## Format (per week)

```
## Week N — <short title>
- **One-liner** (slide-title length)
- **Narration** (a short spoken paragraph)
- **Analogy** (the everyday comparison that makes it click)
- **Coming next**: (optional one-liner)
```

**Authoring rule**: when the user pastes a polished recap in chat, file it **verbatim**. Do not rewrite, paraphrase, or "improve" the wording — it's calibrated for a non-technical audience.

---

## Week 1 — Getting the data into shape
- **One-liner**: "We took two public datasets of real battery tests and cleaned them into one consistent, analysis-ready format."
- **Narration**: Two public datasets of real lithium-ion cells were downloaded — batteries run through simulated driving patterns at various temperatures. The raw files were messy and inconsistent, so a loader was built to standardize everything: same column names, same units, one reading per second across all 278 test records. Basic sanity plots confirmed the data is physically reasonable, and a few broken files were flagged and set aside. The crucial honesty point that shapes the whole project: these datasets only ever measure the battery's surface temperature — never the internal core temperature, which is exactly what we ultimately want to predict.
- **Analogy**: It's like measuring someone's skin temperature when what you really care about is their core body temperature during a fever. The whole project exists to bridge that gap.

## Week 2 (in progress) — Building the resting-voltage curve
- **One-liner**: "We built the curve that tells us how much heat the battery generates — the foundation for the whole physics model."
- **Narration**: To estimate how much heat a battery produces, you first need its "resting voltage" at every charge level — the voltage it would settle to if you stopped using it and let it relax. That's the open-circuit voltage. We extracted it from an extremely slow discharge test — 18 hours, slow enough that the cell is essentially at rest the whole time — producing a lookup table: at any given charge level, here's the resting voltage. The gap between this resting voltage and the actual voltage under load, multiplied by the current, is the power being lost as heat. Sanity checks confirmed the curve has the correct textbook shape and rises smoothly as charge increases.
- **Analogy**: The resting voltage is the battery's "calm pulse." When you put it under load, its voltage sags away from that calm value — and the size of that sag tells you how hard it's straining internally, which is what turns into heat. No resting-voltage curve, no heat estimate; no heat estimate, no temperature model.

## Week 2 (continued) — Finding the parameter we can't measure
- **One-liner**: "We discovered that the one quantity our method can't pin down is the one that matters most — and turned that into a measured uncertainty instead of a hidden flaw."
- **Narration**: Our model splits the battery into two connected parts — a hot core and a cooler surface — and the whole point is to estimate the temperature gap between them, since only the surface can be measured. That gap depends on a parameter describing how easily heat flows from core to surface. The catch: because we can only ever measure the surface, that exact parameter is the hardest one to determine from our data — and two independent research papers confirmed this is a fundamental limitation, not a gap in our effort. Rather than hide that, we anchor the parameter to physically reasonable values and then deliberately vary it across its plausible range, which gives us an honest uncertainty band around our predictions instead of a falsely precise single number.
- **Analogy**: It's like estimating how hot an engine's core is from a thermometer taped to its casing. The casing reading is real, but converting it to the core temperature depends on how well heat travels through the metal — and you can't measure that from the outside. The honest move isn't to pretend you know it exactly; it's to say "it's somewhere in this range, so the core is somewhere in this band."
- **Why this one matters in a talk**: it's the rare result that's simultaneously impressive and humble — "we found our method's weakest point and quantified it" reads as rigor, not failure. Lead with it when an audience is technical enough to appreciate that naming your own limitation is the credible move.

*Cross-ref: this is the audience-facing surface of the headline caveat in [report-notes.md](report-notes.md) — `R_int` identifiability finding + mitigation plan.*

## Week 2 (continued) — Calibrating the model, and learning to distrust a good-looking result
- **One-liner**: "We tuned the physics model to match reality — and the hard part wasn't the tuning, it was catching three different ways the numbers could look right while being wrong."
- **Narration**: The model has four physical dials — two for how much heat the battery's core and surface can hold, two for how easily heat moves between them and escapes to the air. We set them by matching the model's predicted surface temperature to the real measured surface temperature. The honest story is in what went wrong along the way. First, the model lost to a trivial "just guess the room temperature" baseline — and the cause was that the dataset reported the thermostat's setting, not the temperature the battery actually sat at, which were about a degree apart. Once we measured the true resting temperature from the data itself, the model beat the baseline. But then we caught ourselves cheating: that "true temperature" had accidentally been calculated using one of the very test cycles we were supposed to be holding back for grading. After fixing that, we found the easy test cycles were too gentle to prove anything — the battery barely heated up, so "just guess the resting temperature" was already nearly perfect. Only when we tested on aggressive, cold-weather cycles did we get a real verdict: the model works well in the conditions it was tuned for, and breaks down in cold conditions it never saw. We report all of it.
- **Analogy**: It's like sighting in a rifle. The first group was off — but because the scope was misaligned, not the shooter. We fixed the scope. Then we realized we'd been checking our aim using the same shots we were trying to grade. We re-graded honestly. Then we noticed the target was so close that anyone would hit it, so we moved it back to a real distance — and only then learned where the shots actually land.
- **Why this one matters in a talk**: the whole arc is self-correction — a data bug, then our own accidental cheat, then a too-easy test exposed. An audience trusts "here are the three mistakes we caught" far more than "it worked." It's the most credible thing in the project.

*Cross-ref: the "breaks down in cold conditions" point is the audience surface of the 2026-06-18b cold-ambient limitation in [report-notes.md](report-notes.md) — the severity-stratified result showing 0 °C drive cycles fall outside the model's reliable regime.*

## Week 3 — Teaching a model to predict the hidden temperature
- **One-liner**: "We trained machine-learning models to predict the battery's core temperature — and found a simple model does just as well as a complex one, because the real uncertainty lives in the physics, not the algorithm."
- **Narration**: We took the core-temperature labels from the physics model and trained machine-learning models to reproduce them using only the signals a real battery system can actually measure — current, voltage, surface temperature, and their recent history. We tested honestly: every model was scored only on drive cycles it had never seen, never on scattered moments from cycles it had already learned. A simple linear model and a much more flexible one — an ensemble of decision trees — came out essentially tied, both predicting the unseen core temperature to within about half a degree. That tie is the real finding. It means the link between what we can measure and the hidden core temperature is simple enough that a basic model captures it, and reaching for something fancier adds nothing. And that half-degree the model is unsure about is small next to the several-degree uncertainty we already measured in the physics itself — the part we fundamentally cannot observe. The machine learning was never the bottleneck; the unmeasurable internal property always was.
- **Analogy**: It's like predicting someone's core body temperature from their skin temperature and how hard they've been exercising. Once you have those two things, a rule of thumb works about as well as an elaborate formula — and either way, your real problem isn't the method, it's that you can never directly check the core reading to confirm it. Saying that out loud is the whole point.
- **Coming next**: writing it all up for the public project page — the method, the honest limitations, and the headline that our biggest uncertainty is physical, not algorithmic.

*Cross-ref: the "uncertainty is physical, not algorithmic" headline is the audience-facing surface of the ridge≈HGBR finding and the R_cs label band in [report-notes.md](report-notes.md).*
