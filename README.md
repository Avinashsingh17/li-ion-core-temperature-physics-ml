# Li-ion Core Temperature: a Physics-ML Hybrid

I estimate the internal **core** temperature of a lithium-ion cell — the quantity that governs battery safety — from the signals a real battery management system can actually measure: current, voltage, and surface temperature. No public dataset measures core temperature directly, so a calibrated two-state physics model generates the core-temperature labels, and a machine-learning model learns to reproduce them.

**Every core temperature in this project is model-derived, not measured.** That single fact shapes the whole project, and two findings follow from it. First, the dominant uncertainty is physical, not algorithmic: the parameter that sets the core-surface temperature gap (`R_cs`) is unidentifiable from surface data, leaving a roughly 5 °C uncertainty band on the labels themselves — an order of magnitude larger than the ML's own prediction error. Second, a simple linear model matches gradient boosting to within 0.01 °C, because that unobservable physics, not model choice, is the ceiling.

📄 **The full write-up is in [`report/writeup.md`](report/writeup.md)** — that's the place to start if you want the reasoning rather than the code.

## The pipeline

Two stages. A calibrated physics model turns measurable signals into core-temperature labels; an ML model then learns to reproduce those labels from the same signals, so a deployed system could skip the physics and predict the core directly.

1. **Physics.** A two-state lumped thermal model (a hot core node, a cooler surface node, coupled by internal resistance `R_cs`), with heat `Q = I·(V − V_OCV)`. It's calibrated against measured *surface* temperature using a constrained fit — pin what's physically known, fit only what must be fit — and then used to generate the core labels.
2. **Surrogate.** Ridge regression (reported) and HistGradientBoosting (checked alternative) learn core temperature from current, voltage, surface temperature, and their recent history. Evaluation never random-splits: train and test are always whole held-out drive cycles, scored leave-one-cycle-out.

All reported numbers come from a single locked run, **`2026-06-18b`**. The locked calibration is committed to the repo, and the figure-rendering script refuses to run if it drifts from those values.

## Reproducing the results

Everything is run from the **repository root**. Python **≥ 3.10**.

```bash
pip install -r requirements.txt
```

**1. Get the data.** The raw dataset is not redistributed here (see [Data](#data)). Download it from Mendeley, place the archive (`LG 18650HG2 Li-ion Battery Data.zip`) in the repository root, then unpack:

```bash
python extract_lg.py
```

**2. Reproduce the published figures (default).** This uses the committed, locked `2026-06-18b` calibration — it does *not* re-run the optimizer:

```bash
python data_loading.py        # raw → uniform 1 Hz processed records
python build_ocv.py           # V_OCV(SOC) lookup from the C/20 characterization
python generate_labels.py     # model-derived T_core labels (locked calibration)
python train_ridge.py         # ridge stage  → data/ml/
python train_hgbr.py          # HGBR stage   → data/ml/  (reads ridge outputs)
python report/figures/make_figures.py   # renders F1–F7
```

`make_figures.py` asserts the on-disk calibration against the locked `2026-06-18b` values and stops if they differ — that guard is intentional, so the figures can only ever be rendered from the locked run.

**3. Re-derive the calibration (exploration only).** If you want to re-run the constrained fit from scratch rather than use the locked parameters:

```bash
python calibrate.py           # re-optimizes; overwrites data/calibration/calibration_results.json
```

Re-optimization may not bit-match the locked parameters, in which case `make_figures.py` will correctly halt. Restore the committed `calibration_results.json` to get back to the published figures.

Off the main path: `thermal_model.py` (single-cycle sanity run, `--cycle`), `eda.py` (diagnostic plots), `inspect_mat.py` (raw `.mat` inspection).

## Data

This project uses the **LG 18650HG2** dataset (Kollmeyer et al.). All reported results come from its 25 °C drive cycles — the eleven-cycle set of US06, LA92, UDDS, and eight mixed cycles:

> Kollmeyer, P., Vidal, C., Naguib, M., & Skells, M. (2020). *LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script*. Mendeley Data, V3. https://doi.org/10.17632/cp3473x7xv.3

The data is hosted on Mendeley Data and is **not** included in this repository; download it from the link above and cite it if you use it. See the Mendeley page for license terms.

## Repository layout

```
.
├── report/
│   ├── writeup.md                 # the full write-up — start here
│   └── figures/                   # F1–F7 (committed) + the scripts that render them
├── knowledge/                     # the project's working notebook (see below)
├── build_ocv.py  data_loading.py  calibrate.py  generate_labels.py
├── thermal_model.py  features_and_split.py  train_ridge.py  train_hgbr.py
├── eda.py  inspect_mat.py  extract_lg.py
├── data/calibration/calibration_results.json   # the locked 2026-06-18b calibration (committed)
├── requirements.txt
└── README.md
```

## Working notes (`knowledge/`)

`knowledge/` is the project's working notebook, published deliberately: design **decisions** with their rationale, plain-language **concept** notes on the physics and ML, **source** notes on the key papers, and an honest **log** and **report-notes** file recording caveats and the mistakes I caught along the way. It documents *why* the project is built the way it is, including its limitations. (The source-paper PDFs are not redistributed; only my own notes on them.)

## A note on honesty

The core temperatures here are model-derived, never measured — there is no core thermocouple in the data. A low ML error means the surrogate faithfully reproduces the labels; it does **not** certify that the labels' absolute core values are correct. The `R_cs` band is the project's way of stating exactly how much it can't know. That's the point of the project, not a footnote to it.
