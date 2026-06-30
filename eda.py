"""
eda.py — sanity-check plots on the resampled 1 Hz Parquet records.

For one representative cycle from each dataset, plots:
  - Current vs time
  - Voltage vs time
  - Surface (and ambient if measured) temperature vs time
  - Current histogram

Run:  python eda.py
Outputs to ./data/eda_plots/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED = Path("data/processed")
OUT_DIR = Path("data/eda_plots")


def plot_signals(parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(parquet)
    name = parquet.stem

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df["time_s"], df["current_A"], lw=0.7)
    axes[0].axhline(0, c="k", lw=0.3)
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title(name, fontsize=10)

    axes[1].plot(df["time_s"], df["voltage_V"], lw=0.7, color="C1")
    axes[1].set_ylabel("Voltage (V)")

    axes[2].plot(df["time_s"], df["T_surface_C"], lw=0.8, color="C3",
                 label="T_surface")
    # Plot ambient only if the dataset has a real measurement (not constant).
    if df["T_ambient_C"].notna().any() and df["T_ambient_C"].std() > 1e-3:
        axes[2].plot(df["time_s"], df["T_ambient_C"], lw=0.7, ls="--",
                     color="C2", label="T_ambient")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_signals.png", dpi=140)
    plt.close(fig)


def plot_current_hist(parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(parquet)
    name = parquet.stem
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["current_A"].dropna(), bins=80)
    ax.axvline(0, c="k", lw=0.4)
    ax.set_xlabel("Current (A)  (negative = discharge)")
    ax.set_ylabel("count")
    ax.set_title(f"{name}  —  current distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_current_hist.png", dpi=140)
    plt.close(fig)


def pick_representative(dataset: str, cycle: str, ambient_C: int = 25) -> Path | None:
    """Pick the first available parquet matching a dataset/cycle/ambient combo."""
    pattern = f"{dataset}_{ambient_C}degC_{cycle}_*.parquet"
    matches = sorted(PROCESSED.glob(pattern))
    return matches[0] if matches else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("LG", "UDDS", 25),
        ("Panasonic", "Cycle1", 25),
    ]
    for ds, cyc, t in targets:
        p = pick_representative(ds, cyc, t)
        if p is None:
            print(f"  no parquet found for {ds} / {cyc} / {t}°C — skipping")
            continue
        print(f"  plotting {p.name}")
        plot_signals(p, OUT_DIR)
        plot_current_hist(p, OUT_DIR)
    print(f"\nplots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
