"""
data_loading.py — Battery dataset loader.

Loads LG 18650HG2 (Kollmeyer 2020) and Panasonic 18650PF (Kollmeyer 2017)
drive-cycle and pause/charge data into a unified, uniform 1 Hz pandas
DataFrame schema and writes per-record Parquet files.

Sign convention: current < 0 = discharge, > 0 = charge/regen (raw, unchanged).

Standardized columns (in order):
    time_s, voltage_V, current_A, power_W, T_surface_C, T_ambient_C,
    Ah, Wh, dataset, cycle, ambient_setpoint_C, file_id

Run:  python data_loading.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
LG_ROOT = (
    PROJECT_ROOT / "data" / "LG"
    / "LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020"
)
PAN_ROOT_TOP = PROJECT_ROOT / "Panasonic 18650PF Data"
# Panasonic 25degC lives in a duplicated nested folder from the original zip.
PAN_ROOT_NESTED = PAN_ROOT_TOP / "Panasonic 18650PF Data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
COLS_CONT = ["voltage_V", "current_A", "power_W", "T_surface_C", "T_ambient_C"]
COLS_CUM = ["Ah", "Wh"]
TAG_COLS = ["dataset", "cycle", "ambient_setpoint_C", "file_id"]
STANDARD_COLS = ["time_s"] + COLS_CONT + COLS_CUM + TAG_COLS


# ---------------------------------------------------------------------------
# Filename classification
# ---------------------------------------------------------------------------
# Drive-cycle tokens anchored by separators so we don't false-match inside words.
_DC_TOKEN = r"(UDDS|HWFET|LA92|US06|Mixed\d?|Mix\d?|Cycle_\d|NN)"
DRIVE_CYCLE_RE = re.compile(rf"(?:^|[_\s]){_DC_TOKEN}(?=[_\.\s]|$)", re.IGNORECASE)
PAUSE_CHARGE_RE = re.compile(
    r"(?:^|[_\s])(Charge\d*|Pause\d*|PreChg)(?=[_\.\s]|$)", re.IGNORECASE
)
# Tests we exclude per current scope (drive cycles + pause/charge only).
EXCLUDE_RE = re.compile(
    r"(HPPC|EIS|C20DisCh|Cap_\d|Dis_\d|PausCycl|Trise|5 ?pulse)", re.IGNORECASE
)


def parse_ambient_from_folder(folder_name: str) -> int | None:
    """'25degC' -> 25;  'n10degC' or '-10degC' -> -10;  unrelated -> None."""
    s = folder_name.strip()
    m = re.fullmatch(r"(?:n|-)?(\d+)degC", s)
    if not m:
        return None
    val = int(m.group(1))
    return -val if s.startswith(("n", "-")) else val


def classify_cycle(name: str) -> str | None:
    """Map a filename to a canonical cycle label, or None if it's out of scope."""
    if EXCLUDE_RE.search(name):
        return None
    dc = [m.upper() for m in DRIVE_CYCLE_RE.findall(name)]
    if len(dc) > 1:
        # Panasonic stores some long contiguous files containing multiple cycles.
        return "Contiguous"
    if dc:
        tok = dc[0]
        if tok.startswith("MIX"):
            num = re.search(r"\d", tok)
            return f"Mixed{num.group() if num else ''}"
        if tok.startswith("CYCLE_"):
            return tok.replace("_", "").title()  # CYCLE_1 -> Cycle1
        return tok
    pc = PAUSE_CHARGE_RE.search(name)
    if pc:
        return pc.group(1).capitalize()
    return None


# ---------------------------------------------------------------------------
# LG loaders
# ---------------------------------------------------------------------------
LG_CSV_HEADER_SKIPROWS = 28  # rows 1..28 are metadata; row 29 is the column header
LG_CSV_RENAME = {
    "Voltage": "voltage_V",
    "Current": "current_A",
    "Temperature": "T_surface_C",
    "Capacity": "Ah",
    "WhAccu": "Wh",
}


def _hms_to_seconds(s: pd.Series) -> pd.Series:
    """'HH:MM:SS.fff' -> float seconds via pandas Timedelta."""
    return pd.to_timedelta(s.astype(str), errors="coerce").dt.total_seconds()


def load_lg_csv(path: Path) -> pd.DataFrame:
    """Read an LG cycler CSV.

    The file has 28 metadata lines, then a column-names row, then a units row
    (e.g., '[V]', '[A]'), then data. We skip metadata, treat row 29 as the
    header, then drop the units row.
    """
    df = pd.read_csv(path, skiprows=LG_CSV_HEADER_SKIPROWS, header=0,
                     low_memory=False)
    # Drop the units row if present (always the first data row after the header).
    if df.iloc[0].astype(str).str.contains(r"\[.+\]", regex=True).any():
        df = df.iloc[1:].reset_index(drop=True)
    # Drop any trailing unnamed column from a stray comma.
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    # Coerce numerics.
    for c in ["Voltage", "Current", "Temperature", "Capacity", "WhAccu"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Time from 'Prog Time' (HH:MM:SS.fff); zero-base it.
    if "Prog Time" in df.columns:
        secs = _hms_to_seconds(df["Prog Time"])
        df["time_s"] = secs - secs.iloc[0]
    df = df.rename(columns=LG_CSV_RENAME)
    # CSV has no Power column; derive it (raw sign convention preserved).
    df["power_W"] = df["voltage_V"] * df["current_A"]
    return df


def _load_mat_meas(path: Path) -> dict[str, np.ndarray]:
    """Load .mat and return the 'meas' struct fields as a dict of arrays."""
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "meas" not in md:
        raise ValueError(f"{path.name}: missing 'meas' (top keys: {list(md)})")
    meas = md["meas"]
    return {fn: np.asarray(getattr(meas, fn)) for fn in meas._fieldnames}


def load_lg_mat(path: Path) -> pd.DataFrame:
    """LG .mat -> DataFrame. No chamber probe; T_ambient_C left for caller."""
    d = _load_mat_meas(path)
    return pd.DataFrame({
        "time_s":      d["Time"].astype(float),
        "voltage_V":   d["Voltage"].astype(float),
        "current_A":   d["Current"].astype(float),
        "Ah":          d["Ah"].astype(float),
        "Wh":          d["Wh"].astype(float),
        "power_W":     d["Power"].astype(float),
        "T_surface_C": d["Battery_Temp_degC"].astype(float),
    })


def load_panasonic_mat(path: Path) -> pd.DataFrame:
    """Panasonic .mat -> DataFrame (uniquely has Chamber_Temp_degC)."""
    d = _load_mat_meas(path)
    return pd.DataFrame({
        "time_s":      d["Time"].astype(float),
        "voltage_V":   d["Voltage"].astype(float),
        "current_A":   d["Current"].astype(float),
        "Ah":          d["Ah"].astype(float),
        "Wh":          d["Wh"].astype(float),
        "power_W":     d["Power"].astype(float),
        "T_surface_C": d["Battery_Temp_degC"].astype(float),
        # Chamber_Temp is stored as uint8 in the source — 1°C resolution only.
        "T_ambient_C": d["Chamber_Temp_degC"].astype(float),
    })


# ---------------------------------------------------------------------------
# Resampling to uniform 1 Hz
# ---------------------------------------------------------------------------
def resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Linear-interp every numeric column onto a uniform 1-second grid.

    A single np.interp pass handles both directions:
      - Up-sampling slow charge/pause segments (sometimes 1 sample / 5 s).
      - Down-sampling 10 Hz drive-cycle data (0.1 s native step).
    Linear interpolation is appropriate because the physical signals
    (voltage, current, surface temp) are smooth at these timescales, and
    the cumulative signals (Ah, Wh) are integrals — linear interp between
    samples preserves their monotonic structure.

    Tag columns (constant strings/ints) are broadcast across the new grid.
    Duplicate timestamps (a rare cycler-log quirk) are dropped first.
    """
    t = df["time_s"].to_numpy(dtype=float)
    # Drop duplicate timestamps.
    _, uidx = np.unique(t, return_index=True)
    if len(uidx) != len(t):
        df = df.iloc[np.sort(uidx)].reset_index(drop=True)
        t = df["time_s"].to_numpy(dtype=float)
    if len(t) < 2:
        return df.copy()

    t_grid = np.arange(0.0, t[-1] + 1e-9, 1.0)
    out: dict[str, np.ndarray] = {"time_s": t_grid}
    for c in df.columns:
        if c == "time_s":
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            out[c] = np.interp(t_grid, t, col.to_numpy(dtype=float))
        else:
            # Tag columns: constant -> broadcast first value across the grid.
            v = col.iloc[0]
            out[c] = np.full(t_grid.shape, v, dtype=object)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
@dataclass
class FileMeta:
    path: Path
    dataset: str          # "LG" or "Panasonic"
    ambient_C: int        # folder-encoded ambient setpoint, °C
    cycle: str            # canonical label (UDDS, HWFET, Charge3, ...)
    file_id: str          # filename stem of the source .mat (or .csv if CSV-only)
    source: str           # "csv" or "mat"


def discover_lg_files(root: Path = LG_ROOT) -> list[FileMeta]:
    """Walk LG temperature folders. Prefer CSV; fall back to .mat.

    LG ships both .mat and matching .csv for many tests; the CSV is used when
    a matching one exists, because it carries Step/Status columns useful for
    later filtering. Files without a CSV (typical for HPPC etc., though those
    are excluded by scope) fall back to .mat.
    """
    out: list[FileMeta] = []
    if not root.exists():
        return out
    for amb_dir in sorted(root.iterdir()):
        if not amb_dir.is_dir():
            continue
        ambient = parse_ambient_from_folder(amb_dir.name)
        if ambient is None:
            continue
        for mat in amb_dir.glob("*.mat"):
            cycle = classify_cycle(mat.name)
            if cycle is None:
                continue
            tid = re.search(r"\s(\d{3})_", mat.name)
            test_id = tid.group(1) if tid else None
            csv = None
            if test_id is not None:
                # Match CSV by both test_id AND cycle keyword: each test_id may
                # have several CSVs (one per cycle subsegment).
                cands = [
                    p for p in amb_dir.glob(f"{test_id}_*.csv")
                    if classify_cycle(p.name) == cycle
                ]
                csv = cands[0] if cands else None
            out.append(FileMeta(
                path=csv or mat,
                dataset="LG",
                ambient_C=ambient,
                cycle=cycle,
                file_id=mat.stem,
                source="csv" if csv else "mat",
            ))
    return out


def discover_panasonic_files(top: Path = PAN_ROOT_TOP,
                             nested: Path = PAN_ROOT_NESTED) -> list[FileMeta]:
    """Walk Panasonic temperature folders (top + nested), excluding Trise variants."""
    out: list[FileMeta] = []
    seen: set[tuple] = set()
    for root in (top, nested):
        if not root.exists():
            continue
        for amb_dir in sorted(root.iterdir()):
            if not amb_dir.is_dir():
                continue
            ambient = parse_ambient_from_folder(amb_dir.name)
            if ambient is None or "Trise" in amb_dir.name:
                continue
            for sub_name in ("Drive cycles", "Drive Cycles", "Charges and Pauses"):
                sub = amb_dir / sub_name
                if not sub.exists():
                    continue
                for mat in sub.glob("*.mat"):
                    cycle = classify_cycle(mat.name)
                    if cycle is None:
                        continue
                    key = (ambient, cycle, mat.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(FileMeta(
                        path=mat,
                        dataset="Panasonic",
                        ambient_C=ambient,
                        cycle=cycle,
                        file_id=mat.stem,
                        source="mat",
                    ))
    return out


# ---------------------------------------------------------------------------
# Per-record loader + quality checks
# ---------------------------------------------------------------------------
def load_record(meta: FileMeta) -> pd.DataFrame:
    """Read one file, standardize columns, tag, resample to 1 Hz."""
    if meta.dataset == "LG":
        df = load_lg_csv(meta.path) if meta.source == "csv" else load_lg_mat(meta.path)
    else:
        df = load_panasonic_mat(meta.path)

    # Ensure every standard physical column exists.
    for c in COLS_CONT + COLS_CUM:
        if c not in df.columns:
            df[c] = np.nan
    # LG has no chamber probe — use the folder setpoint as ambient.
    if df["T_ambient_C"].isna().all():
        df["T_ambient_C"] = float(meta.ambient_C)

    df["dataset"] = meta.dataset
    df["cycle"] = meta.cycle
    df["ambient_setpoint_C"] = meta.ambient_C
    df["file_id"] = meta.file_id

    return resample_1hz(df[STANDARD_COLS])


def quality_check(df: pd.DataFrame, meta: FileMeta) -> list[str]:
    """Return a list of human-readable flags — never silently paper over."""
    flags: list[str] = []
    if len(df) < 10:
        flags.append("very short record (<10 s)")
    if df["current_A"].abs().max() < 0.05:
        # Catches files like LG 551_HWFET_25degC where logger captured only the pause.
        flags.append("no current activity (pause-only file?)")
    if df[["voltage_V", "current_A", "T_surface_C"]].isna().any().any():
        flags.append("NaNs in physical signals after resample")
    # Sanity check sign convention: a drive cycle should swing current both ways
    # (regen at warm temps) or strongly negative (cold, no regen).
    if meta.cycle in {"UDDS", "HWFET", "LA92", "US06"}:
        if df["current_A"].min() > -0.5:
            flags.append("drive cycle but current never goes meaningfully negative")
    return flags


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def out_path_for(meta: FileMeta, out_dir: Path = OUTPUT_DIR) -> Path:
    return out_dir / f"{meta.dataset}_{meta.ambient_C}degC_{meta.cycle}_{_safe(meta.file_id)}.parquet"


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------
def process_all(out_dir: Path = OUTPUT_DIR, *, limit: int | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Discover -> load -> resample -> write Parquet for every in-scope file.

    `limit` processes only the first N files (smoke test).
    Returns an index DataFrame (also persisted as `_index.parquet`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = discover_lg_files() + discover_panasonic_files()
    if limit:
        metas = metas[:limit]

    rows: list[dict] = []
    for i, m in enumerate(metas, 1):
        try:
            df = load_record(m)
            flags = quality_check(df, m)
            out = out_path_for(m, out_dir)
            df.to_parquet(out, index=False)
            rows.append({
                "dataset": m.dataset, "ambient_C": m.ambient_C, "cycle": m.cycle,
                "file_id": m.file_id, "source": m.source,
                "n_rows_1Hz": len(df), "out": out.name,
                "flags": "; ".join(flags),
            })
            if verbose:
                tag = f"   [FLAG: {rows[-1]['flags']}]" if flags else ""
                print(f"  ({i:>3}/{len(metas)})  {out.name}{tag}")
        except Exception as e:
            warnings.warn(f"FAILED {m.path}: {e}")
            rows.append({
                "dataset": m.dataset, "ambient_C": m.ambient_C, "cycle": m.cycle,
                "file_id": m.file_id, "source": m.source,
                "n_rows_1Hz": 0, "out": "", "flags": f"FAILED: {e}",
            })

    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / "_index.parquet", index=False)
    return index


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="process first N files only (smoke test)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    idx = process_all(args.out, limit=args.limit)
    ok = (idx["n_rows_1Hz"] > 0).sum()
    print(f"\nWrote {ok} of {len(idx)} records to {args.out}")

    flagged = idx[(idx["flags"] != "") & (idx["n_rows_1Hz"] > 0)]
    if len(flagged):
        print(f"\n{len(flagged)} records flagged:")
        for _, r in flagged.iterrows():
            print(f"  {r['dataset']:9} {r['ambient_C']:>4}degC  {r['cycle']:11}  "
                  f"{r['file_id'][:55]:55}  -> {r['flags']}")
    failed = idx[idx["flags"].str.startswith("FAILED")]
    if len(failed):
        print(f"\n{len(failed)} records FAILED to load:")
        for _, r in failed.iterrows():
            print(f"  {r['dataset']:9} {r['ambient_C']:>4}degC  {r['file_id']}  -> {r['flags']}")
