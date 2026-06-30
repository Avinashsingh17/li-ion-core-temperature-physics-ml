"""Read-only schema inspection of one drive-cycle .mat from each dataset."""
from pathlib import Path
import scipy.io as sio
import numpy as np

FILES = {
    "LG_HWFET_25C": Path(
        "data/LG/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC/"
        "10-29-18_01.36 551_HWFET_25degC_LGHG2.mat"
    ),
    "Panasonic_HWFET_25C": Path(
        "Panasonic 18650PF Data/Panasonic 18650PF Data/25degC/Drive cycles/"
    ),
}


def header_bytes(p: Path) -> str:
    with open(p, "rb") as f:
        return f.read(128).decode("latin-1", errors="replace").strip("\x00 ")


def inspect(label: str, path: Path) -> None:
    print(f"\n{'=' * 60}\n{label}  ->  {path}\n{'=' * 60}")
    print("MATLAB header:", header_bytes(path)[:80])
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"  scipy.io.loadmat FAILED: {e}")
        return
    keys = [k for k in d if not k.startswith("__")]
    print("Top-level keys:", keys)
    for k in keys:
        v = d[k]
        print(f"  [{k}] type={type(v).__name__}", end="")
        if isinstance(v, np.ndarray):
            print(f" dtype={v.dtype} shape={v.shape}")
            if v.dtype.names:
                print(f"    struct fields: {v.dtype.names}")
        elif hasattr(v, "_fieldnames"):
            print(f"  struct fields: {v._fieldnames}")
            for fn in v._fieldnames:
                arr = getattr(v, fn)
                if isinstance(arr, np.ndarray):
                    print(f"      .{fn:18s} shape={arr.shape} dtype={arr.dtype}", end="")
                    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                        try:
                            print(f"  range=[{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]")
                        except Exception:
                            print()
                    else:
                        print()
                else:
                    print(f"      .{fn} = {arr!r:.60}")
        else:
            print(f"  value={v!r:.80}")


def find_panasonic_drive_cycle() -> Path | None:
    d = Path("Panasonic 18650PF Data/Panasonic 18650PF Data/25degC/Drive cycles")
    if not d.exists():
        return None
    cands = sorted(d.glob("*HWFET*.mat"))
    return cands[0] if cands else next(iter(d.glob("*.mat")), None)


if __name__ == "__main__":
    inspect("LG  HWFET 25degC", FILES["LG_HWFET_25C"])
    pan = find_panasonic_drive_cycle()
    if pan:
        inspect(f"Panasonic 25degC  ({pan.name})", pan)
    else:
        print("Could not locate Panasonic drive cycle file")
