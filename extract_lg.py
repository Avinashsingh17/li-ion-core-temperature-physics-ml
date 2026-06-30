"""
One-shot extractor for the LG 18650HG2 Mendeley zip.

Why this exists:
- Windows Explorer's built-in zip tool fails partway through this archive
  because internal paths exceed MAX_PATH (260 chars) once combined with
  the OneDrive working directory.
- Python's zipfile handles it, AND we strip the redundant 90-char top-level
  folder name so extracted paths stay well under MAX_PATH for downstream
  tools that aren't long-path aware.

Run once; safe to re-run (skips files that already exist).
"""
from pathlib import Path
import zipfile

ZIP_PATH = Path("LG 18650HG2 Li-ion Battery Data.zip")
OUT_DIR = Path("data") / "LG"
# This redundant top folder gets stripped during extraction.
STRIP_PREFIX = (
    "LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network "
    "xEV SOC Estimator Script/"
)


def extract(zip_path: Path, out_dir: Path, strip: str = "") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if strip and name.startswith(strip):
                name = name[len(strip):]
            if not name or name.endswith("/"):
                continue
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def main() -> None:
    print(f"Extracting {ZIP_PATH} -> {OUT_DIR} (stripping top prefix)")
    n = extract(ZIP_PATH, OUT_DIR, STRIP_PREFIX)
    print(f"  wrote {n} files")

    # Nested zip: prepared dataset used by the FNN script.
    nested = OUT_DIR / "LG_HG2_Prepared_Dataset_McMasterUniversity_Jan_2020" / "LGHG2@n10C_to_25degC.zip"
    if nested.exists():
        nested_out = nested.parent / "LGHG2_prepared"
        print(f"Extracting nested {nested.name} -> {nested_out}")
        n2 = extract(nested, nested_out)
        print(f"  wrote {n2} files")
    else:
        print(f"  WARN: expected nested zip not found at {nested}")


if __name__ == "__main__":
    main()
