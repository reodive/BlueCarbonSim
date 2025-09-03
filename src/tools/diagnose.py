import csv
import os
import sys
from typing import List, Dict, Optional


def summarize_injection_counts(path: str) -> None:
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
        cols = hdr[1:] if len(hdr) > 1 else []
        totals = [0 for _ in cols]
        rows = 0
        for row in r:
            rows += 1
            for i, v in enumerate(row[1:]):
                try:
                    totals[i] += int(v)
                except Exception:
                    pass
        print(f"[diagnose] file={os.path.basename(path)} rows={rows}")
        if cols:
            for name, total in zip(cols, totals):
                print(f"  {name}: total={total}")


def summarize_species_diag(path: str) -> None:
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
        idx_vis = hdr.index("visits") if "visits" in hdr else -1
        idx_elig = hdr.index("eligible") if "eligible" in hdr else -1
        idx_abs = hdr.index("absorptions") if "absorptions" in hdr else -1
        rows = 0
        s_vis = s_elig = s_abs = 0
        for row in r:
            rows += 1
            if idx_vis >= 0:
                try: s_vis += int(row[idx_vis])
                except Exception: pass
            if idx_elig >= 0:
                try: s_elig += int(row[idx_elig])
                except Exception: pass
            if idx_abs >= 0:
                try: s_abs += int(row[idx_abs])
                except Exception: pass
        print(f"[diagnose] file={os.path.basename(path)} rows={rows} visits={s_vis} eligible={s_elig} absorptions={s_abs}")


def main(argv: Optional[List[str]] = None) -> int:
    args = argv or []
    target = args[0] if args else None
    default_dir = os.path.join("outputs", "diagnostics")

    if not target:
        # No path provided: list available diagnostics
        if os.path.isdir(default_dir):
            files = sorted(os.listdir(default_dir))
            if not files:
                print("[diagnose] no diagnostics files found under outputs/diagnostics")
                return 0
            print("[diagnose] available:")
            for name in files:
                print(f"  {name}")
            return 0
        print("[diagnose] no diagnostics directory; run the simulation with diagnostics enabled.")
        return 0

    # If a directory is given, scan inside it
    if os.path.isdir(target):
        for name in sorted(os.listdir(target)):
            path = os.path.join(target, name)
            if not os.path.isfile(path):
                continue
            if name == "injection_counts.csv":
                summarize_injection_counts(path)
            elif name.startswith("species_diag_") and name.endswith(".csv"):
                summarize_species_diag(path)
        return 0

    # Single file
    if not os.path.exists(target):
        # Be forgiving with a common mistaken path
        alt = os.path.join(default_dir, os.path.basename(target))
        if os.path.exists(alt):
            target = alt
        else:
            print(f"[diagnose] path not found: {target}")
            return 1

    name = os.path.basename(target)
    if name == "injection_counts.csv":
        summarize_injection_counts(target)
    elif name.startswith("species_diag_") and name.endswith(".csv"):
        summarize_species_diag(target)
    else:
        # Fallback: print the first few lines
        print(f"[diagnose] preview of {name}:")
        try:
            with open(target, newline="") as f:
                r = csv.reader(f)
                for i, row in enumerate(r):
                    print(",".join(map(str, row)))
                    if i >= 10:
                        break
        except Exception as e:
            print(f"[diagnose] unable to read CSV: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

