import itertools
import json
import os
import shutil
from typing import Dict, List, Tuple

from ..simulation import run_simulation
from ..utils.config import load_config


def write_cfg_overrides(path: str, overrides: Dict[str, object]) -> None:
    lines: List[str] = []
    seen = set()
    if os.path.exists(path):
        # robust read for non-UTF8 files
        with open(path, "rb") as fb:
            raw = fb.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp932", errors="ignore")
        for ln in text.splitlines(True):
            key = ln.split(":", 1)[0].strip()
            if key in overrides:
                continue
            lines.append(ln)
    for k, v in overrides.items():
        if isinstance(v, bool):
            vv = "true" if v else "false"
        else:
            vv = str(v)
        lines.append(f"{k}: {vv}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def read_diag_counts(diag_dir: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    if not os.path.isdir(diag_dir):
        return out
    for name in os.listdir(diag_dir):
        if not name.startswith("species_diag_") or not name.endswith(".csv"):
            continue
        sp = name[len("species_diag_") : -len(".csv")]
        visits = eligible = absorptions = 0
        p = os.path.join(diag_dir, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                next(f, None)  # header
                for ln in f:
                    parts = ln.strip().split(",")
                    if len(parts) >= 4:
                        try:
                            v = int(parts[1]); e = int(parts[2]); a = int(parts[3])
                        except Exception:
                            v = e = a = 0
                        visits += v; eligible += e; absorptions += a
        except Exception:
            pass
        out[sp] = {"visits": visits, "eligible": eligible, "absorptions": absorptions}
    return out


def summarize(diag: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for sp, d in diag.items():
        v = max(d.get("visits", 0), 0)
        e = max(d.get("eligible", 0), 0)
        a = max(d.get("absorptions", 0), 0)
        res[sp] = {
            "visits": float(v),
            "eligible": float(e),
            "absorptions": float(a),
            "eligibility_rate": (e / v if v > 0 else 0.0),
            "capture_given_eligible": (a / e if e > 0 else 0.0),
        }
    return res


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep diagnostics across a few parameter variants")
    parser.add_argument("--steps", type=int, default=100, help="Total steps per run")
    parser.add_argument("--variants", type=str, default="", help="JSON of overrides list")
    parser.add_argument("--out", type=str, default="outputs/diagnostics/sweep_summary.csv")
    args = parser.parse_args()

    cfg_path = "config.yaml"
    base_cfg = load_config(cfg_path)

    if args.variants:
        try:
            variants: List[Dict[str, object]] = json.loads(args.variants)
            assert isinstance(variants, list)
        except Exception as e:
            print(f"[error] invalid --variants JSON: {e}")
            return 2
    else:
        variants = [
            {"plankton_capture_radius_m": 2.0, "microalgae_min_eff": 0.10, "injection_sigma_px": 1.0},
            {"plankton_capture_radius_m": 2.5, "microalgae_min_eff": 0.15, "injection_sigma_px": 1.5},
            {"plankton_capture_radius_m": 3.0, "microalgae_min_eff": 0.20, "injection_sigma_px": 2.0},
        ]

    # Always enable diagnostics for the sweep
    for v in variants:
        v["diag_enabled"] = True

    rows: List[List[str]] = []
    header = [
        "variant_index",
        "plankton_capture_radius_m",
        "microalgae_min_eff",
        "injection_sigma_px",
        "reflect_boundaries",
        "chl_visits", "chl_eligible", "chl_absorptions",
        "chl_elig_rate", "chl_cap_given_elig",
        "nanno_visits", "nanno_eligible", "nanno_absorptions",
        "nanno_elig_rate", "nanno_cap_given_elig",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    try:
        for idx, ov in enumerate(variants):
            # Write overrides
            write_cfg_overrides(cfg_path, ov)
            # Run a shorter sim per variant
            try:
                run_simulation(total_steps=int(args.steps))
            except Exception as e:
                print(f"[warn] simulation failed for variant {idx}: {e}")
            diag = read_diag_counts("outputs/diagnostics")
            summ = summarize(diag)
            chl = summ.get("Chlorella_vulgaris", summ.get("Chlorella vulgaris", {}))
            nan = summ.get("Nannochloropsis_gaditana", summ.get("Nannochloropsis gaditana", {}))
            row = [
                str(idx),
                str(ov.get("plankton_capture_radius_m", "")),
                str(ov.get("microalgae_min_eff", "")),
                str(ov.get("injection_sigma_px", "")),
                str(ov.get("reflect_boundaries", base_cfg.get("reflect_boundaries", False))),
                str(int(chl.get("visits", 0))), str(int(chl.get("eligible", 0))), str(int(chl.get("absorptions", 0))),
                f"{chl.get('eligibility_rate', 0.0):.3f}", f"{chl.get('capture_given_eligible', 0.0):.3f}",
                str(int(nan.get("visits", 0))), str(int(nan.get("eligible", 0))), str(int(nan.get("absorptions", 0))),
                f"{nan.get('eligibility_rate', 0.0):.3f}", f"{nan.get('capture_given_eligible', 0.0):.3f}",
            ]
            rows.append(row)
    finally:
        # Restore baseline keys (best-effort)
        restore = {k: base_cfg.get(k) for k in [
            "plankton_capture_radius_m", "microalgae_min_eff", "injection_sigma_px", "diag_enabled"
        ] if k in base_cfg}
        if restore:
            write_cfg_overrides(cfg_path, restore)

    # Write summary CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"[sweep] wrote {args.out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
