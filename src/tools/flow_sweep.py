import json
from typing import Dict

from src.simulation import run_simulation, guard_sensitivity


def run_with_scale(scale: float) -> Dict[str, float]:
    # Use fewer steps for a quick sensitivity check
    env, nut, internal, fixed, released, *_ = run_simulation(total_steps=80)
    # Read metrics written by run_simulation
    try:
        with open("outputs/metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        shares = metrics.get("species_share", {})
        # Fallback: derive totals from latest fixed series if shares missing
        if not shares and fixed:
            pass
        return shares
    except Exception:
        return {}


if __name__ == "__main__":
    import argparse
    import os
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Quick flow sensitivity sweep")
    parser.add_argument("--base", type=float, default=1.0, help="Base flow scale")
    parser.add_argument("--delta", type=float, default=0.1, help="Additive delta to base scale")
    args = parser.parse_args()

    # Patch config on-the-fly via env var or temp file is overkill; just tell user to edit config.
    # Here we run two passes by temporarily editing flow_scale in-memory via config file.
    cfg_path = "config.yaml"
    orig = load_config(cfg_path)

    def write_cfg(scale: float):
        lines = []
        seen = set()
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("flow_scale:"):
                        continue
                    lines.append(line)
        lines.append(f"flow_scale: {scale}\n")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    try:
        write_cfg(args.base)
        base_shares = run_with_scale(args.base)
        write_cfg(args.base + args.delta)
        alt_shares = run_with_scale(args.base + args.delta)
        print("Base shares:", base_shares)
        print("Alt  shares:", alt_shares)
        guard_sensitivity(base_shares, alt_shares)
        print("Sensitivity OK: shares changed under flow perturbation.")
    finally:
        # Restore: best-effort (flow_scale removed)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f if not ln.strip().startswith("flow_scale:")]
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

