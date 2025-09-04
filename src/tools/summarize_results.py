import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_species_diag(diag_dir: Path) -> Dict[str, List[Tuple[int, int, int, int]]]:
    data: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for p in diag_dir.glob("species_diag_*.csv"):
        name = p.stem[len("species_diag_"):]
        rows: List[Tuple[int, int, int, int]] = []
        try:
            with p.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        step = int(row.get("step", 0))
                        v = int(row.get("visits", 0))
                        e = int(row.get("eligible", 0))
                        a = int(row.get("absorptions", 0))
                        rows.append((step, v, e, a))
                    except Exception:
                        # skip malformed rows
                        continue
        except Exception:
            continue
        if rows:
            rows.sort(key=lambda t: t[0])
            data[name] = rows
    return data


def cumulative(values: List[int]) -> List[int]:
    s = 0
    out: List[int] = []
    for x in values:
        s += int(x)
        out.append(s)
    return out


def find_plateau_step(series: List[int], frac: float = 0.95) -> int:
    if not series:
        return -1
    total = series[-1]
    if total <= 0:
        return -1
    thresh = total * frac
    for i, v in enumerate(series):
        if v >= thresh:
            return i
    return len(series) - 1


def read_slug_name_map(results_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    idx = results_dir / "index.csv"
    if not idx.exists():
        return mapping
    try:
        with idx.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                slug = str(row.get("slug") or "").strip()
                species = str(row.get("species") or "").strip()
                if slug:
                    mapping[slug] = species or slug
    except Exception:
        pass
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Summarize BlueCarbonSim diagnostics into one-look CSVs")
    ap.add_argument("--diagdir", default="outputs/diagnostics", help="Directory containing diagnostics CSVs")
    ap.add_argument("--outdir", default="results/_summary", help="Output directory for summary CSVs")
    args = ap.parse_args()

    diag_dir = Path(args.diagdir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    species_rows = read_species_diag(diag_dir)
    if not species_rows:
        # Create placeholder with guidance
        with (out_dir / "anomalies.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["level", "species", "code", "details", "suggestion"])
            w.writerow([
                "FAIL",
                "*",
                "no_diagnostics",
                f"no species_diag_*.csv found in '{diag_dir}'",
                "Enable diagnostics_enabled=true in config.yaml and re-run simulation",
            ])
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["species", "visits_sum", "eligible_sum", "absorptions_sum", "P(abs|eligible)", "plateau_step"])
        return

    # Summary per species
    # slug->display name mapping (optional)
    name_map = read_slug_name_map(Path("results"))

    summary: List[Tuple[str, int, int, int, float, int]] = []
    for slug, rows in species_rows.items():
        visits = [v for (_, v, _, _) in rows]
        elig = [e for (_, _, e, _) in rows]
        abso = [a for (_, _, _, a) in rows]
        v_sum = int(sum(visits))
        e_sum = int(sum(elig))
        a_sum = int(sum(abso))
        p_abs_elig = (a_sum / e_sum) if e_sum > 0 else 0.0
        cum_abs = cumulative(abso)
        plateau = find_plateau_step(cum_abs, 0.95)
        disp = name_map.get(slug, slug)
        summary.append((disp, v_sum, e_sum, a_sum, p_abs_elig, plateau))

    # Write summary.csv
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "visits_sum", "eligible_sum", "absorptions_sum", "P(abs|eligible)", "plateau_step"])
        for row in summary:
            w.writerow(row)

    # Anomaly detection
    species_names = [s for (s, *_rest) in summary]
    v_values = [v for (_s, v, *_rest) in summary]
    anomalies: List[List[str]] = []

    # 1) Gate too strong/weak
    for (s, v_sum, e_sum, a_sum, p_abs_elig, plateau) in summary:
        if e_sum == 0 or a_sum == 0:
            anomalies.append([
                "FAIL", s, "zero_capture",
                f"eligible={e_sum}, absorptions={a_sum}",
                "Check radius/layer/eff; for microalgae raise plankton_capture_radius_m to 2.5–3.0 and set microalgae_min_eff to 0.10–0.20",
            ])
        elif p_abs_elig > 0.95:
            anomalies.append([
                "WARN", s, "two_stage_gate_weak",
                f"P(abs|eligible)={p_abs_elig:.3f}",
                "Increase eligible_radius 1.5x; set min_contact_steps to 2–3; apply probabilistic efficiency",
            ])

    # 2) Microalgae specific zeros
    micro_slugs = {"Chlorella vulgaris", "Nannochloropsis gaditana", "Chlorella_vulgaris", "Nannochloropsis_gaditana"}
    for (s, v_sum, e_sum, a_sum, p_abs_elig, plateau) in summary:
        if s in micro_slugs and (e_sum == 0 or a_sum == 0):
            anomalies.append([
                "FAIL", s, "microalgae_zero",
                f"eligible={e_sum}, absorptions={a_sum}",
                "Increase plankton_capture_radius_m (2.5–3.0); set microalgae_min_eff 0.10–0.20; align layer/light",
            ])

    # 3) All visits identical
    if len(set(v_values)) == 1 and len(v_values) > 1:
        anomalies.append([
            "WARN", "*", "visits_uniform",
            "all species have identical visits_sum",
            "Ensure species-specific arrays are used; assert v>=e>=a when writing",
        ])

    # 4) Per-step ordering check (optional)
    for slug, rows in species_rows.items():
        broken = False
        for (_step, v, e, a) in rows:
            if not (v >= e >= a >= 0):
                broken = True
                break
        if broken:
            anomalies.append([
                "FAIL", slug, "ordering_broken",
                "violates visits>=eligible>=absorptions per step",
                "Fix writer to preserve ordering; add asserts",
            ])

    with (out_dir / "anomalies.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "species", "code", "details", "suggestion"])
        for row in anomalies:
            w.writerow(row)


if __name__ == "__main__":
    main()
