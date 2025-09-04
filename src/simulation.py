# -*- coding: utf-8 -*-
# src/simulation.py
import csv
import json
import os
import random
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Defaults for injectable knobs (overridden by config inside run_simulation)
injection_sigma_px = 1.0
injection_drift_px_per_step = 0.0
microalgae_min_eff = 0.0

from .environment import compute_efficiency_score, get_environmental_factors
from .models.particle import initialize_particles
from .models.plant import Plant
from .particle_flow import (
    diffuse_particles,
    generate_dynamic_flow_field,
    inject_particles,
)
from .terrain import create_terrain
from .utils.config import load_config
from .utils.profile_adapter import normalize_profiles
from math import hypot

# ===== Guards & Metrics Helpers =====
def _segment_circle_intersect(x0, y0, x1, y1, cx, cy, r):
    """Return True if segment (x0,y0)-(x1,y1) intersects circle centered at (cx,cy) with radius r."""
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    fx = float(x0) - float(cx)
    fy = float(y0) - float(cy)
    a = dx * dx + dy * dy
    if a <= 1e-12:
        return (fx * fx + fy * fy) <= float(r) * float(r)
    t = -(fx * dx + fy * dy) / a
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = float(x0) + t * dx
    qy = float(y0) + t * dy
    qdx = qx - float(cx)
    qdy = qy - float(cy)
    return (qdx * qdx + qdy * qdy) <= float(r) * float(r)
def validate_geometry(
    sources: List[Tuple[int, int]],
    plants: List[Plant],
    width: int,
    height: int,
    *,
    margin_px: int = 3,
    auto_shift: bool = True,
    y_shift_candidates: Tuple[int, ...] = (-4, -2, -1, 1, 2, 4),
) -> List[Tuple[int, int]]:
    """Ensure injection sources are not within any plant radius+margin.
    If too close, try auto y-shifts; otherwise raise ValueError.
    Returns possibly adjusted list of sources (tuples).
    """
    adjusted = list(sources)
    for i, (sx, sy) in enumerate(list(adjusted)):
        def ok(x, y, p):
            return hypot(x - p.x, y - p.y) > (p.radius + margin_px)
        bad = any(not ok(sx, sy, p) for p in plants)
        if not bad:
            continue
        if auto_shift:
            shifted_ok = False
            # First try a few small, human-chosen nudges
            for dy in y_shift_candidates:
                ny = int(max(0, min(sy + dy, height - 1)))
                if all(ok(sx, ny, p) for p in plants):
                    adjusted[i] = (sx, ny)
                    shifted_ok = True
                    break
            # Fallback: expand search radius along y until safe or bounds reached
            if not shifted_ok:
                max_span = max(height, 1)
                for dd in range(margin_px, max_span):
                    for dy in (-dd, dd):
                        ny = int(max(0, min(sy + dy, height - 1)))
                        if all(ok(sx, ny, p) for p in plants):
                            adjusted[i] = (sx, ny)
                            shifted_ok = True
                            break
                    if shifted_ok:
                        break
            # As a last resort, try x-direction shifts too
            if not shifted_ok:
                max_span_x = max(width, 1)
                for dd in range(margin_px, max_span_x):
                    for dx in (-dd, dd):
                        nx = int(max(0, min(sx + dx, width - 1)))
                        if all(ok(nx, sy, p) for p in plants):
                            adjusted[i] = (nx, sy)
                            shifted_ok = True
                            break
                    if shifted_ok:
                        break
            if not shifted_ok:
                raise ValueError(f"Injection source {i} too close to vegetation; auto-shift failed")
        else:
            raise ValueError(f"Injection source {i} too close to vegetation")
    return adjusted


def assert_gate_effective(abs_dy_samples: List[float], min_median: float = 0.8):
    if len(abs_dy_samples) == 0:
        return
    if np.median(np.asarray(abs_dy_samples)) < float(min_median):
        raise AssertionError("Vertical gate ineffective: median |dy| too small")


_dom_state = {}

def guard_dominance(species_share: Dict[str, float], max_share: float = 0.65, streak: int = 3):
    """Fail if any species exceeds max_share for 'streak' consecutive checks (per-name state)."""
    if not species_share:
        return
    top, val = max(species_share.items(), key=lambda kv: kv[1])
    k = f"dom_{top}"
    _dom_state[k] = 1 + _dom_state.get(k, 0) if val > max_share else 0
    if _dom_state[k] >= streak:
        raise AssertionError(f"Dominance: {top} {val:.2%} > {max_share:.0%} for {streak} runs")


def save_pref1_snapshot(grid_shape, plants, sources, particles0, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h, w = grid_shape
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()
        ax.set_facecolor("#eef7ff")
        for p in plants:
            circ = plt.Circle((p.x, p.y), p.radius, edgecolor="green", facecolor="none", lw=1.5)
            ax.add_patch(circ)
            ax.plot([p.x], [p.y], marker="o", color="green", ms=3)
        for sx, sy in sources:
            ax.plot([sx], [sy], marker="x", color="red", ms=6)
        if particles0 is not None and len(particles0) > 0:
            ax.scatter([p.x for p in particles0], [p.y for p in particles0], c="#00bcd4", s=5, alpha=0.7)
        ax.set_title("Pref1 snapshot")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "debug_pref1.png"))
        plt.close(fig)

        if particles0 is not None and len(particles0) > 0 and len(plants) > 0:
            dists = []
            dys = []
            for p in particles0:
                dmin = min(hypot(p.x - q.x, p.y - q.y) for q in plants)
                dists.append(dmin)
                dys.append(min(abs(p.y - q.y) for q in plants))
            # separate hist files for CI
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.hist(dys, bins=20, color="#795548"); ax1.set_title("|dy| at injection")
            plt.tight_layout(); fig1.savefig(os.path.join(outdir, "dy_hist.png")); plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.hist(dists, bins=20, color="#607d8b"); ax2.set_title("dist to nearest plant")
            plt.tight_layout(); fig2.savefig(os.path.join(outdir, "dist_hist.png")); plt.close(fig2)
    except Exception as e:
        print(f"[warn] Pref1 snapshot failed: {e}")


def guard_sensitivity(baseline: Dict[str, float], alt: Dict[str, float], eps: float = 1e-6):
    """Fail if totals are indistinguishable under a perturbation (gate not effective)."""
    keys = set(baseline.keys()) & set(alt.keys())
    if all(abs(float(baseline[k]) - float(alt[k])) < eps for k in keys):
        raise AssertionError("Sensitivity guard: no change detected after gate width perturbation")

def apply_depth_filter(eff: float, plant, env) -> float:
    """
    雎ｺ・ｱ陟趣ｽｦ邵ｺ・ｮ鬩包ｽｩ陜ｨ・ｰ郢晁ｼ斐≦郢晢ｽｫ郢ｧ・ｿ邵ｲ繝ｻ
    雋趣ｽｮ鬮｢轣假ｽｸ・ｯ繝ｻ繝ｻpartina/Rhizophora繝ｻ蟲ｨ繝ｻ雎鯉ｽｴ鬮ｱ・｢騾ｶ・ｴ闕ｳ螂・ｽｽ讒ｫ・ｹ・ｲ陷・ｽｺ陝ｶ・ｯ邵ｺ・ｮ邵ｺ貅假ｽ∬ｱｺ・ｱ陟趣ｽｦ郢晁ｼ斐≦郢晢ｽｫ郢ｧ・ｿ郢ｧ蟶昶・騾包ｽｨ邵ｺ蜉ｱ竊醍ｸｺ繝ｻﾂ繝ｻ
    """
    if getattr(plant, "name", "") in ("Spartina alterniflora", "Rhizophora spp."):
        return eff  # 雎ｺ・ｱ陟趣ｽｦ隴夲ｽ｡闔会ｽｶ郢ｧ雋橸ｽ､謔ｶ笘・    dmin, dmax = getattr(plant, "model_depth_range", (0, 999))
    depth_m = float(env.get("depth_m", 0.0))
    return 0.0 if not (dmin <= depth_m <= dmax) else eff

def seasonal_inflow(step, total_steps, base_mgC_per_step=30.0, particle_mass_mgC=1.0):
    """Return seasonal inflow as particle count (mgC/step -> count)."""
    cycle = 2 * np.pi * step / total_steps
    mgc = base_mgC_per_step * (0.5 + 0.5 * np.sin(cycle))
    count = int(round(mgc / max(particle_mass_mgC, 1e-9)))
    return max(count, 0)
with open("data/plants.json") as f:
    profiles_raw = json.load(f)
profiles = normalize_profiles(profiles_raw)


def _slugify(name: str) -> str:
    """Safe slug for filenames."""
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s


def run_simulation(
    total_steps: int = 150,
    num_particles: int = 1000,
    width: int = 100,
    height: int = 100,
    *,
    guard_config_path: str = "config/guard.yaml",
    schema_path: str = "config/schema.json",
    pref1: bool = False,
):
    """Run the simulation and return series and outputs."""

    # 陜ｨ・ｰ陟厄ｽ｢邵ｺ・ｨ雎ｺ・ｱ陟趣ｽｦ郢晄ｧｭ繝｣郢昴・
    terrain, depth_map = create_terrain(width, height)

    # 陷・ｽｺ陷牙ｸｷ逡醍ｹｧ・ｷ郢晢ｽｪ郢晢ｽｼ郢ｧ・ｺ
    env_series: List[float] = []
    nutrient_series: List[float] = []

    # 驍擾ｽｯ驕ｨ謳ｾ・ｼ莠･繝ｻ闖ｴ髮｣・ｼ繝ｻ
    internal_series: List[float] = []  # 陷茨ｽｨ驕橸ｽｮ邵ｺ・ｮ total_growth 陷ｷ驛・ｽｨ闌ｨ・ｼ閧ｲ・ｴ・ｯ驕ｨ謳ｾ・ｼ繝ｻ
    fixed_series: List[float] = []     # 陷茨ｽｨ驕橸ｽｮ邵ｺ・ｮ total_fixed 陷ｷ驛・ｽｨ闌ｨ・ｼ閧ｲ・ｴ・ｯ驕ｨ謳ｾ・ｼ繝ｻ
    released_series: List[float] = []  # 闔閧ｲ・ｴ繝ｻ・ｼ蝓滓ざ闖ｴ・ｿ騾包ｽｨ繝ｻ繝ｻ
    carbon_series: List[float] = []    # 陷ｷ蠕｡・ｸ螂・ｽｼ莠包ｽｺ閧ｲ・ｴ繝ｻ・ｼ繝ｻ

    # 郢ｧ・ｹ郢昴・繝｣郢晉軸・ｯ雜｣・ｼ莠･繝ｻ闖ｴ髮｣・ｼ繝ｻ
    step_absorbed_series: List[float] = []
    step_fixed_series: List[float] = []
    step_growth_series: List[float] = []
    particle_count_series: List[int] = []

    # 闔会ｽ｣髯ｦ・ｨ3驕橸ｽｮ繝ｻ莠･・ｾ譴ｧ蟀ｿ闔蜻磯共騾包ｽｨ繝ｻ繝ｻ
    zostera_fixed_series, kelp_fixed_series, chlorella_fixed_series = [], [], []
    zostera_growth_series, kelp_growth_series, chlorella_growth_series = [], [], []
    zostera_absorbed_series, kelp_absorbed_series, chlorella_absorbed_series = [], [], []
    os.makedirs("results", exist_ok=True)

    # 髫ｪ・ｭ陞ｳ螟奇ｽｪ・ｭ邵ｺ・ｿ髴趣ｽｼ邵ｺ・ｿ繝ｻ莠･閻ｰ闖ｴ蜥ｲ・ｭ莨夲ｽｼ繝ｻ
    cfg = load_config()
    particle_mass_mgC = float(cfg.get("particle_mass_mgC", 1.0))
    inflow_mgC_per_step_base = float(cfg.get("inflow_mgC_per_step_base", 30.0))
    chl_mortality = float(cfg.get("chl_mortality_rate", 0.02))  # 2%/step
    live_plot_interval = int(cfg.get("live_plot_interval", 0))   # 0邵ｺ・ｪ郢ｧ蟲ｨﾎ帷ｹｧ・､郢晏戟邱帝包ｽｻ邵ｺ・ｪ邵ｺ繝ｻ
    show_plots = bool(cfg.get("show_plots", False))              # True邵ｺ・ｧplt.show
    debug_mass_check = bool(cfg.get("debug_mass_check", False))  # 雎亥ｼｱ縺帷ｹ昴・繝｣郢晏干繝ｻ髮会ｽｪ鬩･荳茨ｽｿ譎擾ｽｭ蛟･繝｡郢ｧ・ｧ郢昴・縺・    flow_scale = float(cfg.get("flow_scale", 1.0))               # 雎ｬ繝ｻﾂ貅倥○郢ｧ・ｱ郢晢ｽｼ郢晢ｽｫ繝ｻ蝓溪楳陟趣ｽｦ驕抵ｽｺ髫ｱ蜥ｲ逡代・繝ｻ
    # 郢ｧ・ｵ郢昜ｹ昴Θ郢ｧ・｣騾包ｽｨ邵ｺ・ｮ陜ｮ繝ｻ・ｸﾂ郢晢ｽ｢郢晢ｽｼ郢晏ｳｨ繝ｻ隴幄・笏・    sanity_uniform = bool(cfg.get("sanity_uniform", False))
    # 雎包ｽｨ陷茨ｽ･陋ｻ繝ｻ・ｸ繝ｻ・ｼ閧ｲ・ｩ・ｺ鬮｢阮吶Σ郢ｧ・､郢ｧ・｢郢ｧ・ｹ驍ｱ・ｩ陷･讙守舞繝ｻ繝ｻ    injection_sigma_px = float(cfg.get("injection_sigma_px", 1.0))
    # 豕ｨ蜈･蛻・ｸ・ｼ育ｩｺ髢薙ヰ繧､繧｢繧ｹ邱ｩ蜥檎畑・・    injection_drift_px_per_step = float(cfg.get("injection_drift_px_per_step", 0.0))
    microalgae_min_eff = float(cfg.get("microalgae_min_eff", 0.0))
    tidal_period_steps = int(cfg.get("tidal_period_steps", 24))
    intertidal_submergence_fraction = float(cfg.get("intertidal_submergence_fraction", 0.35))
    intertidal_shallow_band_m = float(cfg.get("intertidal_shallow_band_m", 1.0))
    # Kelp band widths (configurable)
    kelp_bottom_band_m = float(cfg.get("kelp_bottom_band_m", 5.0))
    kelp_surface_band_m = float(cfg.get("kelp_surface_band_m", 3.0))

    # Injection sweep (vertical sinusoidal sweep of injection y)
    injection_sweep_amp_px = float(cfg.get("injection_sweep_amp_px", 0.0))
    injection_sweep_period_steps = int(cfg.get("injection_sweep_period_steps", 0))
    # Encounter/uptake knobs (configurable)
    use_swept_contact = bool(cfg.get("use_swept_contact", False))
    seagrass_min_contact_steps = int(cfg.get("seagrass_min_contact_steps", 2))
    microalgae_uptake_scale = float(cfg.get("microalgae_uptake_scale", 1.0))

    # Eelgrass-favoring knobs (optional)
    eelgrass_shallow_bonus = float(cfg.get("eelgrass_shallow_bonus", 1.2))
    eelgrass_shallow_cutoff_m = float(cfg.get("eelgrass_shallow_cutoff_m", 3.0))
    eelgrass_low_nutrient_bonus = float(cfg.get("eelgrass_low_nutrient_bonus", 0.3))
    eelgrass_mid_boost_factor = float(cfg.get("eelgrass_mid_boost_factor", 2.0))
    eelgrass_mid_boost_center_frac = float(cfg.get("eelgrass_mid_boost_center_frac", 0.5))
    eelgrass_mid_boost_k = float(cfg.get("eelgrass_mid_boost_k", 10.0))

    # Microalgae seasonality knobs
    chl_mortality_amp = float(cfg.get("chl_mortality_amp", 0.05))
    chl_uptake_season_amp = float(cfg.get("chl_uptake_season_amp", 0.3))

    # 鬩溷調・ｽ・ｮ繝ｻ繝ｻ驕橸ｽｮ邵ｺ・ｮ隶蜥ｲ鮟・ｸｺ・ｮ郢晢ｽｬ郢ｧ・､郢ｧ・｢郢ｧ・ｦ郢晁肩・ｼ繝ｻ
    plant_positions = {
        # 闖ｴ荳ｻ・｡・ｩ遶雁ｸ晢ｽｫ莨懶ｽ｡・ｩ邵ｺ・ｮ闕ｳ・ｭ鬮｢隰趣ｽｻ・･闕ｳ鄙ｫ竏磯§・ｻ陷榊桁・ｼ繝ｻ 邵ｺ・ｯ邵ｺ譏ｴ繝ｻ邵ｺ・ｾ邵ｺ・ｾ/陷企宦・ｾ繝ｻ・ｰ莉｣・陟・干・・ｸｺ蜻ｻ・ｼ繝ｻ
        "Zostera marina":            {"x": 30, "y": 28, "radius": 11},  # 陷企宦・ｾ繝ｻ・・邵ｺ・ｰ邵ｺ驢搾ｽｸ・ｮ陝・・
        "Halophila ovalis":          {"x": 45, "y": 26, "radius": 10},  # 陷企宦・ｾ繝ｻ2邵ｺ・ｧ鬩包ｽｭ鬩輔・邏ｫ遶頑汚・ｼ驕ｺ鄙・5 PSU繝ｻ繝ｻ
        "Posidonia oceanica":        {"x": 95, "y": 30, "radius": 12},  # 陷企宦・ｾ繝ｻ蜒題棔・ｧ邵ｺ・ｧ鬩包ｽｭ鬩輔・邏ｫ遶翫・
        "Macrocystis pyrifera":      {"x": 85, "y": 80, "radius": 9},   # 陷企宦・ｾ繝ｻ蜒題棔・ｧ邵ｺ・ｧ鬩包ｽｭ鬩輔・邏ｫ遶翫・
        "Saccharina japonica":       {"x": 75, "y": 82, "radius": 9},   # 陷企宦・ｾ繝ｻ蜒題棔・ｧ邵ｺ・ｧ鬩包ｽｭ鬩輔・邏ｫ遶翫・
        "Chlorella vulgaris":        {"x": 10, "y": 14, "radius": 3},   # 0遯ｶ繝ｻ0 PSU
        "Nannochloropsis gaditana":  {"x": 70, "y": 16, "radius": 4},   # 遶包ｽ･20 PSU繝ｻ繝ｻK繝ｻ繝ｻ
        "Spartina alterniflora":     {"x": 30, "y": 6,  "radius": 8},   # 遶包ｽ･10 PSU繝ｻ莠･・ｷ・ｦ邵ｺ荵晢ｽ芽愾・ｳ邵ｺ・ｸ繝ｻ繝ｻ
        "Rhizophora spp.":           {"x": 20, "y": 7,  "radius": 8},   # 遶包ｽ･5 PSU繝ｻ莠･・ｰ莉｣・陷ｿ・ｳ邵ｺ・ｸ繝ｻ繝ｻ
    }

    # 驍願ｲ橸ｽｭ闊後・雎包ｽｨ陷茨ｽ･闖ｴ蜥ｲ・ｽ・ｮ繝ｻ莠･・｢繝ｻ髦懶ｾ・懶ｽｱ・､邵ｺ・ｫ陜暦ｽｺ陞ｳ螢ｹﾂ繧会ｽｨ・ｮ騾ｶ・ｴ闕ｳ鄙ｫ繝ｻ闖ｴ・ｿ郢ｧ荳岩・邵ｺ繝ｻ・ｼ繝ｻ
    injection_sources = [
        (2, int(height * 0.20)),   # 陝ｾ・ｦ郢晢ｽｻ髯ｦ・ｨ陞ｻ・､繝ｻ繝ｻhlorella/Halophila 陝・ｽｾ陟｢諛ｶ・ｼ繝ｻ
        (2, int(height * 0.50)),   # 陝ｾ・ｦ郢晢ｽｻ闕ｳ・ｭ陞ｻ・､
        (width - 2, int(height * 0.15)),  # 陷ｿ・ｳ郢晢ｽｻ髯ｦ・ｨ陞ｻ・､
        (width - 2, int(height * 0.30)),  # 陷ｿ・ｳ郢晢ｽｻ闕ｳ・ｭ雎ｬ繝ｻ
        (width - 2, int(height * 0.80)),  # 陷ｿ・ｳ郢晢ｽｻ雎ｺ・ｱ郢ｧ繝ｻ
    ]

    # ===== Guard config & schema (lightweight validation) =====
    guard_cfg = load_config(guard_config_path) if os.path.exists(guard_config_path) else {}
    min_dist_px = int(guard_cfg.get("min_dist_px", 3))
    gate_enabled = bool(guard_cfg.get("gate_enabled", False))
    # diagnostics flag can come from guard.yaml or fallback to main config
    diag_enabled = bool(guard_cfg.get(
        "diag_enabled",
        guard_cfg.get(
            "diagnostics_enabled",
            bool(cfg.get("diag_enabled", cfg.get("diagnostics_enabled", False)))
        ),
    ))
    min_median_abs_dy = float(guard_cfg.get("vertical_gate_min_median_abs_dy_px", 0.8))
    dom_max_share = float(guard_cfg.get("dominance_max_share", 0.65))
    dom_streak = int(guard_cfg.get("dominance_streak", 3))

    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            if "min_dist_px" in schema:
                if min_dist_px < int(schema["min_dist_px"].get("min", 0)):
                    raise ValueError("min_dist_px below schema minimum")
            if "vertical_gate_min_median_abs_dy_px" in schema:
                if min_median_abs_dy < float(schema["vertical_gate_min_median_abs_dy_px"].get("min", 0.0)):
                    raise ValueError("vertical_gate_min_median_abs_dy_px below schema minimum")
        except Exception as e:
            raise ValueError(f"Guard schema validation failed: {e}")

    # 陝・ｽｾ髮趣ｽ｡驕橸ｽｮ郢ｧ繝ｻ驕橸ｽｮ邵ｺ・ｫ隲｡・｡陟托ｽｵ
    target_species = [
        "Zostera marina",
        "Halophila ovalis",
        "Posidonia oceanica",
        "Macrocystis pyrifera",
        "Saccharina japonica",
        "Chlorella vulgaris",
        "Nannochloropsis gaditana",
        "Spartina alterniflora",
        "Rhizophora spp."
    ]
    plants = []
    for plant_type in target_species:
        profile = profiles.get(plant_type, {})
        pos = plant_positions[plant_type]
        p = Plant(
            name=plant_type,
            fixation_ratio=profile.get("fixation_ratio", 0.7),
            release_ratio=profile.get("release_ratio", 0.05),
            structure_density=profile.get("structure_density", 1.0),
            opt_temp=profile.get("opt_temp", 20),
            light_tolerance=profile.get("light_tolerance", 1.0),
            light_half_saturation=profile.get("light_half_saturation", 0.5),
            salinity_range=tuple(profile.get("salinity_range", (20, 35))),
            absorption_efficiency=profile.get("absorption_efficiency", 1.0),
            growth_rate=profile.get("growth_rate", 1.0),
            x=pos["x"],
            y=pos["y"],
            radius=pos["radius"],
        )
        # Optional per-config adjustment: boost eelgrass footprint radius
        if p.name == "Zostera marina":
            try:
                zrad = cfg.get("zostera_radius", None)
                if zrad is not None:
                    p.radius = int(zrad)
            except Exception:
                pass
        # Nerf Chlorella via config override for fixation ratio
        if p.name == "Chlorella vulgaris":
            try:
                fr = cfg.get("chlorella_fixation_ratio", None)
                if fr is not None:
                    p.fixation_ratio = float(fr)
            except Exception:
                pass
        # depth range 郢ｧ雋橸ｽｱ讓環・ｧ邵ｺ・ｨ邵ｺ蜉ｱ窶ｻ闔牙・ｽｸ繝ｻ
        setattr(p, "model_depth_range", tuple(profile.get("model_depth_range", (1, 6))))
        plants.append(p)

    # 陋ｻ譎・ｄ驍願ｲ橸ｽｭ繝ｻ
    particles = initialize_particles(num_particles, terrain)

    # 陝ｯ・ｩ郢ｧ・ｪ郢晄じ縺夂ｹｧ・ｧ郢ｧ・ｯ郢昴・
    rocks = [
        {"x": width // 2, "y": int(height * 0.5), "w": 12, "h": 8},
        {"x": int(width * 0.7), "y": int(height * 0.3), "w": 8, "h": 5},
    ]

    # 陟取・・ｳ・ｪ郢晄ｧｭ繝｣郢昴・
    bottom_type_map = np.full((height, width), "mud", dtype=object)
    bottom_type_map[90:100, 0:33] = "mud"
    bottom_type_map[90:100, 33:66] = "sand"
    bottom_type_map[90:100, 66:100] = "rock"

    seed_val = int(cfg.get("seed", 42))
    np.random.seed(seed_val)
    random.seed(seed_val)

    # Geometry guard: adjust or abort if any source collides with plant footprints
    injection_sources = validate_geometry(injection_sources, plants, width, height, margin_px=min_dist_px)

    # 郢晁ｼ斐♂郢ｧ・｢郢晞亂縺幄怙蝓ｼ纃ｾ邵ｺ・ｿ闔牙･・邵ｺ・ｯ髯ｦ蠕鯉ｽ冗ｸｺ・ｪ邵ｺ繝ｻ・ｼ莠包ｽｾ蟶ｷ・ｵ・ｦ邵ｺ・ｯ霑ｺ・ｰ陟・・竊楢嵩譎擾ｽｭ蛟･ﾂ竏ｫ・ｨ・ｮ邵ｺ・ｫ邵ｺ・ｯ關捺剌・ｭ蛟･・邵ｺ・ｪ邵ｺ繝ｻ・ｼ繝ｻ

    # Pref1: one-step visualization and exit (no main loop)
    if pref1:
        try:
            from .models.particle import Particle
        except Exception:
            Particle = None
        p0 = []
        if Particle is not None:
            for sx, sy in injection_sources:
                for _ in range(50):
                    x = sx + np.random.normal(scale=injection_sigma_px)
                    y = sy + np.random.normal(scale=injection_sigma_px)
                    if 0 <= x < width and 0 <= y < height and terrain[int(y), int(x)] == 1:
                        p0.append(Particle(x=x, y=y, mass=1.0, origin="pref1", x0=x, y0=y))
        save_pref1_snapshot((height, width), plants, injection_sources, p0, outdir="outputs")
        return (), (), (), (), (), [], [], [], [], [], []

    # ===== 郢晢ｽ｡郢ｧ・､郢晢ｽｳ郢晢ｽｫ郢晢ｽｼ郢昴・=====
    # 霑夲ｽｩ騾・・・ｰ・ｺ陟趣ｽｦ繝ｻ繝ｻnvironment 邵ｺ・ｨ隰ｨ・ｴ陷ｷ闌ｨ・ｼ繝ｻ
    KD = float(cfg.get("kd_m_inv", 0.8))
    MAX_DEPTH_M = 8.0
    meters_per_pixel = MAX_DEPTH_M / max((height - 1), 1)
    # 郢晏干ﾎ帷ｹ晢ｽｳ郢ｧ・ｯ郢晏現ﾎｦ邵ｺ・ｮ雎鯉ｽｴ陝ｷ・ｳ陟弱・窶ｲ郢ｧ螂・ｽｼ繝ｻ繝ｻ菫・・ 郢晄鱒縺醍ｹｧ・ｻ郢晢ｽｫ隰蟶ｷ・ｮ繝ｻ
    plankton_radius_m = float(cfg.get("plankton_radius_m", 0.9))
    plankton_radius_px = plankton_radius_m / max(meters_per_pixel, 1e-9)
    # 蠕ｮ邏ｰ阯ｻ縺ｮ謐墓拷蜊雁ｾ・ｼ亥ｺ・ａ縺ｮ譌｢螳壹ｒ險ｱ螳ｹ・・    plankton_capture_radius_px = float(cfg.get("plankton_capture_radius_m", max(plankton_radius_m, 2.0))) / max(meters_per_pixel, 1e-9)
    # 郢晁ｼ斐°郢晏現縺郢晢ｽｼ郢晢ｽｳ髴大床・ｼ・ｼ繝ｻ繝ｻ^-kd z = 0.1 遶翫・z 遶輔・2.3/kd繝ｻ繝ｻ
    euphotic_depth_m = 2.3 / max(KD, 1e-6)
    euphotic_px = int(euphotic_depth_m / meters_per_pixel)
    intertidal_shallow_band_px = intertidal_shallow_band_m / max(meters_per_pixel, 1e-9)

    # 髮会ｽｪ鬩･荳翫Σ郢晢ｽｩ郢晢ｽｳ郢ｧ・ｹ騾包ｽｨ繝ｻ繝ｻgC陷雁・ｽｽ謳ｾ・ｼ繝ｻ
    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_initial = float(len(particles)) * float(particle_mass_mgC)
    loss_quant_mgC = 0.0  # reinjection rounding loss (mgC)
    # 驕橸ｽｮ陋ｻ・･邵ｺ譁絶・邵ｺ・ｮ驍擾ｽｯ驕ｨ髦ｪ縺咏ｹ晢ｽｪ郢晢ｽｼ郢ｧ・ｺ繝ｻ莠･繝ｻ郢ｧ・ｹ郢昴・繝｣郢晄圜・ｼ繝ｻ
    species_series: Dict[str, Dict[str, List[float]]] = {
        name: {"total_absorbed": [], "total_fixed": [], "total_growth": []}
        for name in target_species
    }

    # Metrics accumulators
    source_labels = [f"src{i}" for i in range(len(injection_sources))]
    capture_matrix: Dict[str, Dict[str, float]] = {lab: {name: 0.0 for name in target_species} for lab in source_labels + ["init"]}
    abs_dy_samples: List[float] = []
    # 髫ｪ・ｺ隴・ｽｭ郢晁・繝｣郢晁ｼ斐＜
    diag_series: Dict[str, Dict[str, List[int]]] = (
        {name: {"visits": [], "eligible": [], "absorptions": []} for name in target_species}
        if diag_enabled else {}
    )
    inj_series: List[List[int]] = []
    inj_xy_rows: List[List[float]] = []
    # Index rows for results/index.csv (species name, slug, summary filename, timeseries filename)
    index_rows: List[List[str]] = []
    travel_weighted_sum = 0.0
    travel_weight = 0.0

    for step in range(total_steps):
        # 髮会ｽｪ鬩･謫ｾ・ｼ繝ｻgC繝ｻ蟲ｨ縺帷ｹ晉ｿｫ繝｣郢晏干縺咏ｹ晢ｽｧ郢昴・繝ｨ繝ｻ蛹ｻ縺帷ｹ昴・繝｣郢晄懃√繝ｻ繝ｻ
        prev_particles_mass_mgC = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
        prev_loss_quant_mgC = float(loss_quant_mgC)

        # 郢ｧ・ｹ郢昴・繝｣郢晏干・・ｸｺ・ｨ邵ｺ・ｫ霑ｺ・ｰ陟・・・ｩ遨ゑｽｾ・｡郢ｧ蛛ｵ縺冗ｹ晢ｽ｣郢昴・縺咏ｹ晢ｽ･
        plant_env, plant_eff = {}, {}
        for i, plant in enumerate(plants):
            if sanity_uniform:
                env = get_environmental_factors(
                    plant.x, plant.y, step,
                    total_steps=total_steps, width=width, height=height,
                    salinity_mode="constant",
                    S_mean=28.0, S_amp=0.0,
                    kd_m_inv=0.0, max_depth_m=MAX_DEPTH_M,
                    T_mean=20.0, T_amp=0.0, I0_daylen_frac=1.0,
                    nutrient_mean=0.5, nutrient_amp=0.0, nutrient_pulse_period=0,
                )
            else:
                env = get_environmental_factors(
                    plant.x, plant.y, step,
                    total_steps=total_steps, width=width, height=height,
                    salinity_mode="linear_x",  # 雎趣ｽｽ雎鯉ｽｴ陜薙・ 陝ｾ・ｦ闖ｴ荳ｻ・｡・ｩ遶願ｲ樊価鬯ｮ莨懶ｽ｡・ｩ
                    S_min=0.0, S_max=35.0,      # 陝ｾ・ｦ驕ｶ・ｯ郢ｧ蜑・ｽｽ荳ｻ・｡・ｩ繝ｻ蝓滂ｽｷ・｡雎鯉ｽｴ陝・・・翫・蟲ｨ竊馴坎・ｭ陞ｳ繝ｻ
                    kd_m_inv=KD, max_depth_m=MAX_DEPTH_M,
                )
            px, py = int(plant.x), int(plant.y)
            bottom_type = bottom_type_map[py, px] if (0 <= py < height and 0 <= px < width) else "mud"

            eff = compute_efficiency_score(plant, env, bottom_type=bottom_type)
            eff = apply_depth_filter(eff, plant, env)  # 雎ｺ・ｱ陟趣ｽｦ郢晢ｽｬ郢晢ｽｳ郢ｧ・ｸ陞滓じ繝ｻ0
            plant_env[plant.name] = env
            plant_eff[plant.name] = eff
            # Favor eelgrass (Zostera marina) in shallow, low-nutrient conditions
            if plant.name == "Zostera marina":
                try:
                    d = float(env.get("depth_m", 0.0))
                    if d <= float(eelgrass_shallow_cutoff_m):
                        plant_eff[plant.name] = float(min(max(plant_eff[plant.name] * float(eelgrass_shallow_bonus), 0.0), 1.0))
                except Exception:
                    pass
                try:
                    nutr = float(env.get("nutrient", 1.0))
                    bonus = float(1.0 + eelgrass_low_nutrient_bonus * (1.0 - max(min(nutr, 1.0), 0.0)))
                    plant_eff[plant.name] = float(min(max(plant_eff[plant.name] * bonus, 0.0), 1.0))
                except Exception:
                    pass

            if i == 0:
                env_series.append(eff)
                nutrient_series.append(env["nutrient"])

        # 隶蜥ｲ鮟・ｸｺ譁絶・邵ｺ・ｮ驍擾ｽｯ驕ｨ蝓ｼ纃ｼ
        # 驍擾ｽｯ驕ｨ髦ｪ縺咏ｹ晢ｽｪ郢晢ｽｼ郢ｧ・ｺ繝ｻ莠包ｽｻ・｣髯ｦ・ｨ3驕橸ｽｮ邵ｺ・ｯ陟墓｢ｧ蟀ｿ闔蜻磯共陷ｷ蜥ｲ・ｧ・ｰ邵ｺ・ｫ陷ｷ蛹ｻ・冗ｸｺ蟶吮ｻ髫ｪ蛟ｬ鮖ｸ繝ｻ繝ｻ
        # Zostera marina: index 0, Macrocystis pyrifera: index 3, Chlorella vulgaris: index 5
        zostera_fixed_series.append(plants[0].total_fixed)
        kelp_fixed_series.append(plants[3].total_fixed)
        chlorella_fixed_series.append(plants[5].total_fixed)
        zostera_growth_series.append(plants[0].total_growth)
        kelp_growth_series.append(plants[3].total_growth)
        chlorella_growth_series.append(plants[5].total_growth)
        zostera_absorbed_series.append(plants[0].total_absorbed)
        kelp_absorbed_series.append(plants[3].total_absorbed)
        chlorella_absorbed_series.append(plants[5].total_absorbed)

        # 驍願ｲ橸ｽｭ蜈亥ヱ隰ｨ・｣繝ｻ逎ｯ蟷戊・・髦懃ｸｺ・ｧ雎ｬ竏昴・郢ｧ蛛ｵ縺咲ｹｧ・ｦ郢晢ｽｳ郢晁肩・ｼ繝ｻ
        if sanity_uniform:
            flow_field = np.zeros((height, width, 2))
            flow_field[:, :, 0] = 0.05 * float(flow_scale)
        else:
            flow_field = generate_dynamic_flow_field(width, height, step, scale=flow_scale)
        particles, outflow_mass_step = diffuse_particles(
            particles, terrain, flow_field,
            reflect_boundaries=bool(cfg.get("reflect_boundaries", False))
        )
        outflow_mgC_step = float(outflow_mass_step) * float(particle_mass_mgC)
        mass_outflow += outflow_mgC_step

        # 陷ｷ・ｸ陷ｿ荳ｻ繝ｻ騾・・・ｼ莠包ｽｿ譎擾ｽｭ莨懈抄郢ｧ雋橸ｽｮ蛹ｻ・狗ｹ晢ｽｻ驕ｶ・ｶ陷ｷ蝓滓ｽ碑崕繝ｻ・ｼ繝ｻ
        debug_hits = {p.name: {"visits": 0, "eligible": 0, "absorptions": 0} for p in plants}
        remaining_particles = []
        step_absorbed = 0.0
        step_fixed = 0.0
        step_growth = 0.0
        for particle in particles:
            # 1) 邵ｺ阮吶・驍願ｲ橸ｽｭ闊娯・陝・ｽｾ邵ｺ蜉ｱ窶ｻ陷ｷ・ｸ陷ｿ荳ｻ蠎・妙・ｽ邵ｺ・ｪ隶蜥ｲ鮟・屐蜻ｵ・｣諛奇ｽ定崕邇ｲ雖・            candidates = []  # (plant, uptake_ratio)
            for plant in plants:
                name = plant.name
                env = plant_env[name]
                eff = plant_eff[name]
                # 陟包ｽｮ驍擾ｽｰ髦ｯ・ｻ邵ｺ・ｯ visit/eligible 陋ｻ・､陞ｳ螢ｹ繝ｻ邵ｺ貅假ｽ・eff<=0 邵ｺ・ｧ郢ｧ繧・ｽｹ・ｾ闖ｴ蜍滓・陞ｳ螢ｹ・帝勗蠕娯鴬
                if eff <= 0.0 and name not in ("Chlorella vulgaris", "Nannochloropsis gaditana"):
                    continue
                dx = particle.x - plant.x
                dy = particle.y - plant.y
                r2 = dx * dx + dy * dy
                within_radius = r2 <= (plant.radius ** 2)
                # Swept-contact detection for fast pass-through (optional, seagrasses)
                swept_hit = False
                if use_swept_contact and name in ("Zostera marina", "Halophila ovalis"):
                    try:
                        swept_hit = _segment_circle_intersect(
                            getattr(particle, "x_prev", particle.x), getattr(particle, "y_prev", particle.y),
                            particle.x, particle.y, plant.x, plant.y, plant.radius
                        )
                    except Exception:
                        swept_hit = False
                # 髴醍ｬｬ逎∬ｰｿ・ｵ鬮ｫ雜｣・ｼ驛・ｽｨ・ｺ隴・ｽｭ騾包ｽｨ繝ｻ繝ｻ                visited = False
                eligible_flag = False
                if name in ("Chlorella vulgaris", "Nannochloropsis gaditana"):
                    cap2 = (plankton_capture_radius_px ** 2)
                    visited = (particle.y <= euphotic_px * 1.2) and (r2 <= cap2 * 4.0)
                    eligible_flag = (particle.y <= euphotic_px) and (r2 <= cap2 * 1.5)
                elif name in ("Macrocystis pyrifera", "Saccharina japonica"):
                    visited = (r2 <= (plant.radius * plant.radius * 4.0))
                    eligible_flag = (r2 <= (plant.radius * plant.radius * 2.25))
                elif name in ("Spartina alterniflora", "Rhizophora spp."):
                    shallow_ok = particle.y <= intertidal_shallow_band_px
                    visited = (r2 <= (plant.radius * plant.radius * 4.0))
                    eligible_flag = shallow_ok and (r2 <= (plant.radius * plant.radius * 2.25))
                else:
                    visited = (r2 <= (plant.radius * plant.radius * 4.0))
                    eligible_flag = (r2 <= (plant.radius * plant.radius * 2.25))
                # visits 縺ｯ allowed 遒ｺ螳壼ｾ後↓郢ｰ繧贋ｸ翫￡蜿肴丐縺吶ｋ

                allowed = False
                if name in ("Chlorella vulgaris",):
                    # 隴帷甥繝ｻ陞ｻ・､邵ｺ荵昶命郢ｧ・ｳ郢晢ｽｭ郢昜ｹ昴・陷企宦・ｾ繝ｻ繝ｻ邵ｺ・ｮ邵ｺ・ｿ陷ｷ・ｸ陷ｿ繝ｻ
                    within_capture = r2 <= (plankton_capture_radius_px ** 2)
                    allowed = (particle.y <= euphotic_px) and within_capture
                elif name in ("Macrocystis pyrifera", "Saccharina japonica"):
                    # 郢ｧ・ｳ郢晢ｽｳ郢ｧ・ｻ郢晏干繝ｨ: 郢晏ｸ厥晉ｹ晏ｳｨ繝ｵ郢ｧ・｡郢ｧ・ｹ郢晁肩・ｼ蝓滂ｽｵ・ｷ陟守ｩゑｽｻ蛟ｩ・ｿ謇假ｽｼ蟲ｨ竊堤ｹｧ・ｭ郢晢ｽ｣郢晏ｼｱ繝ｴ郢晢ｽｼ繝ｻ驛・ｽ｡・ｨ陞ｻ・､繝ｻ蟲ｨ縲堤ｸｺ・ｮ隰仙｢捺狭郢ｧ雋槭・邵ｺ莉｣・狗ｸｲ繝ｻ
                    kelp_band_m = float(kelp_bottom_band_m)          # 雎ｬ・ｷ陟取・・ｿ螟ｧ・咲ｸｺ・ｮ鬩ｩ蟶ｷ蟲ｩ闖ｴ諛・舞陝ｶ・ｯ繝ｻ謖会ｽｱm, 郢ｧ繝ｻ・・ｫ｡・｡陞滂ｽｧ繝ｻ繝ｻ
                    surface_band_m = float(kelp_surface_band_m)       # 髯ｦ・ｨ陞ｻ・､邵ｺ・ｮ闖ｴ諛・舞陝ｶ・ｯ繝ｻ繝ｻ..m, 郢ｧ・ｭ郢晢ｽ｣郢晏ｼｱ繝ｴ郢晢ｽｼ陷ｴ螢ｹ・√・繝ｻ
                    kelp_band_px = kelp_band_m / meters_per_pixel
                    surface_band_px = surface_band_m / meters_per_pixel
                    within_band = abs(dy) <= kelp_band_px
                    near_surface = particle.y <= surface_band_px
                    horizontal_ok = abs(dx) <= plant.radius
                    # 雎ｬ・ｷ陟取・・ｿ螟ｧ・・ 陞ｳ謔溘・邵ｺ・ｪ陷企宦・ｾ繝ｻ繝ｻ AND 闖ｴ諛・舞陝ｶ・ｯ陷繝ｻ
                    # 髯ｦ・ｨ陞ｻ・､陝ｶ・ｯ: 鬩ｩ蟶ｷ蟲ｩ髴肴辨螻ｬ邵ｺ・ｯ闕ｳ讎頑牒邵ｲ竏ｵ・ｰ・ｴ陝ｷ・ｳ陷企宦・ｾ繝ｻ繝ｻ邵ｺ・ｧ邵ｺ繧・ｽ檎ｸｺ・ｰ陷ｷ・ｸ陷ｿ荳ｻ蠎・・蛹ｻ縺冗ｹ晢ｽ｣郢晏ｼｱ繝ｴ郢晢ｽｼ繝ｻ繝ｻ
                    allowed = (within_radius and within_band) or (horizontal_ok and near_surface)
                else:
                    if name in ("Spartina alterniflora", "Rhizophora spp."):
                        # 雋趣ｽｮ鬮｢轣假ｽｸ・ｯ: 髯ｦ・ｨ陞ｻ・､郢晢ｽｻ陷企宦・ｾ繝ｻ繝ｻ邵ｺ・ｯ陋溷揃・｣諛ｷ蝟ｧ邵ｺ蜉ｱﾂ竏晉ｲｾ陷ｿ蜿厄ｽｯ譁絶・鬨ｾ・｣驍ｯ螢ｹ縺皮ｹ晢ｽｼ郢晏現・定ｬ怜ｸ呻ｿ郢ｧ繝ｻ
                        shallow_ok = particle.y <= intertidal_shallow_band_px
                        allowed = shallow_ok and within_radius
                    else:
                        # 闕ｳﾂ髣奇ｽｬ邵ｺ・ｮ雎ｬ・ｷ髣輔・ 陷企宦・ｾ繝ｻ繝ｻ + 郢ｧ繝ｻ・・弱・・樣ｩｩ蟶ｷ蟲ｩ陝ｶ・ｯ繝ｻ謖会ｽｱ4 m繝ｻ繝ｻ
                        sg_band_m = 4.0
                        if name == "Zostera marina":
                            sg_band_m = 4.0
                        sg_band_px = sg_band_m / meters_per_pixel
                        base_contact = (within_radius or swept_hit)
                        allowed = base_contact and (abs(dy) <= sg_band_px)

                # 蜊倩ｪｿ諤ｧ菫晁ｨｼ: allowed 縺ｧ縺ゅｌ縺ｰ eligible/visited 繧ら悄縺ｫ縺吶ｋ
                if allowed:
                    eligible_flag = True
                    visited = True
                if visited:
                    debug_hits[name]["visits"] += 1
                if not allowed:
                    continue

                # 遞ｮ蛻･縺ｮ荳矩剞蜉ｹ邇・ｒ驕ｩ逕ｨ・亥ｾｮ邏ｰ阯ｻ縺ｮ荳肴ｴｻ諤ｧ蛹門屓驕ｿ・・                eff_local = eff
                if name in ("Chlorella vulgaris", "Nannochloropsis gaditana"):
                    eff_local = max(eff_local, float(microalgae_min_eff))
                uptake_ratio = eff_local * getattr(plant, "absorption_efficiency", 1.0)
                # Seasonal shaping: microalgae summer boost; eelgrass mid-game boost
                if name == "Chlorella vulgaris":
                    s = 0.5 + 0.5 * np.sin(2.0 * np.pi * (step / float(max(total_steps, 1))))
                    uptake_ratio *= float(1.0 + chl_uptake_season_amp * s)
                # Attenuate microalgae to prevent full upstream depletion (configurable)
                if name in ("Chlorella vulgaris", "Nannochloropsis gaditana") and microalgae_uptake_scale != 1.0:
                    uptake_ratio *= float(microalgae_uptake_scale)
                if name == "Zostera marina":
                    t = step / float(max(total_steps, 1))
                    center = float(eelgrass_mid_boost_center_frac)
                    k = float(eelgrass_mid_boost_k)
                    L = 1.0 / (1.0 + np.exp(-k * (t - center)))
                    factor = 1.0 + (float(eelgrass_mid_boost_factor) - 1.0) * float(L)
                    uptake_ratio *= float(factor)
                # 雋趣ｽｮ鬮｢轣假ｽｸ・ｯ邵ｺ・ｮ鬨ｾ・｣驍ｯ螢ｹ縺皮ｹ晢ｽｼ郢晁肩・ｼ繝ｻ..1繝ｻ繝ｻ
                if name in ("Spartina alterniflora", "Rhizophora spp."):
                    phase = 2.0 * np.pi * ((step % max(tidal_period_steps, 1)) / max(tidal_period_steps, 1))
                    submergence = 0.5 + 0.5 * np.sin(phase)
                    thr = 1.0 - intertidal_submergence_fraction
                    gate = (submergence - thr) / max(intertidal_submergence_fraction, 1e-9)
                    gate = float(min(max(gate, 0.0), 1.0))
                    uptake_ratio *= gate
                uptake_ratio = float(min(max(uptake_ratio, 0.0), 1.0))
                if eligible_flag and allowed:
                    debug_hits[name]["eligible"] += 1

                # Two-stage gate: require base contact for 2+ consecutive steps and a stochastic pass
                if not hasattr(particle, "contact_steps"):
                    particle.contact_steps = {}
                prev_cs = int(particle.contact_steps.get(name, 0))
                if name in ("Zostera marina", "Halophila ovalis"):
                    # seagrass: allow swept-contact and configurable dwell requirement
                    base_contact = (within_radius or swept_hit)
                    particle.contact_steps[name] = (prev_cs + 1) if base_contact else 0
                    need_steps = int(max(seagrass_min_contact_steps, 0))
                    passes_dwell = int(particle.contact_steps.get(name, 0)) >= need_steps
                    contact_ok = base_contact
                else:
                    particle.contact_steps[name] = (prev_cs + 1) if within_radius else 0
                    passes_dwell = int(particle.contact_steps.get(name, 0)) >= 2
                    contact_ok = within_radius

                if eligible_flag and contact_ok and passes_dwell:
                    if uptake_ratio > 0.0 and random.random() < uptake_ratio:
                        candidates.append((plant, uptake_ratio))

            # 2) 陋溷揃・｣諛岩ｲ霎滂ｽ｡邵ｺ莉｣・檎ｸｺ・ｰ邵ｺ譏ｴ繝ｻ邵ｺ・ｾ邵ｺ・ｾ隹ｿ荵昶・
            if not candidates or particle.mass <= 1e-12:
                if particle.mass > 1e-12:
                    remaining_particles.append(particle)
                continue

            # 3) 陷ｷ・ｸ陷ｿ蠑ｱ・定屐蜻ｵ・｣諞ｺ菫｣邵ｺ・ｧ隰也甥繝ｻ繝ｻ逎ｯ纃ｾ邵ｺ・ｿ=uptake_ratio繝ｻ繝ｻ
            total_u = sum(u for _, u in candidates)
            if total_u <= 1e-12:
                remaining_particles.append(particle)
                continue

            # 驍願ｲ橸ｽｭ闊個ｰ郢ｧ迚呻ｽｼ霈披ｳ隰壽㈱・･驍ｱ蝓弱詐邵ｺ・ｯ "驍願ｲ橸ｽｭ蜊・ｳ・ｪ鬩･繝ｻ・・・min(total_u, 1)"
            take_total = particle.mass * min(total_u, 1.0)
            # 陷ｷ繝ｻﾂ蜻ｵ・｣諛岩・邵ｺ・ｮ鬩滓ｦ翫・
            for plant, u in candidates:
                share = (u / total_u) * take_total
                if share <= 0.0:
                    continue
                absorbed, fixed, growth = plant.absorb(share)
                debug_hits[plant.name]["absorptions"] += 1
                # plant.absorb 邵ｺ・ｯ share 郢ｧ蛛ｵ笳守ｸｺ・ｮ邵ｺ・ｾ邵ｺ・ｾ雎ｸ驛・ｽｲ・ｻ邵ｺ蜷ｶ・玖恆閧ｴ鄂ｲ繝ｻ莠包ｽｿ譎擾ｽｭ莨懈抄繝ｻ蟲ｨﾂ繝ｻ
                step_absorbed += absorbed
                step_fixed += fixed
                step_growth += growth
                # Metrics: capture by origin and travel distance
                origin = getattr(particle, "origin", None) or "init"
                if origin not in capture_matrix:
                    capture_matrix[origin] = {name: 0.0 for name in target_species}
                capture_matrix[origin][plant.name] = capture_matrix[origin].get(plant.name, 0.0) + float(absorbed)
                try:
                    dx0 = float(particle.x) - float(particle.x0)
                    dy0 = float(particle.y) - float(particle.y0)
                    travel = float(np.hypot(dx0, dy0))
                    travel_weighted_sum += float(absorbed) * travel
                    travel_weight += float(absorbed)
                except Exception:
                    pass

            particle.mass -= take_total
            if particle.mass > 1e-12:
                remaining_particles.append(particle)

        particles = np.array(remaining_particles, dtype=object)
        # 郢ｧ・ｹ郢昴・繝｣郢晉､ｼ・ｵ繧会ｽｫ・ｯ邵ｺ・ｧ髫ｪ・ｺ隴・ｽｭ郢ｧ・ｷ郢晢ｽｪ郢晢ｽｼ郢ｧ・ｺ邵ｺ・ｫ髴托ｽｽ髫ｪ繝ｻ
        if diag_enabled:
            for p in plants:
                ds = debug_hits[p.name]
                diag_series[p.name]["visits"].append(int(ds["visits"]))
                diag_series[p.name]["eligible"].append(int(ds["eligible"]))
                diag_series[p.name]["absorptions"].append(int(ds["absorptions"]))

        # 隴・ｽｰ髫穂ｹ暦ｽｵ竏昴・繝ｻ閧ｲ・ｭ迚吶・鬩滓誓・ｼ繝ｻ 隘搾ｽｷ雋・・ﾎ帷ｹ晏生ﾎ晁脂蛟･窶ｳ邵ｺ・ｧ雎包ｽｨ陷茨ｽ･邵ｺ蜉ｱﾂ竏ｫ・ｸ・ｦ郢ｧ・ｲ郢晢ｽｼ郢晏現繝ｻ隴帷甥譟題ｫ､・ｧ郢ｧ繧奇ｽｨ蝓滂ｽｸ・ｬ
        num_new = seasonal_inflow(step, total_steps, base_mgC_per_step=inflow_mgC_per_step_base, particle_mass_mgC=particle_mass_mgC)
        if num_new > 0:
            # 鬩･髦ｪ竏ｩ闔牙･・繝ｻ莠･・ｷ・ｦ髯ｦ・ｨ陞ｻ・､邵ｺ・ｫ30%郢ｧ蟶昴・陋ｻ繝ｻﾂ竏ｽ・ｻ謔ｶ繝ｻ陜ｮ繝ｻ・ｭ莨夲ｽｼ繝ｻ
            weights = [0.30, 0.70 / 4.0, 0.70 / 4.0, 0.70 / 4.0, 0.70 / 4.0]  # TODO: make configurable via CSV
            wsum = sum(weights) or 1.0
            ideal = [w / wsum * num_new for w in weights]
            counts = [int(x) for x in ideal]
            deficit = num_new - sum(counts)
            if deficit > 0:
                order = sorted(range(len(ideal)), key=lambda i: ideal[i] - counts[i], reverse=True)
                for i in range(deficit):
                    counts[order[i % len(order)]] += 1

            new_particles = []
            added_total = 0
            from .models.particle import Particle
            for si, (sx, sy) in enumerate(injection_sources):
                count = counts[si]
                lab = source_labels[si]
                for _ in range(count):
                    sy_eff = sy + step * injection_drift_px_per_step
                    sy_eff = sy + step * injection_drift_px_per_step
                    x = sx + np.random.normal(scale=injection_sigma_px)
                    y = sy_eff + np.random.normal(scale=injection_sigma_px)
                    if sy_eff < 0:
                        sy_eff = max(0, min(sy_eff, height - 1))
                        if diag_enabled:
                            inj_xy_rows.append([int(step), float(x), float(y), lab])
                        try:
                            abs_dy = min(abs(y - p.y) for p in plants)
                            abs_dy_samples.append(float(abs_dy))
                        except Exception:
                            pass
                        added_total += 1
            if diag_enabled:
                inj_series.append([int(step)] + [int(c) for c in counts])
            if len(new_particles) > 0:
                if isinstance(particles, list):
                    particles.extend(new_particles)
                elif len(particles) == 0:
                    particles = np.array(new_particles, dtype=object)
                else:
                    particles = np.concatenate((particles, np.array(new_particles, dtype=object)))
            inflow_mgC_step = float(added_total) * float(particle_mass_mgC)
            mass_inflow += inflow_mgC_step
        if gate_enabled and step == 0:
            assert_gate_effective(abs_dy_samples, min_median=min_median_abs_dy)

        # 郢晏干ﾎ帷ｹ晢ｽｳ郢ｧ・ｯ郢晏現ﾎｦ繝ｻ繝ｻhlorella繝ｻ蟲ｨ繝ｻ髢ｾ・ｪ霎滂ｽｶ雎・ｽｻ闔・｡郢晢ｽｻ陷閧ｴ蜿幄怎・ｺ繝ｻ繝ｻO2陟包ｽｩ陝ｶ・ｰ繝ｻ繝ｻ
        reinj_mgC_step = 0.0
        for plant in plants:
            if plant.name == "Chlorella vulgaris" and plant.total_growth > 0:
                # seasonally modulated mortality: higher outside summer
                s = 0.5 + 0.5 * np.sin(2.0 * np.pi * (step / float(max(total_steps, 1))))
                rate_t = float(chl_mortality + chl_mortality_amp * (1.0 - s))
                rate_t = float(min(max(rate_t, 0.0), 0.95))
                mortal = plant.total_growth * rate_t
                if mortal > 0:
                    plant.total_growth -= mortal
                    # mgC 郢ｧ蝣､・ｲ雋橸ｽｭ蜈育・邵ｺ・ｫ陞溽判驪､邵ｺ蜉ｱ窶ｻ邵ｲ竏壺落邵ｺ・ｮ陜｣・ｴ邵ｺ・ｫ陷閧ｴ・ｳ・ｨ陷茨ｽ･
                    n_rel = int(round(mortal / max(particle_mass_mgC, 1e-9)))
                    # 陷閧ｴ・ｳ・ｨ陷茨ｽ･邵ｺ・ｮ闕ｳ・ｸ郢ｧ竏ｬ・ｪ・､陝ｾ・ｮ繝ｻ繝ｻgC繝ｻ蟲ｨ・帝里繝ｻ・ｩ謳ｾ・ｼ螢ｽ・ｭ・｣髮具｣ｰ邵ｺ・ｩ邵ｺ・｡郢ｧ蟲ｨ・り愾謔ｶ・顔ｸｺ繝ｻ・・                    loss_quant_mgC += float(mortal) - float(n_rel) * float(particle_mass_mgC)
                    if n_rel > 0:
                        # 邵ｺ譏ｴ繝ｻ陜｣・ｴ邵ｺ・ｫ隘搾ｽｷ雋・・ﾎ帷ｹ晏生ﾎ晁脂蛟･窶ｳ邵ｺ・ｧ陷閧ｴ・ｳ・ｨ陷茨ｽ･繝ｻ繝ｻrigin="reinj_<plant>")
                        from .models.particle import Particle
                        added_rel = 0
                        for _ in range(n_rel):
                            x = float(plant.x) + np.random.normal(scale=0.5)
                            y = float(plant.y) + np.random.normal(scale=0.5)
                            if 0 <= x < width and 0 <= y < height and terrain[int(y), int(x)] == 1:
                                origin_lab = f"reinj_{_slugify(plant.name)}"
                                pnew = Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0, origin=origin_lab, x0=x, y0=y)
                                if isinstance(particles, list):
                                    particles.append(pnew)
                                elif len(particles) == 0:
                                    particles = np.array([pnew], dtype=object)
                                else:
                                    particles = np.concatenate((particles, np.array([pnew], dtype=object)))
                                added_rel += 1
                        reinj_mgC = float(added_rel) * float(particle_mass_mgC)
                        reinj_mgC_step += reinj_mgC
                        mass_inflow += reinj_mgC

        # 郢ｧ・ｹ郢昴・繝｣郢晉軸蠢ｰ邵ｺ・ｮ髮会ｽｪ鬩･荳茨ｽｿ譎擾ｽｭ蛟･繝｡郢ｧ・ｧ郢昴・縺代・莠包ｽｻ・ｻ隲｢謫ｾ・ｼ繝ｻ
        if debug_mass_check:
            curr_particles_mass_mgC = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
            absorbed_mgC_step = float(step_absorbed) * float(particle_mass_mgC)
            quant_delta_mgC = float(loss_quant_mgC) - float(prev_loss_quant_mgC)
            inflow_total_step = float(inflow_mgC_step if 'inflow_mgC_step' in locals() else 0.0) + float(reinj_mgC_step)
            left = prev_particles_mass_mgC + inflow_total_step
            right = curr_particles_mass_mgC + outflow_mgC_step + absorbed_mgC_step - quant_delta_mgC
            denom = max(left, right, 1.0)
            rel_err = abs(left - right) / denom
            if rel_err > 1e-6:
                print(f"[mass-check] step={step} rel_err={rel_err:.3e} (prev={prev_particles_mass_mgC:.4f}, inflow={inflow_total_step:.4f}, outflow={outflow_mgC_step:.4f}, absorbed={absorbed_mgC_step:.4f}, quant・弱・{quant_delta_mgC:.4f}, curr={curr_particles_mass_mgC:.4f})")

        # 陷ｿ・ｯ髫暮摩蝟ｧ繝ｻ莠包ｽｻ・ｻ隲｢繝ｻ郢晢ｽｩ郢ｧ・､郢晏私・ｼ繝ｻ
        if live_plot_interval > 0 and (step % live_plot_interval == 0):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor("#d0f7ff")
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw / 2, ry - rh / 2), rw, rh, color="gray", alpha=0.7)
                ax.add_patch(rock_patch)
            # 隶蜥ｲ鮟・ｸｺ・ｮ闖ｴ蜥ｲ・ｽ・ｮ
            for plant in plants:
                circ = plt.Circle((plant.x, plant.y), plant.radius, color="green", alpha=0.3)
                ax.add_patch(circ)
            # 驍願ｲ橸ｽｭ繝ｻ
            if len(particles) > 0:
                ax.scatter([p.x for p in particles], [p.y for p in particles], c="cyan", s=5)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_xlabel("X")
            ax.set_ylabel("Depth (down)")
            ax.invert_yaxis()
            ax.set_title(f"Step {step} (Live)")
            plt.tight_layout()
            if show_plots:
                plt.show(block=False)
                plt.pause(0.01)
            plt.close(fig)

        # 陷ｷ驛・ｽｨ莠･ﾂ・､
        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)

        # 郢ｧ・ｹ郢昴・繝｣郢晏干・・ｸｺ・ｨ邵ｺ・ｮ陷ｷ驛・ｽｨ蛹ｻ・帝坎蛟ｬ鮖ｸ
        step_absorbed_series.append(step_absorbed)
        step_fixed_series.append(step_fixed)
        step_growth_series.append(step_growth)
        particle_count_series.append(int(len(particles)))

        # 驕橸ｽｮ邵ｺ譁絶・邵ｺ・ｮ驍擾ｽｯ驕ｨ髦ｪ・帝坎蛟ｬ鮖ｸ
        for plant in plants:
            series = species_series[plant.name]
            series["total_absorbed"].append(plant.total_absorbed)
            series["total_fixed"].append(plant.total_fixed)
            series["total_growth"].append(plant.total_growth)

    # ===== 驍ｨ蜈域｣｡鬮ｮ繝ｻ・ｨ繝ｻ=====
    species_fixed_totals = {plant.name: (plant.total_fixed * float(particle_mass_mgC)) for plant in plants}
    print("\n=== 陷ｷ驛・ｽｨ莠･蟠玖楜蜥ｾO2鬩･謫ｾ・ｼ蝓滂ｽ､蜥ｲ鮟・◇・ｮ陋ｻ・･繝ｻ繝ｻ===")
    for species, total_mgC in species_fixed_totals.items():
        print(f"{species}: {total_mgC:.2f} mgC")

    # 髮会ｽｪ鬩･荳槫ｺｶ隰ｾ・ｯ郢昶・縺臥ｹ昴・縺代・莠･繝ｻ隴帶ｻゑｽｼ蛹ｺ・ｵ竏昴・繝ｻ譎・ｽｮ遏ｩ纃ｼ繝ｻ蛹ｺ・ｵ竏昴・繝ｻ繝ｻ
    current_particle_mass = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
    plant_absorbed = float(sum(p.total_absorbed for p in plants)) * float(particle_mass_mgC)
    plant_fixed = float(sum(p.total_fixed for p in plants)) * float(particle_mass_mgC)

    total_injected = float(mass_initial + mass_inflow)
    total_outflow = float(mass_outflow)
    total_remaining = float(current_particle_mass + plant_absorbed + loss_quant_mgC)

    # 陷ｿ蜿夜ｫｪ髫ｱ・､陝ｾ・ｮ郢ｧ蝣､・ｮ諤懊・邵ｺ蜉ｱ窶ｻ髯ｦ・ｨ驕会ｽｺ繝ｻ繝ｻgC陷雁・ｽｽ謳ｾ・ｼ繝ｻ
    balance_error = 0.0 if total_injected <= 1e-9 else abs(total_injected - (total_remaining + total_outflow)) / total_injected
    print(
        f"Mass balance: Injected={total_injected:.2f} mgC, "
        f"Absorbed={plant_absorbed:.2f} mgC, Fixed={plant_fixed:.2f} mgC, "
        f"Outflow={total_outflow:.2f} mgC, Remaining={total_remaining:.2f} mgC, "
        f"Quantization={loss_quant_mgC:.2f} mgC, Error={balance_error*100:.2f}%"
    )

    # 驍ｨ蜈域｣｡CSV闖ｫ譎擾ｽｭ蛛・ｽｼ蛹ｻ縺礼ｹ晄ｧｭﾎ・& 驕橸ｽｮ陋ｻ・･邵ｺ譁絶・邵ｺ・ｮ隴弱ｉ・ｳ・ｻ陋ｻ繝ｻ郢ｧ・ｵ郢晄ｧｭﾎ懊・繝ｻ
    os.makedirs("results", exist_ok=True)

    # 隴鯉ｽｧ闔蜻磯共: 陷ｷ繝ｻ・ｨ・ｮ邵ｺ・ｮ陷ｷ驛・ｽｨ闌ｨ・ｼ繝ｻ髯ｦ髱ｴSV繝ｻ繝ｻ
    for plant in plants:
        slug = _slugify(plant.name)
        mgc_path = os.path.join("results", f"result_{slug}_mgC.csv")
        with open(mgc_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_absorbed_mgC", "total_fixed_mgC", "total_growth_mgC"])
            writer.writerow([
                plant.total_absorbed * float(particle_mass_mgC),
                plant.total_fixed * float(particle_mass_mgC),
                plant.total_growth * float(particle_mass_mgC),
            ])
        # index 諠・ｱ繧定塘遨搾ｼ亥錐遘ｰ/繧ｹ繝ｩ繝・げ/繝輔ぃ繧､繝ｫ繝代せ・・        
        index_rows.append([
            plant.name,
            slug,
            f"result_{slug}_mgC.csv",
            f"time_series_{slug}_mgC.csv",
        ])

    # 隴・ｽｰ: 陷茨ｽｨ闖ｴ阮吶＠郢晄ｧｭﾎ・    with open(os.path.join("results", "summary_totals.csv"), "w", newline="") as f:
    with open(os.path.join("results", "summary_totals.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "total_absorbed_mgC", "total_fixed_mgC", "total_growth_mgC"])
        for plant in plants:
            writer.writerow([
                plant.name,
                plant.total_absorbed * float(particle_mass_mgC),
                plant.total_fixed * float(particle_mass_mgC),
                plant.total_growth * float(particle_mass_mgC),
            ])

    # 隴・ｽｰ: 陷茨ｽｨ闖ｴ阮吶・隴弱ｉ・ｳ・ｻ陋ｻ繝ｻ
    with open(os.path.join("results", "overall_timeseries.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "env_eff_sample",
            "nutrient_sample",
            "absorbed_step",
            "fixed_step",
            "growth_step",
            "particle_count",
            "cum_fixed_all",
            "cum_growth_all",
        ])
        for i in range(total_steps):
            writer.writerow([
                i,
                env_series[i] if i < len(env_series) else "",
                nutrient_series[i] if i < len(nutrient_series) else "",
                step_absorbed_series[i] if i < len(step_absorbed_series) else 0.0,
                step_fixed_series[i] if i < len(step_fixed_series) else 0.0,
                step_growth_series[i] if i < len(step_growth_series) else 0.0,
                particle_count_series[i] if i < len(particle_count_series) else 0,
                fixed_series[i] if i < len(fixed_series) else 0.0,
                internal_series[i] if i < len(internal_series) else 0.0,
            ])

    # 隴・ｽｰ: 驕橸ｽｮ陋ｻ・･邵ｺ譁絶・邵ｺ・ｮ隴弱ｉ・ｳ・ｻ陋ｻ證ｦ・ｼ閧ｲ・ｴ・ｯ驕ｨ謳ｾ・ｼ繝ｻ
    for name, series in species_series.items():
        slug = _slugify(name)
        out_path = os.path.join("results", f"time_series_{slug}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total_absorbed", "total_fixed", "total_growth"])
            for i in range(total_steps):
                writer.writerow([
                    i,
                    series["total_absorbed"][i],
                    series["total_fixed"][i],
                    series["total_growth"][i],
                ])
        # mgC郢ｧ・ｹ郢ｧ・ｱ郢晢ｽｼ郢晢ｽｫ邵ｺ・ｮ闕ｳ・ｦ髯ｦ謔溘・陷峨・
        out_path_mgc = os.path.join("results", f"time_series_{slug}_mgC.csv")
        with open(out_path_mgc, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total_absorbed_mgC", "total_fixed_mgC", "total_growth_mgC"])
            for i in range(total_steps):
                writer.writerow([
                    i,
                    series["total_absorbed"][i] * float(particle_mass_mgC),
                    series["total_fixed"][i] * float(particle_mass_mgC),
                series["total_growth"][i] * float(particle_mass_mgC),
            ])

    # 繧､繝ｳ繝・ャ繧ｯ繧ｹ・医←縺ｮ繝輔ぃ繧､繝ｫ繧定ｦ九ｌ縺ｰ繧医＞縺九ｒ荳隕ｧ蛹厄ｼ・    try:
with open(os.path.join("results", "index.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["species", "slug", "summary_file_mgC", "timeseries_file_mgC"])
    for row in index_rows:
        writer.writerow(row)
    except Exception as e:
        print(f"[warn] results index write failed: {e}")

    # Consolidated per-species cumulative absorbed mgC timeseries
    try:
        os.makedirs("results", exist_ok=True)
        all_path = os.path.join("results", "all_species_timeseries.csv")
        species_names = list(species_series.keys())
        with open(all_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + species_names)
            for i in range(total_steps):
                row = [i]
                for name in species_names:
                    val = 0.0
                    if i < len(species_series[name]["total_absorbed"]):
                        val = float(species_series[name]["total_absorbed"][i]) * float(particle_mass_mgC)
                    row.append(val)
                writer.writerow(row)
    except Exception as e:
        print(f"[warn] write all_species_timeseries failed: {e}")

    # Consolidated per-species cumulative fixed mgC timeseries
    try:
        os.makedirs("results", exist_ok=True)
        fix_path = os.path.join("results", "all_species_fixation_timeseries.csv")
        species_names = list(species_series.keys())
        with open(fix_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + species_names)
            for i in range(total_steps):
                row = [i]
                for name in species_names:
                    val = 0.0
                    if i < len(species_series[name]["total_fixed"]):
                        val = float(species_series[name]["total_fixed"][i]) * float(particle_mass_mgC)
                    row.append(val)
                writer.writerow(row)
    except Exception as e:
        print(f"[warn] write all_species_fixation_timeseries failed: {e}")

    # Optional permanence (decay of fixed pool) timeseries
    try:
        permanence_enabled = bool(cfg.get("permanence_enabled", True))
        if permanence_enabled:
            default_half_life = {
                "Zostera marina": 300,
                "Halophila ovalis": 250,
                "Posidonia oceanica": 350,
                "Macrocystis pyrifera": 120,
                "Saccharina japonica": 120,
                "Chlorella vulgaris": 15,
                "Nannochloropsis gaditana": 15,
                "Spartina alterniflora": 600,
                "Rhizophora spp.": 1000,
            }
            def _half_life_for(name: str) -> float:
                slug = _slugify(name)
                key = f"half_life_{slug}_steps"
                try:
                    return float(cfg.get(key, default_half_life.get(name, 300)))
                except Exception:
                    return float(default_half_life.get(name, 300))
            perm_ts = {name: [0.0] * total_steps for name in species_series.keys()}
            for name, series in species_series.items():
                hl = max(_half_life_for(name), 1.0)
                lam = float(np.log(2.0) / hl)
                prev = 0.0
                last_cum = 0.0
                for i in range(total_steps):
                    cum = float(series["total_fixed"][i]) * float(particle_mass_mgC)
                    inc = cum - last_cum if i > 0 else cum
                    if inc < 0:
                        inc = 0.0
                    prev = prev * (1.0 - lam) + inc
                    perm_ts[name][i] = prev
                    last_cum = cum
            perm_path = os.path.join("results", "all_species_permanence_timeseries.csv")
            with open(perm_path, "w", newline="") as f:
                writer = csv.writer(f)
                names = list(species_series.keys())
                writer.writerow(["step"] + names)
                for i in range(total_steps):
                    row = [i] + [perm_ts[n][i] for n in names]
                    writer.writerow(row)
    except Exception as e:
        print(f"[warn] permanence timeseries failed: {e}")

    # 陜暦ｽｳ邵ｺ・ｮ闖ｫ譎擾ｽｭ蛛・ｽｼ驛・ｽｦ荵晢ｽ・ｸｺ蜷ｶ・櫁愾・ｯ髫暮摩蝟ｧ繝ｻ繝ｻ
    try:
        # 1) 驕橸ｽｮ陋ｻ・･邵ｺ譁絶・邵ｺ・ｮ陷ｷ驛・ｽｨ闌ｨ・ｼ蝓滂ｽ｣蛛ｵ縺堤ｹ晢ｽｩ郢晏桁・ｼ蟲ｨﾂ繧・・闖ｴ髦ｪ繝ｻ mgC 邵ｺ・ｫ驍ｨ・ｱ闕ｳﾂ
        species_fixed_totals_mgC = {p.name: (p.total_fixed * float(particle_mass_mgC)) for p in plants}
        fig, ax = plt.subplots(figsize=(10, 5))
        species = list(species_fixed_totals_mgC.keys())
        totals = [species_fixed_totals_mgC[s] for s in species]
        import numpy as _np_for_ticks
        idx = _np_for_ticks.arange(len(species))
        ax.bar(idx, totals)
        ax.set_xticks(idx)
        ax.set_xticklabels(species, rotation=45, ha="right")
        ax.set_ylabel("Total Fixed CO2 [mgC]")
        ax.set_title("Total CO2 Fixed by Species")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "total_fixed_by_species.png"))
        plt.close(fig)

        # 2) 陷茨ｽｨ闖ｴ阮吶・隴弱ｉ・ｳ・ｻ陋ｻ證ｦ・ｼ繝ｻ郢ｧ・ｹ郢昴・繝｣郢晏干竕邵ｺ貅假ｽ翫・繝ｻ
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.asarray(step_absorbed_series, dtype=float) * float(particle_mass_mgC), label="absorbed/step [mgC]")
        ax.plot(np.asarray(step_fixed_series, dtype=float) * float(particle_mass_mgC), label="fixed/step [mgC]")
        ax.plot(np.asarray(step_growth_series, dtype=float) * float(particle_mass_mgC), label="growth/step [mgC]")
        ax.set_xlabel("Step")
        ax.set_ylabel("mgC")
        ax.set_title("System Totals per Step")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "overall_timeseries.png"))
        plt.close(fig)

        # 3) 莉｣陦ｨ3遞ｮ縺ｮ邏ｯ遨搾ｼ育ｷ夲ｼ・        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.asarray(zostera_fixed_series, dtype=float) * float(particle_mass_mgC), label="Zostera marina (fixed)")
        ax.plot(np.asarray(kelp_fixed_series, dtype=float) * float(particle_mass_mgC), label="Macrocystis pyrifera (fixed)")
        ax.plot(np.asarray(chlorella_fixed_series, dtype=float) * float(particle_mass_mgC), label="Chlorella vulgaris (fixed)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Fixed [mgC]")
        ax.set_title("Cumulative Fixed CO2: Selected Species")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "selected_species_cumulative_fixed.png"))
        plt.close(fig)

        # 4) 蜈ｨ遞ｮ縺ｮ邏ｯ遨榊精蜿朱㍼・亥推遞ｮ・・        fig, ax = plt.subplots(figsize=(10, 6))
        for name, series in species_series.items():
            vals = np.asarray(series["total_absorbed"], dtype=float) * float(particle_mass_mgC)
            ax.plot(vals, label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Absorbed CO2 [mgC]")
        ax.set_title("CO2 Absorption Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "all_species_absorption.png"))
        plt.close(fig)

        # 5) 蜈ｨ遞ｮ縺ｮ邏ｯ遨榊崋螳夐㍼・亥推遞ｮ・・        fig, ax = plt.subplots(figsize=(10, 6))
        for name, series in species_series.items():
            vals = np.asarray(series["total_fixed"], dtype=float) * float(particle_mass_mgC)
            ax.plot(vals, label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Fixed CO2 [mgC]")
        ax.set_title("CO2 Fixation Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "all_species_fixation.png"))
        plt.close(fig)
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    # ===== Diagnostics output =====
    if diag_enabled:
        try:
            os.makedirs("outputs/diagnostics", exist_ok=True)
            # per-species
            for name in target_species:
                slug = _slugify(name)
                path = os.path.join("outputs/diagnostics", f"species_diag_{slug}.csv")
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "visits", "eligible", "absorptions"])  # eff0陷繝ｻ・ｨ・ｳ邵ｺ・ｯ陟｢繝ｻ・ｦ竏ｵ蜃ｾ邵ｺ・ｫ髴托ｽｽ陷会｣ｰ
                    steps = range(total_steps)
                    vs = diag_series[name]["visits"] if name in diag_series else []
                    es = diag_series[name]["eligible"] if name in diag_series else []
                    as_ = diag_series[name]["absorptions"] if name in diag_series else []
                    for i in steps:
                        v = vs[i] if i < len(vs) else 0
                        e = es[i] if i < len(es) else 0
                        a = as_[i] if i < len(as_) else 0
                        if not (v >= e and e >= a):
                            print(f"[warn] diag monotonicity failed {name} step={i}: v={v} e={e} a={a}")
                        w.writerow([i, v, e, a])
            # injection counts
            if inj_series:
                inj_path = os.path.join("outputs/diagnostics", "injection_counts.csv")
                with open(inj_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step"] + source_labels)
                    for row in inj_series:
                        w.writerow(row)
            if inj_xy_rows:
                xy_path = os.path.join("outputs/diagnostics", "injection_xy.csv")
                with open(xy_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "x", "y", "source"])
                    for row in inj_xy_rows:
                        w.writerow(row)
        except Exception as e:
            print(f"[warn] diagnostics output failed: {e}")

    # ===== Metrics output & dominance guard =====
    try:
        os.makedirs("outputs", exist_ok=True)
        total_abs = sum(p.total_absorbed for p in plants)
        species_share = {p.name: (p.total_absorbed / total_abs if total_abs > 0 else 0.0) for p in plants}
        # dominance guard (configurable)
        guard_dominance(species_share, max_share=dom_max_share, streak=dom_streak)
        mean_travel = (travel_weighted_sum / travel_weight) if travel_weight > 0 else 0.0
        metrics = {
            "capture_matrix": capture_matrix,
            "median_abs_dy_at_injection": (float(np.median(np.asarray(abs_dy_samples))) if len(abs_dy_samples) > 0 else None),
            "mean_travel_before_capture_px": float(mean_travel),
            "species_share": species_share,
        }
        with open(os.path.join("outputs", "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] metrics output failed: {e}")

    return (
        env_series,
        nutrient_series,
        internal_series,
        fixed_series,
        released_series,
        zostera_fixed_series,
        kelp_fixed_series,
        chlorella_fixed_series,
        zostera_absorbed_series,
        kelp_absorbed_series,
        chlorella_absorbed_series,
    )


def simulate_step(plants, step, total_steps=150, width=100, height=100, co2=100.0):
    """
    陷雁･縺帷ｹ昴・繝｣郢晁挙・ｩ遨ゑｽｾ・｡繝ｻ莠包ｽｸ・ｻ邵ｺ・ｫ郢昴・縺帷ｹ晁ご逡代・蟲ｨﾂ繝ｻ
    plants: dict繝ｻ莠･謗ｨ驕橸ｽｮ邵ｺ・ｮ郢昜ｻ｣ﾎ帷ｹ晢ｽ｡郢晢ｽｼ郢ｧ・ｿ邵ｲ繝ｻSON邵ｺ・ｮ郢ｧ・ｭ郢晢ｽｼ陷ｷ髦ｪ竊楢惺蛹ｻ・冗ｸｺ蟶呻ｽ九・繝ｻ
    co2: 邵ｺ阮吶・郢ｧ・ｹ郢昴・繝｣郢晏干縲帝圦遨ゑｽｾ・｡陝・ｽｾ髮趣ｽ｡邵ｺ・ｫ闕ｳ蠑ｱ竏ｴ郢ｧ迢励￥驍擾｣ｰ鬩･謫ｾ・ｼ莠･驟碑叉ﾂ陷雁・ｽｽ髦ｪ縲偵・繝ｻ
    """
    results = {}
    nutrient_series = []

    # 騾ｶ・ｸ陝・ｽｾ鬩滓ｦ翫・邵ｺ・ｮ邵ｺ貅假ｽ∫ｸｲ竏晄耳驕橸ｽｮ邵ｺ・ｮ鬩･髦ｪ竏ｩ繝ｻ繝ｻbsorption_rate 邵ｺ・ｾ邵ｺ貅倥・ absorption_efficiency繝ｻ蟲ｨ・定惺驛・ｽｨ繝ｻ
    total_abs_rate = 0.0
    for _name, _params in plants.items():
        total_abs_rate += float(_params.get("absorption_rate", _params.get("absorption_efficiency", 1.0)))
    total_abs_rate = max(total_abs_rate, 1e-9)  # 0陷托ｽｲ鬮ｦ・ｲ雎・ｽ｢

    step_abs_total = 0.0   # 邵ｺ阮吶・郢ｧ・ｹ郢昴・繝｣郢晏干縲堤ｸｺ・ｮ驍ｱ荳樒ｲｾ陷ｿ繝ｻ
    step_fix_total = 0.0   # 邵ｺ阮吶・郢ｧ・ｹ郢昴・繝｣郢晏干縲堤ｸｺ・ｮ驍ｱ荳槫ｴ玖楜繝ｻ

    for plant_name, params in plants.items():
        x = params.get("x", 0)
        y = params.get("y", 0)
        env = get_environmental_factors(x, y, step, total_steps=total_steps, width=width, height=height)

        temp_sigma = 5.0
        temp_eff = np.exp(-0.5 * ((env["temperature"] - params.get("opt_temp", 20)) / temp_sigma) ** 2)
        light_eff = min(env["light"] / max(params.get("light_tolerance", 1.0), 1e-9), 1.0)

        sal_min, sal_max = params.get("salinity_range", (20, 35))
        salinity = env["salinity"]
        if sal_min <= salinity <= sal_max:
            sal_eff = 1.0
        elif salinity < sal_min:
            sal_eff = max(0.0, 1 - (sal_min - salinity) / 10)
        else:
            sal_eff = max(0.0, 1 - (salinity - sal_max) / 10)

        efficiency = float(temp_eff) * float(light_eff) * float(sal_eff)

        # JSON邵ｺ・ｮ郢ｧ・ｭ郢晢ｽｼ陷ｷ髦ｪ竊楢惺蛹ｻ・冗ｸｺ蟶呻ｽ九・閧ｲ蠍瑚汞・ｾ鬩滓ｦ翫・繝ｻ繝ｻ
        weight   = float(params.get("absorption_rate", params.get("absorption_efficiency", 1.0)))
        share    = weight / total_abs_rate
        fix_rate = params.get("fixation_rate", params.get("fixation_ratio", 0.7))
        growth_r = params.get("growth_rate", 0.0)

        # 邵ｺ譏ｴ繝ｻ驕橸ｽｮ邵ｺ・ｮ陷ｷ・ｸ陷ｿ譛ｱ纃ｼ = 邵ｺ阮吶・郢ｧ・ｹ郢昴・繝｣郢晏干繝ｻCO2 ・・・陷会ｽｹ驍・・・・・騾ｶ・ｸ陝・ｽｾ郢ｧ・ｷ郢ｧ・ｧ郢ｧ・｢
        absorbed = float(co2) * float(efficiency) * float(share)
        fixed = absorbed * float(fix_rate)
        growth = float(growth_r) * absorbed

        step_abs_total += absorbed
        step_fix_total += fixed
        nutrient_series.append(env["nutrient"])

        results[plant_name] = {
            "absorbed": absorbed,
            "growth": growth,
            "fixed": fixed,
            "efficiency": efficiency,
            "env": env,
        }

    # 邵ｺ阮吶・郢ｧ・ｹ郢昴・繝｣郢晄懊・邵ｺ・ｮ陷ｷ驛・ｽｨ蛹ｻ・る恆譁絶・繝ｻ莠･蠎ｶ隰ｾ・ｯ鬮ｮ繝ｻ・ｨ閧ｲ逡代・繝ｻ
    return results, nutrient_series, step_abs_total, step_fix_total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref1", action="store_true", help="Run 1-step preflight visualization and exit")
    args = parser.parse_args()
    run_simulation(pref1=bool(args.pref1))
    # 鬮ｱ讚・ｽ｡・ｨ驕会ｽｺ郢晢ｽ｢郢晢ｽｼ郢晏ｳｨ縲堤ｸｺ・ｯ鬮ｱ讒ｫ・ｯ・ｾ髫ｧ・ｱ郢晁・繝｣郢ｧ・ｯ郢ｧ・ｨ郢晢ｽｳ郢昜ｼ夲ｽｼ繝ｻI/郢ｧ・ｵ郢晢ｽｼ郢晁・繝ｻ陷ｷ莉｣・繝ｻ繝ｻ
    try:
        if not show_plots:
            plt.switch_backend("Agg")
    except Exception:
        pass

    # 郢晢ｽｩ郢晢ｽｳ邵ｺ譁絶・邵ｺ・ｫ隰ｾ・ｯ鬩溷調邏ｫ郢ｧ・ｬ郢晢ｽｼ郢晁・諞ｾ隲ｷ荵晢ｽ堤ｹ晢ｽｪ郢ｧ・ｻ郢昴・繝ｨ
    try:
        _dom_state.clear()
    except Exception:
        pass




