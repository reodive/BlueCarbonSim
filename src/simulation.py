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
    豺ｱ蠎ｦ縺ｮ驕ｩ蝨ｰ繝輔ぅ繝ｫ繧ｿ縲・
    貎ｮ髢灘ｸｯ・・partina/Rhizophora・峨・豌ｴ髱｢逶ｴ荳奇ｽ槫ｹｲ蜃ｺ蟶ｯ縺ｮ縺溘ａ豺ｱ蠎ｦ繝輔ぅ繝ｫ繧ｿ繧帝←逕ｨ縺励↑縺・・
    """
    if getattr(plant, "name", "") in ("Spartina alterniflora", "Rhizophora spp."):
        return eff  # 豺ｱ蠎ｦ譚｡莉ｶ繧貞､悶☆
    dmin, dmax = getattr(plant, "model_depth_range", (0, 999))
    depth_m = float(env.get("depth_m", 0.0))
    return 0.0 if not (dmin <= depth_m <= dmax) else eff

def seasonal_inflow(step, total_steps, base_mgC_per_step=30.0, particle_mass_mgC=1.0):
    """蟄｣遽諤ｧ縺ｮCO竄よｵ∝・・・gC/step・峨ｒ邊貞ｭ先焚縺ｫ螟画鋤縺励※霑斐☆"""
    cycle = 2 * np.pi * step / total_steps
    mgc = base_mgC_per_step * (0.5 + 0.5 * np.sin(cycle))
    count = int(round(mgc / max(particle_mass_mgC, 1e-9)))
    return max(count, 0)
with open("data/plants.json") as f:
    profiles_raw = json.load(f)
profiles = normalize_profiles(profiles_raw)


def _slugify(name: str) -> str:
    """繝輔ぃ繧､繝ｫ蜷咲畑縺ｫ螳牙・縺ｪ繧ｹ繝ｩ繝・げ縺ｸ螟画鋤"""
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
    """繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繧貞ｮ溯｡後＠縲∝推遞ｮ繧ｷ繝ｪ繝ｼ繧ｺ縺ｨ邨先棡繧定ｿ斐☆"""

    # 蝨ｰ蠖｢縺ｨ豺ｱ蠎ｦ繝槭ャ繝・
    terrain, depth_map = create_terrain(width, height)

    # 蜃ｺ蜉帷畑繧ｷ繝ｪ繝ｼ繧ｺ
    env_series: List[float] = []
    nutrient_series: List[float] = []

    # 邏ｯ遨搾ｼ亥・菴難ｼ・
    internal_series: List[float] = []  # 蜈ｨ遞ｮ縺ｮ total_growth 蜷郁ｨ茨ｼ育ｴｯ遨搾ｼ・
    fixed_series: List[float] = []     # 蜈ｨ遞ｮ縺ｮ total_fixed 蜷郁ｨ茨ｼ育ｴｯ遨搾ｼ・
    released_series: List[float] = []  # 莠育ｴ・ｼ域悴菴ｿ逕ｨ・・
    carbon_series: List[float] = []    # 蜷御ｸ奇ｼ井ｺ育ｴ・ｼ・

    # 繧ｹ繝・ャ繝玲ｯ趣ｼ亥・菴難ｼ・
    step_absorbed_series: List[float] = []
    step_fixed_series: List[float] = []
    step_growth_series: List[float] = []
    particle_count_series: List[int] = []

    # 莉｣陦ｨ3遞ｮ・亥ｾ梧婿莠呈鋤逕ｨ・・
    zostera_fixed_series, kelp_fixed_series, chlorella_fixed_series = [], [], []
    zostera_growth_series, kelp_growth_series, chlorella_growth_series = [], [], []
    zostera_absorbed_series, kelp_absorbed_series, chlorella_absorbed_series = [], [], []
    os.makedirs("results", exist_ok=True)

    # 險ｭ螳夊ｪｭ縺ｿ霎ｼ縺ｿ・亥腰菴咲ｭ会ｼ・
    cfg = load_config()
    particle_mass_mgC = float(cfg.get("particle_mass_mgC", 1.0))
    inflow_mgC_per_step_base = float(cfg.get("inflow_mgC_per_step_base", 30.0))
    chl_mortality = float(cfg.get("chl_mortality_rate", 0.02))  # 2%/step
    live_plot_interval = int(cfg.get("live_plot_interval", 0))   # 0縺ｪ繧峨Λ繧､繝匁緒逕ｻ縺ｪ縺・
    show_plots = bool(cfg.get("show_plots", False))              # True縺ｧplt.show
    debug_mass_check = bool(cfg.get("debug_mass_check", False))  # 豈弱せ繝・ャ繝励・雉ｪ驥丈ｿ晏ｭ倥メ繧ｧ繝・け
    flow_scale = float(cfg.get("flow_scale", 1.0))               # 豬・溘せ繧ｱ繝ｼ繝ｫ・域─蠎ｦ遒ｺ隱咲畑・・
    # 繧ｵ繝九ユ繧｣逕ｨ縺ｮ蝮・ｸ繝｢繝ｼ繝峨・譛臥┌
    sanity_uniform = bool(cfg.get("sanity_uniform", False))
    # 豕ｨ蜈･蛻・ｸ・ｼ育ｩｺ髢薙ヰ繧､繧｢繧ｹ邱ｩ蜥檎畑・・    injection_sigma_px = float(cfg.get("injection_sigma_px", 1.0))
    # 注入分布（空間バイアス緩和用）
    injection_drift_px_per_step = float(cfg.get("injection_drift_px_per_step", 0.0))
    microalgae_min_eff = float(cfg.get("microalgae_min_eff", 0.0))
    tidal_period_steps = int(cfg.get("tidal_period_steps", 24))
    intertidal_submergence_fraction = float(cfg.get("intertidal_submergence_fraction", 0.35))
    intertidal_shallow_band_m = float(cfg.get("intertidal_shallow_band_m", 1.0))

    # 驟咲ｽｮ・・遞ｮ縺ｮ讀咲黄縺ｮ繝ｬ繧､繧｢繧ｦ繝茨ｼ・
    plant_positions = {
        # 菴主｡ｩ竊帝ｫ伜｡ｩ縺ｮ荳ｭ髢謎ｻ･荳翫∈遘ｻ蜍包ｼ・ 縺ｯ縺昴・縺ｾ縺ｾ/蜊雁ｾ・ｰ代＠蠅励ｄ縺呻ｼ・
        "Zostera marina":            {"x": 30, "y": 28, "radius": 11},  # 蜊雁ｾ・ｒ1縺縺醍ｸｮ蟆・
        "Halophila ovalis":          {"x": 45, "y": 26, "radius": 10},  # 蜊雁ｾ・2縺ｧ驕ｭ驕・紫竊托ｼ遺翁15 PSU・・
        "Posidonia oceanica":        {"x": 95, "y": 30, "radius": 12},  # 蜊雁ｾ・僑螟ｧ縺ｧ驕ｭ驕・紫竊・
        "Macrocystis pyrifera":      {"x": 85, "y": 80, "radius": 9},   # 蜊雁ｾ・僑螟ｧ縺ｧ驕ｭ驕・紫竊・
        "Saccharina japonica":       {"x": 75, "y": 82, "radius": 9},   # 蜊雁ｾ・僑螟ｧ縺ｧ驕ｭ驕・紫竊・
        "Chlorella vulgaris":        {"x": 10, "y": 14, "radius": 3},   # 0窶・0 PSU
        "Nannochloropsis gaditana":  {"x": 70, "y": 16, "radius": 4},   # 竕･20 PSU・・K・・
        "Spartina alterniflora":     {"x": 30, "y": 6,  "radius": 8},   # 竕･10 PSU・亥ｷｦ縺九ｉ蜿ｳ縺ｸ・・
        "Rhizophora spp.":           {"x": 20, "y": 7,  "radius": 8},   # 竕･5 PSU・亥ｰ代＠蜿ｳ縺ｸ・・
    }

    # 邊貞ｭ舌・豕ｨ蜈･菴咲ｽｮ・亥｢・阜ﾃ怜ｱ､縺ｫ蝗ｺ螳壹らｨｮ逶ｴ荳翫・菴ｿ繧上↑縺・ｼ・
    injection_sources = [
        (2, int(height * 0.20)),   # 蟾ｦ繝ｻ陦ｨ螻､・・hlorella/Halophila 蟇ｾ蠢懶ｼ・
        (2, int(height * 0.50)),   # 蟾ｦ繝ｻ荳ｭ螻､
        (width - 2, int(height * 0.15)),  # 蜿ｳ繝ｻ陦ｨ螻､
        (width - 2, int(height * 0.30)),  # 蜿ｳ繝ｻ荳ｭ豬・
        (width - 2, int(height * 0.80)),  # 蜿ｳ繝ｻ豺ｱ繧・
    ]

    # ===== Guard config & schema (lightweight validation) =====
    guard_cfg = load_config(guard_config_path) if os.path.exists(guard_config_path) else {}
    min_dist_px = int(guard_cfg.get("min_dist_px", 3))
    gate_enabled = bool(guard_cfg.get("gate_enabled", False))
    # 診断の有効化（guard.yaml優先、無ければconfig.yaml）
    diag_enabled = bool(
        guard_cfg.get(
            "diag_enabled",
            guard_cfg.get(
                "diagnostics_enabled",
                bool(cfg.get("diag_enabled", cfg.get("diagnostics_enabled", False)))
            ),
        )
    )
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

    # 蟇ｾ雎｡遞ｮ繧・遞ｮ縺ｫ諡｡蠑ｵ
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
        # depth range 繧貞ｱ樊ｧ縺ｨ縺励※莉倅ｸ・
        setattr(p, "model_depth_range", tuple(profile.get("model_depth_range", (1, 6))))
        plants.append(p)

    # 蛻晄悄邊貞ｭ・
    particles = initialize_particles(num_particles, terrain)

    # 蟯ｩ繧ｪ繝悶ず繧ｧ繧ｯ繝・
    rocks = [
        {"x": width // 2, "y": int(height * 0.5), "w": 12, "h": 8},
        {"x": int(width * 0.7), "y": int(height * 0.3), "w": 8, "h": 5},
    ]

    # 蠎戊ｳｪ繝槭ャ繝・
    bottom_type_map = np.full((height, width), "mud", dtype=object)
    bottom_type_map[90:100, 0:33] = "mud"
    bottom_type_map[90:100, 33:66] = "sand"
    bottom_type_map[90:100, 66:100] = "rock"

    seed_val = int(cfg.get("seed", 42))
    np.random.seed(seed_val)
    random.seed(seed_val)

    # Geometry guard: adjust or abort if any source collides with plant footprints
    injection_sources = validate_geometry(injection_sources, plants, width, height, margin_px=min_dist_px)

    # 繝輔ぉ繧｢繝阪せ蜀埼㍾縺ｿ莉倥￠縺ｯ陦後ｏ縺ｪ縺・ｼ井ｾ帷ｵｦ縺ｯ迺ｰ蠅・↓萓晏ｭ倥∫ｨｮ縺ｫ縺ｯ萓晏ｭ倥＠縺ｪ縺・ｼ・

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

    # ===== 繝｡繧､繝ｳ繝ｫ繝ｼ繝・=====
    # 迚ｩ逅・ｰｺ蠎ｦ・・nvironment 縺ｨ謨ｴ蜷茨ｼ・
    KD = float(cfg.get("kd_m_inv", 0.8))
    MAX_DEPTH_M = 8.0
    meters_per_pixel = MAX_DEPTH_M / max((height - 1), 1)
    # 繝励Λ繝ｳ繧ｯ繝医Φ縺ｮ豌ｴ蟷ｳ蠎・′繧奇ｼ・・俄・ 繝斐け繧ｻ繝ｫ謠帷ｮ・
    plankton_radius_m = float(cfg.get("plankton_radius_m", 0.9))
    plankton_radius_px = plankton_radius_m / max(meters_per_pixel, 1e-9)
    # 微細藻の捕捉半径（広めの既定を許容）
    plankton_capture_radius_px = float(cfg.get("plankton_capture_radius_m", max(plankton_radius_m, 2.0))) / max(meters_per_pixel, 1e-9)
    # 繝輔か繝医だ繝ｼ繝ｳ霑台ｼｼ・・^-kd z = 0.1 竊・z 竕・2.3/kd・・
    euphotic_depth_m = 2.3 / max(KD, 1e-6)
    euphotic_px = int(euphotic_depth_m / meters_per_pixel)
    intertidal_shallow_band_px = intertidal_shallow_band_m / max(meters_per_pixel, 1e-9)

    # 雉ｪ驥上ヰ繝ｩ繝ｳ繧ｹ逕ｨ・・gC蜊倅ｽ搾ｼ・
    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_initial = float(len(particles)) * float(particle_mass_mgC)
    loss_quant_mgC = 0.0  # reinjection rounding loss (mgC)
    # 遞ｮ蛻･縺斐→縺ｮ邏ｯ遨阪す繝ｪ繝ｼ繧ｺ・亥・繧ｹ繝・ャ繝暦ｼ・
    species_series: Dict[str, Dict[str, List[float]]] = {
        name: {"total_absorbed": [], "total_fixed": [], "total_growth": []}
        for name in target_species
    }

    # Metrics accumulators
    source_labels = [f"src{i}" for i in range(len(injection_sources))]
    capture_matrix: Dict[str, Dict[str, float]] = {lab: {name: 0.0 for name in target_species} for lab in source_labels + ["init"]}
    abs_dy_samples: List[float] = []
    # 險ｺ譁ｭ繝舌ャ繝輔ぃ
    diag_series: Dict[str, Dict[str, List[int]]] = (
        {name: {"visits": [], "eligible": [], "absorptions": []} for name in target_species}
        if diag_enabled else {}
    )
    inj_series: List[List[int]] = []
    inj_xy_rows: List[List[float]] = []
    travel_weighted_sum = 0.0
    travel_weight = 0.0

    for step in range(total_steps):
        # 雉ｪ驥擾ｼ・gC・峨せ繝翫ャ繝励す繝ｧ繝・ヨ・医せ繝・ャ繝怜燕・・
        prev_particles_mass_mgC = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
        prev_loss_quant_mgC = float(loss_quant_mgC)

        # 繧ｹ繝・ャ繝励＃縺ｨ縺ｫ迺ｰ蠅・ｩ穂ｾ｡繧偵く繝｣繝・す繝･
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
                    salinity_mode="linear_x",  # 豎ｽ豌ｴ蝓・ 蟾ｦ菴主｡ｩ竊貞承鬮伜｡ｩ
                    S_min=0.0, S_max=35.0,      # 蟾ｦ遶ｯ繧剃ｽ主｡ｩ・域ｷ｡豌ｴ蟇・ｊ・峨↓險ｭ螳・
                    kd_m_inv=KD, max_depth_m=MAX_DEPTH_M,
                )
            px, py = int(plant.x), int(plant.y)
            bottom_type = bottom_type_map[py, px] if (0 <= py < height and 0 <= px < width) else "mud"

            eff = compute_efficiency_score(plant, env, bottom_type=bottom_type)
            eff = apply_depth_filter(eff, plant, env)  # 豺ｱ蠎ｦ繝ｬ繝ｳ繧ｸ螟悶・0
            plant_env[plant.name] = env
            plant_eff[plant.name] = eff

            if i == 0:
                env_series.append(eff)
                nutrient_series.append(env["nutrient"])

        # 讀咲黄縺斐→縺ｮ邏ｯ遨埼㍼
        # 邏ｯ遨阪す繝ｪ繝ｼ繧ｺ・井ｻ｣陦ｨ3遞ｮ縺ｯ蠕梧婿莠呈鋤蜷咲ｧｰ縺ｫ蜷医ｏ縺帙※險倬鹸・・
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

        # 邊貞ｭ先僑謨｣・磯幕蠅・阜縺ｧ豬∝・繧偵き繧ｦ繝ｳ繝茨ｼ・
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

        # 蜷ｸ蜿主・逅・ｼ井ｿ晏ｭ伜援繧貞ｮ医ｋ繝ｻ遶ｶ蜷域潔蛻・ｼ・
        debug_hits = {p.name: {"visits": 0, "eligible": 0, "absorptions": 0} for p in plants}
        remaining_particles = []
        step_absorbed = 0.0
        step_fixed = 0.0
        step_growth = 0.0
        for particle in particles:
            # 1) 縺薙・邊貞ｭ舌↓蟇ｾ縺励※蜷ｸ蜿主庄閭ｽ縺ｪ讀咲黄蛟呵｣懊ｒ蛻玲嫌
            candidates = []  # (plant, uptake_ratio)
            for plant in plants:
                name = plant.name
                env = plant_env[name]
                eff = plant_eff[name]
                # 蠕ｮ邏ｰ阯ｻ縺ｯ visit/eligible 蛻､螳壹・縺溘ａ eff<=0 縺ｧ繧ょｹｾ菴募愛螳壹ｒ陦後≧
                if eff <= 0.0 and name not in ("Chlorella vulgaris", "Nannochloropsis gaditana"):
                    continue
                dx = particle.x - plant.x
                dy = particle.y - plant.y
                r2 = dx * dx + dy * dy
                within_radius = r2 <= (plant.radius ** 2)
                # 霑第磁谿ｵ髫趣ｼ郁ｨｺ譁ｭ逕ｨ・・                visited = False
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
                # visits は allowed 確定後に繰り上げ反映する

                allowed = False
                if name in ("Chlorella vulgaris",):
                    # 譛牙・螻､縺九▽繧ｳ繝ｭ繝九・蜊雁ｾ・・縺ｮ縺ｿ蜷ｸ蜿・
                    within_capture = r2 <= (plankton_capture_radius_px ** 2)
                    allowed = (particle.y <= euphotic_px) and within_capture
                elif name in ("Macrocystis pyrifera", "Saccharina japonica"):
                    # 繧ｳ繝ｳ繧ｻ繝励ヨ: 繝帙Ν繝峨ヵ繧｡繧ｹ繝茨ｼ域ｵｷ蠎穂ｻ倩ｿ托ｼ峨→繧ｭ繝｣繝弱ヴ繝ｼ・郁｡ｨ螻､・峨〒縺ｮ謐墓拷繧貞・縺代ｋ縲・
                    kelp_band_m = 5.0          # 豬ｷ蠎戊ｿ大ｍ縺ｮ驩帷峩菴懃畑蟶ｯ・按ｱm, 繧・ｄ諡｡螟ｧ・・
                    surface_band_m = 3.0       # 陦ｨ螻､縺ｮ菴懃畑蟶ｯ・・..m, 繧ｭ繝｣繝弱ヴ繝ｼ蜴壹ａ・・
                    kelp_band_px = kelp_band_m / meters_per_pixel
                    surface_band_px = surface_band_m / meters_per_pixel
                    within_band = abs(dy) <= kelp_band_px
                    near_surface = particle.y <= surface_band_px
                    horizontal_ok = abs(dx) <= plant.radius
                    # 豬ｷ蠎戊ｿ大ｍ: 螳悟・縺ｪ蜊雁ｾ・・ AND 菴懃畑蟶ｯ蜀・
                    # 陦ｨ螻､蟶ｯ: 驩帷峩霍晞屬縺ｯ荳榊撫縲∵ｰｴ蟷ｳ蜊雁ｾ・・縺ｧ縺ゅｌ縺ｰ蜷ｸ蜿主庄・医く繝｣繝弱ヴ繝ｼ・・
                    allowed = (within_radius and within_band) or (horizontal_ok and near_surface)
                else:
                    if name in ("Spartina alterniflora", "Rhizophora spp."):
                        # 貎ｮ髢灘ｸｯ: 陦ｨ螻､繝ｻ蜊雁ｾ・・縺ｯ蛟呵｣懷喧縺励∝精蜿取ｯ斐↓騾｣邯壹ご繝ｼ繝医ｒ謗帙￠繧・
                        shallow_ok = particle.y <= intertidal_shallow_band_px
                        allowed = shallow_ok and within_radius
                    else:
                        # 荳闊ｬ縺ｮ豬ｷ闕・ 蜊雁ｾ・・ + 繧・ｄ蠎・＞驩帷峩蟶ｯ・按ｱ4 m・・
                        sg_band_m = 4.0
                        if name == "Zostera marina":
                            sg_band_m = 4.0
                        sg_band_px = sg_band_m / meters_per_pixel
                        allowed = within_radius and (abs(dy) <= sg_band_px)

                # 単調性保証: allowed であれば eligible/visited も真にする
                if allowed:
                    eligible_flag = True
                    visited = True
                if visited:
                    debug_hits[name]["visits"] += 1
                if not allowed:
                    continue

                # 種別の下限効率を適用（微細藻の不活性化回避）
                eff_local = eff
                if name in ("Chlorella vulgaris", "Nannochloropsis gaditana"):
                    eff_local = max(eff_local, float(microalgae_min_eff))
                uptake_ratio = eff_local * getattr(plant, "absorption_efficiency", 1.0)
                # 貎ｮ髢灘ｸｯ縺ｮ騾｣邯壹ご繝ｼ繝茨ｼ・..1・・
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
                if within_radius:
                    particle.contact_steps[name] = prev_cs + 1
                else:
                    particle.contact_steps[name] = 0

                if eligible_flag and within_radius and int(particle.contact_steps.get(name, 0)) >= 2:
                    if uptake_ratio > 0.0 and random.random() < uptake_ratio:
                        candidates.append((plant, uptake_ratio))

            # 2) 蛟呵｣懊′辟｡縺代ｌ縺ｰ縺昴・縺ｾ縺ｾ谿九☆
            if not candidates or particle.mass <= 1e-12:
                if particle.mass > 1e-12:
                    remaining_particles.append(particle)
                continue

            # 3) 蜷ｸ蜿弱ｒ蛟呵｣憺俣縺ｧ謖牙・・磯㍾縺ｿ=uptake_ratio・・
            total_u = sum(u for _, u in candidates)
            if total_u <= 1e-12:
                remaining_particles.append(particle)
                continue

            # 邊貞ｭ舌°繧牙ｼ輔″謚懊￥邱城㍼縺ｯ "邊貞ｭ占ｳｪ驥・ﾃ・min(total_u, 1)"
            take_total = particle.mass * min(total_u, 1.0)
            # 蜷・呵｣懊∈縺ｮ驟榊・
            for plant, u in candidates:
                share = (u / total_u) * take_total
                if share <= 0.0:
                    continue
                absorbed, fixed, growth = plant.absorb(share)
                debug_hits[plant.name]["absorptions"] += 1
                # plant.absorb 縺ｯ share 繧偵◎縺ｮ縺ｾ縺ｾ豸郁ｲｻ縺吶ｋ蜑肴署・井ｿ晏ｭ伜援・峨・
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
        # 繧ｹ繝・ャ繝礼ｵらｫｯ縺ｧ險ｺ譁ｭ繧ｷ繝ｪ繝ｼ繧ｺ縺ｫ霑ｽ險・
        if diag_enabled:
            for p in plants:
                ds = debug_hits[p.name]
                diag_series[p.name]["visits"].append(int(ds["visits"]))
                diag_series[p.name]["eligible"].append(int(ds["eligible"]))
                diag_series[p.name]["absorptions"].append(int(ds["absorptions"]))

        # 譁ｰ隕乗ｵ∝・・育ｭ牙・驟搾ｼ・ 襍ｷ貅舌Λ繝吶Ν莉倥″縺ｧ豕ｨ蜈･縺励∫ｸｦ繧ｲ繝ｼ繝医・譛牙柑諤ｧ繧りｨ域ｸｬ
        num_new = seasonal_inflow(step, total_steps, base_mgC_per_step=inflow_mgC_per_step_base, particle_mass_mgC=particle_mass_mgC)
        if num_new > 0:
            # 驥阪∩莉倥￠・亥ｷｦ陦ｨ螻､縺ｫ30%繧帝・蛻・∽ｻ悶・蝮・ｭ会ｼ・
            weights = [0.30, 0.70 / 4.0, 0.70 / 4.0, 0.70 / 4.0, 0.70 / 4.0]
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
                    sy_eff = max(0, min(sy_eff, height - 1))
                    x = sx + np.random.normal(scale=injection_sigma_px)
                    y = sy_eff + np.random.normal(scale=injection_sigma_px)
                    if 0 <= x < width and 0 <= y < height and terrain[int(y), int(x)] == 1:
                        new_particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0, origin=lab, x0=x, y0=y))
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

        # 繝励Λ繝ｳ繧ｯ繝医Φ・・hlorella・峨・閾ｪ辟ｶ豁ｻ莠｡繝ｻ蜀肴叛蜃ｺ・・O2蠕ｩ蟶ｰ・・
        reinj_mgC_step = 0.0
        for plant in plants:
            if plant.name == "Chlorella vulgaris" and plant.total_growth > 0:
                mortal = plant.total_growth * chl_mortality
                if mortal > 0:
                    plant.total_growth -= mortal
                    # mgC 繧堤ｲ貞ｭ先焚縺ｫ螟画鋤縺励※縲√◎縺ｮ蝣ｴ縺ｫ蜀肴ｳｨ蜈･
                    n_rel = int(round(mortal / max(particle_mass_mgC, 1e-9)))
                    # 蜀肴ｳｨ蜈･縺ｮ荳ｸ繧∬ｪ､蟾ｮ・・gC・峨ｒ闢・ｩ搾ｼ壽ｭ｣雋縺ｩ縺｡繧峨ｂ蜿悶ｊ縺・ｋ
                    loss_quant_mgC += float(mortal) - float(n_rel) * float(particle_mass_mgC)
                    if n_rel > 0:
                        # 縺昴・蝣ｴ縺ｫ襍ｷ貅舌Λ繝吶Ν莉倥″縺ｧ蜀肴ｳｨ蜈･・・rigin="reinj_<plant>")
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

        # 繧ｹ繝・ャ繝玲忰縺ｮ雉ｪ驥丈ｿ晏ｭ倥メ繧ｧ繝・け・井ｻｻ諢擾ｼ・
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
                print(f"[mass-check] step={step} rel_err={rel_err:.3e} (prev={prev_particles_mass_mgC:.4f}, inflow={inflow_total_step:.4f}, outflow={outflow_mgC_step:.4f}, absorbed={absorbed_mgC_step:.4f}, quantﾎ・{quant_delta_mgC:.4f}, curr={curr_particles_mass_mgC:.4f})")

        # 蜿ｯ隕門喧・井ｻｻ諢・繝ｩ繧､繝厄ｼ・
        if live_plot_interval > 0 and (step % live_plot_interval == 0):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor("#d0f7ff")
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw / 2, ry - rh / 2), rw, rh, color="gray", alpha=0.7)
                ax.add_patch(rock_patch)
            # 讀咲黄縺ｮ菴咲ｽｮ
            for plant in plants:
                circ = plt.Circle((plant.x, plant.y), plant.radius, color="green", alpha=0.3)
                ax.add_patch(circ)
            # 邊貞ｭ・
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

        # 蜷郁ｨ亥､
        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)

        # 繧ｹ繝・ャ繝励＃縺ｨ縺ｮ蜷郁ｨ医ｒ險倬鹸
        step_absorbed_series.append(step_absorbed)
        step_fixed_series.append(step_fixed)
        step_growth_series.append(step_growth)
        particle_count_series.append(int(len(particles)))

        # 遞ｮ縺斐→縺ｮ邏ｯ遨阪ｒ險倬鹸
        for plant in plants:
            series = species_series[plant.name]
            series["total_absorbed"].append(plant.total_absorbed)
            series["total_fixed"].append(plant.total_fixed)
            series["total_growth"].append(plant.total_growth)

    # ===== 邨先棡髮・ｨ・=====
    species_fixed_totals = {plant.name: (plant.total_fixed * float(particle_mass_mgC)) for plant in plants}
    print("\n=== 蜷郁ｨ亥崋螳咾O2驥擾ｼ域､咲黄遞ｮ蛻･・・===")
    for species, total_mgC in species_fixed_totals.items():
        print(f"{species}: {total_mgC:.2f} mgC")

    # 雉ｪ驥丞庶謾ｯ繝√ぉ繝・け・亥・譛滂ｼ区ｵ∝・・晄ｮ矩㍼・区ｵ∝・・・
    current_particle_mass = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
    plant_absorbed = float(sum(p.total_absorbed for p in plants)) * float(particle_mass_mgC)
    plant_fixed = float(sum(p.total_fixed for p in plants)) * float(particle_mass_mgC)

    total_injected = float(mass_initial + mass_inflow)
    total_outflow = float(mass_outflow)
    total_remaining = float(current_particle_mass + plant_absorbed + loss_quant_mgC)

    # 蜿取髪隱､蟾ｮ繧堤ｮ怜・縺励※陦ｨ遉ｺ・・gC蜊倅ｽ搾ｼ・
    balance_error = 0.0 if total_injected <= 1e-9 else abs(total_injected - (total_remaining + total_outflow)) / total_injected
    print(
        f"Mass balance: Injected={total_injected:.2f} mgC, "
        f"Absorbed={plant_absorbed:.2f} mgC, Fixed={plant_fixed:.2f} mgC, "
        f"Outflow={total_outflow:.2f} mgC, Remaining={total_remaining:.2f} mgC, "
        f"Quantization={loss_quant_mgC:.2f} mgC, Error={balance_error*100:.2f}%"
    )

    # 邨先棡CSV菫晏ｭ假ｼ医し繝槭Μ & 遞ｮ蛻･縺斐→縺ｮ譎らｳｻ蛻・繧ｵ繝槭Μ・・
    os.makedirs("results", exist_ok=True)

    # 譌ｧ莠呈鋤: 蜷・ｨｮ縺ｮ蜷郁ｨ茨ｼ・陦靴SV・・
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
        # index 情報を蓄積（名称/スラッグ/ファイルパス）
        index_rows.append([
            plant.name,
            slug,
            f"result_{slug}_mgC.csv",
            f"time_series_{slug}_mgC.csv",
        ])

    # 譁ｰ: 蜈ｨ菴薙し繝槭Μ
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

    # 譁ｰ: 蜈ｨ菴薙・譎らｳｻ蛻・
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

    # 譁ｰ: 遞ｮ蛻･縺斐→縺ｮ譎らｳｻ蛻暦ｼ育ｴｯ遨搾ｼ・
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
        # mgC繧ｹ繧ｱ繝ｼ繝ｫ縺ｮ荳ｦ陦悟・蜉・
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

    # インデックス（どのファイルを見ればよいかを一覧化）
    try:
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

    # 蝗ｳ縺ｮ菫晏ｭ假ｼ郁ｦ九ｄ縺吶＞蜿ｯ隕門喧・・
    try:
        # 1) 遞ｮ蛻･縺斐→縺ｮ蜷郁ｨ茨ｼ域｣偵げ繝ｩ繝包ｼ峨ょ腰菴阪・ mgC 縺ｫ邨ｱ荳
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

        # 2) 蜈ｨ菴薙・譎らｳｻ蛻暦ｼ・繧ｹ繝・ャ繝励≠縺溘ｊ・・
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

        # 3) 莉｣陦ｨ3遞ｮ縺ｮ邏ｯ遨搾ｼ育ｷ夲ｼ・
        fig, ax = plt.subplots(figsize=(10, 5))
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

        # 4) 蜷・ｨｮ縺ｮ邏ｯ遨榊精蜿朱㍼・亥・9遞ｮ・・
        fig, ax = plt.subplots(figsize=(10, 6))
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
                    w.writerow(["step", "visits", "eligible", "absorptions"])  # eff0蜀・ｨｳ縺ｯ蠢・ｦ∵凾縺ｫ霑ｽ蜉
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
    蜊倥せ繝・ャ繝苓ｩ穂ｾ｡・井ｸｻ縺ｫ繝・せ繝育畑・峨・
    plants: dict・亥推遞ｮ縺ｮ繝代Λ繝｡繝ｼ繧ｿ縲・SON縺ｮ繧ｭ繝ｼ蜷阪↓蜷医ｏ縺帙ｋ・・
    co2: 縺薙・繧ｹ繝・ャ繝励〒隧穂ｾ｡蟇ｾ雎｡縺ｫ荳弱∴繧狗く邏驥擾ｼ亥酔荳蜊倅ｽ阪〒・・
    """
    results = {}
    nutrient_series = []

    # 逶ｸ蟇ｾ驟榊・縺ｮ縺溘ａ縲∝推遞ｮ縺ｮ驥阪∩・・bsorption_rate 縺ｾ縺溘・ absorption_efficiency・峨ｒ蜷郁ｨ・
    total_abs_rate = 0.0
    for _name, _params in plants.items():
        total_abs_rate += float(_params.get("absorption_rate", _params.get("absorption_efficiency", 1.0)))
    total_abs_rate = max(total_abs_rate, 1e-9)  # 0蜑ｲ髦ｲ豁｢

    step_abs_total = 0.0   # 縺薙・繧ｹ繝・ャ繝励〒縺ｮ邱丞精蜿・
    step_fix_total = 0.0   # 縺薙・繧ｹ繝・ャ繝励〒縺ｮ邱丞崋螳・

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

        # JSON縺ｮ繧ｭ繝ｼ蜷阪↓蜷医ｏ縺帙ｋ・育嶌蟇ｾ驟榊・・・
        weight   = float(params.get("absorption_rate", params.get("absorption_efficiency", 1.0)))
        share    = weight / total_abs_rate
        fix_rate = params.get("fixation_rate", params.get("fixation_ratio", 0.7))
        growth_r = params.get("growth_rate", 0.0)

        # 縺昴・遞ｮ縺ｮ蜷ｸ蜿朱㍼ = 縺薙・繧ｹ繝・ャ繝励・CO2 ﾃ・蜉ｹ邇・ﾃ・逶ｸ蟇ｾ繧ｷ繧ｧ繧｢
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

    # 縺薙・繧ｹ繝・ャ繝怜・縺ｮ蜷郁ｨ医ｂ霑斐☆・亥庶謾ｯ髮・ｨ育畑・・
    return results, nutrient_series, step_abs_total, step_fix_total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref1", action="store_true", help="Run 1-step preflight visualization and exit")
    args = parser.parse_args()
    run_simulation(pref1=bool(args.pref1))
    # 髱櫁｡ｨ遉ｺ繝｢繝ｼ繝峨〒縺ｯ髱槫ｯｾ隧ｱ繝舌ャ繧ｯ繧ｨ繝ｳ繝会ｼ・I/繧ｵ繝ｼ繝舌・蜷代￠・・
    try:
        if not show_plots:
            plt.switch_backend("Agg")
    except Exception:
        pass

    # 繝ｩ繝ｳ縺斐→縺ｫ謾ｯ驟咲紫繧ｬ繝ｼ繝臥憾諷九ｒ繝ｪ繧ｻ繝・ヨ
    try:
        _dom_state.clear()
    except Exception:
        pass

