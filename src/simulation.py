# src/simulation.py
import csv
import json
import os
import random
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
    深度の適地フィルタ。
    潮間帯（Spartina/Rhizophora）は水面直上～干出帯のため深度フィルタを適用しない。
    """
    if getattr(plant, "name", "") in ("Spartina alterniflora", "Rhizophora spp."):
        return eff  # 深度条件を外す
    dmin, dmax = getattr(plant, "model_depth_range", (0, 999))
    depth_m = float(env.get("depth_m", 0.0))
    return 0.0 if not (dmin <= depth_m <= dmax) else eff

def seasonal_inflow(step, total_steps, base_mgC_per_step=30.0, particle_mass_mgC=1.0):
    """季節性のCO₂流入（mgC/step）を粒子数に変換して返す"""
    cycle = 2 * np.pi * step / total_steps
    mgc = base_mgC_per_step * (0.5 + 0.5 * np.sin(cycle))
    count = int(round(mgc / max(particle_mass_mgC, 1e-9)))
    return max(count, 0)
with open("data/plants.json") as f:
    profiles_raw = json.load(f)
profiles = normalize_profiles(profiles_raw)


def _slugify(name: str) -> str:
    """ファイル名用に安全なスラッグへ変換"""
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
    """シミュレーションを実行し、各種シリーズと結果を返す"""

    # 地形と深度マップ
    terrain, depth_map = create_terrain(width, height)

    # 出力用シリーズ
    env_series: List[float] = []
    nutrient_series: List[float] = []

    # 累積（全体）
    internal_series: List[float] = []  # 全種の total_growth 合計（累積）
    fixed_series: List[float] = []     # 全種の total_fixed 合計（累積）
    released_series: List[float] = []  # 予約（未使用）
    carbon_series: List[float] = []    # 同上（予約）

    # ステップ毎（全体）
    step_absorbed_series: List[float] = []
    step_fixed_series: List[float] = []
    step_growth_series: List[float] = []
    particle_count_series: List[int] = []

    # 代表3種（後方互換用）
    zostera_fixed_series, kelp_fixed_series, chlorella_fixed_series = [], [], []
    zostera_growth_series, kelp_growth_series, chlorella_growth_series = [], [], []
    zostera_absorbed_series, kelp_absorbed_series, chlorella_absorbed_series = [], [], []
    os.makedirs("results", exist_ok=True)

    # 設定読み込み（単位等）
    cfg = load_config()
    particle_mass_mgC = float(cfg.get("particle_mass_mgC", 1.0))
    inflow_mgC_per_step_base = float(cfg.get("inflow_mgC_per_step_base", 30.0))
    chl_mortality = float(cfg.get("chl_mortality_rate", 0.02))  # 2%/step
    live_plot_interval = int(cfg.get("live_plot_interval", 0))   # 0ならライブ描画なし
    show_plots = bool(cfg.get("show_plots", False))              # Trueでplt.show
    debug_mass_check = bool(cfg.get("debug_mass_check", False))  # 毎ステップの質量保存チェック
    flow_scale = float(cfg.get("flow_scale", 1.0))               # 流速スケール（感度確認用）
    # サニティ用の均一モードの有無
    sanity_uniform = bool(cfg.get("sanity_uniform", False))
    # 潮間帯の稼働率（Spartina/Rhizophora の水没時間をモデル化）
    tidal_period_steps = int(cfg.get("tidal_period_steps", 24))
    intertidal_submergence_fraction = float(cfg.get("intertidal_submergence_fraction", 0.35))
    intertidal_shallow_band_m = float(cfg.get("intertidal_shallow_band_m", 1.0))

    # 配置（9種の植物のレイアウト）
    plant_positions = {
        # 低塩→高塩の中間以上へ移動（y はそのまま/半径少し増やす）
        "Zostera marina":            {"x": 30, "y": 28, "radius": 11},  # 半径を1だけ縮小
        "Halophila ovalis":          {"x": 45, "y": 26, "radius": 10},  # 半径+2で遭遇率↑（≥15 PSU）
        "Posidonia oceanica":        {"x": 95, "y": 30, "radius": 12},  # 半径拡大で遭遇率↑
        "Macrocystis pyrifera":      {"x": 85, "y": 80, "radius": 9},   # 半径拡大で遭遇率↑
        "Saccharina japonica":       {"x": 75, "y": 82, "radius": 9},   # 半径拡大で遭遇率↑
        "Chlorella vulgaris":        {"x": 10, "y": 14, "radius": 3},   # 0–10 PSU
        "Nannochloropsis gaditana":  {"x": 70, "y": 16, "radius": 4},   # ≥20 PSU（OK）
        "Spartina alterniflora":     {"x": 30, "y": 6,  "radius": 8},   # ≥10 PSU（左から右へ）
        "Rhizophora spp.":           {"x": 20, "y": 7,  "radius": 8},   # ≥5 PSU（少し右へ）
    }

    # 粒子の注入位置（境界×層に固定。種直上は使わない）
    injection_sources = [
        (2, int(height * 0.20)),   # 左・表層（Chlorella/Halophila 対応）
        (2, int(height * 0.50)),   # 左・中層
        (width - 2, int(height * 0.15)),  # 右・表層
        (width - 2, int(height * 0.30)),  # 右・中浅
        (width - 2, int(height * 0.80)),  # 右・深め
    ]

    # ===== Guard config & schema (lightweight validation) =====
    guard_cfg = load_config(guard_config_path) if os.path.exists(guard_config_path) else {}
    min_dist_px = int(guard_cfg.get("min_dist_px", 3))
    gate_enabled = bool(guard_cfg.get("gate_enabled", False))
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

    # 対象種を9種に拡張
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
        # depth range を属性として付与
        setattr(p, "model_depth_range", tuple(profile.get("model_depth_range", (1, 6))))
        plants.append(p)

    # 初期粒子
    particles = initialize_particles(num_particles, terrain)

    # 岩オブジェクト
    rocks = [
        {"x": width // 2, "y": int(height * 0.5), "w": 12, "h": 8},
        {"x": int(width * 0.7), "y": int(height * 0.3), "w": 8, "h": 5},
    ]

    # 底質マップ
    bottom_type_map = np.full((height, width), "mud", dtype=object)
    bottom_type_map[90:100, 0:33] = "mud"
    bottom_type_map[90:100, 33:66] = "sand"
    bottom_type_map[90:100, 66:100] = "rock"

    np.random.seed(42)
    random.seed(42)

    # Geometry guard: adjust or abort if any source collides with plant footprints
    injection_sources = validate_geometry(injection_sources, plants, width, height, margin_px=min_dist_px)

    # フェアネス再重み付けは行わない（供給は環境に依存、種には依存しない）

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
                    x = sx + np.random.normal(scale=1.0)
                    y = sy + np.random.normal(scale=1.0)
                    if 0 <= x < width and 0 <= y < height and terrain[int(y), int(x)] == 1:
                        p0.append(Particle(x=x, y=y, mass=1.0, origin="pref1", x0=x, y0=y))
        save_pref1_snapshot((height, width), plants, injection_sources, p0, outdir="outputs")
        return (), (), (), (), (), [], [], [], [], [], []

    # ===== メインループ =====
    # 物理尺度（environment と整合）
    KD = float(cfg.get("kd_m_inv", 0.8))
    MAX_DEPTH_M = 8.0
    meters_per_pixel = MAX_DEPTH_M / max((height - 1), 1)
    # プランクトンの水平広がり（m）→ ピクセル換算
    plankton_radius_m = float(cfg.get("plankton_radius_m", 0.9))
    plankton_radius_px = plankton_radius_m / max(meters_per_pixel, 1e-9)
    # フォトゾーン近似（e^-kd z = 0.1 → z ≈ 2.3/kd）
    euphotic_depth_m = 2.3 / max(KD, 1e-6)
    euphotic_px = int(euphotic_depth_m / meters_per_pixel)
    intertidal_shallow_band_px = intertidal_shallow_band_m / max(meters_per_pixel, 1e-9)

    # 質量バランス用（mgC単位）
    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_initial = float(len(particles)) * float(particle_mass_mgC)
    loss_quant_mgC = 0.0  # reinjection rounding loss (mgC)
    # 種別ごとの累積シリーズ（全ステップ）
    species_series: Dict[str, Dict[str, List[float]]] = {
        name: {"total_absorbed": [], "total_fixed": [], "total_growth": []}
        for name in target_species
    }

    # Metrics accumulators
    source_labels = [f"src{i}" for i in range(len(injection_sources))]
    capture_matrix: Dict[str, Dict[str, float]] = {lab: {name: 0.0 for name in target_species} for lab in source_labels + ["init"]}
    abs_dy_samples: List[float] = []
    travel_weighted_sum = 0.0
    travel_weight = 0.0

    for step in range(total_steps):
        # 質量（mgC）スナップショット（ステップ前）
        prev_particles_mass_mgC = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
        prev_loss_quant_mgC = float(loss_quant_mgC)

        # ステップごとに環境評価をキャッシュ
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
                    salinity_mode="linear_x",  # 汽水域: 左低塩→右高塩
                    S_min=0.0, S_max=35.0,      # 左端を低塩（淡水寄り）に設定
                    kd_m_inv=KD, max_depth_m=MAX_DEPTH_M,
                )
            px, py = int(plant.x), int(plant.y)
            bottom_type = bottom_type_map[py, px] if (0 <= py < height and 0 <= px < width) else "mud"

            eff = compute_efficiency_score(plant, env, bottom_type=bottom_type)
            eff = apply_depth_filter(eff, plant, env)  # 深度レンジ外は0
            plant_env[plant.name] = env
            plant_eff[plant.name] = eff

            if i == 0:
                env_series.append(eff)
                nutrient_series.append(env["nutrient"])

        # 植物ごとの累積量
        # 累積シリーズ（代表3種は後方互換名称に合わせて記録）
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

        # 粒子拡散（開境界で流出をカウント）
        if sanity_uniform:
            flow_field = np.zeros((height, width, 2))
            flow_field[:, :, 0] = 0.05 * float(flow_scale)
        else:
            flow_field = generate_dynamic_flow_field(width, height, step, scale=flow_scale)
        particles, outflow_mass_step = diffuse_particles(particles, terrain, flow_field)
        outflow_mgC_step = float(outflow_mass_step) * float(particle_mass_mgC)
        mass_outflow += outflow_mgC_step

        # 吸収処理（保存則を守る・競合按分）
        debug_hits = {p.name: {"visits": 0, "eligible": 0, "absorptions": 0} for p in plants}
        remaining_particles = []
        step_absorbed = 0.0
        step_fixed = 0.0
        step_growth = 0.0
        for particle in particles:
            # 1) この粒子に対して吸収可能な植物候補を列挙
            candidates = []  # (plant, uptake_ratio)
            for plant in plants:
                debug_hits[plant.name]["visits"] += 1
                env = plant_env[plant.name]
                eff = plant_eff[plant.name]
                if eff <= 0.0:
                    continue
                name = plant.name
                dx = particle.x - plant.x
                dy = particle.y - plant.y
                r2 = dx * dx + dy * dy
                within_radius = r2 <= (plant.radius ** 2)

                allowed = False
                if name in ("Chlorella vulgaris",):
                    # 有光層かつコロニー半径内のみ吸収
                    within_colony = r2 <= (plankton_radius_px ** 2)
                    allowed = (particle.y <= euphotic_px) and within_colony
                elif name in ("Macrocystis pyrifera", "Saccharina japonica"):
                    # コンセプト: ホルドファスト（海底付近）とキャノピー（表層）での捕捉を分ける。
                    kelp_band_m = 5.0          # 海底近傍の鉛直作用帯（±m, やや拡大）
                    surface_band_m = 3.0       # 表層の作用帯（0..m, キャノピー厚め）
                    kelp_band_px = kelp_band_m / meters_per_pixel
                    surface_band_px = surface_band_m / meters_per_pixel
                    within_band = abs(dy) <= kelp_band_px
                    near_surface = particle.y <= surface_band_px
                    horizontal_ok = abs(dx) <= plant.radius
                    # 海底近傍: 完全な半径内 AND 作用帯内
                    # 表層帯: 鉛直距離は不問、水平半径内であれば吸収可（キャノピー）
                    allowed = (within_radius and within_band) or (horizontal_ok and near_surface)
                else:
                    if name in ("Spartina alterniflora", "Rhizophora spp."):
                        # 潮間帯: 表層・半径内は候補化し、吸収比に連続ゲートを掛ける
                        shallow_ok = particle.y <= intertidal_shallow_band_px
                        allowed = shallow_ok and within_radius
                    else:
                        # 一般の海草: 半径内 + やや広い鉛直帯（±4 m）
                        sg_band_m = 4.0
                        if name == "Zostera marina":
                            sg_band_m = 4.0
                        sg_band_px = sg_band_m / meters_per_pixel
                        allowed = within_radius and (abs(dy) <= sg_band_px)

                if not allowed:
                    continue

                uptake_ratio = eff * getattr(plant, "absorption_efficiency", 1.0)
                # 潮間帯の連続ゲート（0..1）
                if name in ("Spartina alterniflora", "Rhizophora spp."):
                    phase = 2.0 * np.pi * ((step % max(tidal_period_steps, 1)) / max(tidal_period_steps, 1))
                    submergence = 0.5 + 0.5 * np.sin(phase)
                    thr = 1.0 - intertidal_submergence_fraction
                    gate = (submergence - thr) / max(intertidal_submergence_fraction, 1e-9)
                    gate = float(min(max(gate, 0.0), 1.0))
                    uptake_ratio *= gate
                uptake_ratio = float(min(max(uptake_ratio, 0.0), 1.0))
                if uptake_ratio > 0.0 and allowed:
                    debug_hits[plant.name]["eligible"] += 1
                if uptake_ratio > 0.0:
                    candidates.append((plant, uptake_ratio))

            # 2) 候補が無ければそのまま残す
            if not candidates or particle.mass <= 1e-12:
                if particle.mass > 1e-12:
                    remaining_particles.append(particle)
                continue

            # 3) 吸収を候補間で按分（重み=uptake_ratio）
            total_u = sum(u for _, u in candidates)
            if total_u <= 1e-12:
                remaining_particles.append(particle)
                continue

            # 粒子から引き抜く総量は "粒子質量 × min(total_u, 1)"
            take_total = particle.mass * min(total_u, 1.0)
            # 各候補への配分
            for plant, u in candidates:
                share = (u / total_u) * take_total
                if share <= 0.0:
                    continue
                absorbed, fixed, growth = plant.absorb(share)
                debug_hits[plant.name]["absorptions"] += 1
                # plant.absorb は share をそのまま消費する前提（保存則）。
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
        # ステップ終端で診断シリーズに追記
        if diag_enabled:
            for p in plants:
                ds = debug_hits[p.name]
                diag_series[p.name]["visits"].append(int(ds["visits"]))
                diag_series[p.name]["eligible"].append(int(ds["eligible"]))
                diag_series[p.name]["absorptions"].append(int(ds["absorptions"]))

        # 新規流入（等分配）: 起源ラベル付きで注入し、縦ゲートの有効性も計測
        num_new = seasonal_inflow(step, total_steps, base_mgC_per_step=inflow_mgC_per_step_base, particle_mass_mgC=particle_mass_mgC)
        if num_new > 0:
            # 重み付け（左表層に30%を配分、他は均等）
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
                    x = sx + np.random.normal(scale=1.0)
                    y = sy + np.random.normal(scale=1.0)
                    if 0 <= x < width and 0 <= y < height and terrain[int(y), int(x)] == 1:
                        new_particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0, origin=lab, x0=x, y0=y))
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

        # プランクトン（Chlorella）の自然死亡・再放出（CO2復帰）
        reinj_mgC_step = 0.0
        for plant in plants:
            if plant.name == "Chlorella vulgaris" and plant.total_growth > 0:
                mortal = plant.total_growth * chl_mortality
                if mortal > 0:
                    plant.total_growth -= mortal
                    # mgC を粒子数に変換して、その場に再注入
                    n_rel = int(round(mortal / max(particle_mass_mgC, 1e-9)))
                    # 再注入の丸め誤差（mgC）を蓄積：正負どちらも取りうる
                    loss_quant_mgC += float(mortal) - float(n_rel) * float(particle_mass_mgC)
                    if n_rel > 0:
                        # その場に起源ラベル付きで再注入（origin="reinj_<plant>")
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

        # ステップ末の質量保存チェック（任意）
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
                print(f"[mass-check] step={step} rel_err={rel_err:.3e} (prev={prev_particles_mass_mgC:.4f}, inflow={inflow_total_step:.4f}, outflow={outflow_mgC_step:.4f}, absorbed={absorbed_mgC_step:.4f}, quantΔ={quant_delta_mgC:.4f}, curr={curr_particles_mass_mgC:.4f})")

        # 可視化（任意/ライブ）
        if live_plot_interval > 0 and (step % live_plot_interval == 0):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor("#d0f7ff")
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw / 2, ry - rh / 2), rw, rh, color="gray", alpha=0.7)
                ax.add_patch(rock_patch)
            # 植物の位置
            for plant in plants:
                circ = plt.Circle((plant.x, plant.y), plant.radius, color="green", alpha=0.3)
                ax.add_patch(circ)
            # 粒子
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

        # 合計値
        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)

        # ステップごとの合計を記録
        step_absorbed_series.append(step_absorbed)
        step_fixed_series.append(step_fixed)
        step_growth_series.append(step_growth)
        particle_count_series.append(int(len(particles)))

        # 種ごとの累積を記録
        for plant in plants:
            series = species_series[plant.name]
            series["total_absorbed"].append(plant.total_absorbed)
            series["total_fixed"].append(plant.total_fixed)
            series["total_growth"].append(plant.total_growth)

    # ===== 結果集計 =====
    species_fixed_totals = {plant.name: (plant.total_fixed * float(particle_mass_mgC)) for plant in plants}
    print("\n=== 合計固定CO2量（植物種別） ===")
    for species, total_mgC in species_fixed_totals.items():
        print(f"{species}: {total_mgC:.2f} mgC")

    # 質量収支チェック（初期＋流入＝残量＋流出）
    current_particle_mass = (float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0) * float(particle_mass_mgC)
    plant_absorbed = float(sum(p.total_absorbed for p in plants)) * float(particle_mass_mgC)
    plant_fixed = float(sum(p.total_fixed for p in plants)) * float(particle_mass_mgC)

    total_injected = float(mass_initial + mass_inflow)
    total_outflow = float(mass_outflow)
    total_remaining = float(current_particle_mass + plant_absorbed + loss_quant_mgC)

    # 収支誤差を算出して表示（mgC単位）
    balance_error = 0.0 if total_injected <= 1e-9 else abs(total_injected - (total_remaining + total_outflow)) / total_injected
    print(
        f"Mass balance: Injected={total_injected:.2f} mgC, "
        f"Absorbed={plant_absorbed:.2f} mgC, Fixed={plant_fixed:.2f} mgC, "
        f"Outflow={total_outflow:.2f} mgC, Remaining={total_remaining:.2f} mgC, "
        f"Quantization={loss_quant_mgC:.2f} mgC, Error={balance_error*100:.2f}%"
    )

    # 結果CSV保存（サマリ & 種別ごとの時系列/サマリ）
    os.makedirs("results", exist_ok=True)

    # 旧互換: 各種の合計（1行CSV）
    for plant in plants:
        # Raw units (particle counts); kept for backward compatibility
        with open(os.path.join("results", f"result_{plant.name}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_absorbed", "total_fixed", "total_growth"])  # raw units
            writer.writerow([plant.total_absorbed, plant.total_fixed, plant.total_growth])
        # Slugged filename in mgC (new)
        slug = _slugify(plant.name)
        with open(os.path.join("results", f"result_{slug}_mgC.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_absorbed_mgC", "total_fixed_mgC", "total_growth_mgC"])
            writer.writerow([
                plant.total_absorbed * float(particle_mass_mgC),
                plant.total_fixed * float(particle_mass_mgC),
                plant.total_growth * float(particle_mass_mgC),
            ])

    # 新: 全体サマリ
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

    # 新: 全体の時系列
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

    # 新: 種別ごとの時系列（累積）
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
        # mgCスケールの並行出力
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

    # 図の保存（見やすい可視化）
    try:
        # 1) 種別ごとの合計（棒グラフ）。単位は mgC に統一
        species_fixed_totals_mgC = {p.name: (p.total_fixed * float(particle_mass_mgC)) for p in plants}
        fig, ax = plt.subplots(figsize=(10, 5))
        species = list(species_fixed_totals_mgC.keys())
        totals = [species_fixed_totals_mgC[s] for s in species]
        ax.bar(species, totals)
        ax.set_xticklabels(species, rotation=45, ha="right")
        ax.set_ylabel("Total Fixed CO2 [mgC]")
        ax.set_title("Total CO2 Fixed by Species")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "total_fixed_by_species.png"))
        plt.close(fig)

        # 2) 全体の時系列（1ステップあたり）
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(step_absorbed_series, label="absorbed/step")
        ax.plot(step_fixed_series, label="fixed/step")
        ax.plot(step_growth_series, label="growth/step")
        ax.set_xlabel("Step")
        ax.set_ylabel("mgC")
        ax.set_title("System Totals per Step")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "overall_timeseries.png"))
        plt.close(fig)

        # 3) 代表3種の累積（線）
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(zostera_fixed_series, label="Zostera marina (fixed)")
        ax.plot(kelp_fixed_series, label="Macrocystis pyrifera (fixed)")
        ax.plot(chlorella_fixed_series, label="Chlorella vulgaris (fixed)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Fixed [mgC]")
        ax.set_title("Cumulative Fixed CO2: Selected Species")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "selected_species_cumulative_fixed.png"))
        plt.close(fig)

        # 4) 各種の累積吸収量（全9種）
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, series in species_series.items():
            ax.plot(series["total_absorbed"], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Absorbed CO2")
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
                    w.writerow(["step", "visits", "eligible", "absorptions"])  # eff0内訳は必要時に追加
                    steps = range(total_steps)
                    vs = diag_series[name]["visits"] if name in diag_series else []
                    es = diag_series[name]["eligible"] if name in diag_series else []
                    as_ = diag_series[name]["absorptions"] if name in diag_series else []
                    for i in steps:
                        v = vs[i] if i < len(vs) else 0
                        e = es[i] if i < len(es) else 0
                        a = as_[i] if i < len(as_) else 0
                        w.writerow([i, v, e, a])
            # injection counts
            if inj_series:
                inj_path = os.path.join("outputs/diagnostics", "injection_counts.csv")
                with open(inj_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step"] + source_labels)
                    for row in inj_series:
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
    単ステップ評価（主にテスト用）。
    plants: dict（各種のパラメータ。JSONのキー名に合わせる）
    co2: このステップで評価対象に与える炭素量（同一単位で）
    """
    results = {}
    nutrient_series = []

    # 相対配分のため、各種の重み（absorption_rate または absorption_efficiency）を合計
    total_abs_rate = 0.0
    for _name, _params in plants.items():
        total_abs_rate += float(_params.get("absorption_rate", _params.get("absorption_efficiency", 1.0)))
    total_abs_rate = max(total_abs_rate, 1e-9)  # 0割防止

    step_abs_total = 0.0   # このステップでの総吸収
    step_fix_total = 0.0   # このステップでの総固定

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

        # JSONのキー名に合わせる（相対配分）
        weight   = float(params.get("absorption_rate", params.get("absorption_efficiency", 1.0)))
        share    = weight / total_abs_rate
        fix_rate = params.get("fixation_rate", params.get("fixation_ratio", 0.7))
        growth_r = params.get("growth_rate", 0.0)

        # その種の吸収量 = このステップのCO2 × 効率 × 相対シェア
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

    # このステップ分の合計も返す（収支集計用）
    return results, nutrient_series, step_abs_total, step_fix_total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref1", action="store_true", help="Run 1-step preflight visualization and exit")
    args = parser.parse_args()
    run_simulation(pref1=bool(args.pref1))
    # 非表示モードでは非対話バックエンド（CI/サーバー向け）
    try:
        if not show_plots:
            plt.switch_backend("Agg")
    except Exception:
        pass

    # ランごとに支配率ガード状態をリセット
    try:
        _dom_state.clear()
    except Exception:
        pass
