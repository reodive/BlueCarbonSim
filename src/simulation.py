# src/simulation.py
import csv
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from models.particle import initialize_particles
from models.plant import Plant
from environment import get_environmental_factors, compute_efficiency_score
from terrain import create_terrain
from particle_flow import diffuse_particles, inject_particles, generate_dynamic_flow_field
from utils.config import load_config

def apply_depth_filter(eff: float, plant, env) -> float:
    """
    深度の適地フィルタ。environment が返す depth_m と
    plant.model_depth_range を使って、範囲外なら効率を0にする。
    """
    dmin, dmax = getattr(plant, "model_depth_range", (0, 999))
    depth_m = float(env.get("depth_m", 0.0))
    return 0.0 if not (dmin <= depth_m <= dmax) else eff

def seasonal_inflow(step, total_steps, base_mgC_per_step=30.0, particle_mass_mgC=1.0):
    """季節性のCO₂流入（mgC/step）を粒子数に変換して返す"""
    cycle = 2 * np.pi * step / total_steps
    mgc = base_mgC_per_step * (0.5 + 0.5 * np.sin(cycle))
    count = int(round(mgc / max(particle_mass_mgC, 1e-9)))
    return max(count, 0)

from utils.profile_adapter import normalize_profiles
with open("data/plants.json") as f:
    profiles_raw = json.load(f)
profiles = normalize_profiles(profiles_raw)
def run_simulation(total_steps: int = 150, num_particles: int = 1000, width: int = 100, height: int = 100):
    """シミュレーションを実行し、各種シリーズと結果を返す"""

    # 地形と深度マップ
    terrain, depth_map = create_terrain(width, height)

    # 出力用シリーズ
    env_series, nutrient_series = [], []
    internal_series, fixed_series, released_series, carbon_series = [], [], [], []
    zostera_fixed_series, kelp_fixed_series, chlorella_fixed_series = [], [], []
    zostera_growth_series, kelp_growth_series, chlorella_growth_series = [], [], []
    zostera_absorbed_series, kelp_absorbed_series, chlorella_absorbed_series = [], [], []
    os.makedirs("results", exist_ok=True)

    # 設定読み込み（単位等）
    cfg = load_config()
    particle_mass_mgC = float(cfg.get("particle_mass_mgC", 1.0))
    inflow_mgC_per_step_base = float(cfg.get("inflow_mgC_per_step_base", 30.0))
    chl_mortality = float(cfg.get("chl_mortality_rate", 0.02))  # 2%/step

    # 配置（Kelp は rock 帯、Zostera は mud 帯、Chlorella は表層）
    plant_positions = {
        "Zostera marina": {"x": 20, "y": 95, "radius": 5},
        "Macrocystis pyrifera": {"x": 85, "y": 85, "radius": 7},
        "Chlorella vulgaris": {"x": 80, "y": 10, "radius": 3},
    }

    # 対象種を3種に絞る（シリーズ整合のため）
    target_species = ["Zostera marina", "Macrocystis pyrifera", "Chlorella vulgaris"]
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

    # ===== メインループ =====
    # 物理尺度（environment と整合）
    KD = 0.8
    MAX_DEPTH_M = 8.0
    meters_per_pixel = MAX_DEPTH_M / max((height - 1), 1)
    # フォトゾーン近似（e^-kd z = 0.1 → z ≈ 2.3/kd）
    euphotic_depth_m = 2.3 / max(KD, 1e-6)
    euphotic_px = int(euphotic_depth_m / meters_per_pixel)

    # 質量バランス用
    mass_initial = float(len([] if particles is None else []))  # set later after init
    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_initial = float(num_particles)
    for step in range(total_steps):

        # ステップごとに環境評価をキャッシュ
        plant_env, plant_eff = {}, {}
        for i, plant in enumerate(plants):
            env = get_environmental_factors(
                plant.x, plant.y, step,
                total_steps=total_steps, width=width, height=height,
                salinity_mode="linear_x",  # 汽水域: 左低塩→右高塩
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
        zostera_fixed_series.append(plants[0].total_fixed)
        kelp_fixed_series.append(plants[1].total_fixed)
        chlorella_fixed_series.append(plants[2].total_fixed)
        zostera_growth_series.append(plants[0].total_growth)
        kelp_growth_series.append(plants[1].total_growth)
        chlorella_growth_series.append(plants[2].total_growth)
        zostera_absorbed_series.append(plants[0].total_absorbed)
        kelp_absorbed_series.append(plants[1].total_absorbed)
        chlorella_absorbed_series.append(plants[2].total_absorbed)

        # 粒子拡散（開境界で流出をカウント）
        flow_field = generate_dynamic_flow_field(width, height, step)
        particles, outflow_mass_step = diffuse_particles(particles, terrain, flow_field)
        mass_outflow += float(outflow_mass_step)

        # 吸収処理（保存則を守る）
        remaining_particles = []
        absorbed_total = 0.0
        for particle in particles:
            absorbed_here = False
            for plant in plants:
                env = plant_env[plant.name]
                eff = plant_eff[plant.name]
                if eff <= 0.0:
                    continue
                # 種別別の吸収ゾーン
                name = plant.name
                # 横方向の近接（根系/群落の空間範囲）
                dx = particle.x - plant.x
                dy = particle.y - plant.y
                r2 = dx * dx + dy * dy
                within_radius = r2 <= (plant.radius ** 2)

                allowed = False
                if name in ("Chlorella vulgaris",):
                    # プランクトン: フォトゾーン（上層）で吸収可能
                    allowed = (particle.y <= euphotic_px)
                elif name in ("Macrocystis pyrifera", "Saccharina japonica"):
                    # ケルプ: 群落周囲 + 表層キャノピー
                    kelp_band_m = 4.0
                    surface_band_m = 1.5
                    kelp_band_px = kelp_band_m / meters_per_pixel
                    surface_band_px = surface_band_m / meters_per_pixel
                    within_band = abs(dy) <= kelp_band_px and within_radius
                    near_surface = particle.y <= surface_band_px
                    allowed = within_band or near_surface
                else:
                    # 海草: 群落の周囲の浅場のみ
                    sg_band_m = 2.0
                    sg_band_px = sg_band_m / meters_per_pixel
                    allowed = within_radius and (abs(dy) <= sg_band_px)

                if not allowed:
                    continue

                uptake_ratio = eff * getattr(plant, "absorption_efficiency", 1.0)
                uptake_ratio = min(max(uptake_ratio, 0.0), 1.0)
                if uptake_ratio > 0 and particle.mass > 0:
                    absorb_amount = particle.mass * uptake_ratio
                    plant.absorb(absorb_amount)
                    particle.mass -= absorb_amount
                    absorbed_total += absorb_amount
                    absorbed_here = True
                    if particle.mass <= 1e-12:
                        break
            if particle.mass > 1e-12:
                remaining_particles.append(particle)
        particles = np.array(remaining_particles, dtype=object)

        # 新規流入
        num_new = seasonal_inflow(step, total_steps, base_mgC_per_step=inflow_mgC_per_step_base, particle_mass_mgC=particle_mass_mgC)
        particles = inject_particles(particles, terrain, num_new_particles=num_new)
        mass_inflow += float(num_new)  # 1.0 質量/粒子仮定

        # プランクトン（Chlorella）の自然死亡・再放出（CO2復帰）
        for plant in plants:
            if plant.name == "Chlorella vulgaris" and plant.total_growth > 0:
                mortal = plant.total_growth * chl_mortality
                if mortal > 0:
                    plant.total_growth -= mortal
                    # mgC を粒子数に変換して、その場に再注入
                    n_rel = int(round(mortal / max(particle_mass_mgC, 1e-9)))
                    if n_rel > 0:
                        particles = inject_particles(particles, terrain, num_new_particles=n_rel, sources=[(plant.x, plant.y)])
                        mass_outflow += 0.0  # 再注入なので流出ではない

        # 可視化（任意）
        if step % 10 == 0:
            plt.clf()
            fig, ax = plt.subplots()
            ax.set_facecolor("#d0f7ff")
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw / 2, ry - rh / 2), rw, rh,
                                           color="gray", alpha=0.7)
                ax.add_patch(rock_patch)
            for i, plant in enumerate(plants):
                ax.plot([20 + i * 30, 20 + i * 30], [height - 1, height],
                        color="green", linewidth=2)
            ax.scatter([p.x for p in particles], [p.y for p in particles], c="cyan", s=5)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_xlabel("Horizontal Position")
            ax.set_ylabel("Depth (Downward)")
            ax.invert_yaxis()
            ax.set_title(f"Step {step} (Side View)")
            plt.tight_layout()
            plt.pause(0.01)
            plt.close(fig)

        # 合計値
        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)

    # ===== 結果集計 =====
    species_fixed_totals = {plant.name: plant.total_fixed for plant in plants}
    print("\n=== 合計固定CO2量（植物種別） ===")
    for species, total in species_fixed_totals.items():
        print(f"{species}: {total:.2f}")

    # 質量収支チェック
    current_particle_mass = float(sum(p.mass for p in particles)) if len(particles) > 0 else 0.0
    plant_absorbed = float(sum(p.total_absorbed for p in plants))
    expected_total = mass_initial + mass_inflow - mass_outflow
    accounted = current_particle_mass + plant_absorbed
    balance_error = 0.0 if expected_total <= 1e-9 else abs(accounted - expected_total) / expected_total
    print(f"Mass balance error: {balance_error*100:.2f}% (<=2% target)")

    # 結果CSV保存
    os.makedirs("results", exist_ok=True)
    for plant in plants:
        with open(os.path.join("results", f"result_{plant.name}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_absorbed", "total_fixed", "total_growth"])
            writer.writerow([plant.total_absorbed, plant.total_fixed, plant.total_growth])

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
    plants: dict形式の簡易パラメータセット
    """
    results = {}
    nutrient_series = []
    for plant_name, params in plants.items():
        x = params.get("x", 0)
        y = params.get("y", 0)
        env = get_environmental_factors(x, y, step, total_steps=total_steps, width=width, height=height)

        temp_sigma = 5.0
        temp_eff = np.exp(-0.5 * ((env["temperature"] - params.get("opt_temp", 20)) / temp_sigma) ** 2)
        light_eff = min(env["light"] / params.get("light_tolerance", 1.0), 1.0)

        sal_min, sal_max = params.get("salinity_range", (20, 35))
        salinity = env["salinity"]
        if sal_min <= salinity <= sal_max:
            sal_eff = 1.0
        elif salinity < sal_min:
            sal_eff = max(0, 1 - (sal_min - salinity) / 10)
        else:
            sal_eff = max(0, 1 - (salinity - sal_max) / 10)

        efficiency = temp_eff * light_eff * sal_eff
        absorbed = params.get("absorption_efficiency", 1.0) * co2 * efficiency
        growth = params.get("growth_rate", 1.0) * absorbed
        fixed = absorbed * params.get("fixation_ratio", 0.7)

        nutrient_series.append(env["nutrient"])
        results[plant_name] = {
            "absorbed": absorbed,
            "growth": growth,
            "fixed": fixed,
            "efficiency": efficiency,
            "env": env,
        }

    return results, nutrient_series
