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

def apply_depth_filter(eff: float, plant, env) -> float:
    """
    深度の適地フィルタ。environment が返す depth_m と
    plant.model_depth_range を使って、範囲外なら効率を0にする。
    """
    dmin, dmax = getattr(plant, "model_depth_range", (0, 999))
    depth_m = float(env.get("depth_m", 0.0))
    return 0.0 if not (dmin <= depth_m <= dmax) else eff

def seasonal_inflow(step, total_steps, base=30):
    """季節性のCO₂流入量（正弦波変動）"""
    cycle = 2 * np.pi * step / total_steps
    return int(base * (0.5 + 0.5 * np.sin(cycle)))

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

    # 植物プロファイル読み込み
    with open("data/plants.json", "r") as f:
        profiles = json.load(f)

    plant_positions = {
        "Seagrass": {"x": 20, "y": 95, "radius": 5},
        "Kelp": {"x": 50, "y": 85, "radius": 7},
        "Chlorella": {"x": 80, "y": 10, "radius": 3},
    }

    plants = []
    for plant_type, profile in profiles.items():
        pos = plant_positions.get(plant_type, {"x": 50, "y": 95, "radius": 3})
        plants.append(
            Plant(
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
        )

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
    for step in range(total_steps):

        # ステップごとに環境評価をキャッシュ
        plant_env, plant_eff = {}, {}
        for i, plant in enumerate(plants):
            env = get_environmental_factors(
                plant.x, plant.y, step,
                total_steps=total_steps, width=width, height=height,
                salinity_mode="estuary"  # 汽水域にしたいときだけ
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

        # 粒子拡散
        flow_field = generate_dynamic_flow_field(width, height, step)
        particles = diffuse_particles(particles, terrain, flow_field)

        # 吸収処理（保存則を守る）
        remaining_particles = []
        for particle in particles:
            if particle.y > height - 5:
                for plant in plants:
                    env = plant_env[plant.name]
                    eff = plant_eff[plant.name]
                    uptake_ratio = eff * getattr(plant, "absorption_efficiency", 1.0)
                    uptake_ratio = min(max(uptake_ratio, 0.0), 1.0)
                    if uptake_ratio > 0:
                        absorb_amount = particle.mass * uptake_ratio
                        plant.absorb(absorb_amount, efficiency_score=eff)
                        particle.mass -= absorb_amount
                        if particle.mass <= 1e-12:
                            break
                if particle.mass > 1e-12:
                    remaining_particles.append(particle)
            else:
                remaining_particles.append(particle)
        particles = np.array(remaining_particles, dtype=object)

        # 新規流入
        num_new = seasonal_inflow(step, total_steps, base=30)
        particles = inject_particles(particles, terrain, num_new_particles=num_new)

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
