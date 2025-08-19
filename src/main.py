import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import math
# NOTE: Refactored baseline: field-based deterministic absorption.

GRID_WIDTH = 50
GRID_HEIGHT = 50

# 新しい地形生成関数
def generate_depth_map(mode="random", width=GRID_WIDTH, height=GRID_HEIGHT):
    if mode == "flat":
        return np.full((width, height), 1.0)
    elif mode == "slope":
        return np.tile(np.linspace(0.2, 1.0, width), (height, 1)).T
    elif mode == "bay":
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2)
        return 1.0 - np.clip(radius, 0, 1)
    else:  # random
        return np.random.uniform(0.2, 1.0, size=(width, height))

# 地形マップ（0.0〜1.0 の正規化値）
depth_map_raw = generate_depth_map(mode="bay")  # Choose from: 'flat', 'bay', 'slope', 'random'
# 実際の水深(m) 1〜10m にスケーリング
depth_map = 1.0 + 9.0 * depth_map_raw  # 1–10 m
# 光量マップ（簡略: 表層1.0, 深さ線形減衰後クリップ）
light_map = 1.0 - 0.08 * depth_map_raw  # raw (0–1) を利用
light_map = np.clip(light_map, 0.0, 1.0)


import json
with open("data/plants.json", "r") as f:
    plant_params_dict = json.load(f)

class Particle:
    def __init__(self, x, y, **kwargs):
        self.x = x  # horizontal position (0 to width)
        self.y = y  # vertical position (0 at surface, increases with depth)
        self.mass = kwargs.get("mass", 1.0)
        self.form = kwargs.get("form", "CO2")
        self.reactivity = kwargs.get("reactivity", 1.0)

def create_terrain(width=100, height=100):
    # For the side view, the "terrain" is just water (1) everywhere, with optional rocks
    terrain = np.ones((height, width))
    # Add some "rock" obstacles (optional; rectangles/ellipses in the water column)
    # Example: a rock in the middle
    terrain[40:60, 45:55] = 0  # rock obstacle
    # Depth is just the y-index (vertical, 0 at surface, height-1 at bottom)
    depth_map = np.tile(np.arange(height).reshape((height, 1)), (1, width)) / (height - 1)
    return terrain, depth_map

def initialize_particles(num_particles, terrain):
    height, width = terrain.shape
    particles = []
    while len(particles) < num_particles:
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, 3)  # start near the surface (y = 0 ~ 3)
        if terrain[int(y), int(x)] == 1:
            particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0))
    return np.array(particles)

# 植物クラス（吸収効率・成長スピード・固定率を個別に持つ）
class Plant:
    def __init__(self, name, absorb_efficiency, growth_speed, fixation_ratio, release_ratio, structure_density,
                 opt_temp=20, light_tolerance=1.0, salinity_range=(20, 35), absorption_efficiency=1.0, growth_rate=1.0,
                 x=50, y=95, radius=3):
        self.name = name
        self.absorb_efficiency = absorb_efficiency  # 吸収効率
        self.growth_speed = growth_speed            # 成長スピード
        self.fixation_ratio = fixation_ratio        # 固定率
        self.release_ratio = release_ratio          # 再放出率
        self.structure_density = structure_density  # 密度係数
        self.opt_temp = opt_temp                   # 最適温度
        self.light_tolerance = light_tolerance     # 光許容度
        self.salinity_range = salinity_range       # 許容塩分範囲
        self.absorption_efficiency = absorption_efficiency # CO2吸収効率
        self.growth_rate = growth_rate
        self.x = x
        self.y = y
        self.radius = radius
        self.total_absorbed = 0
        self.total_fixed = 0
        self.total_growth = 0

    def absorb(self, base_absorption, efficiency_score=1.0):
        absorbed = self.absorb_efficiency * base_absorption * efficiency_score
        self.total_absorbed += absorbed
        fixed = absorbed * self.fixation_ratio
        self.total_fixed += fixed
        growth = absorbed * self.growth_speed
        self.total_growth += growth
        return absorbed, fixed, growth
def get_environmental_factors(x, y, step, total_steps=150, width=100, height=100):
    temp = 20 + 10 * np.sin(2 * np.pi * step / total_steps)
    day_night_factor = 0.8 + 0.2 * math.sin(2 * math.pi * step / 100)
    # Daily light cycle: day (steps 0-0.6), night (steps 0.6-1.0)
    day_frac = (step % (total_steps // 10)) / (total_steps // 10)
    light_surface = 1.0 if day_frac < 0.6 else 0.2
    depth_norm = y / (height - 1)
    base_light_intensity = light_surface * np.exp(-3 * depth_norm)
    light = base_light_intensity * day_night_factor
    # Salinity: river (x=0, 15 PSU) to ocean (x=width-1, 35 PSU)
    salinity = 15 + (35 - 15) * (x / (width - 1))
    # Nutrient: fixed for now
    nutrient = 1.0
    return {
        "temperature": temp,
        "light": light,
        "salinity": salinity,
        "nutrient": nutrient,
        "base_light_intensity": base_light_intensity,
        "day_night_factor": day_night_factor
    }

def compute_efficiency_score(plant, env, salt_tolerance=None, current_salt=None, bottom_type=None):
    # Temperature efficiency: Gaussian around plant.opt_temp
    temp_sigma = 5.0  # width of optimal range
    temp_eff = math.exp(-0.5 * ((env["temperature"] - plant.opt_temp) / temp_sigma) ** 2)
    # Light efficiency: scaled by plant.light_tolerance (max at 1.0)
    light_eff = min(env["light"] / plant.light_tolerance, 1.0)
    # Salinity efficiency: 1 if within range, else drop off
    sal_min, sal_max = plant.salinity_range
    if sal_min <= env["salinity"] <= sal_max:
        sal_eff = 1.0
    else:
        # Efficiency drops off linearly outside range (up to 0 at ±10 PSU)
        if env["salinity"] < sal_min:
            sal_eff = max(0, 1 - (sal_min - env["salinity"]) / 10)
        else:
            sal_eff = max(0, 1 - (env["salinity"] - sal_max) / 10)
    # Nutrient: currently fixed at 1.0
    nutrient_eff = env.get("nutrient", 1.0)

    # --- 塩分ストレスによる吸収率の減少 ---
    # plantにsalt_toleranceがあれば、それを使い、なければsalinity_rangeの中央値
    salt_tolerance_value = getattr(plant, "salt_tolerance", None)
    if salt_tolerance_value is None:
        salt_tolerance_value = (plant.salinity_range[0] + plant.salinity_range[1]) / 2
    salt_diff = abs(env["salinity"] - salt_tolerance_value)
    salt_penalty = max(0, 1 - 0.05 * salt_diff)  # 塩分差1あたり5%減

    # --- 底質タイプによる定着率・初期吸収効率への補正 ---
    # bottom_type: 'mud', 'sand', 'rock', etc.
    fixation_factor = 1.0
    if bottom_type is not None:
        if bottom_type == 'mud':
            fixation_factor = 1.0
        elif bottom_type == 'sand':
            fixation_factor = 0.7
        elif bottom_type == 'rock':
            fixation_factor = 0.85
        else:
            fixation_factor = 0.5

    # Combine all
    score = temp_eff * light_eff * sal_eff * nutrient_eff * salt_penalty * fixation_factor
    return max(0.0, min(1.0, score))

def get_nutrient_factor(step, total_steps):
    """
    栄養（窒素など）の濃度に応じた係数を返す。
    実装例：sin波で周期的に濃度が変化（春や秋に高くなる想定）
    """
    # 周期：全体の1/2周期、最大1.2、最小0.6程度
    return 0.6 + 0.6 * np.sin(2 * np.pi * step / total_steps)

def get_environmental_factor(step, total_steps, plant, depth_map):
    sunlight = 1.0 if step < total_steps * 0.5 else 0.3
    temperature = 15 + (step / total_steps) * 20

    # 植物ごとの最適温度に応じて吸収効率を変化
    temp_diff = (temperature - plant.optimal_temp) / plant.temp_range if hasattr(plant, 'optimal_temp') and hasattr(plant, 'temp_range') else 0
    temp_factor = np.exp(-0.5 * (temp_diff) ** 2)

    ix, iy = int(plant.x), int(plant.y)
    depth = depth_map[iy, ix]
    light_attenuation = np.exp(-3 * depth)

    return sunlight * light_attenuation * temp_factor

def seasonal_inflow(step, total_steps, base=30):
    # 流入量を時間によって変動させる（1周期のsin波）
    cycle = 2 * np.pi * step / total_steps
    return int(base * (0.5 + 0.5 * np.sin(cycle)))

def get_flow_vector(step, base_flow=(0.05, 0.02), noise_scale=0.02):
    """
    水流ベクトルにノイズを加えて自然な揺らぎを表現
    """
    np.random.seed(step)  # 再現性確保（任意）
    noise = np.random.normal(0, noise_scale, 2)
    flow_x = base_flow[0] + noise[0]
    flow_y = base_flow[1] + noise[1]
    return (flow_x, flow_y)

def diffuse_particles(particles, terrain, flow_field):
    height, width = terrain.shape
    new_particles = []
    for particle in particles:
        x, y = particle.x, particle.y
        ix, iy = int(y), int(x)
        if 0 <= iy < flow_field.shape[0] and 0 <= ix < flow_field.shape[1]:
            flow_x, flow_y = flow_field[iy, ix]
        else:
            flow_x, flow_y = 0, 0
        dx = np.random.normal(flow_x, 0.5)
        dy = np.random.normal(flow_y, 0.5)
        new_x = np.clip(x + dx, 0, width - 1)
        new_y = np.clip(y + dy, 0, height - 1)
        # terrain check
        if terrain[int(new_y), int(new_x)] == 1:
            particle.x = new_x
            particle.y = new_y
        new_particles.append(particle)
    return np.array(new_particles)

# 動的フローフィールド生成関数
def generate_dynamic_flow_field(width, height, step):
    # Creates a dynamic 2D flow field with periodic variation
    flow_field = np.zeros((height, width, 2))
    for y in range(height):
        for x in range(width):
            angle = np.sin((x + step * 0.1) * 0.1) + np.cos((y + step * 0.1) * 0.1)
            flow_x = 0.1 * np.cos(angle)
            flow_y = 0.1 * np.sin(angle)
            flow_field[y, x] = [flow_x, flow_y]
    return flow_field

def mark_absorption_area(absorption_map, x, y, radius):
    """
    粒子が吸収された位置を中心に、植物の吸収半径でヒートマップに記録。
    """
    height, width = absorption_map.shape
    for dx in range(-int(radius), int(radius) + 1):
        for dy in range(-int(radius), int(radius) + 1):
            px = int(x + dx)
            py = int(y + dy)
            if 0 <= px < width and 0 <= py < height:
                if dx**2 + dy**2 <= radius**2:
                    absorption_map[py, px] += 1

# 粒子流入関数（大気・河川由来）
def inject_particles(particles, terrain, num_new_particles=20, sources=None):
    """
    指定した位置（sources）に新しい粒子を追加。
    sources: [(x1, y1), (x2, y2), ...]
    """
    height, width = terrain.shape
    new_particles = []

    if sources is None:
        sources = [(0, 50), (99, 20)]  # デフォルトで左端中央と右端上

    for sx, sy in sources:
        for _ in range(num_new_particles // len(sources)):
            # 周辺に少しばらけさせる
            x = sx + np.random.normal(scale=1.0)
            y = sy + np.random.normal(scale=1.0)
            if 0 <= int(y) < height and 0 <= int(x) < width:
                if terrain[int(y), int(x)] == 1:  # 海上のみ
                    new_particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0))

    # 既存粒子と合体して返す
    return np.hstack([particles, new_particles])

plt.figure(figsize=(7, 7))

def get_light_level(step, total_steps):
    return 1.0 if step < total_steps * 0.5 else 0.3

def get_temperature(step, total_steps):
    return 15 + (step / total_steps) * 20

# --- Side-view simulation model ---

# 新しい光効率関数（時間変化を取り入れる）
def get_light_efficiency(depth, step, total_steps):
    base_efficiency = np.exp(-depth / 10)  # 深さによる光減衰
    seasonal_variation = 0.5 + 0.5 * np.sin(2 * np.pi * step / total_steps)  # 時間による変動
    return base_efficiency * seasonal_variation

def run_simulation():
    width, height = 100, 100
    terrain, depth_map = create_terrain(width, height)
    env_series = []
    nutrient_series = []
    total_steps = 150
    num_steps = 150
    num_particles = 1000
    dt = 1.0
    time = np.arange(total_steps)
    internal_series = []
    fixed_series = []
    released_series = []
    carbon_series = []
    nutrient_series = []
    total_steps = num_steps
    zostera_fixed_series = []
    kelp_fixed_series = []
    chlorella_fixed_series = []
    zostera_growth_series = []
    kelp_growth_series = []
    chlorella_growth_series = []
    zostera_absorbed_series = []
    kelp_absorbed_series = []
    chlorella_absorbed_series = []

    # 植物ごとのパラメータを外部ファイルから読み込む
    profiles = load_plant_profiles("plants.json")
    plants = []
    # 位置・半径はここで定義（例: Seagrass, Kelp, Chlorella）
    plant_positions = {
        "Seagrass": {"x": 20, "y": 95, "radius": 5},
        "Kelp": {"x": 50, "y": 85, "radius": 7},
        "Chlorella": {"x": 80, "y": 10, "radius": 3}
    }
    for plant_type in profiles:
        profile = profiles[plant_type]
        pos = plant_positions.get(plant_type, {"x":50, "y":95, "radius":3})
        plants.append(
            Plant(
                name=plant_type,
                absorb_efficiency=profile.get("absorb_efficiency", 1.0),
                growth_speed=profile.get("growth_speed", 1.0),
                fixation_ratio=profile.get("fixation_ratio", 0.7),
                release_ratio=profile.get("release_ratio", 0.05),
                structure_density=profile.get("structure_density", 1.0),
                opt_temp=profile.get("opt_temp", 20),
                light_tolerance=profile.get("light_tolerance", 1.0),
                salinity_range=tuple(profile.get("salinity_range", (20, 35))),
                absorption_efficiency=profile.get("absorption_efficiency", 1.0),
                growth_rate=profile.get("growth_rate", 1.0),
                x=pos["x"], y=pos["y"], radius=pos["radius"]
            )
        )

    # --- Particle initialization: start at surface ---
    particles = initialize_particles(num_particles, terrain)

    # --- Optional: define rocks (rectangles or ellipses) in the water column ---
    rocks = [
        {"x": width//2, "y": int(height*0.5), "w": 12, "h": 8},  # center rock
        {"x": int(width*0.7), "y": int(height*0.3), "w": 8, "h": 5}
    ]

    # --- 地形マップ（底質タイプ）を作成 ---
    # 例: 0: mud, 1: sand, 2: rock
    bottom_type_map = np.full((height, width), 'mud', dtype=object)
    bottom_type_map[90:100, 0:33] = 'mud'
    bottom_type_map[90:100, 33:66] = 'sand'
    bottom_type_map[90:100, 66:100] = 'rock'

    # ランダム性のブレを防ぐためにseed固定
    np.random.seed(42)
    random.seed(42)
    for step in range(total_steps):
        # --- Plant absorption/environmental update ---
        for i, plant in enumerate(plants):
            # Get environmental factors at plant's position
            env = get_environmental_factors(plant.x, plant.y, step, total_steps=total_steps, width=width, height=height)
            # 位置に応じて底質タイプを取得
            px, py = int(plant.x), int(plant.y)
            if 0 <= py < height and 0 <= px < width:
                bottom_type = bottom_type_map[py, px]
            else:
                bottom_type = 'mud'
            # 吸収効率計算（塩分ストレス・底質補正・昼夜補正）
            efficiency = compute_efficiency_score(plant, env, bottom_type=bottom_type)
            # 吸収時の昼夜による光量周期変動を反映
            base_absorption = dt * 1.0
            # effective_lightはget_environmental_factorsで計算済み
            effective_light = env["base_light_intensity"] * env["day_night_factor"]
            # 吸収効率にeffective_lightを乗算
            total_efficiency = efficiency * effective_light
            absorbed, fixed, growth = plant.absorb(base_absorption, efficiency_score=total_efficiency)
            # For plotting environmental factors, log for first plant
            if i == 0:
                env_series.append(total_efficiency)
                nutrient_series.append(env["nutrient"])
        zostera_fixed_series.append(plants[0].total_fixed)
        kelp_fixed_series.append(plants[1].total_fixed)
        chlorella_fixed_series.append(plants[2].total_fixed)
        zostera_growth_series.append(plants[0].total_growth)
        kelp_growth_series.append(plants[1].total_growth)
        chlorella_growth_series.append(plants[2].total_growth)
        zostera_absorbed_series.append(plants[0].total_absorbed)
        kelp_absorbed_series.append(plants[1].total_absorbed)
        chlorella_absorbed_series.append(plants[2].total_absorbed)

        # --- Particle movement: diffusion and lateral movement (no gravity) ---
        new_particles = []
        for particle in particles:
            drift_x = np.random.normal(0, 0.7)
            drift_y = np.random.normal(0, 0.7)
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                if ((particle.x - rx) / (rw/2)) ** 2 + ((particle.y - ry) / (rh/2)) ** 2 < 1.0:
                    drift_x += np.random.choice([-2, 2])
            new_y = np.clip(particle.y + drift_y, 0, height-1)
            new_x = np.clip(particle.x + drift_x, 0, width-1)
            if terrain[int(new_y), int(new_x)] == 1:
                particle.x = new_x
                particle.y = new_y
            else:
                particle.x = np.clip(particle.x + np.random.choice([-2, 2]), 0, width-1)
            new_particles.append(particle)
        particles = np.array(new_particles)

        # --- 粒子吸収処理: 各植物のパラメータで吸収 ---
        remaining_particles = []
        for particle in particles:
            absorbed = False
            for i, plant in enumerate(plants):
                # ここでは簡単に、粒子が下部に到達したら吸収
                if particle.y > height-5:  # 海底付近
                    # base_absorptionを粒子ごとに設定して吸収
                    base_absorption = dt * 1.0
                    # 吸収位置の底質タイプ
                    px, py = int(plant.x), int(plant.y)
                    if 0 <= py < height and 0 <= px < width:
                        bottom_type = bottom_type_map[py, px]
                    else:
                        bottom_type = 'mud'
                    env = get_environmental_factors(plant.x, plant.y, step, total_steps=total_steps, width=width, height=height)
                    efficiency = compute_efficiency_score(plant, env, bottom_type=bottom_type)
                    effective_light = env["base_light_intensity"] * env["day_night_factor"]
                    total_efficiency = efficiency * effective_light
                    absorbed_amt, fixed_amt, growth_amt = plant.absorb(base_absorption, efficiency_score=total_efficiency)
                    absorbed = True
                    break
            if not absorbed:
                remaining_particles.append(particle)
        particles = np.array(remaining_particles)

        # --- Periodic inflow: inject new particles at surface ---
        num_new = seasonal_inflow(step, total_steps, base=30)
        for _ in range(num_new):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, 3)
            if terrain[int(y), int(x)] == 1:
                particles = np.hstack([particles, [Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0)]])

        # --- Visualization: vertical cross-section (side view) ---
        if step % 10 == 0:
            plt.clf()
            fig, ax = plt.subplots()
            ax.set_facecolor('#d0f7ff')
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw/2, ry - rh/2), rw, rh, color='gray', alpha=0.7)
                ax.add_patch(rock_patch)
            for i, plant in enumerate(plants):
                ax.plot([20 + i*30, 20 + i*30], [height-1, height], color='green', linewidth=2)
            scat = ax.scatter([p.x for p in particles], [p.y for p in particles], c='cyan', s=5)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_xlabel("Horizontal Position")
            ax.set_ylabel("Depth (Downward)")
            ax.invert_yaxis()
            ax.set_title(f"Step {step} (Side View)")
            plt.tight_layout()
            plt.pause(0.01)
            plt.close(fig)

        # --- Carbon tracking (for compatibility with previous code) ---
        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)
    # 各植物種ごとの合計固定CO2量を計算して出力
    species_fixed_totals = {}
    for plant in plants:
        species = plant.name
        if species not in species_fixed_totals:
            species_fixed_totals[species] = 0.0
        species_fixed_totals[species] += plant.total_fixed

    print("\n=== 合計固定CO2量（植物種別） ===")
    for species, total in species_fixed_totals.items():
        print(f"{species}: {total:.2f}")

    # 各植物のシミュレーション結果を個別に記録
    for plant in plants:
        total_absorbed = plant.total_absorbed
        with open(f"result_{plant.name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PlantType", "TotalAbsorbedCO2"])
            writer.writerow([plant.name, total_absorbed])

    return env_series, nutrient_series, internal_series, fixed_series, released_series, zostera_fixed_series, kelp_fixed_series, chlorella_fixed_series, zostera_absorbed_series, kelp_absorbed_series, chlorella_absorbed_series



def simulate_step(plants, step, total_steps=150, width=100, height=100, co2=100.0):
    results = {}
    for plant_name, params in plants.items():
        # Assume params is a dict loaded from JSON
        # If needed, supply dummy x/y or use fixed positions
        x = params.get("x", 0)
        y = params.get("y", 0)
        env = get_environmental_factors(x, y, step, total_steps=total_steps, width=width, height=height)
        # Compute efficiency using available params
        # For demonstration, use opt_temp, light_tolerance, salinity_range
        temp_sigma = 5.0
        temp_eff = np.exp(-0.5 * ((env["temperature"] - params.get("opt_temp", 20)) / temp_sigma) ** 2)
        light_eff = min(env["light"] / params.get("light_tolerance", 1.0), 1.0)
        sal_min, sal_max = params.get("salinity_range", (20, 35))
        salinity = env["salinity"]
        if sal_min <= salinity <= sal_max:
            sal_eff = 1.0
        else:
            if salinity < sal_min:
                sal_eff = max(0, 1 - (sal_min - salinity) / 10)
            else:
                sal_eff = max(0, 1 - (salinity - sal_max) / 10)
        efficiency = temp_eff * light_eff * sal_eff
        absorbed = params["absorption_rate"] * co2 * efficiency
        growth = params["growth_rate"] * absorbed
        fixed = absorbed * params["fixation_rate"]
        results[plant_name] = {
            "absorbed": absorbed,
            "growth": growth,
            "fixed": fixed,
            "efficiency": efficiency,
            "env": env
        }
    return results

if __name__ == "__main__":
    width, height = 50, 50
    steps = 100
    SUBSTEPS = 5  # 吸収処理を滑らかにするための細分化
    random.seed(42)
    np.random.seed(42)
    absorption_map = np.zeros((width, height))
    time_series_absorption = []

    import pandas as pd

    def move_particles(carbon_particles):
        """
        粒子移動処理（水流に偏りを持たせた自然なランダム移動）。
        flow_weightsで指定した方向確率に基づいて、粒子を移動させる。
        """
        new_particles = np.zeros_like(carbon_particles)
        # 偏りのある移動方向（例：右方向に流れやすい）
        flow_weights = [
            (0, 1, 0.35),   # 右
            (0, -1, 0.15),  # 左
            (1, 0, 0.25),   # 下
            (-1, 0, 0.20),  # 上
            (0, 0, 0.05)    # 留まる（微小確率）
        ]
        directions = [(dx, dy) for dx, dy, _ in flow_weights]
        probabilities = [weight for _, _, weight in flow_weights]
        for x in range(width):
            for y in range(height):
                val = carbon_particles[x, y]
                if val > 0:
                    # valが1未満の場合も考慮して、整数化して分配
                    n_particles = int(val)
                    frac = val - n_particles
                    # 整数部分
                    for _ in range(n_particles):
                        dx, dy = random.choices(directions, weights=probabilities, k=1)[0]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            new_particles[nx, ny] += 1
                        else:
                            new_particles[x, y] += 1  # はみ出すなら移動しない
                    # 残りの端数（valが1未満の場合や端数部分）は、同じく方向抽選
                    if frac > 0:
                        dx, dy = random.choices(directions, weights=probabilities, k=1)[0]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            new_particles[nx, ny] += frac
                        else:
                            new_particles[x, y] += frac
        return new_particles



    def absorb_particles_field(carbon_particles, grid, absorption_efficiency, params,
                               absorption_map, total_growth_fixed, plant_params,
                               absorption_count_map=None):
        """
        吸収処理: grid上の植物セルごと (deterministic proportional uptake)
        plant_params: dict with absorption_rate, fixation_rate, growth_rate, light_tolerance, depth_range.
        """
        absorbed = 0.0
        total_growth, total_fixed = 0.0, 0.0
        global depth_map, light_map
        for x in range(width):
            for y in range(height):
                if grid[x, y] <= 0:
                    continue
                cell_total_absorption = 0.0  # accumulate all neighbor absorption for fixation
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < carbon_particles.shape[0] and 0 <= ny < carbon_particles.shape[1]):
                            continue
                        try:
                            dval = depth_map[nx, ny]
                            lval = light_map[nx, ny]
                        except Exception:
                            dval = 5
                            lval = 1.0
                        depth_range = plant_params.get("model_depth_range",
                               plant_params.get("depth_range", (1.0, 10.0)))
# dval は 1〜10m (meters). Prefer model_depth_range if present.
                        # dval は 1〜10m に正規化済み（x25 プログレス）
                        if not (depth_range[0] <= dval <= depth_range[1]):
                            continue
                        min_light = plant_params.get("light_tolerance", 0.0)
                        if lval < min_light:
                            continue
                        absorb_fraction = plant_params['absorption_rate'] * absorption_efficiency * grid[x, y]
                        effective_grid = grid[x, y]
                        # Warm-up phase: neutralize grid advantage
                        if 'current_step' in globals() and current_step < 10:
                            effective_grid = 1.0
                        absorb_fraction = plant_params['absorption_rate'] * absorption_efficiency * effective_grid
                        # （元の absorb_fraction 行を差し替えるなら必要）
                        if absorb_fraction > 1.0:
                            absorb_fraction = 1.0
                        if absorb_fraction <= 0:
                            continue
                        potential = carbon_particles[nx, ny] * absorb_fraction
                        absorption = min(carbon_particles[nx, ny], potential)
                        if absorption <= 0:
                            continue
                        carbon_particles[nx, ny] -= absorption
                        absorption_map[nx, ny] += absorption
                        if absorption_count_map is not None:
                            absorption_count_map[nx, ny] += 1
                        absorbed += absorption
                        cell_total_absorption += absorption
                        # Growth only once (central cell proxy for biomass accumulation)
                        if dx == 0 and dy == 0:
                            adjusted_growth = plant_params.get("growth_rate", 0.0) * lval
                            if adjusted_growth > 0:
                                grid[x, y] += adjusted_growth
                                total_growth += adjusted_growth
                # Apply fixation for all absorbed carbon in this cell neighborhood
                if 'cell_total_absorption' in locals() and cell_total_absorption > 0:
                    fixation_rate = plant_params.get('fixation_rate', 0.0)
                    fixed_amount = cell_total_absorption * fixation_rate
                    total_fixed += fixed_amount
        total_growth_fixed[0] += total_growth
        total_growth_fixed[1] += total_fixed
        # After processing all neighbors for this cell, fix carbon proportionally
        # (aggregate fixation for entire 3x3 neighborhood absorption)
        # Use fixation_rate once per plant cell.
        # 'cell_total_absorption' was last set inside the loops; we need to sum per cell.
        # To ensure per-cell fixation, accumulate a separate variable. We added 'cell_total_absorption' inside the loop.
        # We must wrap the fixation addition inside the x,y loops: move this logic inside those loops.
        return absorbed

    def run_simulation(plant_name, params):
        total_absorption = 0
        grid = np.zeros((width, height))
        carbon_particles = np.random.rand(width, height)
        total_absorbed = 0
        total_fixed = 0
        total_growth = 0
        absorption_history = []
        plant_efficiencies = {
            "Chlorella": 0.1,
            "Kelp": 0.07,
            "Zostera marina": 0.05,
            "Zostera": 0.05,
            "Macrocystis pyrifera": 0.07
        }
        absorption_efficiency = plant_efficiencies.get(plant_name, getattr(params, "absorption_rate", 0.05))
        np.random.seed(42)
        positions = np.random.choice([0, 1], size=(width, height), p=[0.98, 0.02])
        grid[positions == 1] = 1
        new_particles_per_step = 10
        for step in range(steps):
            # Add new CO2 particles every step
            for _ in range(new_particles_per_step):
                x, y = np.random.randint(0, width, size=2)
                carbon_particles[x, y] += 1
            # --- CO2 inflow: 毎ステップ一定量流入 ---
            for _ in range(50):
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                carbon_particles[x, y] += 1.0
            absorbed = 0
            total_growth_fixed = [0, 0]  # [growth, fixed]
            # サブステップで吸収処理・移動処理を細分化
            for sub_step in range(SUBSTEPS):
                carbon_particles = move_particles(carbon_particles)
                sub_absorbed = absorb_particles_field(carbon_particles, grid, absorption_efficiency, params, absorption_map, total_growth_fixed, plant_params=params)
                absorbed += sub_absorbed
            # --- Night respiration / release (net flux adjustment) ---
            day_night_factor = 0.8 + 0.2 * math.sin(2 * math.pi * step / 100)
            if day_night_factor < 0.5:
                respiration_coeff = 0.02
                grid *= (1 - respiration_coeff)
            total_absorbed += absorbed
            total_absorption += absorbed
            total_growth += total_growth_fixed[0]
            total_fixed += total_growth_fixed[1]
            absorption_history.append(total_absorption)
        # Export summary CSV for this plant
        summary_df = pd.DataFrame([{
            "plant": plant_name,
            "total_absorbed": total_absorbed,
            "total_fixed": total_fixed,
            "growth": total_growth,
            "fixed_eff": (total_fixed / total_absorbed) if total_absorbed else 0.0
        }])
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(f"results/summary_{plant_name}.csv", index=False)
        # Export absorption over time plot for this plant
        plt.plot(absorption_history)
        plt.title(f"Absorption Over Time - {plant_name}")
        plt.xlabel("Step")
        plt.ylabel("CO2 Absorbed")
        plt.savefig(f"results/absorption_{plant_name}.png")
        plt.close()
        # Also save per-step CSV like before
        csv_filename = f"results/result_{plant_name}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Absorbed_Step", "Absorbed_Cumulative"])
            cumulative_prev = 0
            for i, value in enumerate(absorption_history):
                absorbed_step = value - cumulative_prev
                writer.writerow([i, absorbed_step, value])
                cumulative_prev = value
        time_series_absorption.append((plant_name, absorption_history))
        return {
            "plant": plant_name,
            "total_absorbed": total_absorbed,
            "total_fixed": total_fixed,
            "total_growth": total_growth,
        }

    chlorella_log = []
    kelp_log = []
    zostera_log = []
    results = []

    # Per-step absorption counters for each plant
    chlorella_absorbed_this_step = 0
    kelp_absorbed_this_step = 0
    zostera_absorbed_this_step = 0

    for plant_name, params in plant_params_dict.items():
        # Reset per-step counters at the start of each plant sim
        chlorella_absorbed_this_step = 0
        kelp_absorbed_this_step = 0
        zostera_absorbed_this_step = 0

        # Patch run_simulation to log per-step absorption
        # NOTE: depth_map values used here are in meters (1–10). Ensure plants.json depth_range matches this scale.
        def run_simulation_with_step_logging(plant_name, params):
            grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=int)
            import math
            heatmap = {}  # Dictionary to track absorption amount per coarse grid location
            # Initialize absorption count map
            absorption_count_map = np.zeros_like(grid)
            total_absorption = 0
            grid = np.zeros((width, height))
            carbon_particles = np.random.rand(width, height)
            total_absorbed = 0
            total_fixed = 0
            total_growth = 0
            absorption_history = []
            # Use plant parameters from JSON
            absorption_efficiency = params["absorption_rate"]
            np.random.seed(42)
            positions = np.random.choice([0, 1], size=(width, height), p=[0.98, 0.02])
            # ---- Initial valid cells (using model_depth_range if present) ----
            depth_range_effective = params.get("model_depth_range",
                                            params.get("depth_range", (1.0, 10.0)))
            min_light_dbg_init = params.get("light_tolerance", 0.0)
            temp_positions = (positions == 1)
            initial_valid_cells = 0
            for xi in range(width):
                for yi in range(height):
                    if not temp_positions[xi, yi]:
                        continue
                    dval_i = depth_map[xi, yi]
                    lval_i = light_map[xi, yi]
                    if depth_range_effective[0] <= dval_i <= depth_range_effective[1] and lval_i >= min_light_dbg_init:
                        initial_valid_cells += 1
            grid[positions == 1] = 1
            grid[positions == 1] = 1
            new_particles_per_step = 10
            # --- グリッドサイズを指定（例: 5×5区画） ---
            grid_size = 5
            # ---------- 改善 2: 季節によるCO2流入変動 ----------
            # 定数設定：1年のステップ数（例: 365日）
            seasonal_cycle_length = 365
            seasonal_amplitude = 0.3  # 振幅30%
            base_inflow = 50  # 以前の固定流入量
            def generate_particles(amount, shape):
                # 粒子をランダムな位置に均等に分布
                arr = np.zeros(shape)
                for _ in range(amount):
                    x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                    arr[x, y] += 1.0
                return arr
            for step in range(steps):
                # 種別 + ステップに依存しない再現性確保（種間比較公平性）
                base_seed = 42
                random.seed(base_seed + hash(plant_name) % 10000)
                np.random.seed(base_seed + (hash(plant_name) % 10000))
                # Reset per-step counter at the start of each step
                step_absorbed = 0
                # Add new CO2 particles every step
                for _ in range(new_particles_per_step):
                    x, y = np.random.randint(0, width, size=2)
                    carbon_particles[x, y] += 1
                # --- CO2 inflow: 季節変動を導入 ---
                seasonal_factor = 1 + seasonal_amplitude * math.sin(2 * math.pi * step / seasonal_cycle_length)
                inflow_amount = int(base_inflow * seasonal_factor)
                # デバッグ: 初期5ステップで有効セル数を計測
                if step < 5:
                    valid_cells = 0
                    depth_range_dbg = params.get("model_depth_range",
                             params.get("depth_range", (1.0, 10.0)))
                    min_light_dbg = params.get("light_tolerance", 0.0)
                    for x_dbg in range(width):
                        for y_dbg in range(height):
                            if grid[x_dbg, y_dbg] <= 0:
                                continue
                            dval_dbg = depth_map[x_dbg, y_dbg]
                            lval_dbg = light_map[x_dbg, y_dbg]
                            if depth_range_dbg[0] <= dval_dbg <= depth_range_dbg[1] and lval_dbg >= min_light_dbg:
                                valid_cells += 1
                    if step == 0:
                        print(f"[DEBUG] {plant_name}: depth_range={depth_range_dbg} light_tol={min_light_dbg} initial_valid_cells={initial_valid_cells}")
                    print(f"[DEBUG] {plant_name} step {step}: valid_cells={valid_cells}")
                carbon_particles += generate_particles(inflow_amount, carbon_particles.shape)
                absorbed = 0
                total_growth_fixed = [0, 0]  # [growth, fixed]
                # サブステップで吸収処理・移動処理を細分化
                for sub_step in range(SUBSTEPS):
                    # --- 粒子移動（2Dフィールドをnumpyとして保持） ---
                    # ここでは単純にランダムウォーク＋右方向へ僅かなバイアス！
                    shifted = np.zeros_like(carbon_particles)
                    # ランダム方向（-1,0,1）を各セルに与える
                    dx_field = np.random.randint(-1, 2, size=carbon_particles.shape)
                    dy_field = np.random.randint(-1, 2, size=carbon_particles.shape)
                    bias = 0.2  # 右方向バイアス
                    # 移動
                    for x in range(width):
                        for y in range(height):
                            val = carbon_particles[x, y]
                            if val == 0:
                                continue
                            dx = dx_field[x, y]
                            dy = dy_field[x, y]
                            # 右方向バイアス
                            if random.random() < bias:
                                dx = 1
                            nx = min(width - 1, max(0, x + dx))
                            ny = min(height - 1, max(0, y + dy))
                            shifted[nx, ny] += val
                    carbon_particles = shifted
                    # --- 吸収前のスナップショット（numpy配列のまま） ---
                    before_absorption = carbon_particles.copy()
                    # 吸収
                    sub_absorbed = absorb_particles_field(
                        carbon_particles,
                        grid,
                        absorption_efficiency,
                        params,
                        absorption_map,
                        total_growth_fixed,
                        plant_params=params,
                        absorption_count_map=absorption_count_map
                    )
                    absorbed += sub_absorbed
                    step_absorbed += sub_absorbed
                    # --- 吸収差分をヒートマップへ集計 ---
                    delta_grid = before_absorption - carbon_particles
                    for x in range(width):
                        for y in range(height):
                            delta = delta_grid[x, y]
                            if delta > 0:
                                x_bin = x // grid_size
                                y_bin = y // grid_size
                                key = (x_bin, y_bin)
                                heatmap[key] = heatmap.get(key, 0) + delta
                # --- Night respiration / release (net flux adjustment) ---
                day_night_factor = 0.8 + 0.2 * math.sin(2 * math.pi * step / 100)
                if day_night_factor < 0.5:
                    respiration_coeff = 0.02
                    grid *= (1 - respiration_coeff)
                total_absorbed += absorbed
                total_absorption += absorbed
                total_growth += total_growth_fixed[0]
                total_fixed += total_growth_fixed[1]
                absorption_history.append(total_absorbed)
                # After each step, log per-step absorption for this plant
                if plant_name == "Chlorella":
                    chlorella_log.append(step_absorbed)
                elif plant_name == "Kelp" or plant_name == "Macrocystis pyrifera":
                    kelp_log.append(step_absorbed)
                elif plant_name == "Zostera marina" or plant_name == "Zostera":
                    zostera_log.append(step_absorbed)
            # Export summary CSV for this plant
            summary_df = pd.DataFrame([{
                "plant": plant_name,
                "total_absorbed": total_absorbed,
                "total_fixed": total_fixed,
                "growth": total_growth,
                "fixed_eff": (total_fixed / total_absorbed) if total_absorbed else 0.0,
                "initial_valid_cells": initial_valid_cells
            }])
            os.makedirs("results", exist_ok=True)
            summary_df.to_csv(f"results/summary_{plant_name}.csv", index=False)
            # Export absorption over time plot for this plant
            plt.plot(absorption_history)
            plt.title(f"Absorption Over Time - {plant_name}")
            plt.xlabel("Step")
            plt.ylabel("CO2 Absorbed")
            plt.savefig(f"results/absorption_{plant_name}.png")
            plt.close()
            # Also save per-step CSV like before
            csv_filename = f"results/result_{plant_name}.csv"
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Absorbed_Step", "Absorbed_Cumulative"])
                cumulative_prev = 0
                for i, value in enumerate(absorption_history):
                    absorbed_step = value - cumulative_prev
                    writer.writerow([i, absorbed_step, value])
                    cumulative_prev = value
            # Save absorption heatmap (output: x_bin, y_bin, amount)
            heatmap_path = f"results/heatmap_{plant_name}.csv"
            with open(heatmap_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x_bin", "y_bin", "amount"])
                for (x_bin, y_bin), amount in heatmap.items():
                    writer.writerow([x_bin, y_bin, amount])
            # Save absorption count map as a CSV heatmap
            heatmap_path = f"results/heatmap_density_{plant_name}.csv"
            np.savetxt(heatmap_path, absorption_count_map, delimiter=",")
            time_series_absorption.append((plant_name, absorption_history))
            return {
                "plant": plant_name,
                "total_absorbed": total_absorbed,
                "total_fixed": total_fixed,
                "total_growth": total_growth,
                "initial_valid_cells": initial_valid_cells
            }

        # run_simulation functionのパッチ使う！！
        result = run_simulation_with_step_logging(plant_name, params)
        results.append(result)
    for plant_name, history in time_series_absorption:
        plt.plot(history, label=plant_name)
    plt.xlabel("Step")
    plt.ylabel("Absorbed CO2")
    plt.title("CO2 Absorption Over Time")
    plt.legend()
    plt.savefig("results/absorption_over_time.png")
    plt.clf()
    plt.imshow(absorption_map, cmap='hot', interpolation='nearest')
    plt.title("Absorption Heatmap")
    plt.colorbar(label="Absorbed CO2")
    plt.savefig("results/absorption_heatmap.png")
    plt.clf()
    with open("results/summary_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plant", "Total_Absorbed", "Total_Fixed", "Total_Growth", "Fixed_Eff", "Initial_Valid_Cells"])
        for r in results:
            fixed_eff = r["total_fixed"]/r["total_absorbed"] if r["total_absorbed"] else 0.0
            init_cells = r.get("initial_valid_cells", 0)
            writer.writerow([r["plant"], r["total_absorbed"], r["total_fixed"], r["total_growth"], fixed_eff, init_cells])
    print("=== Summary generation complete. If any species shows 0 totals, check depth_range vs depth_map scaling (now 1–10m) and light_tolerance thresholds. ===")