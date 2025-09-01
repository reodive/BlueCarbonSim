# src/simulation.py
import csv
import json
import os
import random
import re
from typing import Dict, List

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

    # 配置（9種の植物のレイアウト）
    plant_positions = {
        "Zostera marina":            {"x": 18, "y": 72, "radius": 5},
        "Halophila ovalis":          {"x": 25, "y": 68, "radius": 5},
        "Posidonia oceanica":        {"x": 60, "y": 78, "radius": 6},
        "Macrocystis pyrifera":      {"x": 85, "y": 80, "radius": 7},
        "Saccharina japonica":       {"x": 75, "y": 82, "radius": 7},
        "Chlorella vulgaris":        {"x": 10, "y": 14, "radius": 3},
        "Nannochloropsis gaditana":  {"x": 55, "y": 16, "radius": 3},
        "Spartina alterniflora":     {"x": 12, "y": 95, "radius": 4},
        "Rhizophora spp.":           {"x": 15, "y": 92, "radius": 4},
    }

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
    KD = float(cfg.get("kd_m_inv", 0.8))
    MAX_DEPTH_M = 8.0
    meters_per_pixel = MAX_DEPTH_M / max((height - 1), 1)
    # フォトゾーン近似（e^-kd z = 0.1 → z ≈ 2.3/kd）
    euphotic_depth_m = 2.3 / max(KD, 1e-6)
    euphotic_px = int(euphotic_depth_m / meters_per_pixel)

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

    for step in range(total_steps):

        # ステップごとに環境評価をキャッシュ
        plant_env, plant_eff = {}, {}
        for i, plant in enumerate(plants):
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
        flow_field = generate_dynamic_flow_field(width, height, step)
        particles, outflow_mass_step = diffuse_particles(particles, terrain, flow_field)
        mass_outflow += float(outflow_mass_step) * float(particle_mass_mgC)

        # 吸収処理（保存則を守る）
        remaining_particles = []
        step_absorbed = 0.0
        step_fixed = 0.0
        step_growth = 0.0
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
                    absorbed, fixed, growth = plant.absorb(absorb_amount)
                    particle.mass -= absorbed
                    step_absorbed += absorbed
                    step_fixed += fixed
                    step_growth += growth
                    absorbed_here = True
                    if particle.mass <= 1e-12:
                        break
            if particle.mass > 1e-12:
                remaining_particles.append(particle)
        particles = np.array(remaining_particles, dtype=object)

        # 新規流入
        num_new = seasonal_inflow(step, total_steps, base_mgC_per_step=inflow_mgC_per_step_base, particle_mass_mgC=particle_mass_mgC)
        particles, added = inject_particles(particles, terrain, num_new_particles=num_new)
        # 実際に追加できた粒子数で流入を加算（質量保存）
        mass_inflow += float(added) * float(particle_mass_mgC)

        # プランクトン（Chlorella）の自然死亡・再放出（CO2復帰）
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
                        particles, added_rel = inject_particles(
                            particles, terrain, num_new_particles=n_rel, sources=[(plant.x, plant.y)]
                        )
                        # 再注入（流入）として会計。丸め分は loss_quant_mgC に反映済み。
                        mass_inflow += float(added_rel) * float(particle_mass_mgC)

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
    species_fixed_totals = {plant.name: plant.total_fixed for plant in plants}
    print("\n=== 合計固定CO2量（植物種別） ===")
    for species, total in species_fixed_totals.items():
        print(f"{species}: {total:.2f}")

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
        with open(os.path.join("results", f"result_{plant.name}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_absorbed", "total_fixed", "total_growth"])
            writer.writerow([plant.total_absorbed, plant.total_fixed, plant.total_growth])

    # 新: 全体サマリ
    with open(os.path.join("results", "summary_totals.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "total_absorbed", "total_fixed", "total_growth"])
        for plant in plants:
            writer.writerow([plant.name, plant.total_absorbed, plant.total_fixed, plant.total_growth])

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

    # 図の保存（見やすい可視化）
    try:
        # 1) 種別ごとの合計（棒グラフ）
        fig, ax = plt.subplots(figsize=(10, 5))
        species = list(species_fixed_totals.keys())
        totals = [species_fixed_totals[s] for s in species]
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
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

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

        # JSONのキー名に合わせる
        abs_rate = params.get("absorption_rate", params.get("absorption_efficiency", 1.0))
        fix_rate = params.get("fixation_rate", params.get("fixation_ratio", 0.7))
        growth_r = params.get("growth_rate", 0.0)

        absorbed = abs_rate * co2 * efficiency
        fixed = absorbed * fix_rate
        growth = growth_r * absorbed

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
    # デフォルト設定で一度だけ走らせる
    run_simulation()
