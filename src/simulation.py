import csv
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from models.particle import Particle, initialize_particles
from models.plant import Plant
from environment import get_environmental_factors, compute_efficiency_score
from terrain import create_terrain
from particle_flow import diffuse_particles, inject_particles, generate_dynamic_flow_field


def seasonal_inflow(step, total_steps, base=30):
    cycle = 2 * np.pi * step / total_steps
    return int(base * (0.5 + 0.5 * np.sin(cycle)))


def run_simulation():
    width, height = 100, 100
    terrain, depth_map = create_terrain(width, height)
    env_series = []
    nutrient_series = []
    total_steps = 150
    num_particles = 1000
    dt = 1.0
    internal_series = []
    fixed_series = []
    released_series = []
    carbon_series = []
    zostera_fixed_series = []
    kelp_fixed_series = []
    chlorella_fixed_series = []
    zostera_growth_series = []
    kelp_growth_series = []
    chlorella_growth_series = []
    zostera_absorbed_series = []
    kelp_absorbed_series = []
    chlorella_absorbed_series = []

    with open("data/plants.json", "r") as f:
        profiles = json.load(f)
    plants = []
    plant_positions = {
        "Seagrass": {"x": 20, "y": 95, "radius": 5},
        "Kelp": {"x": 50, "y": 85, "radius": 7},
        "Chlorella": {"x": 80, "y": 10, "radius": 3},
    }
    for plant_type in profiles:
        profile = profiles[plant_type]
        pos = plant_positions.get(plant_type, {"x": 50, "y": 95, "radius": 3})
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
                x=pos["x"],
                y=pos["y"],
                radius=pos["radius"],
            )
        )

    particles = initialize_particles(num_particles, terrain)

    rocks = [
        {"x": width // 2, "y": int(height * 0.5), "w": 12, "h": 8},
        {"x": int(width * 0.7), "y": int(height * 0.3), "w": 8, "h": 5},
    ]

    bottom_type_map = np.full((height, width), "mud", dtype=object)
    bottom_type_map[90:100, 0:33] = "mud"
    bottom_type_map[90:100, 33:66] = "sand"
    bottom_type_map[90:100, 66:100] = "rock"

    np.random.seed(42)
    random.seed(42)
    for step in range(total_steps):
        for i, plant in enumerate(plants):
            env = get_environmental_factors(
                plant.x, plant.y, step, total_steps=total_steps, width=width, height=height
            )
            px, py = int(plant.x), int(plant.y)
            if 0 <= py < height and 0 <= px < width:
                bottom_type = bottom_type_map[py, px]
            else:
                bottom_type = "mud"
            efficiency = compute_efficiency_score(plant, env, bottom_type=bottom_type)
            effective_light = env["base_light_intensity"] * env["day_night_factor"]
            total_efficiency = efficiency * effective_light
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

        flow_field = generate_dynamic_flow_field(width, height, step)
        particles = diffuse_particles(particles, terrain, flow_field)

        remaining_particles = []
        for particle in particles:
            absorbed = False
            for i, plant in enumerate(plants):
                if particle.y > height - 5:
                    base_absorption = dt * 1.0
                    px, py = int(plant.x), int(plant.y)
                    if 0 <= py < height and 0 <= px < width:
                        bottom_type = bottom_type_map[py, px]
                    else:
                        bottom_type = "mud"
                    env = get_environmental_factors(
                        plant.x, plant.y, step, total_steps=total_steps, width=width, height=height
                    )
                    efficiency = compute_efficiency_score(plant, env, bottom_type=bottom_type)
                    effective_light = env["base_light_intensity"] * env["day_night_factor"]
                    total_efficiency = efficiency * effective_light
                    plant.absorb(base_absorption, efficiency_score=total_efficiency)
                    absorbed = True
                    break
            if not absorbed:
                remaining_particles.append(particle)
        particles = np.array(remaining_particles)

        num_new = seasonal_inflow(step, total_steps, base=30)
        particles = inject_particles(particles, terrain, num_new_particles=num_new)

        if step % 10 == 0:
            plt.clf()
            fig, ax = plt.subplots()
            ax.set_facecolor("#d0f7ff")
            for rock in rocks:
                rx, ry, rw, rh = rock["x"], rock["y"], rock["w"], rock["h"]
                rock_patch = plt.Rectangle((rx - rw / 2, ry - rh / 2), rw, rh, color="gray", alpha=0.7)
                ax.add_patch(rock_patch)
            for i, plant in enumerate(plants):
                ax.plot([20 + i * 30, 20 + i * 30], [height - 1, height], color="green", linewidth=2)
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

        carbon_series.append(sum(p.total_fixed for p in plants))
        internal_series.append(sum(p.total_growth for p in plants))
        fixed_series.append(sum(p.total_fixed for p in plants))
        released_series.append(0)

    species_fixed_totals = {}
    for plant in plants:
        species = plant.name
        if species not in species_fixed_totals:
            species_fixed_totals[species] = 0.0
        species_fixed_totals[species] += plant.total_fixed

    print("\n=== 合計固定CO2量（植物種別） ===")
    for species, total in species_fixed_totals.items():
        print(f"{species}: {total:.2f}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    for plant in plants:
        total_absorbed = plant.total_absorbed
        filepath = os.path.join(results_dir, f"result_{plant.name}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PlantType", "TotalAbsorbedCO2"])
            writer.writerow([plant.name, total_absorbed])

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
    results = {}
    for plant_name, params in plants.items():
        x = params.get("x", 0)
        y = params.get("y", 0)
        env = get_environmental_factors(
            x, y, step, total_steps=total_steps, width=width, height=height
        )
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
            "env": env,
        }
    return results
