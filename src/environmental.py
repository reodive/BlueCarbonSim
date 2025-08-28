import math
import numpy as np


def get_environmental_factors(x, y, step, total_steps=150, width=100, height=100):
    temp = 20 + 10 * np.sin(2 * np.pi * step / total_steps)
    day_night_factor = 0.8 + 0.2 * math.sin(2 * math.pi * step / 100)
    day_frac = (step % (total_steps // 10)) / (total_steps // 10)
    light_surface = 1.0 if day_frac < 0.6 else 0.2
    depth_norm = y / (height - 1)
    base_light_intensity = light_surface * np.exp(-3 * depth_norm)
    light = base_light_intensity * day_night_factor
    salinity = 15 + (35 - 15) * (x / (width - 1))
    nutrient = 1.0
    return {
        "temperature": temp,
        "light": light,
        "salinity": salinity,
        "nutrient": nutrient,
        "base_light_intensity": base_light_intensity,
        "day_night_factor": day_night_factor,
    }


def compute_efficiency_score(plant, env, bottom_type=None):
    temp_sigma = 5.0
    temp_eff = math.exp(-0.5 * ((env["temperature"] - plant.opt_temp) / temp_sigma) ** 2)
    light_eff = min(env["light"] / plant.light_tolerance, 1.0)
    sal_min, sal_max = plant.salinity_range
    if sal_min <= env["salinity"] <= sal_max:
        sal_eff = 1.0
    else:
        if env["salinity"] < sal_min:
            sal_eff = max(0, 1 - (sal_min - env["salinity"]) / 10)
        else:
            sal_eff = max(0, 1 - (env["salinity"] - sal_max) / 10)
    nutrient_eff = env.get("nutrient", 1.0)
    salt_tolerance_value = getattr(plant, "salt_tolerance", None)
    if salt_tolerance_value is None:
        salt_tolerance_value = (plant.salinity_range[0] + plant.salinity_range[1]) / 2
    salt_diff = abs(env["salinity"] - salt_tolerance_value)
    salt_penalty = max(0, 1 - 0.05 * salt_diff)
    fixation_factor = 1.0
    if bottom_type is not None:
        if bottom_type == "mud":
            fixation_factor = 1.0
        elif bottom_type == "sand":
            fixation_factor = 0.7
        elif bottom_type == "rock":
            fixation_factor = 0.85
        else:
            fixation_factor = 0.5
    score = temp_eff * light_eff * sal_eff * nutrient_eff * salt_penalty * fixation_factor
    return max(0.0, min(1.0, score))


def get_nutrient_factor(step, total_steps):
    return 0.6 + 0.6 * np.sin(2 * np.pi * step / total_steps)


def get_light_level(step, total_steps):
    return 1.0 if step < total_steps * 0.5 else 0.3


def get_temperature(step, total_steps):
    return 15 + (step / total_steps) * 20


def get_light_efficiency(depth, step, total_steps):
    base_efficiency = np.exp(-depth / 10)
    seasonal_variation = 0.5 + 0.5 * np.sin(2 * np.pi * step / total_steps)
    return base_efficiency * seasonal_variation