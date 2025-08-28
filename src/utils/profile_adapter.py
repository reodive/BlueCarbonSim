import math

def normalize_profiles(raw_profiles: dict):
    # 1) absorption_efficiency を 0..1 に正規化（種間相対比較）
    max_abs_rate = max(v.get("absorption_rate", 1.0) for v in raw_profiles.values())
    max_abs_rate = max(max_abs_rate, 1e-9)

    fixed = {}
    for name, p in raw_profiles.items():
        # キー変換
        fixation_ratio = p.get("fixation_rate", 0.7)
        absorption_efficiency = p.get("absorption_rate", 1.0) / max_abs_rate  # 0..1
        # 光の半飽和「しきい値」へ変換（0..1 で「これ以上で飽和」）
        # 既存値が小さすぎるので、"しきい値"としては大きめに補正
        light_half_sat = max(p.get("light_tolerance", 0.35), 0.2)  # 下限0.2

        fixed[name] = {
            "fixation_ratio": float(fixation_ratio),
            "absorption_efficiency": float(min(max(absorption_efficiency, 0.0), 1.0)),
            "growth_rate": float(p.get("growth_rate", 0.01)),
            "salinity_range": tuple(p.get("salinity_range", (20, 35))),
            "light_half_saturation": float(min(max(light_half_sat, 0.0), 1.0)),
            "opt_temp": float(p.get("opt_temp", 20.0)),
            "temp_sigma": float(p.get("temp_sigma", 5.0)),
            # 深度レンジはそのまま保持（後段でフィルタに使用）
            "depth_range": tuple(p.get("depth_range", (0.5, 5))),
            "lit_depth_range_m": tuple(p.get("lit_depth_range_m", (0, 10))),
            "model_depth_range": tuple(p.get("model_depth_range", (1, 6))),
        }
    return fixed
