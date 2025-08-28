# src/environment.py
# ------------------
# 現実寄りの環境モデル（季節/日内の分離、Beer–Lambert、0..1正規化）
# 単位と仮定を明示し、魔法定数は関数引数として外出し可能にしている。

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Optional


# ===== 基本ヘルパ =====

def seasonal_sine(step: int, period_steps: int, mean: float, amp: float, phase: float = 0.0) -> float:
    """季節波形（1年=period_steps）"""
    return mean + amp * np.sin(2.0 * np.pi * (step / float(period_steps) - phase))


def diurnal_half_sine(hour_frac: float, day_length_frac: float = 0.5) -> float:
    """
    日内変動（0..1）: 日中のみ半波正弦、夜間は0。
    hour_frac: 0..1 の一日内の位置
    day_length_frac: 日照長の比率（0.5=12時間/日）
    """
    hour_frac = float(hour_frac % 1.0)
    if hour_frac >= day_length_frac:
        return 0.0
    # [0, day_length_frac] を [0, 1] に射影して半波正弦
    return float(np.sin(np.pi * (hour_frac / max(day_length_frac, 1e-9))))


def beer_lambert(I0: float, kd_m_inv: float, depth_m: float) -> float:
    """Beer–Lambert: I(z) = I0 * exp(-kd*z)（kd: m^-1, depth: m）"""
    return float(I0 * math.exp(-max(kd_m_inv, 0.0) * max(depth_m, 0.0)))


def clip01(x: float) -> float:
    """0..1 にクリップ"""
    return float(max(0.0, min(1.0, x)))


# ===== 環境フィールド生成 =====

def get_environmental_factors(
    x: int,
    y: int,
    step: int,
    *,
    total_steps: int = 365,          # 1年=365step を推奨（Δt=1日なら自然）
    width: int = 100,
    height: int = 100,
    # 物理・環境パラメータ（必要に応じて configs から注入）
    kd_m_inv: float = 0.7,           # 光学減衰係数 [m^-1]（沿岸の例）
    cell_depth_m: Optional[float] = None,  # セル代表深度 [m]。未指定ならyから近似
    max_depth_m: float = 5.0,        # 深度近似に使う最大水深 [m]
    T_mean: float = 20.0,            # 平均水温 [°C]
    T_amp: float = 10.0,             # 季節振幅 [°C]
    T_phase: float = 0.0,            # 温度の季節位相
    I0_daylen_frac: float = 0.5,     # 日照長（比率, 0.5=12h）
    salinity_mode: str = "seasonal", # "seasonal" | "linear_x" | "constant"
    S_mean: float = 28.0,            # 平均塩分 [PSU]
    S_amp: float = 2.0,              # 季節振幅 [PSU]
    S_min: float = 15.0,             # 線形勾配の最小（linear_x用）
    S_max: float = 35.0,             # 線形勾配の最大（linear_x用）
    nutrient_mean: float = 0.5,      # 栄養塩（効率ファクタ）の平均（0..1）
    nutrient_amp: float = 0.5,       # 栄養塩の振幅（0..1でクリップ）
    nutrient_phase: float = 0.0,
) -> Dict[str, float]:
    """
    環境フィールドを返す。
    戻り値:
        temperature [°C], light [0..1], salinity [PSU], nutrient [0..1],
        meta: I0_diurnal [0..1], depth_m [m]
    """

    # --- 温度（季節） ---
    temperature = seasonal_sine(step, total_steps, T_mean, T_amp, T_phase)

    # --- 日内光（表層） ---
    # 1日を total_steps/365 日あたりのstepに換算（Δt=1日運用なら簡略に step%1=0）
    steps_per_day = max(total_steps / 365.0, 1.0)
    hour_frac = (step % steps_per_day) / steps_per_day  # 0..1
    I0_diurnal = diurnal_half_sine(hour_frac, I0_daylen_frac)  # 0..1

    # --- 深度決定 & 減衰 ---
    if cell_depth_m is None:
        depth_m = (y / max(height - 1, 1)) * max_depth_m  # 0..max_depth_m
    else:
        depth_m = float(cell_depth_m)
    light = beer_lambert(I0_diurnal, kd_m_inv, depth_m)  # 0..1（I0も0..1想定）

    # --- 塩分 ---
    if salinity_mode == "seasonal":
        salinity = seasonal_sine(step, total_steps, S_mean, S_amp, 0.0)
    elif salinity_mode == "linear_x":
        # 画面座標依存は本来避けたいが、境界条件の近似として残す場合
        salinity = float(S_min + (S_max - S_min) * (x / max(width - 1, 1)))
    else:  # "constant"
        salinity = float(S_mean)

    # --- 栄養塩（0..1に確実に収める） ---
    nutrient_raw = seasonal_sine(step, total_steps, nutrient_mean, nutrient_amp, nutrient_phase)
    nutrient = clip01(nutrient_raw)

    return {
        "temperature": float(temperature),
        "light": clip01(float(light)),      # 念のためクリップ
        "salinity": float(salinity),
        "nutrient": float(nutrient),
        "I0_diurnal": float(I0_diurnal),
        "depth_m": float(depth_m),
    }


# ===== 効率スコア =====

def compute_efficiency_score(
    plant,
    env: Dict[str, float],
    bottom_type: Optional[str] = None,
    *,
    # 温度感度は種ごとに持たせるのが望ましいが、無ければ既定値を使う
    default_temp_sigma: float = 5.0,
    # 底質係数は設定から与えるのが望ましい。無ければ既定表。
    default_substrate_factor: Optional[Dict[str, float]] = None,
) -> float:
    """
    0..1 の効率スコア。要素は乗算だが、各ファクタは0..1に正規化。
    - 温度: 至適温度周りのガウス（σは種 or 既定）
    - 光: plant.light_tolerance で 0..1 正規化
    - 塩分: 許容範囲内=1、外側は線形ペナルティ（重複ペナルティなし）
    - 栄養塩: 0..1
    - 底質: 種×底質で係数、既定は {mud:1.0, sand:0.7, rock:0.85}
    """

    # 光：I / (I + I_half) で 0..1
    I = env["light"]  # 0..1
    I_half = getattr(plant, "light_half_saturation", 0.5)  # 0.2〜0.7 推奨
    light_eff = I / (I + max(I_half, 1e-6))

    # 温度
    sigma = getattr(plant, "temp_sigma", default_temp_sigma)
    opt_temp = getattr(plant, "opt_temp", 20.0)
    temp_eff = math.exp(-0.5 * ((env["temperature"] - opt_temp) / max(sigma, 1e-6)) ** 2)
    temp_eff = clip01(temp_eff)

    # 光
    lt = max(getattr(plant, "light_tolerance", 1.0), 1e-6)
    light_eff = clip01(env["light"] / lt)

    # 塩分（重複ペナルティ禁止）
    smin, smax = getattr(plant, "salinity_range", (20.0, 32.0))
    s = env["salinity"]
    if s < smin:
        sal_eff = clip01(1.0 - (smin - s) / 10.0)
    elif s > smax:
        sal_eff = clip01(1.0 - (s - smax) / 10.0)
    else:
        sal_eff = 1.0

    # 栄養塩
    nutrient_eff = clip01(env.get("nutrient", 1.0))

    # 底質
    if default_substrate_factor is None:
        default_substrate_factor = {"mud": 1.0, "sand": 0.7, "rock": 0.85}
    fixation_factor = default_substrate_factor.get(bottom_type, 1.0)

    score = temp_eff * light_eff * sal_eff * nutrient_eff * fixation_factor
    return clip01(float(score))


# ===== 互換用のダミー関数（旧API呼び出しが残っている場合のため） =====
# 既存コードが参照しているなら、内部で新実装に委譲するか、削除して呼び出し側を修正。

def get_nutrient_factor(step: int, total_steps: int) -> float:
    """後方互換: 0..1 の正規化済み栄養塩ファクタ（平均0.5, 振幅0.5）"""
    return clip01(0.5 + 0.5 * np.sin(2 * np.pi * step / float(max(total_steps, 1))))


def get_light_level(step: int, total_steps: int) -> float:
    """非推奨（段差光）。互換のため残す。0.5時点で0/1を分けるだけ。"""
    return 1.0 if step < total_steps * 0.5 else 0.3


def get_temperature(step: int, total_steps: int) -> float:
    """非推奨（線形上昇）。互換のため残す。"""
    return 15.0 + (step / float(max(total_steps, 1))) * 20.0


def get_light_efficiency(depth_m: float, step: int, total_steps: int, kd_m_inv: float = 0.7) -> float:
    """非推奨。Beer–Lambert を使うこと。互換のため残す。"""
    base = math.exp(-max(depth_m, 0.0) / 10.0)  # 旧ロジック
    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * step / float(max(total_steps, 1)))
    return clip01(base * seasonal)
