"""
SuryaJal - Historical Backtesting Engine
Evaluates irrigation controllers against multi-week historical weather datasets
(30-day Kharif Monsoon, Pre-Monsoon Summer Heatwave, and Rabi Winter).

Team Quantum Cosmic | KLE Technological University, Hubballi
"""

from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import sys
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from simulator import simulate, calculate_metrics, get_default_crop_config
from controllers import FixedRoutine, SmartController, MLController


def generate_historical_season_weather(
    season: str = "kharif_monsoon",
    days: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic multi-week hourly weather data for Hubballi agro-climatic zone.
    
    Seasons:
    - kharif_monsoon: High humidity, intermittent rainstorms, frequent cloud cover
    - summer_heatwave: Intense 38-42C heat, high ET0, clear skies, zero rainfall
    - rabi_winter: Mild 28C days, cool 14C nights, steady solar, minimal rain
    """
    np.random.seed(random_seed)
    total_hours = days * 24
    
    if season == "kharif_monsoon":
        start_date = datetime(2024, 7, 1, 0, 0)
        base_temp = 27.0
        temp_amplitude = 5.0
        max_irradiance = 800.0
        rain_prob = 0.22
        rain_scale = 9.0
        cloud_base = 0.40  # frequent cloud attenuation
    elif season == "summer_heatwave":
        start_date = datetime(2024, 4, 1, 0, 0)
        base_temp = 34.0
        temp_amplitude = 8.0
        max_irradiance = 1020.0
        rain_prob = 0.01
        rain_scale = 1.0
        cloud_base = 0.95  # clear scorching skies
    else:  # rabi_winter
        start_date = datetime(2024, 11, 1, 0, 0)
        base_temp = 24.0
        temp_amplitude = 7.0
        max_irradiance = 880.0
        rain_prob = 0.04
        rain_scale = 4.0
        cloud_base = 0.90

    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    hours = np.array([ts.hour for ts in timestamps])
    day_indices = np.array([i // 24 for i in range(total_hours)])

    # Diurnal temperature cycle
    diurnal = base_temp + temp_amplitude * np.sin((hours - 8.0) * np.pi / 12.0)
    temp_noise = np.random.normal(0, 0.7, total_hours)
    temp_c = np.round(diurnal + temp_noise, 1)

    # Solar Irradiance model
    solar = np.zeros(total_hours)
    for i, h in enumerate(hours):
        if 6 <= h <= 18:
            curve = np.sin((h - 6) * np.pi / 12.0)
            solar[i] = max(0.0, max_irradiance * (curve ** 1.3))
    
    # Cloud attenuation
    cloud_factor = np.random.uniform(cloud_base, 1.0, total_hours)
    solar_irradiance = np.round(solar * cloud_factor, 1)

    # Evapotranspiration ET0
    et0 = (solar_irradiance / 1000.0) * 0.45 + np.maximum(0.0, (temp_c - 20.0) * 0.016) + 0.02
    et0_mm = np.round(np.maximum(0.01, et0), 2)

    # Rainfall events
    rain_events = (np.random.rand(total_hours) < rain_prob).astype(float)
    rain_amounts = np.random.exponential(rain_scale, total_hours) * rain_events
    # Zero out rain during very sunny midday to keep physics realistic
    rain_amounts[solar_irradiance > 500.0] *= 0.2
    rainfall_mm = np.round(rain_amounts, 1)

    return pd.DataFrame({
        "time": timestamps,
        "hour": hours,
        "day": day_indices,
        "temp_c": temp_c,
        "solar_irradiance_wm2": solar_irradiance,
        "et0_mm": et0_mm,
        "rainfall_mm": rainfall_mm,
    })


def run_historical_backtest(
    season: str = "kharif_monsoon",
    days: int = 30,
    farm_area_ha: float = 1.0,
    initial_moisture_mm: float = 65.0,
) -> Dict[str, Any]:
    """
    Executes a walk-forward multi-week backtest comparing FixedRoutine,
    SmartController, and MLController across an entire historical season.
    """
    weather_df = generate_historical_season_weather(season=season, days=days)
    
    crop_cfg = get_default_crop_config()
    crop_cfg["farm_area_ha"] = farm_area_ha
    crop_cfg["initial_moisture_mm"] = initial_moisture_mm

    # 1. Instantiate controllers
    ctrl_fixed = FixedRoutine(start_hour=6, duration_hours=2)
    ctrl_smart = SmartController()
    ctrl_ml = MLController()

    # 2. Simulate all three across the multi-week timeline
    res_fixed = simulate(weather_df, crop_cfg, ctrl_fixed)
    res_smart = simulate(weather_df, crop_cfg, ctrl_smart)
    res_ml = simulate(weather_df, crop_cfg, ctrl_ml)

    # 3. Compute cumulative metrics
    metrics_smart = calculate_metrics(res_smart, res_fixed, farm_area_ha=farm_area_ha)
    metrics_ml = calculate_metrics(res_ml, res_fixed, farm_area_ha=farm_area_ha)

    # 4. Compute cumulative time series for plotting
    res_fixed["cum_water_mm"] = res_fixed["water_applied_mm"].cumsum()
    res_smart["cum_water_mm"] = res_smart["water_applied_mm"].cumsum()
    res_ml["cum_water_mm"] = res_ml["water_applied_mm"].cumsum()

    res_fixed["cum_grid_kwh"] = res_fixed["grid_power_kw"].cumsum()
    res_smart["cum_grid_kwh"] = res_smart["grid_power_kw"].cumsum()
    res_ml["cum_grid_kwh"] = res_ml["grid_power_kw"].cumsum()

    return {
        "season": season,
        "days": days,
        "farm_area_ha": farm_area_ha,
        "weather_df": weather_df,
        "res_fixed": res_fixed,
        "res_smart": res_smart,
        "res_ml": res_ml,
        "metrics_smart": metrics_smart,
        "metrics_ml": metrics_ml,
        "crop_cfg": crop_cfg,
    }


def print_backtest_report(backtest_results: Dict[str, Any]):
    """Prints a clean ASCII comparison table for the backtest."""
    season = backtest_results["season"].replace("_", " ").title()
    days = backtest_results["days"]
    area = backtest_results["farm_area_ha"]
    m_smart = backtest_results["metrics_smart"]
    m_ml = backtest_results["metrics_ml"]
    rf = backtest_results["res_fixed"]
    rs = backtest_results["res_smart"]
    rm = backtest_results["res_ml"]

    print("\n" + "=" * 78)
    print(f"SURYAJAL HISTORICAL BACKTEST REPORT: {season.upper()} ({days} DAYS / {days*24} HOURS)")
    print(f"Location: Hubballi, Karnataka | Farm Area: {area} ha | Pump: 5 HP + 5 kWp Solar PV")
    print("=" * 78)

    headers = f"{'Evaluation Dimension':<30} | {'Fixed Routine':<13} | {'Smart (Rule)':<13} | {'SuryaJal ML':<13}"
    print(headers)
    print("-" * 78)

    w_fix = m_ml["baseline_water_mm"]
    w_smt = m_smart["smart_water_mm"]
    w_ml = m_ml["smart_water_mm"]
    print(f"{'Total Water Applied (mm)':<30} | {w_fix:<13.1f} | {w_smt:<13.1f} | {w_ml:<13.1f}")

    vol_fix = w_fix * 10 * area
    vol_smt = w_smt * 10 * area
    vol_ml = w_ml * 10 * area
    print(f"{'Water Volume (m³)':<30} | {vol_fix:<13.0f} | {vol_smt:<13.0f} | {vol_ml:<13.0f}")

    g_fix = m_ml["baseline_grid_kwh"]
    g_smt = m_smart["smart_grid_kwh"]
    g_ml = m_ml["smart_grid_kwh"]
    print(f"{'Grid Power Drawn (kWh)':<30} | {g_fix:<13.1f} | {g_smt:<13.1f} | {g_ml:<13.1f}")

    cost_fix = g_fix * 7.50
    cost_smt = g_smt * 7.50
    cost_ml = g_ml * 7.50
    print(f"{'Electricity Cost (₹)':<30} | ₹{cost_fix:<12.0f} | ₹{cost_smt:<12.0f} | ₹{cost_ml:<12.0f}")

    s_fix = m_ml["base_solar_share_pct"]
    s_smt = m_smart["smart_solar_share_pct"]
    s_ml = m_ml["smart_solar_share_pct"]
    print(f"{'Solar Pumping Share (%)':<30} | {s_fix:<12.1f}% | {s_smt:<12.1f}% | {s_ml:<12.1f}%")

    st_fix = m_ml["baseline_stress_hours"]
    st_smt = m_smart["smart_stress_hours"]
    st_ml = m_ml["smart_stress_hours"]
    print(f"{'Crop Stress Hours':<30} | {st_fix:<13} | {st_smt:<13} | {st_ml:<13}")

    print("-" * 78)
    print(f"{'WATER SAVED (%)':<30} | {'-':<13} | {m_smart['water_saved_pct']:<12.1f}% | {m_ml['water_saved_pct']:<12.1f}%")
    print(f"{'GRID POWER SAVED (%)':<30} | {'-':<13} | {m_smart['grid_power_saved_pct']:<12.1f}% | {m_ml['grid_power_saved_pct']:<12.1f}%")
    print(f"{'NET COST SAVED (₹)':<30} | {'-':<13} | ₹{m_smart['cost_saved_inr']:<12.0f} | ₹{m_ml['cost_saved_inr']:<12.0f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuryaJal Historical Backtester")
    parser.add_argument(
        "--season",
        choices=["kharif_monsoon", "summer_heatwave", "rabi_winter", "all"],
        default="kharif_monsoon",
        help="Season to backtest (default: kharif_monsoon)",
    )
    parser.add_argument("--days", type=int, default=30, help="Simulation days (default: 30)")
    parser.add_argument("--area", type=float, default=1.0, help="Farm area in hectares (default: 1.0)")
    args = parser.parse_args()

    seasons = [args.season] if args.season != "all" else ["kharif_monsoon", "summer_heatwave", "rabi_winter"]

    for s in seasons:
        results = run_historical_backtest(season=s, days=args.days, farm_area_ha=args.area)
        print_backtest_report(results)
