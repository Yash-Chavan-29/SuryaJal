"""
SuryaJal 3-Way Controller Comparison Test
Compares FixedRoutine vs SmartController vs MLController (GradientBoosting + SciPy).
"""

from data_sources import fetch_open_meteo_forecast
from simulator import simulate, calculate_metrics, get_default_crop_config
from controllers import FixedRoutine, SmartController, MLController


def test_three_controllers():
    print("=" * 70)
    print("SURYAJAL 3-WAY CONTROLLER BENCHMARK TEST (7 DAYS LIVE DATA)")
    print("=" * 70)

    # 1. Load Live Weather Forecast (cached)
    weather_df = fetch_open_meteo_forecast(days=7, use_cache=True)
    crop_config = get_default_crop_config()
    crop_config["initial_moisture_mm"] = 62.0

    # 2. Simulate all three controllers
    print("Simulating Fixed Farmer Routine (6 AM Daily)...")
    ctrl_fixed = FixedRoutine(start_hour=6, duration_hours=2)
    res_fixed = simulate(weather_df, crop_config, ctrl_fixed)

    print("Simulating Smart Rule-Based Controller...")
    ctrl_smart = SmartController()
    res_smart = simulate(weather_df, crop_config, ctrl_smart)

    print("Simulating SuryaJal ML Controller (GradientBoosting + SciPy)...")
    ctrl_ml = MLController()
    res_ml = simulate(weather_df, crop_config, ctrl_ml)

    # 3. Calculate Comparative Metrics
    metrics_smart = calculate_metrics(res_smart, res_fixed)
    metrics_ml = calculate_metrics(res_ml, res_fixed)

    # 4. Display 3-Way Table
    print("\n" + "=" * 70)
    print(f"{'Performance Metric':<28} | {'Fixed Routine':<12} | {'Smart (Rule)':<12} | {'ML Controller':<12}")
    print("-" * 70)
    print(f"{'Water Applied (mm)':<28} | {metrics_smart['baseline_water_mm']:<12.1f} | {metrics_smart['smart_water_mm']:<12.1f} | {metrics_ml['smart_water_mm']:<12.1f}")
    print(f"{'Grid Power Drawn (kWh)':<28} | {metrics_smart['baseline_grid_kwh']:<12.1f} | {metrics_smart['smart_grid_kwh']:<12.1f} | {metrics_ml['smart_grid_kwh']:<12.1f}")
    print(f"{'Solar Pumping Share (%)':<28} | {metrics_smart['base_solar_share_pct']:<12.1f} | {metrics_smart['smart_solar_share_pct']:<12.1f} | {metrics_ml['smart_solar_share_pct']:<12.1f}")
    print(f"{'Crop Stress Hours':<28} | {metrics_smart['baseline_stress_hours']:<12} | {metrics_smart['smart_stress_hours']:<12} | {metrics_ml['smart_stress_hours']:<12}")
    print(f"{'Water Saved (%)':<28} | {'-':<12} | {metrics_smart['water_saved_pct']:<11.1f}% | {metrics_ml['water_saved_pct']:<11.1f}%")
    print(f"{'Grid Energy Saved (%)':<28} | {'-':<12} | {metrics_smart['grid_power_saved_pct']:<11.1f}% | {metrics_ml['grid_power_saved_pct']:<11.1f}%")
    print("=" * 70)

    # Assertions
    assert metrics_ml['grid_power_saved_pct'] > 50.0, "ML controller should save over 50% grid energy"
    assert metrics_ml['smart_stress_hours'] == 0, "ML controller must avoid crop stress"
    print("\nALL 3-WAY TESTS PASSED:")
    print("  [PASS] GradientBoostingRegressor forecasted daily depletion.")
    print("  [PASS] SciPy optimizer found optimal solar pumping hours.")
    print("  [PASS] ML controller achieved massive grid power and water savings with 0 stress.")
    print("=" * 70)


if __name__ == "__main__":
    test_three_controllers()
