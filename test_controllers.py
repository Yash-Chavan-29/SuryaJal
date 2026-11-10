"""
SuryaJal Head-to-Head Controller Test
Compares FixedRoutine vs SmartController on identical 1-week weather data.
"""

from data_sources import generate_sample_weather
from simulator import simulate, calculate_metrics, get_default_crop_config
from controllers import FixedRoutine, SmartController


def test_head_to_head_comparison():
    print("=" * 65)
    print("SURYAJAL HEAD-TO-HEAD CONTROLLER COMPARISON (7 DAYS)")
    print("=" * 65)

    # 1. Generate identical weather
    weather_df = generate_sample_weather(days=7, random_seed=42)
    crop_config = get_default_crop_config()
    crop_config["initial_moisture_mm"] = 60.0  # Standard field condition ready for irrigation cycle

    # 2. Simulate Fixed Routine (6:00 AM - 8:00 AM daily)
    fixed_ctrl = FixedRoutine(start_hour=6, duration_hours=2)
    fixed_results = simulate(weather_df, crop_config, fixed_ctrl)

    # 3. Simulate Smart Controller (Moisture + Solar peak + Rain skip)
    smart_ctrl = SmartController()
    smart_results = simulate(weather_df, crop_config, smart_ctrl)

    # 4. Calculate comparative metrics
    metrics = calculate_metrics(smart_results, fixed_results)

    # 5. Display comparison
    print(f"{'Metric':<30} | {'Fixed Routine':<15} | {'Smart Controller':<15}")
    print("-" * 65)
    print(f"{'Water Applied (mm)':<30} | {metrics['baseline_water_mm']:<15.1f} | {metrics['smart_water_mm']:<15.1f}")
    print(f"{'Grid Energy Drawn (kWh)':<30} | {metrics['baseline_grid_kwh']:<15.1f} | {metrics['smart_grid_kwh']:<15.1f}")
    print(f"{'Crop Stress Hours':<30} | {metrics['baseline_stress_hours']:<15} | {metrics['smart_stress_hours']:<15}")
    print("-" * 65)
    print(f"-> WATER SAVED:       {metrics['water_saved_pct']:.1f}%")
    print(f"-> GRID POWER SAVED:  {metrics['grid_power_saved_pct']:.1f}%")
    print(f"-> STRESS AVOIDED:    {metrics['stress_hours_avoided']} hours")
    print("=" * 65)

    # 6. Physical Assertions
    assert metrics['water_saved_pct'] > 0, "SmartController should save water compared to rigid daily routine"
    assert metrics['grid_power_saved_pct'] > 0, "SmartController should save significant grid power by shifting to solar hours"
    assert metrics['smart_stress_hours'] == 0, "SmartController must maintain crop health with zero stress hours"

    print("ALL ASSERTIONS PASSED:")
    print("  [PASS] Both controllers executed on identical weather inputs.")
    print("  [PASS] SmartController demonstrated measurable water savings.")
    print("  [PASS] SmartController shifted pumping to solar peak, slashing grid draw.")
    print("  [PASS] Crop experienced zero stress hours.")
    print("=" * 65)


if __name__ == "__main__":
    test_head_to_head_comparison()
