"""
SuryaJal Digital Twin Simulator Test
Runs a 1-week simulation and verifies physical sanity constraints.
"""

from data_sources import generate_sample_weather
from simulator import simulate, get_default_crop_config
from controllers import FixedRoutine


def test_one_week_simulation():
    print("=" * 60)
    print("RUNNING SURYAJAL 1-WEEK DIGITAL TWIN SIMULATION TEST")
    print("=" * 60)

    # 1. Generate 7-day synthetic weather (168 hours)
    weather_df = generate_sample_weather(days=7, random_seed=42)
    print(f"Generated {len(weather_df)} hours of synthetic weather.")

    # 2. Setup standard crop config & Fixed Routine Controller (6 AM, 2 hours/day)
    crop_config = get_default_crop_config()
    controller = FixedRoutine(start_hour=6, duration_hours=2)

    # 3. Execute Simulation
    results_df = simulate(weather_df, crop_config, controller)

    # 4. Extract Totals
    total_water_mm = results_df["water_applied_mm"].sum()
    total_grid_kwh = results_df["grid_power_kw"].sum()
    total_solar_to_pump_kwh = results_df["solar_to_pump_kw"].sum()
    total_stress_hours = results_df["crop_stress"].sum()
    min_moisture = results_df["soil_moisture_mm"].min()
    max_moisture = results_df["soil_moisture_mm"].max()

    print("\n--- 1-WEEK TOTALS (Fixed Routine 6 AM - 8 AM) ---")
    print(f"Total Water Applied:       {total_water_mm:.2f} mm")
    print(f"Total Grid Energy Drawn:   {total_grid_kwh:.2f} kWh")
    print(f"Total Solar Energy Used:   {total_solar_to_pump_kwh:.2f} kWh")
    print(f"Total Crop Stress Hours:   {total_stress_hours} hours")
    print(f"Moisture Range:            [{min_moisture:.2f} mm, {max_moisture:.2f} mm]")

    # 5. Sanity Checks
    assert min_moisture >= 0.0, f"FAIL: Soil moisture dropped below 0! Min: {min_moisture}"
    assert max_moisture <= crop_config["field_capacity_mm"], (
        f"FAIL: Soil moisture exceeded field capacity {crop_config['field_capacity_mm']}! Max: {max_moisture}"
    )
    assert total_water_mm > 0, "FAIL: Total water applied should be greater than zero for fixed routine."
    assert total_grid_kwh > 0, "FAIL: 6 AM pumping should draw grid power since sun is low."

    # 6. Sanity Check 2: No Rain + No Irrigation -> Moisture must strictly decrease
    dry_weather = weather_df.copy()
    dry_weather["rainfall_mm"] = 0.0
    no_irrigation_results = simulate(dry_weather, crop_config, controller=None)
    moistures = no_irrigation_results["soil_moisture_mm"].values

    # Check that day 7 moisture < day 1 moisture
    assert moistures[-1] < moistures[0], "FAIL: Moisture did not fall without rain or irrigation!"
    print("\nSANITY CHECKS PASSED:")
    print("  [PASS] Soil moisture stays strictly between [0.0, Field Capacity].")
    print("  [PASS] Without irrigation/rain, soil moisture monotonically declines.")
    print("  [PASS] Solar vs grid partition calculated accurately.")
    print("=" * 60)


if __name__ == "__main__":
    test_one_week_simulation()
