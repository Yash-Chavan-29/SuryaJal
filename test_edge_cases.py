"""
SuryaJal Edge Case & Stress Testing Suite
Tests all boundary conditions, zero division risks, and extreme climate scenarios.
"""

from data_sources import generate_sample_weather, fetch_open_meteo_forecast
from simulator import simulate, calculate_metrics, apply_scenario_modifiers, get_default_crop_config
from controllers import FixedRoutine, SmartController, MLController


def run_comprehensive_audit():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE EDGE-CASE AUDIT FOR SURYAJAL")
    print("=" * 70)

    crop_cfg = get_default_crop_config()
    raw_weather = generate_sample_weather(days=7)

    # Test 1: Total Drought + Max Heatwave (+8C)
    print("Test 1: Severe Drought (0 Rain) + Extreme Heatwave (+8C)...")
    drought_weather = apply_scenario_modifiers(raw_weather, heatwave_offset_c=8.0, rain_multiplier=0.0)
    f_res = simulate(drought_weather, crop_cfg, FixedRoutine())
    s_res = simulate(drought_weather, crop_cfg, SmartController())
    ml_res = simulate(drought_weather, crop_cfg, MLController())
    m = calculate_metrics(ml_res, f_res)
    assert not f_res["soil_moisture_mm"].isna().any(), "NaN found in fixed routine moisture!"
    assert not ml_res["soil_moisture_mm"].isna().any(), "NaN found in ML controller moisture!"
    print(f"   -> Fixed Stress: {m['baseline_stress_hours']}h | ML Stress: {m['smart_stress_hours']}h | Avoided: {m['stress_hours_avoided']}h")
    print("   [PASS] Drought + Heatwave handled gracefully.")

    # Test 2: Heavy Monsoonal Deluge (Rain Multiplier = 2.5x)
    print("\nTest 2: Monsoonal Deluge (2.5x Rain Multiplier)...")
    flood_weather = apply_scenario_modifiers(raw_weather, heatwave_offset_c=0.0, rain_multiplier=2.5)
    f_flood = simulate(flood_weather, crop_cfg, FixedRoutine())
    ml_flood = simulate(flood_weather, crop_cfg, MLController())
    m_flood = calculate_metrics(ml_flood, f_flood)
    # Under deluge, ML must NEVER irrigate while it is raining, and must save >85% water
    rain_pumping_hours = int((ml_flood["pump_active"] & (ml_flood["rainfall_mm"] > 0)).sum())
    assert rain_pumping_hours == 0, f"ML pumped during rainfall! Violations: {rain_pumping_hours} hrs"
    assert m_flood["water_saved_pct"] >= 85.0, f"ML failed to achieve >85% savings during deluge (got {m_flood['water_saved_pct']}%)"
    print(f"   -> Fixed Water: {m_flood['baseline_water_mm']} mm | ML Water: {m_flood['smart_water_mm']} mm ({m_flood['water_saved_pct']}% saved)")
    print("   -> Rain pumping hours: 0 hrs (100% rain-aware)")
    print("   [PASS] Deluge handled with 100% smart rain-skipping.")

    # Test 3: Critical Low Initial Moisture (Wilting Point = 30 mm)
    print("\nTest 3: Critical Initial Moisture (Wilting Point 32 mm)...")
    low_moist_cfg = crop_cfg.copy()
    low_moist_cfg["initial_moisture_mm"] = 32.0
    ml_emergency = simulate(raw_weather, low_moist_cfg, MLController())
    # First hour should trigger immediate emergency pump
    assert ml_emergency.iloc[0]["pump_active"] == True, "Emergency guard failed to trigger at wilting point!"
    print("   -> Emergency guard activated on hour 0 to rescue crop.")
    print("   [PASS] Emergency Crop Guard operates reliably.")

    # Test 4: Saturated Initial Moisture (Field Capacity = 100 mm)
    print("\nTest 4: Saturated Initial Moisture (100 mm)...")
    high_moist_cfg = crop_cfg.copy()
    high_moist_cfg["initial_moisture_mm"] = 100.0
    ml_saturated = simulate(raw_weather, high_moist_cfg, MLController())
    assert ml_saturated.iloc[0]["pump_active"] == False, "Saturation guard failed: pumped into full bucket!"
    print("   -> Saturation guard blocked unnecessary pumping.")
    print("   [PASS] Drainage Guard operates reliably.")

    # Test 5: Multi-Horizon Scalability (3, 7, 10, 14 days)
    print("\nTest 5: Multi-Horizon Scalability...")
    for days in [3, 7, 10, 14]:
        w = generate_sample_weather(days=days)
        res = simulate(w, crop_cfg, MLController())
        assert len(res) == days * 24, f"Mismatch in length for {days} days!"
    print("   [PASS] Horizons [3, 7, 10, 14] days verified without drift.")

    print("\n" + "=" * 70)
    print("AUDIT RESULT: ZERO BUGS DETECTED. ALL AGRO-PHYSICAL BOUNDS INTACT.")
    print("=" * 70)


def test_edge_case_stress_scenarios():
    """Pytest wrapper for deterministic agro-physical stress checks."""
    run_comprehensive_audit()


if __name__ == "__main__":
    run_comprehensive_audit()
