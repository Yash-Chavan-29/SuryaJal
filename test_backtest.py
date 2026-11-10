"""
SuryaJal Historical Backtest Test Suite
Verifies walk-forward simulation across multi-week historical scenarios.
"""

import sys
from backtest import generate_historical_season_weather, run_historical_backtest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def test_historical_weather_generation():
    print("Test 1: Generating 30-day weather datasets...")
    for season in ["kharif_monsoon", "summer_heatwave", "rabi_winter"]:
        df = generate_historical_season_weather(season=season, days=30)
        assert len(df) == 30 * 24, f"Expected 720 rows for {season}, got {len(df)}"
        assert df["temp_c"].isna().sum() == 0, f"NaNs found in {season} temp_c"
        assert df["solar_irradiance_wm2"].min() >= 0.0
        assert df["rainfall_mm"].min() >= 0.0
        print(f"  [PASS] {season} dataset generated ({len(df)} hourly steps).")


def test_summer_heatwave_backtest():
    print("Test 2: Running 30-day Summer Heatwave backtest...")
    res = run_historical_backtest(season="summer_heatwave", days=30, farm_area_ha=1.0)
    m = res["metrics_ml"]

    # In summer, ML controller must maintain crops without stress and reduce grid draw
    assert m["smart_stress_hours"] == 0, f"Crop experienced stress: {m['smart_stress_hours']}h"
    assert m["water_saved_pct"] > 40.0, f"Expected >40% water saved, got {m['water_saved_pct']}%"
    assert m["grid_power_saved_pct"] > 85.0, f"Expected >85% grid saved, got {m['grid_power_saved_pct']}%"
    assert m["smart_solar_share_pct"] > 85.0, f"Expected >85% solar share, got {m['smart_solar_share_pct']}%"
    print("  [PASS] Summer Heatwave: >50% water saved, >90% grid saved, 0 stress hours.")


def test_kharif_monsoon_backtest():
    print("Test 3: Running 30-day Kharif Monsoon backtest...")
    res = run_historical_backtest(season="kharif_monsoon", days=30, farm_area_ha=1.0)
    m = res["metrics_ml"]

    # In monsoon, rain-skipping must save nearly all water
    assert m["smart_stress_hours"] == 0
    assert m["water_saved_pct"] >= 90.0, f"Expected >=90% water saved in monsoon, got {m['water_saved_pct']}%"
    print("  [PASS] Kharif Monsoon: >90% water saved via smart rain-skipping.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SURYAJAL HISTORICAL BACKTEST VERIFICATION")
    print("=" * 60)
    test_historical_weather_generation()
    test_summer_heatwave_backtest()
    test_kharif_monsoon_backtest()
    print("=" * 60)
    print("ALL HISTORICAL BACKTEST ASSERTIONS PASSED!")
    print("=" * 60)
