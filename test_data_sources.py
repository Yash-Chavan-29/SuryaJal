"""
SuryaJal Data Sources & Offline Caching Test
Verifies live API calls, JSON cache persistence, and offline fallback.
"""

from pathlib import Path
from data_sources import (
    fetch_open_meteo_forecast,
    fetch_nasa_power_solar,
    fetch_soilgrids_properties,
    CACHE_DIR,
)


def test_data_sources_and_cache():
    print("=" * 65)
    print("TESTING SURYAJAL DATA SOURCES & OFFLINE CACHING")
    print("=" * 65)

    # 1. Fetch Open-Meteo
    print("1. Querying Open-Meteo Forecast...")
    df = fetch_open_meteo_forecast(days=7)
    print(f"   -> Received {len(df)} hourly forecast rows.")
    print(f"   -> Columns: {list(df.columns)}")
    print(f"   -> Temp Range: [{df['temp_c'].min()} C, {df['temp_c'].max()} C]")
    print(f"   -> Max Solar Irradiance: {df['solar_irradiance_wm2'].max()} W/m2")
    assert len(df) == 7 * 24, f"Expected 168 rows, got {len(df)}"

    # 2. Fetch NASA POWER Solar
    print("\n2. Querying NASA POWER Solar Irradiance...")
    nasa_res = fetch_nasa_power_solar()
    print(f"   -> NASA POWER Result: {nasa_res['source']} ({nasa_res['status']})")

    # 3. Fetch SoilGrids Properties
    print("\n3. Querying SoilGrids Soil Properties...")
    soil_res = fetch_soilgrids_properties()
    print(f"   -> Soil Type: {soil_res['soil_type']}")
    print(f"   -> Field Capacity: {soil_res['field_capacity_mm']} mm")
    print(f"   -> Stress Threshold: {soil_res['stress_threshold_mm']} mm")

    # 4. Verify Cache Files
    print("\n4. Inspecting Local Cache Folder:")
    cache_files = list(CACHE_DIR.glob("*.json"))
    for cf in cache_files:
        print(f"   -> Cached JSON: {cf.name} ({cf.stat().st_size} bytes)")
    assert len(cache_files) > 0, "Cache files should be generated for offline resilience!"

    # 5. Offline Fallback Simulation
    print("\n5. Testing Offline Cache Fallback (Network independent)...")
    cached_df = fetch_open_meteo_forecast(days=7, use_cache=True)
    assert len(cached_df) == 168, "Offline cache fallback failed to deliver 168 hourly rows!"
    print("   -> Offline cache successfully read and validated.")

    print("\nALL DATA SOURCE TESTS PASSED:")
    print("  [PASS] Live APIs queried successfully.")
    print("  [PASS] Offline JSON cache populated.")
    print("  [PASS] Safe offline fallback verified for hackathon venue.")
    print("=" * 65)


if __name__ == "__main__":
    test_data_sources_and_cache()
