"""
SuryaJal Digital Twin Simulator
Hour-by-hour simulation of farm plot soil moisture, solar generation, and pump operations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import config


# --- FAO-56 Crop Database with Growth Stages & Kc Factors ---
CROP_DATABASE = {
    "Cotton": {
        "name_kn": "ಹತ್ತಿ (Cotton)",
        "stages": {
            "Initial / Germination": 0.35,
            "Vegetative Canopy": 0.75,
            "Flowering & Boll Formation": 1.15,
            "Maturity & Harvest": 0.65,
        },
        "default_stage": "Flowering & Boll Formation",
        "root_depth_m": 1.0,
        "critical_moisture_buffer_mm": 14.0,
    },
    "Sugarcane": {
        "name_kn": "ಕಬ್ಬು (Sugarcane)",
        "stages": {
            "Initial Emergence": 0.40,
            "Tillering & Vegetative": 0.80,
            "Grand Growth (Cane Elongation)": 1.25,
            "Ripening": 0.75,
        },
        "default_stage": "Grand Growth (Cane Elongation)",
        "root_depth_m": 1.2,
        "critical_moisture_buffer_mm": 16.0,
    },
    "Maize": {
        "name_kn": "ಮೆಕ್ಕೆಜೋಳ (Maize)",
        "stages": {
            "Initial": 0.30,
            "Vegetative": 0.80,
            "Tasseling & Silking": 1.20,
            "Maturity": 0.60,
        },
        "default_stage": "Tasseling & Silking",
        "root_depth_m": 0.8,
        "critical_moisture_buffer_mm": 12.0,
    },
    "Groundnut": {
        "name_kn": "ಕಡಲೆಕಾಯಿ (Groundnut)",
        "stages": {
            "Initial Emergence": 0.40,
            "Vegetative Growth": 0.70,
            "Pegging & Pod Formation": 1.05,
            "Maturity": 0.60,
        },
        "default_stage": "Pegging & Pod Formation",
        "root_depth_m": 0.6,
        "critical_moisture_buffer_mm": 10.0,
    },
}


def get_default_crop_config(crop_name: str = "Cotton", stage_name: str = None) -> Dict[str, Any]:
    """Returns standard agronomic and hardware parameters for simulation with dynamic Kc."""
    if crop_name not in CROP_DATABASE:
        crop_name = "Cotton"
    
    crop_info = CROP_DATABASE[crop_name]
    stages = crop_info["stages"]
    
    if stage_name not in stages:
        stage_name = crop_info["default_stage"]
        
    kc = stages[stage_name]

    return {
        "crop_name": crop_name,
        "growth_stage": stage_name,
        "kc": kc,  # Dynamic FAO crop coefficient
        "farm_area_ha": 1.0,  # 1 Hectare (10,000 m2; 1 mm depth = 10 m3 = 10,000 Liters)
        "field_capacity_mm": config.DEFAULT_FIELD_CAPACITY_MM,
        "wilting_point_mm": config.DEFAULT_WILTING_POINT_MM,
        "stress_threshold_mm": config.DEFAULT_STRESS_THRESHOLD_MM,
        "initial_moisture_mm": 60.0,
        "pump_flow_rate_mm_hr": config.PUMP_FLOW_RATE_MM_PER_HR,
        "pump_power_kw": config.PUMP_POWER_KW,
        "solar_kw_peak": config.SOLAR_PANEL_KW_PEAK,
        "solar_derating": 0.85,  # Temperature and inverter efficiency
        "electricity_rate_inr_kwh": 7.50,  # Tariff for commercial/grid backup power
    }


def explain_decision(telemetry: Dict[str, Any], crop_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explainable AI (XAI) Causal Engine:
    Constructs a 4-pillar, human-understandable explanation for why the AI decided
    to pump or hold irrigation.
    """
    moisture = float(telemetry.get("moisture_mm", 60.0))
    stress_thresh = float(crop_config.get("stress_threshold_mm", 50.0))
    rain_fc = float(telemetry.get("rain_forecast_mm", 0.0))
    pumping = bool(telemetry.get("pump_active_today", False))
    planned_hours = telemetry.get("planned_hours", [])
    solar_share = float(telemetry.get("solar_share_pct", 90.0))
    cost_saved = float(telemetry.get("cost_saved_inr", 35.0))
    crop_name = crop_config.get("crop_name", "Cotton")
    stage_name = crop_config.get("growth_stage", "Flowering")
    kc = float(crop_config.get("kc", 1.0))

    buffer_mm = round(moisture - stress_thresh, 1)

    if not pumping:
        if rain_fc >= 4.0:
            pillar1 = f"Moisture is {moisture:.1f} mm (+{buffer_mm} mm above danger line)."
            pillar1_kn = f"ಮಣ್ಣಿನಲ್ಲಿ {moisture:.1f} mm ತೇವಾಂಶವಿದೆ (ಅಪಾಯದ ರೇಖೆಗಿಂತ +{buffer_mm} mm ಹೆಚ್ಚು)."
            pillar2 = f"Rain forecast of {rain_fc:.1f} mm arriving within 12h with high probability."
            pillar2_kn = f"ಮುಂದಿನ 12 ಗಂಟೆಗಳಲ್ಲಿ {rain_fc:.1f} mm ಮಳೆ ಬರುವ ನಿರೀಕ್ಷೆ ಇದೆ."
            pillar3 = f"Pumping now would waste ~₹{cost_saved:.0f} in unnecessary energy and leach nutrients."
            pillar3_kn = f"ಈಗ ನೀರು ಹಾಯಿಸಿದರೆ ₹{cost_saved:.0f} ವಿದ್ಯುತ್ ವ್ಯರ್ಥವಾಗುತ್ತದೆ ಮತ್ತು ಪೋಷಕಾಂಶಗಳು ಕೊಚ್ಚಿಹೋಗುತ್ತವೆ."
            pillar4 = "Safety Guard: Standing by. Next automated satellite & soil re-check in 1 hour."
            pillar4_kn = "ಸುರಕ್ಷತಾ ನಿಗಾ: ಸಕ್ರಿಯವಾಗಿದೆ. 1 ಗಂಟೆಯ ನಂತರ ಮರು-ಪರಿಶೀಲನೆ ನಡೆಯಲಿದೆ."
            verdict = "PUMP OFF: Rain Recharge Anticipated"
            verdict_kn = "ಪಂಪ್ ಆಫ್: ಮಳೆಯಿಂದ ಸಹಜ ನೀರುಣಿಕೆ"
        else:
            pillar1 = f"Moisture is {moisture:.1f} mm (+{buffer_mm} mm buffer). Crop root zone is well-hydrated."
            pillar1_kn = f"ತೇವಾಂಶ {moisture:.1f} mm (ಸುರಕ್ಷಿತ ಬಫರ್ +{buffer_mm} mm). ಬೇರುಗಳಲ್ಲಿ ನೀರಿನ ಕೊರತೆಯಿಲ್ಲ."
            pillar2 = f"Zero/light rain ({rain_fc:.1f} mm), but evapotranspiration rate (Kc={kc:.2f}) is manageable."
            pillar2_kn = f"ಮಳೆ ಇಲ್ಲ ({rain_fc:.1f} mm), ಆದರೆ ಬೆಳೆಯ ಇಂಗುವಿಕೆ ದರ (Kc={kc:.2f}) ನಿಯಂತ್ರಣದಲ್ಲಿದೆ."
            pillar3 = f"Preserving groundwater reserves; current soil water is within optimal growth limits."
            pillar3_kn = f"ಅಂತರ್ಜಲ ಉಳಿತಾಯ; ಪ್ರಸ್ತುತ ಮಣ್ಣಿನ ನೀರು ಸೂಕ್ತ ಬೆಳವಣಿಗೆಯ ಮಿತಿಯಲ್ಲಿದೆ."
            pillar4 = "Safety Guard: Will trigger solar pumping if moisture drops below trigger margin."
            pillar4_kn = "ಸುರಕ್ಷತಾ ನಿಗಾ: ತೇವಾಂಶ ಮಿತಿಗಿಂತ ಕಡಿಮೆಯಾದರೆ ಮಧ್ಯಾಹ್ನ ಸೌರ ಪಂಪ್ ಆನ್ ಆಗಲಿದೆ."
            verdict = "PUMP OFF: Soil in Optimal Comfort Zone"
            verdict_kn = "ಪಂಪ್ ಆಫ್: ಮಣ್ಣಿನ ತೇವಾಂಶ ಸಮೃದ್ಧವಾಗಿದೆ"
    else:
        hours_str = ", ".join([f"{h}:00" for h in planned_hours]) if planned_hours else "11:30 - 13:30"
        pillar1 = f"Moisture is {moisture:.1f} mm (approaching danger line {stress_thresh:.1f} mm for {crop_name} {stage_name})."
        pillar1_kn = f"ತೇವಾಂಶ {moisture:.1f} mm ಗೆ ಇಳಿದಿದೆ ({crop_name} {stage_name} ಗೆ ನೀರು ತಕ್ಷಣ ಅಗತ್ಯ)."
        pillar2 = f"No significant rain forecast in the next 18 hours to assist the crop."
        pillar2_kn = f"ಮುಂದಿನ 18 ಗಂಟೆಗಳಲ್ಲಿ ಯಾವುದೇ ಮಳೆಯ ಮುನ್ಸೂಚನೆ ಇಲ್ಲ."
        pillar3 = f"Optimized window at {hours_str} aligns with peak 5 kWp PV sunshine ({solar_share:.0f}% free solar)."
        pillar3_kn = f"{hours_str} ಗಂಟೆಯ ಅವಧಿಯಲ್ಲಿ ಗರಿಷ್ಠ ಸೌರಶಕ್ತಿ ಇರಲಿದ್ದು, {solar_share:.0f}% ಉಚಿತ ವಿದ್ಯುತ್ ಸಿಗಲಿದೆ."
        pillar4 = f"Financial Impact: Running now saves ₹{cost_saved:.0f} vs pumping from grid electricity tonight."
        pillar4_kn = f"ಆರ್ಥಿಕ ಲಾಭ: ರಾತ್ರಿ ಗ್ರಿಡ್ ವಿದ್ಯುತ್ ಬಳಸುವುದಕ್ಕಿಂತ ಈಗ ₹{cost_saved:.0f} ಉಳಿತಾಯವಾಗಲಿದೆ."
        verdict = f"PUMP ON: Peak Solar Window ({hours_str})"
        verdict_kn = f"ಪಂಪ್ ಚಾಲೂ: ಗರಿಷ್ಠ ಸೌರಶಕ್ತಿ ಅವಧಿ ({hours_str})"

    return {
        "verdict": verdict,
        "verdict_kn": verdict_kn,
        "pillar1_soil": pillar1,
        "pillar1_soil_kn": pillar1_kn,
        "pillar2_weather": pillar2,
        "pillar2_weather_kn": pillar2_kn,
        "pillar3_economic": pillar3,
        "pillar3_economic_kn": pillar3_kn,
        "pillar4_safety": pillar4,
        "pillar4_safety_kn": pillar4_kn,
    }


def apply_scenario_modifiers(
    weather_df: pd.DataFrame,
    heatwave_offset_c: float = 0.0,
    rain_multiplier: float = 1.0,
    cloud_reduction_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Applies what-if scenario modifiers to base weather timeline:
    - heatwave_offset_c: Adds temperature delta (°C) and scales ET0 accordingly.
    - rain_multiplier: Scales rainfall depth (0.0 = total drought, 1.0 = normal, 2.0 = excessive rain).
    - cloud_reduction_pct: Diminishes solar irradiance due to overcast skies (0% to 80%).
    """
    df = weather_df.copy()

    # 1. Heatwave modifier
    df["temp_c"] = np.round(df["temp_c"] + heatwave_offset_c, 1)
    # Heatwave increases reference evapotranspiration (~3% per degree C)
    if heatwave_offset_c > 0:
        df["et0_mm"] = np.round(df["et0_mm"] * (1.0 + 0.035 * heatwave_offset_c), 2)

    # 2. Rainfall modifier (Drought vs Monsoonal rain)
    df["rainfall_mm"] = np.round(np.maximum(0.0, df["rainfall_mm"] * rain_multiplier), 1)

    # 3. Cloud cover modifier (Diminishes solar irradiance)
    solar_retention = max(0.1, 1.0 - (cloud_reduction_pct / 100.0))
    df["solar_irradiance_wm2"] = np.round(df["solar_irradiance_wm2"] * solar_retention, 1)

    return df


def simulate(
    weather_df: pd.DataFrame,
    crop_config: Dict[str, Any] = None,
    controller: Any = None,
) -> pd.DataFrame:
    """
    Simulates hourly soil water balance and power usage over the timeline in weather_df.

    Parameters:
        weather_df (pd.DataFrame): Hourly dataframe with columns:
            ['time', 'hour', 'temp_c', 'solar_irradiance_wm2', 'et0_mm', 'rainfall_mm']
        crop_config (dict): Agronomic constraints and pump/solar parameters.
        controller: Any object implementing decide_irrigation(step, moisture, weather_df, crop_config).

    Returns:
        pd.DataFrame: Simulation timeline containing:
            ['time', 'hour', 'temp_c', 'rainfall_mm', 'et0_mm', 'solar_irradiance_wm2',
             'soil_moisture_mm', 'pump_active', 'water_applied_mm', 'drainage_mm',
             'solar_gen_kw', 'solar_to_pump_kw', 'grid_power_kw', 'crop_stress']
    """
    if crop_config is None:
        crop_config = get_default_crop_config()

    field_capacity = float(crop_config.get("field_capacity_mm", 100.0))
    wilting_point = float(crop_config.get("wilting_point_mm", 30.0))
    stress_threshold = float(crop_config.get("stress_threshold_mm", 50.0))
    current_moisture = float(crop_config.get("initial_moisture_mm", 60.0))
    kc = float(crop_config.get("kc", 1.0))

    pump_flow_rate = float(crop_config.get("pump_flow_rate_mm_hr", config.PUMP_FLOW_RATE_MM_PER_HR))
    pump_power = float(crop_config.get("pump_power_kw", config.PUMP_POWER_KW))
    solar_kw_peak = float(crop_config.get("solar_kw_peak", config.SOLAR_PANEL_KW_PEAK))
    solar_derating = float(crop_config.get("solar_derating", 0.85))

    total_steps = len(weather_df)
    results = []

    for step in range(total_steps):
        row = weather_df.iloc[step]
        hour = int(row["hour"])
        temp_c = float(row["temp_c"])
        rainfall = float(row["rainfall_mm"])
        et0 = float(row["et0_mm"])
        irradiance = float(row["solar_irradiance_wm2"])

        # 1. Solar Generation (kW) available at this hour
        solar_gen_kw = max(0.0, solar_kw_peak * (irradiance / 1000.0) * solar_derating)

        # 2. Controller decides whether to run the pump
        if controller is not None:
            pump_active = bool(
                controller.decide_irrigation(
                    current_step=step,
                    current_moisture_mm=current_moisture,
                    weather_df=weather_df,
                    crop_config=crop_config,
                )
            )
        else:
            pump_active = False

        # 3. Pump Power and Water Application
        if pump_active:
            water_applied = pump_flow_rate
            solar_to_pump = min(pump_power, solar_gen_kw)
            grid_power = max(0.0, pump_power - solar_gen_kw)
        else:
            water_applied = 0.0
            solar_to_pump = 0.0
            grid_power = 0.0

        # 4. Soil Water Balance (Bucket Model)
        water_in = rainfall + water_applied
        potential_etc = et0 * kc
        available_water = current_moisture + water_in
        actual_et = min(available_water, potential_etc)

        raw_moisture = current_moisture + water_in - actual_et

        if raw_moisture > field_capacity:
            drainage = raw_moisture - field_capacity
            final_moisture = field_capacity
        else:
            drainage = 0.0
            final_moisture = max(0.0, raw_moisture)

        # 5. Crop Stress Check
        is_stressed = final_moisture < stress_threshold

        results.append({
            "time": row["time"],
            "hour": hour,
            "temp_c": temp_c,
            "rainfall_mm": rainfall,
            "et0_mm": et0,
            "solar_irradiance_wm2": irradiance,
            "soil_moisture_mm": round(final_moisture, 2),
            "pump_active": pump_active,
            "water_applied_mm": round(water_applied, 2),
            "drainage_mm": round(drainage, 2),
            "solar_gen_kw": round(solar_gen_kw, 2),
            "solar_to_pump_kw": round(solar_to_pump, 2),
            "grid_power_kw": round(grid_power, 2),
            "crop_stress": is_stressed,
        })

        current_moisture = final_moisture

    return pd.DataFrame(results)


def calculate_metrics(
    smart_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    farm_area_ha: float = 1.0,
    tariff_inr_per_kwh: float = 7.50,
) -> Dict[str, Any]:
    """
    Computes comprehensive savings and impact metrics comparing SmartController against FixedRoutine.
    """
    base_water_mm = float(baseline_results["water_applied_mm"].sum())
    smart_water_mm = float(smart_results["water_applied_mm"].sum())

    base_grid_kwh = float(baseline_results["grid_power_kw"].sum())
    smart_grid_kwh = float(smart_results["grid_power_kw"].sum())

    base_solar_kwh = float(baseline_results["solar_to_pump_kw"].sum())
    smart_solar_kwh = float(smart_results["solar_to_pump_kw"].sum())

    base_stress_hours = int(baseline_results["crop_stress"].sum())
    smart_stress_hours = int(smart_results["crop_stress"].sum())

    # Pumping hours
    base_pump_hours = int(baseline_results["pump_active"].sum())
    smart_pump_hours = int(smart_results["pump_active"].sum())

    # Solar-hour pumping share (% of pumping powered by solar vs total pumping energy)
    base_total_energy = base_grid_kwh + base_solar_kwh
    smart_total_energy = smart_grid_kwh + smart_solar_kwh

    base_solar_share_pct = (
        (base_solar_kwh / base_total_energy) * 100.0 if base_total_energy > 0 else 0.0
    )
    smart_solar_share_pct = (
        (smart_solar_kwh / smart_total_energy) * 100.0 if smart_total_energy > 0 else 0.0
    )

    # Percentage savings
    water_saved_pct = (
        ((base_water_mm - smart_water_mm) / base_water_mm) * 100.0 if base_water_mm > 0 else 0.0
    )
    grid_power_saved_pct = (
        ((base_grid_kwh - smart_grid_kwh) / base_grid_kwh) * 100.0 if base_grid_kwh > 0 else 0.0
    )
    stress_hours_avoided = max(0, base_stress_hours - smart_stress_hours)

    # Real-world volume translation (1 mm on 1 ha = 10 m3 = 10,000 Liters)
    water_saved_mm = max(0.0, base_water_mm - smart_water_mm)
    water_saved_m3 = water_saved_mm * 10.0 * farm_area_ha
    water_saved_liters = water_saved_m3 * 1000.0

    # Economic impact (electricity bill saved)
    grid_kwh_saved = max(0.0, base_grid_kwh - smart_grid_kwh)
    cost_saved_inr = grid_kwh_saved * tariff_inr_per_kwh

    return {
        "baseline_water_mm": round(base_water_mm, 1),
        "smart_water_mm": round(smart_water_mm, 1),
        "water_saved_mm": round(water_saved_mm, 1),
        "water_saved_pct": round(water_saved_pct, 1),
        "water_saved_m3": round(water_saved_m3, 0),
        "water_saved_liters": int(water_saved_liters),
        "baseline_grid_kwh": round(base_grid_kwh, 1),
        "smart_grid_kwh": round(smart_grid_kwh, 1),
        "grid_kwh_saved": round(grid_kwh_saved, 1),
        "grid_power_saved_pct": round(grid_power_saved_pct, 1),
        "cost_saved_inr": round(cost_saved_inr, 2),
        "base_solar_share_pct": round(base_solar_share_pct, 1),
        "smart_solar_share_pct": round(smart_solar_share_pct, 1),
        "base_pump_hours": base_pump_hours,
        "smart_pump_hours": smart_pump_hours,
        "baseline_stress_hours": base_stress_hours,
        "smart_stress_hours": smart_stress_hours,
        "stress_hours_avoided": stress_hours_avoided,
    }
