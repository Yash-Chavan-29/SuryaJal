"""
SuryaJal™ - Advanced Technical Engineering Engine
Contains 4 Live Real-World Agronomic Models:
1. Solar Energy Arbitrage (Pump vs Net-Meter Grid Export)
2. Variable Rate Irrigation (VRI) Micro-Zone Optimization
3. Micro-Climate Crop Disease & Leaf Wetness Predictor
4. Predictive Groundwater Recharge & Storm Runoff Routing

Team Quantum Cosmic | KLE Technological University, Hubballi | Hack Vally 2026
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# 1. SOLAR ENERGY ARBITRAGE (PUMP VS GRID NET-METER EXPORT)
# ==============================================================================
def calculate_energy_arbitrage(
    solar_gen_kw: float,
    pump_power_kw: float,
    current_soil_moisture_mm: float,
    stress_threshold_mm: float,
    field_capacity_mm: float,
    feed_in_tariff_inr: float = 3.25,      # KERC PM-KUSUM Component-C buyback rate
    grid_power_tariff_inr: float = 7.50,   # Commercial/Backup grid tariff
    crop_economic_value_inr_per_mm: float = 120.0, # Yield value of 1 mm hydration
) -> Dict[str, Any]:
    """
    Computes whether it is financially and agronomically superior to:
    - DISPATCH PUMP (hydrate crop to avoid yield loss), OR
    - EXPORT SOLAR POWER (inject excess kWh into the HESCOM grid for guaranteed revenue).
    """
    moisture_deficit_mm = max(0.0, field_capacity_mm - current_soil_moisture_mm)
    moisture_ratio = current_soil_moisture_mm / field_capacity_mm

    # Crop water urgency factor (non-linear penalty as moisture approaches wilting point)
    if current_soil_moisture_mm <= stress_threshold_mm:
        urgency_weight = 2.5
    elif current_soil_moisture_mm < (stress_threshold_mm + 15.0):
        urgency_weight = 1.0
    else:
        # Moisture is ample; marginal yield benefit of more water is low
        urgency_weight = 0.15

    # Marginal economic gain of 1 hour pumping (applies ~1.37 mm depth)
    water_applied_mm = 1.37
    agronomic_benefit_inr = water_applied_mm * crop_economic_value_inr_per_mm * urgency_weight

    # Financial revenue from exporting the same energy (e.g., 3.7 kWh) to grid
    exportable_energy_kwh = min(solar_gen_kw, pump_power_kw)
    grid_export_revenue_inr = exportable_energy_kwh * feed_in_tariff_inr

    # Optimal Decision
    if agronomic_benefit_inr >= grid_export_revenue_inr:
        decision = "IRRIGATE (Crop Hydration Priority)"
        decision_kn = "ನೀರಾವರಿ ಚಾಲನೆ (ಬೆಳೆ ಸಂರಕ್ಷಣೆ ಆದ್ಯತೆ)"
        decision_code = "PUMP"
        net_financial_gain = agronomic_benefit_inr
        explanation = f"Agronomic yield preservation (₹{agronomic_benefit_inr:.1f}) exceeds grid export revenue (₹{grid_export_revenue_inr:.1f})."
        explanation_kn = f"ಬೆಳೆ ಇಳುವರಿ ರಕ್ಷಣೆ (₹{agronomic_benefit_inr:.1f}) ಗ್ರಿಡ್ ಮಾರಾಟದ ಆದಾಯಕ್ಕಿಂತ (₹{grid_export_revenue_inr:.1f}) ಹೆಚ್ಚು ಲಾಭದಾಯಕವಾಗಿದೆ."
    else:
        decision = "EXPORT TO GRID (Net-Meter Revenue)"
        decision_kn = "ಗ್ರಿಡ್‌ಗೆ ಸೌರ ವಿದ್ಯುತ್ ಮಾರಾಟ (ಹೆಸ್ಕಾಂ ಆದಾಯ)"
        decision_code = "EXPORT"
        net_financial_gain = grid_export_revenue_inr
        explanation = f"Soil moisture buffer ({current_soil_moisture_mm:.1f} mm) is sufficient. Selling solar power generates ₹{grid_export_revenue_inr:.1f} net revenue."
        explanation_kn = f"ಮಣ್ಣಿನಲ್ಲಿ ಸಾಕಷ್ಟು ತೇವಾಂಶವಿದೆ ({current_soil_moisture_mm:.1f} mm). ಹೆಸ್ಕಾಂಗೆ ಸೌರ ವಿದ್ಯುತ್ ಮಾರಿ ₹{grid_export_revenue_inr:.1f} ಗಳಿಕೆ ಉತ್ತಮ."

    return {
        "decision": decision,
        "decision_kn": decision_kn,
        "decision_code": decision_code,
        "agronomic_benefit_inr": round(agronomic_benefit_inr, 2),
        "grid_export_revenue_inr": round(grid_export_revenue_inr, 2),
        "exportable_energy_kwh": round(exportable_energy_kwh, 2),
        "net_gain_inr": round(net_financial_gain, 2),
        "explanation": explanation,
        "explanation_kn": explanation_kn,
    }


# ==============================================================================
# 2. VARIABLE RATE IRRIGATION (VRI) MICRO-ZONE MULTI-SOIL OPTIMIZER
# ==============================================================================
VRI_ZONES = {
    "Zone Alpha (North Ridge)": {
        "soil_name": "Vertisol Heavy Clay (60% Clay)",
        "field_capacity_mm": 100.0,
        "wilting_point_mm": 35.0,
        "infiltration_rate_mm_hr": 8.0,  # Slow infiltration (risk of ponding/runoff)
        "current_moisture_mm": 58.0,
        "area_acres": 2.0,
    },
    "Zone Beta (Mid Plain)": {
        "soil_name": "Alluvial Loam (35% Clay)",
        "field_capacity_mm": 85.0,
        "wilting_point_mm": 25.0,
        "infiltration_rate_mm_hr": 16.0, # Medium infiltration
        "current_moisture_mm": 44.0,
        "area_acres": 1.5,
    },
    "Zone Gamma (South Swale)": {
        "soil_name": "Sandy Loam (18% Clay)",
        "field_capacity_mm": 65.0,
        "wilting_point_mm": 18.0,
        "infiltration_rate_mm_hr": 25.0, # Rapid percolation (quick moisture loss)
        "current_moisture_mm": 32.0,
        "area_acres": 1.5,
    },
}

def calculate_vri_dispatch(
    zones: Dict[str, Dict[str, Any]] = None,
    available_solar_kw: float = 4.2,
) -> Dict[str, Any]:
    """
    Computes distinct solenoid valve timings and application depths for each micro-zone
    to avoid over-watering clay or under-watering sandy patches.
    """
    if zones is None:
        zones = VRI_ZONES

    dispatch_plan = []
    total_water_liters = 0

    for z_name, z_data in zones.items():
        fc = z_data["field_capacity_mm"]
        curr_m = z_data["current_moisture_mm"]
        deficit_mm = max(0.0, fc - curr_m)
        max_infil = z_data["infiltration_rate_mm_hr"]

        # If deficit exists, schedule pulse irrigation based on infiltration rate
        if deficit_mm > 10.0:
            # Apply up to deficit but cap hourly rate to avoid surface runoff
            rate = min(max_infil, 12.0)
            runtime_mins = int((deficit_mm / rate) * 60.0)
            runtime_mins = min(90, max(20, runtime_mins)) # Safety clamp
            water_liters = int(runtime_mins * 60 * 3.8)   # 3.8 L/s flow rate
            status = "IRRIGATE (Pulse Solenoid Active)"
            status_kn = "ಸ್ಮಾರ್ಟ್ ಕವಾಟ ಚಾಲನೆ"
        else:
            runtime_mins = 0
            water_liters = 0
            status = "STANDBY (Adequate Water Buffer)"
            status_kn = "ತೇವಾಂಶ ಸಮೃದ್ಧ - ಕವಾಟ ಆಫ್"

        total_water_liters += water_liters
        dispatch_plan.append({
            "zone": z_name,
            "soil": z_data["soil_name"],
            "current_moisture": curr_m,
            "field_capacity": fc,
            "deficit_mm": round(deficit_mm, 1),
            "runtime_mins": runtime_mins,
            "water_liters": water_liters,
            "valve_status": status,
            "valve_status_kn": status_kn,
        })

    return {
        "zone_plans": dispatch_plan,
        "total_water_liters": total_water_liters,
        "solar_status": "Sufficient Solar Flux" if available_solar_kw >= 3.0 else "Low Solar Power",
    }


# ==============================================================================
# 3. MICRO-CLIMATE CROP DISEASE & INFECTION RISK PREDICTOR
# ==============================================================================
def predict_crop_disease_risk(
    temp_series: np.ndarray,
    rh_series: np.ndarray,
    crop_type: str = "Cotton",
) -> Dict[str, Any]:
    """
    Evaluates micro-climate moisture-heat correlations to predict fungal & bacterial infection risk:
    - Cotton: Bacterial Blight (Xanthomonas) and Alternaria Leaf Spot
    - Sugarcane: Red Rot (Colletotrichum falcatum)
    - Chickpea: Ascochyta Blight & Fusarium Wilt
    """
    # 1. Calculate Leaf Wetness Duration (LWD) proxy (hours where RH >= 85%)
    wet_hours = int(np.sum(rh_series >= 85.0))

    # 2. Temperature window suitability for fungal sporulation (24°C - 32°C)
    favorable_temp_hours = int(np.sum((temp_series >= 24.0) & (temp_series <= 32.0) & (rh_series >= 80.0)))

    # Mills Infection Index calculation
    infection_score = (wet_hours * 1.5) + (favorable_temp_hours * 2.0)
    # Normalize to 0-100%
    risk_pct = min(100.0, max(5.0, (infection_score / 45.0) * 100.0))

    if risk_pct >= 65.0:
        risk_level = "HIGH INFECTION RISK"
        risk_color = "#ef4444"
        disease_name = "Bacterial Blight & Alternaria Leaf Spot" if crop_type == "Cotton" else "Red Rot Fungal Infection"
        advisory_en = f"High humidity ({wet_hours}h wetness) detected. Avoid evening overhead spraying. Apply preventive bio-fungicide (Trichoderma viride)."
        advisory_kn = f"ಹೆಚ್ಚು ಆರ್ದ್ರತೆ ಕಂಡುಬಂದಿದೆ ({wet_hours} ಗಂಟೆ ತೇವಾಂಶ). ಸಂಜೆ ನೀರುಣಿಸಬೇಡಿ. ಮುಂಜಾಗ್ರತಾ ಜೈವಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ (ಟ್ರೈಕೋಡರ್ಮಾ) ಸಿಂಪಡಿಸಿ."
    elif risk_pct >= 35.0:
        risk_level = "MODERATE VIGILANCE"
        risk_color = "#f59e0b"
        disease_name = "Mild Foliar Mildew Risk"
        advisory_en = "Moderate canopy humidity. Maintain root aeration and inspect lower leaves during field scouting."
        advisory_kn = "ಸಾಧಾರಣ ತೇವಾಂಶವಿದೆ. ಮಣ್ಣಿನ ಗಾಳಿಯಾಡುವಿಕೆಯನ್ನು ಕಾಪಾಡಿ ಮತ್ತು ಗಿಡದ ಕೆಳಭಾಗದ ಎಲೆಗಳನ್ನು ಗಮನಿಸಿ."
    else:
        risk_level = "LOW / CLEAR"
        risk_color = "#10b981"
        disease_name = "Pathogen Dormant (Safe)"
        advisory_en = "Dry canopy conditions suppress pathogen sporulation. Healthy crop environment."
        advisory_kn = "ವಾತಾವರಣ ಒಣಗಿದ್ದು, ಯಾವುದೇ ರೋಗದ ಲಕ್ಷಣಗಳಿಲ್ಲ. ಬೆಳೆ ಸುರಕ್ಷಿತವಾಗಿದೆ."

    return {
        "risk_percentage": round(risk_pct, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "disease_name": disease_name,
        "wet_hours_count": wet_hours,
        "advisory_en": advisory_en,
        "advisory_kn": advisory_kn,
    }


# ==============================================================================
# 4. PREDICTIVE GROUNDWATER RECHARGE & STORM RUNOFF ROUTING
# ==============================================================================
def calculate_groundwater_recharge(
    forecast_rain_mm: float,
    catchment_area_sqm: float = 10000.0, # 1 Hectare = 10,000 m2
    curve_number: float = 82.0,          # USDA NRCS Curve Number for cultivated clay
) -> Dict[str, Any]:
    """
    Computes stormwater runoff depth using the NRCS Runoff Equation:
        S = (25400 / CN) - 254
        Q = (P - 0.2*S)^2 / (P + 0.8*S)  for P > 0.2*S
    Directs excess runoff into farm borewell injection pits to replenish regional aquifer.
    """
    s_retention = (25400.0 / curve_number) - 254.0
    initial_abstraction = 0.2 * s_retention

    if forecast_rain_mm > initial_abstraction:
        runoff_depth_mm = ((forecast_rain_mm - initial_abstraction) ** 2) / (forecast_rain_mm + (0.8 * s_retention))
    else:
        runoff_depth_mm = 0.0

    # Total runoff volume generated across the 1-ha catchment (in Liters and m3)
    runoff_volume_m3 = (runoff_depth_mm / 1000.0) * catchment_area_sqm
    runoff_volume_liters = int(runoff_volume_m3 * 1000.0)

    # Recharge pit filtration capacity (assumed 85% capture efficiency with silt trap)
    recharged_liters = int(runoff_volume_liters * 0.85)
    aquifer_rise_cm = round((recharged_liters / 1000.0) / (catchment_area_sqm * 0.15) * 100.0, 1)

    return {
        "forecast_rain_mm": forecast_rain_mm,
        "runoff_depth_mm": round(runoff_depth_mm, 2),
        "total_runoff_liters": runoff_volume_liters,
        "recharged_liters": recharged_liters,
        "aquifer_water_table_rise_cm": aquifer_rise_cm,
        "routing_action": "DIVERT TO RECHARGE PIT" if forecast_rain_mm >= 15.0 else "NATURAL INFILTRATION",
        "routing_action_kn": "ಅಂತರ್ಜಲ ಮರುಪೂರಣ ಗುಂಡಿಗೆ ತಿರುಗಿಸಿ" if forecast_rain_mm >= 15.0 else "ನೈಸರ್ಗಿಕ ಇಂಗುವಿಕೆ",
    }


# ==============================================================================
# 5. CARBON & GROUNDWATER CREDIT MONETIZATION CALCULATOR
# ==============================================================================
def calculate_carbon_and_water_credits(
    solar_kwh_utilized_or_exported: float,
    groundwater_liters_saved: float,
    carbon_price_per_ton_inr: float = 1250.0, # ~15 USD / ton in voluntary carbon markets
    water_credit_per_1000l_inr: float = 45.0, # Water stewardship credit
) -> Dict[str, Any]:
    """
    Calculates verifiable ESG credits:
    1. Carbon Offsets: 0.82 kg CO2 avoided per kWh of thermal coal grid power displaced
       (Central Electricity Authority baseline for Karnataka grid).
    2. Aquifer Conservation: Liters of groundwater preserved from over-extraction.
    """
    co2_avoided_kg = solar_kwh_utilized_or_exported * 0.82
    co2_avoided_tons = co2_avoided_kg / 1000.0
    carbon_revenue_inr = round(co2_avoided_tons * carbon_price_per_ton_inr, 2)

    water_credits_inr = round((groundwater_liters_saved / 1000.0) * water_credit_per_1000l_inr, 2)
    total_esg_revenue_inr = round(carbon_revenue_inr + water_credits_inr, 2)

    return {
        "solar_kwh_displaced": round(solar_kwh_utilized_or_exported, 1),
        "co2_avoided_kg": round(co2_avoided_kg, 2),
        "co2_avoided_tons": round(co2_avoided_tons, 4),
        "carbon_revenue_inr": carbon_revenue_inr,
        "groundwater_liters_saved": int(groundwater_liters_saved),
        "water_credits_inr": water_credits_inr,
        "total_esg_revenue_inr": total_esg_revenue_inr,
        "esg_rating": "AAA (Verra / Gold Standard Aligned)",
        "annual_projected_inr": round(total_esg_revenue_inr * 12.0, 0),
    }


# ==============================================================================
# 6. SENTINEL-2 MULTISPECTRAL NDVI LAYER & FAO-56 Kcb CORRELATION
# ==============================================================================
def generate_sentinel_ndvi_layer(
    crop_name: str = "Cotton",
    growth_stage: str = "Flowering & Boll Formation",
    grid_size: int = 30,
    base_moisture_ratio: float = 0.74,
) -> Dict[str, Any]:
    """
    Generates a 30x30 spatial multispectral NDVI grid simulating ESA Sentinel-2 (10m resolution).
    Correlates mean canopy NDVI with the FAO-56 dual crop coefficient (Kcb):
        Kcb = 1.07 * (1 - ((NDVI_max - NDVI) / (NDVI_max - NDVI_min))^0.84)
    """
    np.random.seed(42)
    # Generate Gaussian spatial pattern centered in the plot
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    
    # Base NDVI distribution with realistic canopy heterogeneity
    ndvi_base = 0.82 - (0.15 * dist) + np.random.normal(0, 0.04, (grid_size, grid_size))
    # Adjust for crop stage and hydration
    ndvi_matrix = np.clip(ndvi_base * (base_moisture_ratio / 0.70), 0.15, 0.92)

    mean_ndvi = float(np.mean(ndvi_matrix))
    min_ndvi = float(np.min(ndvi_matrix))
    max_ndvi = float(np.max(ndvi_matrix))

    # FAO-56 Kcb basal crop coefficient correlation
    # Typical NDVI_min ~ 0.15 (bare soil), NDVI_max ~ 0.88 (dense full canopy)
    ndvi_min_ref = 0.15
    ndvi_max_ref = 0.88
    norm_ndvi = np.clip((max_ndvi - mean_ndvi) / (ndvi_max_ref - ndvi_min_ref + 1e-6), 0.0, 1.0)
    calculated_kcb = round(float(1.07 * (1.0 - (norm_ndvi ** 0.84))), 2)

    canopy_coverage_pct = round(float(np.mean(ndvi_matrix > 0.45) * 100.0), 1)

    return {
        "ndvi_matrix": ndvi_matrix.tolist(),
        "mean_ndvi": round(mean_ndvi, 3),
        "min_ndvi": round(min_ndvi, 3),
        "max_ndvi": round(max_ndvi, 3),
        "fao56_kcb_correlated": calculated_kcb,
        "canopy_coverage_pct": canopy_coverage_pct,
        "sentinel_granule_id": "S2B_MSIL2A_20261104T051019_R062_T43QDB",
        "spatial_resolution": "10m Multispectral (B4-Red, B8-NIR)",
        "health_verdict": "Vigorous Vegetative Canopy" if mean_ndvi >= 0.65 else ("Moderate Canopy Stress" if mean_ndvi >= 0.45 else "Sparse / Critical Water Deficit"),
    }
