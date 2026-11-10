"""
SuryaJal Configuration Module
Handles environment variables, API endpoints, geolocation, and physical defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory & Environment Loading
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# API Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Default Coordinates (Hubballi / Belagavi, Karnataka)
DEFAULT_LATITUDE = 15.3647
DEFAULT_LONGITUDE = 75.1240
LOCATION_NAME = "Hubballi, Karnataka"

# API Endpoints (No key required for weather/solar/soil)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Local Cache Directory for offline hackathon resilience
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Default Agronomic Constants (Soil bucket in mm)
DEFAULT_FIELD_CAPACITY_MM = 100.0   # Max water root zone can hold
DEFAULT_WILTING_POINT_MM = 30.0     # Minimum water before plant wilts
DEFAULT_STRESS_THRESHOLD_MM = 50.0  # Threshold below which crop experiences stress

# Default Solar & Pump Hardware Spec
PUMP_POWER_KW = 3.75                # Standard 5 HP agricultural pump (~3.75 kW)
PUMP_FLOW_RATE_MM_PER_HR = 10.0     # Irrigation application depth (mm/hour)
SOLAR_PANEL_KW_PEAK = 5.0           # 5 kWp agricultural solar panel array

# Multi-Plot Enterprise Database (Agricultural Cooperatives & FPOs)
PLOTS_DATABASE = {
    "Plot Alpha (Plot 14-B)": {
        "plot_id": "Plot 14-B",
        "name": "Plot Alpha (Plot 14-B)",
        "location": "Hubballi-Belagavi Agri Corridor (15.3647°N, 75.1240°E)",
        "lat": 15.3647,
        "lon": 75.1240,
        "area_acres": 2.5,
        "area_ha": 1.01,
        "crop": "Cotton",
        "stage": "Flowering & Boll Formation",
        "soil_type": "Vertisol Heavy Clay (60% Clay)",
        "field_capacity_mm": 100.0,
        "stress_threshold_mm": 50.0,
        "wilting_point_mm": 35.0,
        "infiltration_rate_mm_hr": 5.0,
        "farmer_name": "Prakash S.",
    },
    "Plot Beta (Plot 22-A)": {
        "plot_id": "Plot 22-A",
        "name": "Plot Beta (Plot 22-A)",
        "location": "Dharwad University Farmland (15.4589°N, 75.0078°E)",
        "lat": 15.4589,
        "lon": 75.0078,
        "area_acres": 1.8,
        "area_ha": 0.73,
        "crop": "Maize",
        "stage": "Tasseling & Silking",
        "soil_type": "Alluvial Silt Loam (40% Silt, 35% Sand)",
        "field_capacity_mm": 85.0,
        "stress_threshold_mm": 42.0,
        "wilting_point_mm": 28.0,
        "infiltration_rate_mm_hr": 18.0,
        "farmer_name": "Basavaraj M.",
    },
    "Plot Gamma (Plot 09-C)": {
        "plot_id": "Plot 09-C",
        "name": "Plot Gamma (Plot 09-C)",
        "location": "Ghataprabha River Basin, Belagavi (15.8497°N, 74.4977°E)",
        "lat": 15.8497,
        "lon": 74.4977,
        "area_acres": 3.2,
        "area_ha": 1.29,
        "crop": "Sugarcane",
        "stage": "Grand Growth (Cane Elongation)",
        "soil_type": "Sandy Loam & Alluvium (65% Sand)",
        "field_capacity_mm": 120.0,
        "stress_threshold_mm": 60.0,
        "wilting_point_mm": 40.0,
        "infiltration_rate_mm_hr": 35.0,
        "farmer_name": "Shivanand K.",
    },
}
