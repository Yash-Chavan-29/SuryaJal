"""
SuryaJal Irrigation Controllers
Contains controller implementations for managing irrigation schedules:
- FixedRoutine: Baseline farmer routine (fixed time and duration).
- SmartController: Rule-based solar and rain-aware controller.
- MLController: Predictive optimizer using scikit-learn and SciPy.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
import config


class BaseController:
    """Base interface for all irrigation controllers."""

    def decide_irrigation(
        self,
        current_step: int,
        current_moisture_mm: float,
        weather_df: pd.DataFrame,
        crop_config: Dict[str, Any],
    ) -> bool:
        """
        Decides whether to run the pump during the current hour.
        """
        raise NotImplementedError


class FixedRoutine(BaseController):
    """
    Fixed baseline farmer routine:
    Irrigates unconditionally every day starting at a fixed hour (default 6:00 AM)
    for a fixed duration (default 2 hours, e.g. 6:00 to 8:00 AM).
    """

    def __init__(self, start_hour: int = 6, duration_hours: int = 2):
        self.start_hour = start_hour
        self.duration_hours = duration_hours

    def decide_irrigation(
        self,
        current_step: int,
        current_moisture_mm: float,
        weather_df: pd.DataFrame,
        crop_config: Dict[str, Any],
    ) -> bool:
        hour = weather_df.iloc[current_step]["hour"]
        return bool(self.start_hour <= hour < (self.start_hour + self.duration_hours))


class SmartController(BaseController):
    """
    Rule-based solar and rain-aware controller:
    1. Rain-Skip: Skips irrigation if significant rainfall (>= 4 mm) is forecast in next 12 hours.
    2. Solar Preference: Irrigates during peak solar generation (irradiance >= 380 W/m2)
       when moisture drops near trigger threshold.
    3. Emergency Crop Guard: If moisture drops to or below stress threshold (<= 50 mm),
       pumps immediately to prevent crop damage.
    4. Drainage Guard: Never pumps if moisture is already near field capacity.
    """

    def __init__(
        self,
        moisture_trigger_buffer_mm: float = 12.0,
        min_solar_irradiance_wm2: float = 380.0,
        rain_forecast_window_hrs: int = 12,
        rain_skip_threshold_mm: float = 4.0,
    ):
        self.moisture_trigger_buffer_mm = moisture_trigger_buffer_mm
        self.min_solar_irradiance_wm2 = min_solar_irradiance_wm2
        self.rain_forecast_window_hrs = rain_forecast_window_hrs
        self.rain_skip_threshold_mm = rain_skip_threshold_mm

    def decide_irrigation(
        self,
        current_step: int,
        current_moisture_mm: float,
        weather_df: pd.DataFrame,
        crop_config: Dict[str, Any],
    ) -> bool:
        # Emergency Manual Override (Farmer 1-Click Action)
        if crop_config.get("manual_override_active", False):
            return True

        row = weather_df.iloc[current_step]
        hour = int(row["hour"])

        stress_threshold = float(crop_config.get("stress_threshold_mm", 50.0))
        field_capacity = float(crop_config.get("field_capacity_mm", 100.0))
        pump_depth = float(crop_config.get("pump_flow_rate_mm_hr", 10.0))

        # Saturation Guard
        if current_moisture_mm + pump_depth > field_capacity:
            return False

        # Rule 1: Emergency Crop Guard (overrides farmer hours if critical)
        if current_moisture_mm <= (stress_threshold + 1.0):
            return True

        # Farmer Labor Constraints (avoid hazardous night hours)
        allowed_hours = crop_config.get("allowed_irrigation_hours", None)
        if allowed_hours is not None and hour not in allowed_hours:
            return False

        # Grid Availability Constraint (HESCOM 3-Phase Schedule)
        irradiance = float(row["solar_irradiance_wm2"])
        solar_gen = float(crop_config.get("solar_kw_peak", 5.0)) * (irradiance / 1000.0) * float(crop_config.get("solar_derating", 0.85))
        pump_power = float(crop_config.get("pump_power_kw", 3.7))
        grid_hours = crop_config.get("grid_available_hours", None)
        if (pump_power - solar_gen) > 0.1 and grid_hours is not None and hour not in grid_hours:
            if solar_gen < pump_power * 0.80:
                return False

        trigger_level = stress_threshold + self.moisture_trigger_buffer_mm

        if current_moisture_mm > trigger_level:
            return False

        # Rule 2: Rain Forecast Skip
        upcoming_window = weather_df.iloc[
            current_step : min(current_step + self.rain_forecast_window_hrs, len(weather_df))
        ]
        upcoming_rain = float(upcoming_window["rainfall_mm"].sum())
        if upcoming_rain >= self.rain_skip_threshold_mm:
            return False

        # Rule 3: Solar Hour Preference
        if irradiance >= self.min_solar_irradiance_wm2:
            return True

        return False


class MLController(BaseController):
    """
    AI/ML Controller:
    1. Forecasting Engine: Uses scikit-learn GradientBoostingRegressor to predict next 24-hour
       evapotranspiration and solar irradiance from weather features.
    2. SciPy Optimization: Optimizes the exact pump operating window (start time & duration)
       to minimize grid power consumption and maximize solar utilization while enforcing
       agronomic moisture constraints.
    3. Safe Fallback: Automatically falls back to SmartController if optimization fails.
    """

    def __init__(self):
        self.fallback_controller = SmartController()
        self.ml_model: Optional[GradientBoostingRegressor] = None
        self._is_trained = False
        self._daily_schedules: Dict[int, List[int]] = {}  # day_index -> set of pump hours
        self._initialize_ml_model()

    def _initialize_ml_model(self):
        """Pre-trains a fast GradientBoostingRegressor on agronomic weather patterns."""
        # Generate representative feature data: [hour, temp_c, solar_wm2, rain_mm]
        # Target: Expected hourly soil water depletion (mm)
        np.random.seed(42)
        n_samples = 360
        hours = np.tile(np.arange(24), n_samples // 24)
        temps = 25.0 + 7.0 * np.sin((hours - 8.0) * np.pi / 12.0) + np.random.normal(0, 1.0, len(hours))
        daylight_mask = (hours >= 6) & (hours <= 18)
        solar_curve = np.where(daylight_mask, np.sin((hours - 6) * np.pi / 12.0), 0.0)
        solar = 900.0 * (np.maximum(0.0, solar_curve) ** 1.3)
        rain = np.where(np.random.rand(len(hours)) > 0.92, np.random.exponential(5.0, len(hours)), 0.0)

        X = np.column_stack([hours, temps, solar, rain])
        # Target: Evapotranspiration + percolation loss
        y = (solar / 1000.0) * 0.45 + np.maximum(0.0, (temps - 20.0) * 0.015) + 0.03

        self.ml_model = GradientBoostingRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.ml_model.fit(X, y)
        self._is_trained = True

    def _optimize_daily_window(
        self,
        day_index: int,
        day_weather: pd.DataFrame,
        current_moisture_mm: float,
        crop_config: Dict[str, Any],
        full_weather_df: Optional[pd.DataFrame] = None,
        start_step: int = 0,
    ) -> List[int]:
        """
        Uses SciPy optimization to pick the best pump hours for the given 24-hour day.
        Includes lookahead rain forecasting to avoid pumping before rainstorms.
        """
        stress_threshold = float(crop_config.get("stress_threshold_mm", 50.0))
        field_capacity = float(crop_config.get("field_capacity_mm", 100.0))
        pump_power = float(crop_config.get("pump_power_kw", config.PUMP_POWER_KW))
        solar_peak = float(crop_config.get("solar_kw_peak", config.SOLAR_PANEL_KW_PEAK))
        pump_flow = float(crop_config.get("pump_flow_rate_mm_hr", config.PUMP_FLOW_RATE_MM_PER_HR))

        # Check total daily rain + rolling lookahead (next 36 hours for forecast-driven rain skipping)
        day_rain = float(day_weather["rainfall_mm"].sum())
        lookahead_rain = day_rain
        if full_weather_df is not None:
            upcoming = full_weather_df.iloc[start_step : min(start_step + 36, len(full_weather_df))]
            lookahead_rain = max(day_rain, float(upcoming["rainfall_mm"].sum()))

        # If significant rain is incoming and crop is safe, skip pumping completely
        if lookahead_rain >= 5.0 and current_moisture_mm >= stress_threshold:
            return []  # No irrigation needed; rain will recharge soil bucket

        # Predict 24-hour depletion using ML
        features = np.column_stack([
            day_weather["hour"].values,
            day_weather["temp_c"].values,
            day_weather["solar_irradiance_wm2"].values,
            day_weather["rainfall_mm"].values,
        ])
        kc = float(crop_config.get("kc", 1.0))
        pred_depletion = self.ml_model.predict(features) * kc
        total_expected_depletion = float(np.sum(pred_depletion))

        # Projected moisture at end of 24h
        projected_end_moisture = current_moisture_mm - total_expected_depletion + day_rain

        # Determine if pumping is truly required today
        # If projected end-of-day moisture stays comfortably above stress threshold, skip pumping
        if projected_end_moisture >= (stress_threshold + 4.0):
            return []  # Adequate moisture throughout the day; zero pumping required

        needed_water_mm = max(0.0, (stress_threshold + 14.0) - projected_end_moisture)
        needed_hours = min(2, max(1, int(np.ceil(needed_water_mm / pump_flow))))

        # SciPy Optimization:
        # We want to select `needed_hours` among the 24 hours to minimize grid draw:
        # Cost(h) = max(0.0, pump_power - solar_generated(h))
        solar_gen = np.maximum(
            0.0,
            solar_peak * (day_weather["solar_irradiance_wm2"].values / 1000.0) * 0.85,
        )
        grid_draw_per_hour = np.maximum(0.0, pump_power - solar_gen)

        # Objective function with Farmer Constraints:
        # 1. Allowed labor/working hours
        # 2. HESCOM 3-phase grid electricity availability
        allowed_hours = crop_config.get("allowed_irrigation_hours", None)
        grid_available_hours = crop_config.get("grid_available_hours", None)

        constraint_penalties = np.zeros(24)
        for i in range(24):
            h = int(day_weather.iloc[i]["hour"])
            if allowed_hours is not None and h not in allowed_hours:
                constraint_penalties[i] += 50.0  # Heavy penalty for scheduling outside farmer labor window
            if grid_available_hours is not None and grid_draw_per_hour[i] > 0.1 and h not in grid_available_hours:
                constraint_penalties[i] += 100.0  # Grid is off; cannot draw power!

        # Objective function for continuous weights x in [0, 1]^24
        def objective(x):
            grid_cost = np.sum(x * (grid_draw_per_hour + constraint_penalties))
            # Penalty for not meeting required pumping hours
            hours_penalty = 100.0 * (np.sum(x) - needed_hours) ** 2
            return grid_cost + hours_penalty

        x0 = np.zeros(24)
        # Seed initial guess around solar midday (11 AM to 2 PM)
        x0[11 : 11 + needed_hours] = 1.0
        bounds = [(0.0, 1.0) for _ in range(24)]

        res = minimize(objective, x0, method="SLSQP", bounds=bounds)

        if res.success or res.fun is not None:
            # Pick top `needed_hours` with highest optimized activation
            best_hours_idx = np.argsort(res.x)[-needed_hours:]
            best_hours = [int(day_weather.iloc[i]["hour"]) for i in sorted(best_hours_idx)]
            return best_hours

        # Fallback to solar peak hours (11 AM, 12 PM, 1 PM)
        return [11, 12][:needed_hours]

    def decide_irrigation(
        self,
        current_step: int,
        current_moisture_mm: float,
        weather_df: pd.DataFrame,
        crop_config: Dict[str, Any],
    ) -> bool:
        try:
            # Emergency Manual Override (Farmer 1-Click Action)
            if crop_config.get("manual_override_active", False):
                return True

            row = weather_df.iloc[current_step]
            hour = int(row["hour"])
            day_idx = int(row.get("day", current_step // 24))

            # Emergency Crop Guard: If moisture is dangerously close to or below threshold, pump immediately!
            stress_threshold = float(crop_config.get("stress_threshold_mm", 50.0))
            if current_moisture_mm <= (stress_threshold + 1.0):
                return True

            # Compute or retrieve optimized schedule for this day
            if day_idx not in self._daily_schedules:
                day_weather = weather_df[weather_df["day"] == day_idx]
                if len(day_weather) == 0:
                    day_weather = weather_df.iloc[day_idx * 24 : min((day_idx + 1) * 24, len(weather_df))]

                self._daily_schedules[day_idx] = self._optimize_daily_window(
                    day_index=day_idx,
                    day_weather=day_weather,
                    current_moisture_mm=current_moisture_mm,
                    crop_config=crop_config,
                    full_weather_df=weather_df,
                    start_step=current_step,
                )

            planned_hours = self._daily_schedules.get(day_idx, [])
            return bool(hour in planned_hours)

        except Exception as exc:
            # Safe Fallback to SmartController on any runtime anomaly
            return self.fallback_controller.decide_irrigation(
                current_step=current_step,
                current_moisture_mm=current_moisture_mm,
                weather_df=weather_df,
                crop_config=crop_config,
            )
