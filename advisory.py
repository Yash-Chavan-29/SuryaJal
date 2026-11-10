"""
SuryaJal Bilingual Agronomic Advisory & Telegram Dispatcher
Generates plain-language Kannada & English irrigation advice via Google Gemini API
and dispatches real-time smartphone alerts via Telegram Bot API.
"""

from typing import Dict, Any, Optional
import json
import requests
from google import genai
from google.genai import errors
import config


def generate_bilingual_advisory(telemetry: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates plain-language farmer irrigation advice in Kannada and English
    using the Gemini API (with robust model fallback and offline template resilience).

    Parameters:
        telemetry (dict): Current agronomic state containing:
            - moisture_mm: float
            - stress_threshold_mm: float
            - rain_forecast_mm: float
            - pump_decision: str ("OFF" or "PUMP_HOURS")
            - planned_hours: list
            - solar_share_pct: float
            - water_saved_liters: int
            - cost_saved_inr: float

    Returns:
        dict: {"kannada": str, "english": str, "source": str}
    """
    moisture = telemetry.get("moisture_mm", 60.0)
    stress_thresh = telemetry.get("stress_threshold_mm", 52.0)
    rain_fc = telemetry.get("rain_forecast_mm", 0.0)
    pumping = telemetry.get("pump_active_today", False)
    hours = telemetry.get("planned_hours", [])
    solar_share = telemetry.get("solar_share_pct", 90.0)
    water_saved = telemetry.get("water_saved_liters", 10000)
    cost_saved = telemetry.get("cost_saved_inr", 35.0)

    # 1. Try Gemini API Generation
    if config.GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=config.GEMINI_API_KEY)

            hours_str = (
                ", ".join([f"{h}:00" for h in hours]) if pumping and hours else "None (Pump OFF)"
            )
            decision_context = (
                f"Pump active for {len(hours)} hour(s) during peak solar at {hours_str}. "
                f"{solar_share:.0f}% powered by free solar energy."
                if pumping
                else (
                    f"Pump remains OFF today. Rain forecast of {rain_fc:.1f} mm will recharge soil naturally without wasting electricity."
                    if rain_fc >= 4.0
                    else "Pump remains OFF today as soil moisture is in optimal comfort zone."
                )
            )

            prompt = f"""
You are SuryaJal's friendly AI Agronomist advising smallholder farmers in Hubballi/Belagavi, Karnataka.
Based on the field telemetry below, write a short, clear, respectful 2-sentence advice message for the farmer.

FIELD TELEMETRY:
- Soil Moisture: {moisture:.1f} mm (Stress Danger Line: {stress_thresh:.1f} mm)
- 12-Hour Rain Forecast: {rain_fc:.1f} mm
- Pump Recommendation: {decision_context}
- Impact: {water_saved:,} Liters of water and Rs {cost_saved:.1f} saved.

FORMAT REQUIREMENT:
Output strictly valid JSON with exactly three keys: "kannada", "english", and "hindi".
Do not include markdown code fences or other text outside the JSON.
- "kannada": 2 short sentences in pure colloquial Kannada script (ನುಡಿ / ಕನ್ನಡ).
- "english": 2 short sentences in plain English.
- "hindi": 2 short sentences in simple colloquial Hindi script (देवनागरी / हिंदी).
"""

            # Try primary models with fallback
            models_to_try = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
            for model_name in models_to_try:
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    text = resp.text.strip()
                    # Clean any markdown fences if present
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                        text = text.strip()

                    parsed = json.loads(text)
                    if "kannada" in parsed and "english" in parsed:
                        return {
                            "kannada": parsed["kannada"].strip(),
                            "english": parsed["english"].strip(),
                            "hindi": parsed.get("hindi", "").strip() or f"मिट्टी की नमी {moisture:.1f} मिमी है। सौर पंप का उपयोग करके पानी और बिजली बचाएं।",
                            "source": f"Gemini API ({model_name})",
                        }
                except Exception:
                    continue
        except Exception:
            pass

    # 2. Offline Deterministic Rule-Based Fallback (Hackathon Offline Safety)
    if not pumping:
        if rain_fc >= 4.0:
            return {
                "kannada": f"ಇಂದು ರಾತ್ರಿ {rain_fc:.1f} ಮಿ.ಮೀ ಮಳೆ ಮುನ್ಸೂಚನೆ ಇರುವುದರಿಂದ ಪಂಪ್ ಚಾಲನೆ ಮಾಡಬೇಡಿ. ಮಣ್ಣಿನ ತೇವಾಂಶವು ಸಹಜವಾಗಿ ಹೆಚ್ಚಾಗುತ್ತದೆ ಮತ್ತು ವಿದ್ಯುತ್ ಉಳಿತಾಯವಾಗುತ್ತದೆ.",
                "english": f"Do not irrigate today as {rain_fc:.1f} mm of rainfall is expected. Rain will naturally recharge your soil moisture while saving electricity.",
                "hindi": f"आज आपके खेत में {rain_fc:.1f} मिमी बारिश की संभावना है, इसलिए पंप बंद रखें। बारिश से मिट्टी में प्राकृतिक नमी आएगी और बिजली बचेगी।",
                "source": "SuryaJal Local Rule-Based Engine (Offline Resilience)",
            }
        else:
            return {
                "kannada": f"ನಿಮ್ಮ ಹೊಲದ ಮಣ್ಣಿನಲ್ಲಿ {moisture:.1f} ಮಿ.ಮೀ ಸಮೃದ್ಧ ತೇವಾಂಶವಿದೆ. ಇಂದು ನೀರು ಹಾಯಿಸುವ ಅಗತ್ಯವಿಲ್ಲ; ಸೂರ್ಯಜಲ ನಿಗಾ ಇರಿಸಿದೆ.",
                "english": f"Your field has sufficient soil moisture ({moisture:.1f} mm). No irrigation is needed today; SuryaJal continues monitoring.",
                "hindi": f"आपके खेत की मिट्टी में {moisture:.1f} मिमी पर्याप्त नमी है। आज सिंचाई की आवश्यकता नहीं है; सूर्यजल लगातार निगरानी कर रहा है।",
                "source": "SuryaJal Local Rule-Based Engine (Offline Resilience)",
            }
    else:
        hours_str = ", ".join([f"{h}:00" for h in hours]) if hours else "ಮಧ್ಯಾಹ್ನ 12:00"
        hours_str_hi = ", ".join([f"{h}:00" for h in hours]) if hours else "दोपहर 12:00"
        return {
            "kannada": f"ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಡಿಮೆ ಇರುವುದರಿಂದ ಇಂದು {hours_str} ರ ಗರಿಷ್ಠ ಬಿಸಿಲಿನಲ್ಲಿ ಮಾತ್ರ ಪಂಪ್ ಚಾಲನೆ ಮಾಡಿ. ಸೌರಶಕ್ತಿಯು ನಿಮ್ಮ {solar_share:.0f}% ವಿದ್ಯುತ್ ಬಿಲ್ ಉಳಿಸುತ್ತದೆ.",
            "english": f"Soil moisture is approaching trigger level; run the pump during solar peak at {hours_str}. Free solar energy will power {solar_share:.0f}% of irrigation.",
            "hindi": f"मिट्टी में नमी कम हो रही है; आज दोपहर {hours_str_hi} के समय सोलर पंप चलाएं। मुफ्त सौर ऊर्जा से आपकी {solar_share:.0f}% बिजली की बचत होगी।",
            "source": "SuryaJal Local Rule-Based Engine (Offline Resilience)",
        }


def send_telegram_alert(
    kannada_text: str,
    english_text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sends structured bilingual advisory to Telegram via Telegram Bot API.
    """
    token = bot_token or config.TELEGRAM_BOT_TOKEN
    chat = chat_id or config.TELEGRAM_CHAT_ID

    if not token or not chat:
        return {
            "success": False,
            "message": "Telegram Bot Token or Chat ID is not configured in .env. Enter your Bot Token to receive live phone alerts.",
        }

    stats_str = ""
    if telemetry:
        water_l = telemetry.get("water_saved_liters", 0)
        cost_inr = telemetry.get("cost_saved_inr", 0.0)
        stats_str = (
            f"\n\n📊 *ಇಂದಿನ ಉಳಿತಾಯ / Impact Summary:*\n"
            f"💧 ನೀರು / Water Saved: `{water_l:,} Liters`\n"
            f"💰 ಉಳಿಸಿದ ಹಣ / Cost Saved: `₹{cost_inr:.2f}`"
        )

    formatted_message = (
        f"☀️ *ಸೂರ್ಯಜಲ ಕೃಷಿ ಸಲಹೆ | SuryaJal Advisory*\n"
        f"📍 _{config.LOCATION_NAME}_\n\n"
        f"🌾 *ಕನ್ನಡ:*\n{kannada_text}\n\n"
        f"🌐 *English:*\n{english_text}"
        f"{stats_str}\n\n"
        f"🤖 _Powered by SuryaJal AI Digital Twin_"
    )

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat,
            "text": formatted_message,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            return {
                "success": True,
                "message": "Alert delivered successfully to your phone via Telegram!",
            }
        else:
            return {
                "success": False,
                "message": f"Telegram API returned HTTP {resp.status_code}: {resp.text}",
            }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Telegram request failed: {str(exc)}",
        }
