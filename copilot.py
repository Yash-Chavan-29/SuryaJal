"""
SuryaJal™ - Live Gemini AI Agronomic Co-Pilot & Site Navigator
Powered by Google Gemini 2.5 Flash API.
Eliminates language barriers, guides users across the platform,
and provides natural voice interaction in Kannada, Hindi, and English.
"""

from typing import Dict, Any, List
import json
from google import genai
import config


SYSTEM_NAVIGATOR_PROMPT = """
You are "SuryaJal Sahayaka" (ಸೂರ್ಯಜಲ ಸಹಾಯಕ) — an expert agricultural AI copilot and site navigator for the SuryaJal precision irrigation platform in Hubballi-Belagavi, Karnataka.

You have dual responsibilities:
1. Agronomic Intelligence: Answer ANY farmer or researcher question regarding crops (Cotton, Sugarcane, Maize, Bengal Gram), irrigation timing, solar power, soil moisture, plant diseases, fertilizer schedules, or weather.
2. Platform Navigation Guide: Help the user find anything on this website, tell them which tab to click, how to use features, or guide them through a demo.

CURRENT PLATFORM TELEMETRY & CONTEXT:
{telemetry_context}

AVAILABLE SECTIONS & TOOLS ON THIS PLATFORM:
- "📱 Farmer Phone AI Scout": Snap or upload field photos for real Gemini computer vision crop & soil health diagnosis with spoken voice.
- "⚡ 1. Solar Energy Arbitrage Tab": Interactive sliders calculating whether pumping water or exporting solar power to HESCOM (feed-in tariff ₹3.25/kWh) gives higher profit.
- "🗺️ 2. Variable Rate Irrigation (VRI) Tab": Multi-zone soil maps (Vertisol Clay vs Loam vs Sand) with automated solenoid pulse valve timing.
- "🦠 3. Crop Disease Risk Predictor Tab": Calculates leaf wetness duration (LWD) and forecasts bacterial blight / red rot risk with preventive bio-fungicide advice.
- "💧 4. Stormwater Aquifer Recharge Tab": USDA NRCS runoff model calculating stormwater diversion into borewell recharge pits to elevate groundwater.
- "📊 5. 24h Solar Dispatch & 3D Hydrology Tab": 24-hour solar generation vs pump curve and 3D subsurface hydrology volume.
- "📥 Field Audit Export": Download button for stamped hourly CSV records for PM-KUSUM subsidy compliance.

RESPONSE GUIDELINES:
- Output strictly valid JSON with the following keys:
  - "reply_text": Clear, helpful, respectful 2-4 sentences in the SAME language the user asked (or {target_lang}).
  - "navigation_guide": Short instruction on which button/tab to use next (e.g. "👉 Switch to Tab 1 'Solar Energy Arbitrage' to adjust the export tariff slider").
  - "recommended_tab_index": Integer (0 to 4) if a specific technical tab is recommended, else -1.
- If the user asks in Kannada, reply primarily in fluent, polite Kannada script (ಕನ್ನಡ).
- If the user asks in Hindi, reply in fluent Hindi (हिंदी).
- If the user asks in English, reply in English.
"""

def ask_gemini_copilot(
    user_query: str,
    site_context: Dict[str, Any],
    target_lang: str = "English",
) -> Dict[str, Any]:
    """
    Sends user query and live platform state to Gemini 2.5 Flash,
    returning an intelligent agronomic reply and navigation directions.
    """
    context_str = json.dumps(site_context, indent=2, default=str)

    if config.GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            prompt = SYSTEM_NAVIGATOR_PROMPT.format(
                telemetry_context=context_str,
                target_lang=target_lang,
            ) + f"\n\nUSER QUESTION: {user_query}\n\nRespond ONLY in valid JSON."

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            return {
                "reply_text": parsed.get("reply_text", ""),
                "navigation_guide": parsed.get("navigation_guide", ""),
                "recommended_tab_index": parsed.get("recommended_tab_index", -1),
                "source": "Gemini 2.5 Flash Live Copilot",
            }
        except Exception as e:
            print(f"Gemini Copilot Error: {e}. Falling back to deterministic assistant.")

    # Fallback if offline
    return _deterministic_copilot_fallback(user_query, site_context, target_lang)


def _deterministic_copilot_fallback(query: str, ctx: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Offline rule-based fallback assistant."""
    q = query.lower()
    moist = ctx.get("soil_moisture_mm", 60.0)

    if "arbitrage" in q or "tariff" in q or "sell" in q or "grid" in q or "power" in q:
        return {
            "reply_text": "Our Energy Arbitrage engine evaluates whether pumping water or exporting electricity to the HESCOM grid generates higher financial return based on the current feed-in tariff.",
            "navigation_guide": "👉 Open '⚡ 1. Solar Energy Arbitrage' tab to test live tariff sliders.",
            "recommended_tab_index": 0,
            "source": "SuryaJal Local Assistant (Offline)",
        }
    elif "disease" in q or "blight" in q or "rot" in q or "leaf" in q or "spray" in q:
        return {
            "reply_text": "Micro-climate humidity and temperature are tracked to forecast fungal disease risks like Cotton Bacterial Blight and Sugarcane Red Rot.",
            "navigation_guide": "👉 Open '🦠 3. Crop Disease Risk Predictor' tab to view the infection risk radar.",
            "recommended_tab_index": 2,
            "source": "SuryaJal Local Assistant (Offline)",
        }
    elif "rain" in q or "recharge" in q or "borewell" in q or "water table" in q:
        return {
            "reply_text": "Before heavy storms, excess runoff is diverted into borewell percolation shafts to replenish the regional aquifer using the USDA NRCS model.",
            "navigation_guide": "👉 Open '💧 4. Stormwater Aquifer Recharge' tab to simulate storm diversion.",
            "recommended_tab_index": 3,
            "source": "SuryaJal Local Assistant (Offline)",
        }
    else:
        return {
            "reply_text": f"Current root zone moisture is {moist:.1f} mm. Solar irradiance peaks between 11:30 AM and 01:30 PM for 100% free solar irrigation.",
            "navigation_guide": "👉 Upload a photo in 'AI Field Scout' or explore the 5 live technical engine tabs.",
            "recommended_tab_index": 0,
            "source": "SuryaJal Local Assistant (Offline)",
        }
