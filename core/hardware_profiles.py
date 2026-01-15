"""
PHASE-LOCK: Bitaxe Hardware Profiles
====================================
Defines stock and theoretical limits for various Bitaxe models 
to ensure the Holographic Engine stays calibrated across the fleet.
"""

HARDWARE_PROFILES = {
    "BM1366": {
        "model_name": "Bitaxe Ultra (204)",
        "bridge_type": "bitaxe",
        "stock_hashrate": 500,  # GH/s
        "stock_efficiency": 22.0, # J/TH
        "theoretical_limit": 17.0,
        "default_freq": 500,
        "max_safe_freq": 600,
        "min_safe_freq": 400
    },
    "BM1368": {
        "model_name": "Bitaxe Supra (401)",
        "bridge_type": "bitaxe",
        "stock_hashrate": 650,
        "stock_efficiency": 20.0,
        "theoretical_limit": 17.0,
        "default_freq": 525,
        "max_safe_freq": 650,
        "min_safe_freq": 450
    },
    "BM1370": {
        "model_name": "Bitaxe Gamma (601)",
        "bridge_type": "bitaxe",
        "stock_hashrate": 1100,
        "stock_efficiency": 18.0,
        "theoretical_limit": 15.0,
        "default_freq": 525,
        "max_safe_freq": 575,
        "min_safe_freq": 425
    },
    # --- ENTERPRISE MINERS (Phase 2) ---
    "S19_XP": {
        "model_name": "Antminer S19 XP",
        "bridge_type": "antminer",
        "stock_hashrate": 141000, # GH/s (141 TH/s)
        "stock_efficiency": 21.5,
        "theoretical_limit": 20.0,
        "default_freq": 500,
        "max_safe_freq": 550,
        "min_safe_freq": 450
    },
    "S21_HYD": {
        "model_name": "Antminer S21 Hydro",
        "bridge_type": "antminer",
        "stock_hashrate": 335000, # GH/s (335 TH/s)
        "stock_efficiency": 16.0,
        "theoretical_limit": 14.5,
        "default_freq": 500,
        "max_safe_freq": 550,
        "min_safe_freq": 450
    },
    "M50S": {
        "model_name": "Whatsminer M50S",
        "bridge_type": "whatsminer",
        "stock_hashrate": 126000, # GH/s (126 TH/s)
        "stock_efficiency": 26.0,
        "theoretical_limit": 24.0,
        "default_freq": 500,
        "max_safe_freq": 550,
        "min_safe_freq": 450
    }
}

DEFAULT_PROFILE = HARDWARE_PROFILES["BM1366"]

def get_profile(asic_model):
    """Detect profile based on ASICModel string from AxeOS."""
    # AxeOS usually returns "BM1366", "BM1370", etc.
    return HARDWARE_PROFILES.get(asic_model, DEFAULT_PROFILE)
