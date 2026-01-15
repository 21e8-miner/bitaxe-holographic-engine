#!/usr/bin/env python3
"""
BITAXE HOLOGRAPHIC OVERCLOCK BRIDGE (Phase 5)
=============================================
Links the Bitaxe Miner's ASIC frequency to the Ghost Bars 
Holographic Coherence Engine (HCE).

Logic:
1. Polls NQ Dashboard HCE metrics (esp32 coherence + koopman spectral phase).
2. Calculates optimal ASIC frequency based on environmental coherence.
3. Programmatically updates Bitaxe settings via REST API (192.168.0.23).
"""

import time
import requests
import json
import os
import logging

from hardware_profiles import get_profile

# --- CONFIGURATION ---
BITAXE_IP = "192.168.0.23"
HCE_API_URL = "http://localhost:5001/api/hardware"
NQ_API_URL = "http://localhost:5001/system_coherence.json"

# Frequency Bounds (Default values, will be updated by profile)
MIN_FREQ = 425
BASE_FREQ = 525
MAX_FREQ = 575
CHANGE_THRESHOLD = 15 # MHz delta required to trigger update

# Timing
POLL_INTERVAL = 10 # Seconds
RETRY_DELAY = 5   # Wait if Bitaxe is busy

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("BITAXE_HCE")

def get_hce_metrics():
    """Fetch coherence and spectral metrics from the local HCE server."""
    try:
        resp = requests.get(HCE_API_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'coherence': data.get('combined_boost', 1.0) - 1.0, # 0.0 to 1.0
                'esp_coh': data.get('esp32', {}).get('coherence', 0.5),
                'i7_coh': data.get('i7', {}).get('coherence', 0.5)
            }
    except Exception as e:
        logger.debug(f"HCE API unavailable: {e}")
    
    try:
        resp = requests.get(NQ_API_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'coherence': data.get('coherence', 0.5),
                'status': data.get('status', 'DECOHERED')
            }
    except Exception as e:
        logger.warning(f"Failed to fetch HCE metrics: {e}")
    
    return None

def get_bitaxe_status():
    """Fetch current miner status."""
    try:
        resp = requests.get(f"http://{BITAXE_IP}/api/system/info", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Bitaxe unreachable at {BITAXE_IP}: {e}")
    return None

def set_bitaxe_frequency(freq):
    """Update Bitaxe frequency and restart."""
    logger.info(f"🌀 SPECTRAL OVERCLOCK: Targeting {freq} MHz (Holographic Alignment)")
    try:
        data = {
            "frequency": int(freq),
            "overclockEnabled": 1
        }
        resp = requests.patch(f"http://{BITAXE_IP}/api/system", json=data, timeout=5)
        
        if resp.status_code == 200:
            logger.info("✅ Settings staged. Triggering re-lock (Restart)...")
            requests.post(f"http://{BITAXE_IP}/api/system/restart", timeout=2)
            return True
        else:
            logger.error(f"Failed to patch settings: {resp.text}")
    except Exception as e:
        logger.error(f"API Error: {e}")
    return False

def main():
    logger.info("🚀 BITAXE-HCE BRIDGE INITIALIZED")
    logger.info(f"Targeting Bitaxe at {BITAXE_IP}")
    
    global MIN_FREQ, BASE_FREQ, MAX_FREQ
    
    last_applied_freq = 0
    profile_detected = False
    
    while True:
        metrics = get_hce_metrics()
        status = get_bitaxe_status()
        
        if status and not profile_detected:
            asic_model = status.get('ASICModel', 'BM1366')
            profile = get_profile(asic_model)
            MIN_FREQ = profile['min_safe_freq']
            BASE_FREQ = profile['default_freq']
            MAX_FREQ = profile['max_safe_freq']
            profile_detected = True
            logger.info(f"📋 APPLIED SPECTRAL BOUNDS for {profile['model_name']}: {MIN_FREQ}-{MAX_FREQ} MHz")

        if metrics and status:
            coh = metrics.get('coherence', 0.5)
            current_freq = status.get('frequency', BASE_FREQ)
            
            # Linear mapping of coherence to frequency overdrive
            target_freq = BASE_FREQ + (MAX_FREQ - BASE_FREQ) * coh
            target_freq = max(MIN_FREQ, min(MAX_FREQ, int(target_freq)))
            
            logger.info(f"State: Coherence={coh:.4f} | Current={current_freq}MHz | Ideal={target_freq}MHz")
            
            delta = abs(target_freq - current_freq)
            if delta >= CHANGE_THRESHOLD and abs(target_freq - last_applied_freq) >= 5:
                set_bitaxe_frequency(target_freq)
                last_applied_freq = target_freq
                time.sleep(30)
            else:
                logger.debug("Drift within threshold. Maintaining lock.")
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
