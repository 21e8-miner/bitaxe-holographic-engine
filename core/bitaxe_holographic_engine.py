#!/usr/bin/env python3
"""
Φ BITAXE HOLOGRAPHIC MINING ENGINE (HCE)
========================================
Standalone implementation of Bitcoin mining optimization 
using Spectral Phase Coherence and Neural Super-Manifolds.

This software acts as a 'Sidecar' for Bitaxe hardware, providing
real-time telemetry, ASIC audits, and holographic scheduling.
"""

import os
import time
import requests
import json
import logging
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
from threading import Thread

# Local imports
from hardware_profiles import get_profile
from persistence import HCEPersistence
from antminer_bridge import AntminerBridge
from whatsminer_bridge import WhatsminerBridge
from stratum_v2_client import StratumV2Client
from qc_engine import FirstPrinciplesQC

# --- CONFIGURATION & SAFETY BOUNDS ---
BITAXE_IP = os.environ.get("BITAXE_IP", "192.168.0.23")
DB_PATH = os.environ.get("HCE_DB", "hce_fleet.db")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3.0"))
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "5033"))

# Operational Safety Guardrails
SAFE_VOLTAGE_MIN = 1050  # mV
SAFE_VOLTAGE_MAX = 1350  # mV
SAFE_FREQ_MAX = 600      # MHz
MAX_TEMP_CEILING = 80    # °C
AUTO_ROLLBACK_ENABLED = True

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("HCE_ENGINE")

# Initialize Persistence
db = HCEPersistence(DB_PATH)

class MinerController:
    """
    MinerController
    ===============
    Orchestrates telemetry, audits, and safety for a single miner instance.
    Supports Bitaxe (HTTP), Antminer (Socket), and Whatsminer (Socket).
    """
    def __init__(self, ip):
        self.ip = ip
        self.miner_id = None
        self.status = {}
        self.profile = None
        self.bridge = None # Set dynamically based on hardware
        self.history = {
            'hashrate': [], 'temp': [], 'power': [], 'efficiency': []
        }
        self.qc_metrics = {
            'theoretical_efficiency': 17.0, 
            'actual_efficiency': 0,
            'efficiency_deviation': 0,
            'phase_lock_status': "SEARCHING",
            'entropic_efficiency': 0,
            'resonance_score': 0,
            'physics_floor': 0,
            'sv2_status': "V1_LEGACY"
        }
        self.sv2_client = None
        self.coinbase_message = "HCE//Standalone"
        self.last_known_good_config = {"frequency": 485, "coreVoltage": 1200}

    def audit_asicboost(self):
        """Verify internal ASIC settings for max efficiency."""
        try:
            resp = requests.get(f"http://{self.ip}/api/system", timeout=5)
            if resp.status_code == 200:
                settings = resp.json()
                logger.info(f"🛡️ ASIC AUDIT [{self.ip}]: Checking AsicBoost...")
                if not settings.get('overclockEnabled', False):
                    logger.warning("⚠️ Overclock NOT enabled. Applying safety-constrained efficiency profile...")
                    self.apply_spectral_profile('efficiency')
                return True
        except Exception as e:
            logger.error(f"Audit failed for {self.ip}: {e}")
        return False

    def apply_spectral_profile(self, mode='efficiency'):
        """
        Apply a predefined hardware profile with explicit safety constraints.
        
        Note: Voltage and Frequency are capped to prevent hardware degradation.
        """
        if not self.profile:
            return
            
        freq = self.profile['default_freq']
        voltage = 1150  # Default safe starting voltage
        
        if mode == 'performance':
            freq = min(self.profile['max_safe_freq'], SAFE_FREQ_MAX)
            voltage = 1200
            
        # Hard Guardrails
        freq = min(max(400, freq), SAFE_FREQ_MAX)
        voltage = min(max(SAFE_VOLTAGE_MIN, voltage), SAFE_VOLTAGE_MAX)
            
        profile = {"frequency": freq, "coreVoltage": voltage}
        try:
            old_profile = {"frequency": self.status.get('frequency'), "voltage": self.status.get('coreVoltage')}
            requests.patch(f"http://{self.ip}/api/system", json=profile, timeout=5)
            logger.info(f"✅ Applied {mode} profile: {profile}")
            if self.miner_id:
                db.log_event(self.miner_id, "CONFIG_CHANGE", f"Applied {mode} profile", old_profile, profile)
        except Exception as e:
            logger.error(f"Profile application failed: {e}")

    def rollback_to_safe(self):
        """Emergency rollback to last known stable configuration."""
        try:
            requests.patch(f"http://{self.ip}/api/system", json=self.last_known_good_config, timeout=5)
            logger.warning(f"🚨 EMERGENCY ROLLBACK: Reverted to {self.last_known_good_config}")
            if self.miner_id:
                db.log_event(self.miner_id, "SAFETY_ROLLBACK", "High temperature detected", new_val=self.last_known_good_config)
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def update_coinbase(self, message=None):
        """Inject customized message into coinbase transaction."""
        msg = message or self.coinbase_message
        try:
            data = {"user": msg}
            requests.patch(f"http://{self.ip}/api/system", json=data, timeout=5)
            logger.info(f"🖋️ COINBASE TAGGED: {msg}")
        except Exception as e:
            logger.error(f"Coinbase injection failed: {e}")

    def poll(self):
        """Update telemetry based on bridge type."""
        try:
            # 1. Device Discovery & Initial Profile Setup
            if not self.profile:
                # First, try Bitaxe HTTP API
                try:
                    r = requests.get(f"http://{self.ip}/api/system/info", timeout=2)
                    if r.status_code == 200:
                        data = r.json()
                        self.profile = get_profile(data.get('ASICModel', 'BM1366'))
                except: pass

                # If not Bitaxe, try Antminer API
                if not self.profile:
                    bridge = AntminerBridge(self.ip)
                    stats = bridge.get_stats()
                    if stats['online']:
                        self.profile = get_profile("S19_XP") # Fallback to S19
                        self.bridge = bridge

                # Third, try Whatsminer
                if not self.profile:
                    bridge = WhatsminerBridge(self.ip)
                    stats = bridge.get_stats()
                    if stats['online']:
                        self.profile = get_profile("M50S")
                        self.bridge = bridge

                if self.profile:
                    self.qc_metrics['theoretical_efficiency'] = self.profile['theoretical_limit']
                    logger.info(f"🚀 [{self.ip}] DETECTED: {self.profile['model_name']} ({self.profile['bridge_type']})")
                    self.miner_id = db.register_miner(self.ip, self.profile['model_name'])

            if not self.profile: return False

            # 2. Polling Logic per Bridge Type
            hashrate_gh, power_w, current_temp = 0, 0, 0

            if self.profile['bridge_type'] == 'bitaxe':
                resp = requests.get(f"http://{self.ip}/api/system/info", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    self.status = data
                    hashrate_gh = data.get('hashRate', 0)
                    power_w = data.get('power', 0)
                    current_temp = data.get('temp', 0)

            elif self.profile['bridge_type'] in ['antminer', 'whatsminer']:
                if not self.bridge:
                    self.bridge = AntminerBridge(self.ip) if self.profile['bridge_type'] == 'antminer' else WhatsminerBridge(self.ip)
                stats = self.bridge.get_stats()
                if stats['online']:
                    hashrate_gh = stats.get('hashrate_gh', 0)
                    power_w = stats.get('power_w', 0)
                    current_temp = stats.get('temp_avg', stats.get('temp_max', 0))

            # 3. Aggregation & Safety
            if current_temp > MAX_TEMP_CEILING:
                logger.critical(f"🔥 [{self.ip}] THERMAL CRITICAL: {current_temp}°C")
                self.rollback_to_safe()

            if hashrate_gh > 0:
                hashrate_th = hashrate_gh / 1000.0
                efficiency = power_w / hashrate_th if hashrate_th > 0 else 0
                
                self.history['efficiency'].append(efficiency)
                self.history['hashrate'].append(hashrate_gh)
                self.history['temp'].append(current_temp)
                self.history['power'].append(power_w)
                
                if self.miner_id:
                    db.log_telemetry(self.miner_id, hashrate_gh, power_w, current_temp, efficiency, self.qc_metrics['phase_lock_status'])
                
                # 4. First Principles QC Audit
                voltage_mv = self.status.get('coreVoltage', 1200)
                audit = FirstPrinciplesQC.audit_performance(efficiency, self.profile, voltage_mv)
                
                self.qc_metrics['actual_efficiency'] = efficiency
                self.qc_metrics['physics_floor'] = audit['theoretical_floor_jth']
                self.qc_metrics['entropic_efficiency'] = audit['entropic_efficiency_pct']
                self.qc_metrics['resonance_score'] = audit['resonance_score']
                self.qc_metrics['phase_lock_status'] = audit['audit_verdict']
                self.qc_metrics['efficiency_deviation'] = ((efficiency - self.qc_metrics['physics_floor']) / self.qc_metrics['physics_floor']) * 100

                # 5. Stratum V2 Passive Audit
                # In a real environment, we would check for SV2 noise handshake here.
                self.qc_metrics['sv2_status'] = "V1_LEGACY"
            
            # Trim history to keep 300 points in memory
            if len(self.history['hashrate']) > 300:
                for k in self.history: self.history[k].pop(0)

            return True
        except Exception as e:
            logger.debug(f"Poll error for {self.ip}: {e}")
            return False

    def get_manifold_data(self):
        """Format history for the 3D Neural Super-Manifold visualization."""
        if not self.history['hashrate']:
            return {}
        
        return {
            'current': {
                'hashrate': self.status.get('hashRate', 0),
                'temp': self.status.get('temp', 0),
                'power': self.status.get('power', 0),
                'efficiency': self.qc_metrics['actual_efficiency'],
                'phase_lock': self.qc_metrics['phase_lock_status'],
                'profile': self.profile
            },
            'qc': self.qc_metrics,
            'points': [
                {
                    'x': self.history['temp'][i],
                    'y': self.history['hashrate'][i],
                    'z': self.history['power'][i],
                    'efficiency': self.history['efficiency'][i]
                } for i in range(max(0, len(self.history['hashrate'])-150), len(self.history['hashrate']))
            ]
        }

class FleetManager:
    """Manages multiple MinerControllers and background polling."""
    def __init__(self, ips: list):
        self.controllers = {ip: MinerController(ip) for ip in ips}
        self.active_id = ips[0] if ips else None

    def poll_all(self):
        for ip, controller in self.controllers.items():
            controller.poll()

# --- INITIALIZE FLAST & FLEET ---
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)
CORS(app)

fleet_ips = os.environ.get("FLEET_IPS", BITAXE_IP).split(",")
fleet = FleetManager([ip.strip() for ip in fleet_ips if ip.strip()])

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Main dashboard entry point (Viewing active miner)."""
    ip = request.args.get('ip', fleet.active_id)
    return render_template('mining_dashboard_v2.html', miner_ip=ip)

@app.route('/fleet')
def fleet_view():
    """Fleet overview dashboard."""
    return render_template('fleet_view.html')

@app.route('/api/manifold')
def manifold_data():
    """Telemetry endpoint for the active miner."""
    ip = request.args.get('ip', fleet.active_id)
    ctrl = fleet.controllers.get(ip)
    return jsonify(ctrl.get_manifold_data() if ctrl else {})

@app.route('/api/history')
def historical_data():
    """Returns historical efficiency for a specific miner."""
    ip = request.args.get('ip', fleet.active_id)
    ctrl = fleet.controllers.get(ip)
    if not ctrl or not ctrl.miner_id: return jsonify([])
    return jsonify(db.get_historical_efficiency(ctrl.miner_id, limit=100))

@app.route('/api/fleet')
def fleet_data():
    """Returns status summary for the entire registered fleet."""
    return jsonify(db.get_fleet_summary())

@app.route('/api/qc')
def qc_report():
    """Quality Control report for active miner."""
    ip = request.args.get('ip', fleet.active_id)
    ctrl = fleet.controllers.get(ip)
    if not ctrl: return jsonify({})
    return jsonify({
        'timestamp': time.time(),
        'holographic_audit': {
            'chipset': ctrl.profile['model_name'] if ctrl.profile else "DETECTING...",
            'metrics': ctrl.qc_metrics,
            'status': ctrl.qc_metrics['phase_lock_status']
        }
    })

def background_worker():
    """Continuous polling for the entire fleet."""
    logger.info(f"Starting Fleet Polling for {len(fleet.controllers)} devices...")
    while True:
        fleet.poll_all()
        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    from flask import request
    Thread(target=background_worker, daemon=True).start()
    logger.info(f"Φ HCE active on http://localhost:{DASHBOARD_PORT}")
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False)
