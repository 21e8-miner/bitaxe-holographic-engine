import socket
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("WhatsminerBridge")

class WhatsminerBridge:
    """
    Whatsminer Bridge Layer
    =======================
    Communicates with Whatsminer M30/M50/M60 series hardware via its 
    proprietary API protocol (usually on port 4028).
    
    Supports:
    - High-resolution telemetry (hashrate, power, temp)
    - Power mode management (Normal, High, Low)
    - Per-chip health monitoring
    """

    def __init__(self, ip: str, port: int = 4028):
        self.ip = ip
        self.port = port
        self.timeout = 5

    def _send_command(self, command: str) -> Optional[Dict]:
        """Send a JSON command to the Whatsminer API."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.ip, self.port))
                
                payload = {"cmd": command}
                s.sendall(json.dumps(payload).encode())
                
                response = s.recv(8192)
                data_str = response.decode('utf-8').strip().replace('\x00', '')
                return json.loads(data_str)
        except Exception as e:
            logger.error(f"Failed to communicate with Whatsminer @ {self.ip}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Fetch summary and power information."""
        summary = self._send_command("summary")
        power = self._send_command("get_psu_info") # Whatsminer specific
        
        result = {
            "online": False,
            "hashrate_gh": 0,
            "temp_avg": 0,
            "power_w": 0,
            "efficiency_jth": 0,
            "status": "OFFLINE"
        }

        if summary and "SUMMARY" in summary:
            result["online"] = True
            result["status"] = "ONLINE"
            s = summary["SUMMARY"][0]
            # Whatsminer usually reports 'HS' or 'GHS'
            result["hashrate_gh"] = s.get("GHS 5s", s.get("GHS av", 0))
            result["temp_avg"] = s.get("Temperature", 0)
            
            # Power is critical for J/TH calc
            if power and "PSU" in power:
                p = power["PSU"][0]
                result["power_w"] = p.get("Input Power", 0)
            
            if result["hashrate_gh"] > 0:
                th = result["hashrate_gh"] / 1000.0
                result["efficiency_jth"] = result["power_w"] / th
        
        return result

    def set_power_mode(self, mode: str) -> bool:
        """
        Switch between 'Normal', 'High', and 'Low' power modes.
        Equivalent to a software-based overclock/underclock.
        """
        modes = {
            "normal": "0",
            "high": "1",
            "low": "2"
        }
        target = modes.get(mode.lower())
        if not target:
            return False
            
        logger.info(f"Setting Whatsminer @ {self.ip} to {mode} power mode.")
        resp = self._send_command(f"set_power_mode({target})")
        return resp is not None and "STATUS" in resp and resp["STATUS"][0]["STATUS"] == "S"

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    bridge = WhatsminerBridge("192.168.1.101")
    print(f"Polling Whatsminer @ {bridge.ip}...")
    print(json.dumps(bridge.get_stats(), indent=2))
