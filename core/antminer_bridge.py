import socket
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("AntminerBridge")

class AntminerBridge:
    """
    Antminer Bridge Layer
    =====================
    Communicates with Antminer S19/S21 series hardware via the 
    underlying mining API (cgminer/bmminer compatible).
    
    Supports:
    - Telemetry extraction (hashrate, temperature, fan speed)
    - Chip-level health monitoring
    - Custom frequency/voltage tuning (requires unlocked firmware)
    """

    def __init__(self, ip: str, port: int = 4028):
        self.ip = ip
        self.port = port
        self.timeout = 5

    def _send_command(self, command: str, parameter: str = None) -> Optional[Dict]:
        """Send a JSON command to the miner's API port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.ip, self.port))
                
                payload = {"command": command}
                if parameter:
                    payload["parameter"] = parameter
                
                s.sendall(json.dumps(payload).encode())
                
                # Receive response
                data = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                
                # Clean the response (sometimes there's a trailing null or garbage)
                data_str = data.decode('utf-8').strip().replace('\x00', '')
                return json.loads(data_str)
        except Exception as e:
            logger.error(f"Failed to send command '{command}' to Antminer @ {self.ip}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Fetch comprehensive stats and hashrate."""
        stats = self._send_command("stats")
        summary = self._send_command("summary")
        
        result = {
            "online": False,
            "hashrate_gh": 0,
            "temp_max": 0,
            "fans": [],
            "uptime": 0
        }

        if summary and "SUMMARY" in summary:
            result["online"] = True
            # Antminers usually report GH/s 5s or GH/s av
            s = summary["SUMMARY"][0]
            result["hashrate_gh"] = s.get("GHS 5s", s.get("GHS av", 0))
            result["uptime"] = s.get("Elapsed", 0)

        if stats and "STATS" in stats:
            # Different firmware versions store temp differently
            # Usually in the second or third STATS entry
            for entry in stats["STATS"]:
                if "temp_max" in entry:
                    result["temp_max"] = entry["temp_max"]
                # Collect fan speeds
                for i in range(1, 5):
                    fan_key = f"fan{i}"
                    if fan_key in entry:
                        result["fans"].append(entry[fan_key])
        
        return result

    def set_tuning(self, frequency: int, voltage: int) -> bool:
        """
        Attempts to update frequency/voltage.
        WARNING: This typically requires custom firmware like Vnish, Braiins OS+, or MSK.
        """
        logger.info(f"Targeting tuning for {self.ip}: Freq={frequency}, Volt={voltage}")
        
        # This is a generic implementation. Specific firmwares use different commands:
        # Vnish: "set-freq", "set-volt"
        # Braiins: "tuner-set-profile"
        
        # Example for Vnish/MSK-style API
        freq_resp = self._send_command("set-freq", str(frequency))
        volt_resp = self._send_command("set-volt", str(voltage))
        
        return freq_resp is not None and volt_resp is not None

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    bridge = AntminerBridge("192.168.1.100")
    print(f"Polling Antminer @ {bridge.ip}...")
    stats = bridge.get_stats()
    print(json.dumps(stats, indent=2))
