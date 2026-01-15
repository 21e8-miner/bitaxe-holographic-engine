import sqlite3
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger("HCE_Persistence")

class HCEPersistence:
    """
    HCE Persistence Layer
    ====================
    Handles SQLite-based storage for fleet telemetry, miner configuration, 
    and operational events.
    """

    def __init__(self, db_path: str = "hce_fleet.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Miners Table: Registry of all discovered devices in the fleet
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS miners (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT UNIQUE NOT NULL,
                    model TEXT,
                    mac_address TEXT UNIQUE,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    status TEXT DEFAULT 'UNKNOWN'
                )
            ''')

            # 2. Telemetry Table: High-resolution logs of performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    miner_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hashrate GH_S REAL,
                    power_w REAL,
                    temp_c REAL,
                    efficiency_jth REAL,
                    phase_lock_status TEXT,
                    FOREIGN KEY (miner_id) REFERENCES miners (id)
                )
            ''')

            # 3. Events Table: Audit log for config changes and safety rollbacks
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    miner_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT, -- 'CONFIG_CHANGE', 'SAFETY_ROLLBACK', 'AUDIT'
                    description TEXT,
                    old_value TEXT, -- JSON string
                    new_value TEXT, -- JSON string
                    FOREIGN KEY (miner_id) REFERENCES miners (id)
                )
            ''')
            
            conn.commit()
            logger.info(f"Persistence initialized at {self.db_path}")

    def register_miner(self, ip: str, model: str = "UNKNOWN") -> int:
        """Register or update a miner in the registry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO miners (ip, model, last_seen, status)
                VALUES (?, ?, ?, 'ONLINE')
                ON CONFLICT(ip) DO UPDATE SET
                    model = excluded.model,
                    last_seen = excluded.last_seen,
                    status = 'ONLINE'
                RETURNING id
            ''', (ip, model, datetime.now().isoformat()))
            row = cursor.fetchone()
            conn.commit()
            return row[0] if row else None

    def log_telemetry(self, miner_id: int, hashrate: float, power: float, temp: float, efficiency: float, phase_lock: str):
        """Record a telemetry snapshot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO telemetry (miner_id, hashrate, power_w, temp_c, efficiency_jth, phase_lock_status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (miner_id, hashrate, power, temp, efficiency, phase_lock))
            conn.commit()

    def log_event(self, miner_id: int, event_type: str, description: str, old_val: Any = None, new_val: Any = None):
        """Record an operational event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (miner_id, event_type, description, old_value, new_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                miner_id, 
                event_type, 
                description, 
                json.dumps(old_val) if old_val else None, 
                json.dumps(new_val) if new_val else None
            ))
            conn.commit()

    def get_fleet_summary(self) -> List[Dict]:
        """Retrieve latest status for all registered miners."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT m.*, t.hashrate, t.efficiency_jth, t.phase_lock_status
                FROM miners m
                LEFT JOIN telemetry t ON t.id = (
                    SELECT id FROM telemetry WHERE miner_id = m.id ORDER BY timestamp DESC LIMIT 1
                )
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_historical_efficiency(self, miner_id: int, limit: int = 100) -> List[Dict]:
        """Retrieve historical efficiency data for charting."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, efficiency_jth, hashrate
                FROM telemetry
                WHERE miner_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (miner_id, limit))
            return [dict(row) for row in cursor.fetchall()]

if __name__ == "__main__":
    # Test initialization
    logging.basicConfig(level=logging.INFO)
    db = HCEPersistence("test_fleet.db")
    mid = db.register_miner("192.168.0.23", "Bitaxe Gamma")
    db.log_telemetry(mid, 1150.0, 21.0, 65.0, 18.2, "LOCKED")
    db.log_event(mid, "CONFIG_CHANGE", "Applied Performance Profile", {"freq": 525}, {"freq": 550})
    print(db.get_fleet_summary())
