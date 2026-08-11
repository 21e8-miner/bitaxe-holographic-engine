"""JSONL telemetry + event logging."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from .config import HMEConfig
from .units import NormalizedMetrics


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


class TelemetryLog:
    def __init__(self, cfg: HMEConfig, root: Optional[Path] = None):
        self.cfg = cfg
        base = Path(root) if root else Path.cwd()
        self.dir = (base / cfg.logging.dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.dir / cfg.logging.jsonl_name
        self.events_path = self.dir / cfg.logging.events_name
        self._lock = threading.Lock()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=cfg.logging.history_len)

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def sample(self, m: NormalizedMetrics, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "telemetry",
            "hashrate_ghs": m.hashrate_ghs,
            "hashrate_ths": m.hashrate_ths,
            "power_w": m.power_w,
            "temp_c": m.temp_c,
            "frequency_mhz": m.frequency_mhz,
            "voltage_mv": m.voltage_mv,
            "j_per_th": m.j_per_th,
            "raw_hashrate": m.raw_hashrate,
            "hashrate_unit": m.hashrate_unit_assumed,
        }
        if extra:
            rec.update(_jsonable(extra))
        self._append(self.telemetry_path, rec)
        self.history.append(rec)
        return rec

    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "event",
            "kind": kind,
            **_jsonable(fields),
        }
        self._append(self.events_path, rec)
        return rec
