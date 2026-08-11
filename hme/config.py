"""Load HME configuration from TOML + environment overrides."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# tomllib is stdlib 3.11+; fallback for older
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class DeviceConfig:
    ip: str = "192.168.0.23"
    chip: str = ""
    timeout_sec: float = 8.0
    allow_restart: bool = False


@dataclass
class BoundsConfig:
    min_freq_mhz: int = 425
    max_freq_mhz: int = 575
    base_freq_mhz: int = 500
    min_voltage_mv: int = 1050
    max_voltage_mv: int = 1250
    max_temp_c: float = 70.0
    max_power_w: float = 28.0
    warn_temp_c: float = 65.0


@dataclass
class TunerConfig:
    dry_run: bool = True
    min_change_interval_sec: int = 300
    dwell_sec: int = 120
    poll_sec: float = 5.0
    zero_hash_abort_sec: int = 45
    good_samples: int = 3
    max_jth_regression: float = 0.08
    max_hashrate_drop: float = 0.12
    freq_step_mhz: int = 25
    voltage_steps_mv: List[int] = field(default_factory=list)
    mode: str = "climb"
    max_steps: int = 8


@dataclass
class QCConfig:
    hashrate_unit: str = "auto"
    ref_j_per_th: Dict[str, float] = field(
        default_factory=lambda: {
            "BM1366": 17.0,
            "BM1368": 16.5,
            "BM1370": 15.5,
            "default": 17.0,
        }
    )


@dataclass
class LoggingConfig:
    dir: str = "logs"
    jsonl_name: str = "hme_telemetry.jsonl"
    events_name: str = "hme_events.jsonl"
    history_len: int = 500


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5033


@dataclass
class HMEConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    source_path: Optional[str] = None

    def ref_jth_for_chip(self, chip: str) -> float:
        refs = self.qc.ref_j_per_th or {}
        if chip and chip in refs:
            return float(refs[chip])
        return float(refs.get("default", 17.0))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _apply_env(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Environment overrides (ops-friendly)."""
    d = dict(raw)
    dev = dict(d.get("device") or {})
    tun = dict(d.get("tuner") or {})
    bounds = dict(d.get("bounds") or {})
    qc = dict(d.get("qc") or {})
    server = dict(d.get("server") or {})
    logc = dict(d.get("logging") or {})

    if os.environ.get("HME_BITAXE_IP"):
        dev["ip"] = os.environ["HME_BITAXE_IP"].strip()
    if os.environ.get("HME_CHIP"):
        dev["chip"] = os.environ["HME_CHIP"].strip()
    if os.environ.get("HME_TIMEOUT"):
        dev["timeout_sec"] = float(os.environ["HME_TIMEOUT"])
    b = _env_bool("HME_ALLOW_RESTART")
    if b is not None:
        dev["allow_restart"] = b
    b = _env_bool("HME_DRY_RUN")
    if b is not None:
        tun["dry_run"] = b
    if os.environ.get("HME_MAX_TEMP"):
        bounds["max_temp_c"] = float(os.environ["HME_MAX_TEMP"])
    if os.environ.get("HME_MAX_POWER"):
        bounds["max_power_w"] = float(os.environ["HME_MAX_POWER"])
    if os.environ.get("HME_MIN_FREQ"):
        bounds["min_freq_mhz"] = int(os.environ["HME_MIN_FREQ"])
    if os.environ.get("HME_MAX_FREQ"):
        bounds["max_freq_mhz"] = int(os.environ["HME_MAX_FREQ"])
    if os.environ.get("HME_HASHRATE_UNIT"):
        qc["hashrate_unit"] = os.environ["HME_HASHRATE_UNIT"].strip()
    if os.environ.get("HME_PORT"):
        server["port"] = int(os.environ["HME_PORT"])
    if os.environ.get("HME_LOG_DIR"):
        logc["dir"] = os.environ["HME_LOG_DIR"]

    d["device"] = dev
    d["tuner"] = tun
    d["bounds"] = bounds
    d["qc"] = qc
    d["server"] = server
    d["logging"] = logc
    return d


def _from_raw(raw: Dict[str, Any], source: Optional[str] = None) -> HMEConfig:
    raw = _apply_env(raw)
    dev = raw.get("device") or {}
    bounds = raw.get("bounds") or {}
    tun = raw.get("tuner") or {}
    qc = raw.get("qc") or {}
    logc = raw.get("logging") or {}
    server = raw.get("server") or {}

    ref = qc.get("ref_j_per_th") or {}
    # TOML may give nested tables as dict already
    if not isinstance(ref, dict):
        ref = {}

    return HMEConfig(
        device=DeviceConfig(
            ip=str(dev.get("ip", "192.168.0.23")),
            chip=str(dev.get("chip") or ""),
            timeout_sec=float(dev.get("timeout_sec", 8.0)),
            allow_restart=bool(dev.get("allow_restart", False)),
        ),
        bounds=BoundsConfig(
            min_freq_mhz=int(bounds.get("min_freq_mhz", 425)),
            max_freq_mhz=int(bounds.get("max_freq_mhz", 575)),
            base_freq_mhz=int(bounds.get("base_freq_mhz", 500)),
            min_voltage_mv=int(bounds.get("min_voltage_mv", 1050)),
            max_voltage_mv=int(bounds.get("max_voltage_mv", 1250)),
            max_temp_c=float(bounds.get("max_temp_c", 70.0)),
            max_power_w=float(bounds.get("max_power_w", 28.0)),
            warn_temp_c=float(bounds.get("warn_temp_c", 65.0)),
        ),
        tuner=TunerConfig(
            dry_run=bool(tun.get("dry_run", True)),
            min_change_interval_sec=int(tun.get("min_change_interval_sec", 300)),
            dwell_sec=int(tun.get("dwell_sec", 120)),
            poll_sec=float(tun.get("poll_sec", 5.0)),
            zero_hash_abort_sec=int(tun.get("zero_hash_abort_sec", 45)),
            good_samples=int(tun.get("good_samples", 3)),
            max_jth_regression=float(tun.get("max_jth_regression", 0.08)),
            max_hashrate_drop=float(tun.get("max_hashrate_drop", 0.12)),
            freq_step_mhz=int(tun.get("freq_step_mhz", 25)),
            voltage_steps_mv=[int(x) for x in (tun.get("voltage_steps_mv") or [])],
            mode=str(tun.get("mode", "climb")),
            max_steps=int(tun.get("max_steps", 8)),
        ),
        qc=QCConfig(
            hashrate_unit=str(qc.get("hashrate_unit", "auto")),
            ref_j_per_th={str(k): float(v) for k, v in {
                "BM1366": 17.0, "BM1368": 16.5, "BM1370": 15.5, "default": 17.0, **ref
            }.items()},
        ),
        logging=LoggingConfig(
            dir=str(logc.get("dir", "logs")),
            jsonl_name=str(logc.get("jsonl_name", "hme_telemetry.jsonl")),
            events_name=str(logc.get("events_name", "hme_events.jsonl")),
            history_len=int(logc.get("history_len", 500)),
        ),
        server=ServerConfig(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 5033)),
        ),
        source_path=source,
    )


def find_config_path(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    env = os.environ.get("HME_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    root = _repo_root()
    for name in ("config.toml", "hme.toml", "config.local.toml"):
        p = root / name
        if p.is_file():
            return p
    cwd = Path.cwd()
    for name in ("config.toml", "hme.toml"):
        p = cwd / name
        if p.is_file():
            return p
    return None


def load_config(path: Optional[str] = None) -> HMEConfig:
    """
    Load config. Missing file → defaults + env (safe: dry_run still True by default).
    """
    p = find_config_path(path)
    if p is None:
        return _from_raw({}, source=None)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        data = {}
    return _from_raw(data, source=str(p))


def write_example_if_missing(dest: Optional[Path] = None) -> Path:
    dest = dest or (_repo_root() / "config.toml")
    example = _repo_root() / "config.example.toml"
    if not dest.exists() and example.exists():
        dest.write_text(example.read_text())
    return dest
