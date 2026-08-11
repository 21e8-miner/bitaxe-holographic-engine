"""Hashrate / efficiency unit helpers.

AxeOS has historically reported hashrate in GH/s for Bitaxe boards while some
forks use TH/s. Mis-labeling either way destroys J/TH by 1000×.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class NormalizedMetrics:
    hashrate_ghs: float
    hashrate_ths: float
    power_w: float
    temp_c: float
    frequency_mhz: Optional[float]
    voltage_mv: Optional[float]
    j_per_th: Optional[float]
    raw_hashrate: float
    hashrate_unit_assumed: str  # "ghs" | "ths"


def sniff_hashrate_unit(raw: float, hint: str = "auto") -> str:
    """
    Guess whether raw hashrate is GH/s or TH/s.

    Bitaxe Gamma/Ultra typically report ~500–1500 GH/s (0.5–1.5 TH/s).
    Values in (0, 20] are almost certainly TH/s; values > 50 almost certainly GH/s.
    """
    h = (hint or "auto").lower().strip()
    if h in ("ghs", "gh/s", "gh"):
        return "ghs"
    if h in ("ths", "th/s", "th"):
        return "ths"
    if raw is None or raw <= 0:
        return "ghs"
    if raw <= 20.0:
        return "ths"
    return "ghs"


def to_ghs(raw: float, unit: str) -> float:
    u = sniff_hashrate_unit(raw, unit)
    if u == "ths":
        return float(raw) * 1000.0
    return float(raw)


def j_per_th(power_w: float, hashrate_ghs: float) -> Optional[float]:
    """Joules per terahash = W / (TH/s) = W / (GH/s / 1000)."""
    if power_w is None or hashrate_ghs is None:
        return None
    if power_w <= 0 or hashrate_ghs <= 0:
        return None
    ths = hashrate_ghs / 1000.0
    if ths <= 0:
        return None
    return float(power_w) / ths


def _f(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def _opt_f(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def normalize_axeos_info(info: Dict[str, Any], hashrate_unit: str = "auto") -> NormalizedMetrics:
    """Map AxeOS /api/system/info (and cousins) to normalized metrics."""
    raw_hr = _f(info, "hashRate", "hashrate", "hash_rate", "hr", default=0.0)
    power = _f(info, "power", "power_w", "watts", default=0.0)
    temp = _f(info, "temp", "temperature", "temp_c", default=0.0)
    freq = _opt_f(info, "frequency", "freq", "asicFrequency", "currentFrequency")
    volt = _opt_f(info, "coreVoltage", "core_voltage", "voltage", "coreVoltageActual")
    # some firmwares report actual voltage separately
    if volt is None:
        volt = _opt_f(info, "voltage")

    unit = sniff_hashrate_unit(raw_hr, hashrate_unit)
    ghs = to_ghs(raw_hr, unit)
    ths = ghs / 1000.0
    eff = j_per_th(power, ghs)

    return NormalizedMetrics(
        hashrate_ghs=ghs,
        hashrate_ths=ths,
        power_w=power,
        temp_c=temp,
        frequency_mhz=freq,
        voltage_mv=volt,
        j_per_th=eff,
        raw_hashrate=raw_hr,
        hashrate_unit_assumed=unit,
    )


def efficiency_deviation_pct(actual_jth: Optional[float], ref_jth: float) -> Optional[float]:
    if actual_jth is None or ref_jth <= 0:
        return None
    return ((actual_jth - ref_jth) / ref_jth) * 100.0


def chip_from_info(info: Dict[str, Any]) -> str:
    """Best-effort ASIC family string from AxeOS payload."""
    for k in ("ASICModel", "asicModel", "asic_model", "boardVersion", "deviceModel", "model"):
        v = info.get(k)
        if not v:
            continue
        s = str(v).upper()
        for chip in ("BM1370", "BM1368", "BM1366", "BM1397"):
            if chip in s:
                return chip
        # board names
        if "GAMMA" in s:
            return "BM1370"
        if "SUPRA" in s:
            return "BM1368"
        if "ULTRA" in s or "MAX" in s:
            return "BM1366"
        return str(v)
    # frequency-class heuristic (weak)
    return "unknown"


def score_efficiency(
    m: NormalizedMetrics,
    *,
    w_jth: float = 1.0,
    w_hash: float = 0.35,
    w_temp: float = 0.15,
    temp_soft: float = 65.0,
) -> float:
    """
    Higher is better. Combines lower J/TH, higher hashrate, cooler temp.
    Used by the safe tuner for accept/reject — not a marketing metric.
    """
    if m.hashrate_ghs <= 0 or m.j_per_th is None or m.j_per_th <= 0:
        return -1e9
    # invert J/TH (lower better): 20 J/TH → 50, 15 → 66.7
    jth_score = (1000.0 / m.j_per_th) * w_jth
    hash_score = m.hashrate_ths * 100.0 * w_hash
    temp_pen = max(0.0, m.temp_c - temp_soft) * 5.0 * w_temp
    return jth_score + hash_score - temp_pen
