"""Hashrate / efficiency / scoring helpers.

AxeOS has historically reported hashrate in GH/s for Bitaxe boards while some
forks use TH/s. Mis-labeling either way destroys J/TH by 1000×.

Scoring modes:
  efficiency — lower J/TH preferred (paid power)
  max_hashrate — free power: maximize stable GH/s under thermal/reject gates
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional


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
    # extended (best-effort from AxeOS)
    vr_temp_c: Optional[float] = None
    vin_mv: Optional[float] = None
    shares_accepted: Optional[int] = None
    shares_rejected: Optional[int] = None
    reject_pct: Optional[float] = None
    using_fallback_stratum: Optional[bool] = None
    fan_rpm: Optional[int] = None


def sniff_hashrate_unit(raw: float, hint: str = "auto") -> str:
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


def _opt_i(d: Dict[str, Any], *keys: str) -> Optional[int]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except (TypeError, ValueError):
                continue
    return None


def normalize_axeos_info(info: Dict[str, Any], hashrate_unit: str = "auto") -> NormalizedMetrics:
    """Map AxeOS /api/system/info (and cousins) to normalized metrics."""
    raw_hr = _f(info, "hashRate", "hashrate", "hash_rate", "hr", default=0.0)
    power = _f(info, "power", "power_w", "watts", default=0.0)
    temp = _f(info, "temp", "temperature", "temp_c", default=0.0)
    freq = _opt_f(info, "frequency", "freq", "asicFrequency", "currentFrequency")
    volt = _opt_f(info, "coreVoltage", "core_voltage", "coreVoltageActual")
    if volt is None:
        volt = _opt_f(info, "voltage")  # may be board rail on some builds
    vr = _opt_f(info, "vrTemp", "vr_temp", "vregTemp")
    vin = _opt_f(info, "voltage")  # AxeOS "voltage" is often ~5V rail in mV
    # if coreVoltageActual exists, prefer it for core; rail is separate field
    if info.get("coreVoltageActual") is not None:
        volt = _opt_f(info, "coreVoltageActual", "coreVoltage")
        vin = _opt_f(info, "voltage")

    acc = _opt_i(info, "sharesAccepted", "shares_accepted")
    rej = _opt_i(info, "sharesRejected", "shares_rejected")
    rej_pct = None
    if acc is not None and rej is not None and (acc + rej) > 0:
        rej_pct = 100.0 * rej / (acc + rej)

    unit = sniff_hashrate_unit(raw_hr, hashrate_unit)
    ghs = to_ghs(raw_hr, unit)
    ths = ghs / 1000.0
    eff = j_per_th(power, ghs)
    fan = _opt_i(info, "fanrpm", "fanRpm", "fan_rpm")
    fb = info.get("isUsingFallbackStratum")
    using_fb = bool(fb) if fb is not None else None

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
        vr_temp_c=vr,
        vin_mv=vin,
        shares_accepted=acc,
        shares_rejected=rej,
        reject_pct=rej_pct,
        using_fallback_stratum=using_fb,
        fan_rpm=fan,
    )


def efficiency_deviation_pct(actual_jth: Optional[float], ref_jth: float) -> Optional[float]:
    if actual_jth is None or ref_jth <= 0:
        return None
    return ((actual_jth - ref_jth) / ref_jth) * 100.0


def chip_from_info(info: Dict[str, Any]) -> str:
    for k in ("ASICModel", "asicModel", "asic_model", "boardVersion", "deviceModel", "model"):
        v = info.get(k)
        if not v:
            continue
        s = str(v).upper()
        for chip in ("BM1370", "BM1368", "BM1366", "BM1397"):
            if chip in s:
                return chip
        if "GAMMA" in s:
            return "BM1370"
        if "SUPRA" in s:
            return "BM1368"
        if "ULTRA" in s or "MAX" in s:
            return "BM1366"
        return str(v)
    return "unknown"


def score_efficiency(
    m: NormalizedMetrics,
    *,
    w_jth: float = 1.0,
    w_hash: float = 0.35,
    w_temp: float = 0.15,
    temp_soft: float = 65.0,
) -> float:
    """Paid-power score: lower J/TH + hashrate − thermal penalty."""
    if m.hashrate_ghs <= 0 or m.j_per_th is None or m.j_per_th <= 0:
        return -1e9
    jth_score = (1000.0 / m.j_per_th) * w_jth
    hash_score = m.hashrate_ths * 100.0 * w_hash
    temp_pen = max(0.0, m.temp_c - temp_soft) * 5.0 * w_temp
    return jth_score + hash_score - temp_pen


def score_max_hashrate(
    m: NormalizedMetrics,
    *,
    temp_soft: float = 68.0,
    vr_soft: float = 80.0,
    max_reject_pct: float = 1.0,
) -> float:
    """
    Free-power score: maximize stable hashrate.
    J/TH is only a mild stability hint (wild J/TH often means bad HR units/data).
    Reject% and thermal soft limits penalize.
    """
    if m.hashrate_ghs <= 0:
        return -1e9
    score = m.hashrate_ghs  # primary: GH/s
    # soft thermal penalties (hard gates enforced elsewhere)
    score -= max(0.0, m.temp_c - temp_soft) * 15.0
    if m.vr_temp_c is not None:
        score -= max(0.0, m.vr_temp_c - vr_soft) * 20.0
    if m.reject_pct is not None and m.reject_pct > max_reject_pct:
        score -= (m.reject_pct - max_reject_pct) * 50.0
    # tiny J/TH sanity: if efficiency is absurdly bad, data/HW is sick
    if m.j_per_th is not None and m.j_per_th > 40:
        score -= (m.j_per_th - 40) * 2.0
    return score


def score_metrics(m: NormalizedMetrics, objective: str = "max_hashrate", **kwargs) -> float:
    obj = (objective or "max_hashrate").lower().strip()
    if obj in ("efficiency", "jth", "paid"):
        return score_efficiency(m, **{k: v for k, v in kwargs.items() if k in (
            "w_jth", "w_hash", "w_temp", "temp_soft"
        )})
    return score_max_hashrate(
        m,
        temp_soft=kwargs.get("temp_soft", 68.0),
        vr_soft=kwargs.get("vr_soft", 80.0),
        max_reject_pct=kwargs.get("max_reject_pct", 1.0),
    )


def merge_share_delta(
    m: NormalizedMetrics,
    *,
    d_acc: int,
    d_rej: int,
    elapsed_sec: float,
    share_diff: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Estimate effective hashrate from share deltas.
    Without pool difficulty, report share rates only; if share_diff known, rough GH/s.
    """
    if elapsed_sec <= 0:
        return {"share_accept_per_min": None, "share_reject_per_min": None, "share_hr_ghs_est": None}
    acc_pm = (d_acc / elapsed_sec) * 60.0
    rej_pm = (d_rej / elapsed_sec) * 60.0
    est = None
    # Stratum share difficulty often ~ network-relative; AxeOS exposes stratumDiff
    # Very rough: hashes_per_share ≈ diff * 2^32; GH/s ≈ shares/s * hashes_per_share / 1e9
    if share_diff is not None and share_diff > 0 and d_acc >= 0:
        shares_per_s = d_acc / elapsed_sec
        est = shares_per_s * float(share_diff) * (2**32) / 1e9
    return {
        "share_accept_per_min": acc_pm,
        "share_reject_per_min": rej_pm,
        "share_hr_ghs_est": est,
        "delta_accepted": d_acc,
        "delta_rejected": d_rej,
        "elapsed_sec": elapsed_sec,
    }
