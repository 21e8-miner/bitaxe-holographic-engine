"""Minimal AxeOS HTTP client for Bitaxe-class boards."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from .config import BoundsConfig, DeviceConfig, HMEConfig
from .units import NormalizedMetrics, chip_from_info, normalize_axeos_info

log = logging.getLogger("hme.client")


class BitaxeError(RuntimeError):
    pass


@dataclass
class ApplyResult:
    ok: bool
    dry_run: bool
    payload: Dict[str, Any]
    restarted: bool
    message: str
    before: Optional[NormalizedMetrics] = None
    after: Optional[NormalizedMetrics] = None


class BitaxeClient:
    def __init__(self, cfg: HMEConfig):
        self.cfg = cfg
        self.device: DeviceConfig = cfg.device
        self.bounds: BoundsConfig = cfg.bounds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "hme-safe-tuner/1.5"})

    @property
    def base(self) -> str:
        ip = self.device.ip.strip()
        if ip.startswith("http://") or ip.startswith("https://"):
            return ip.rstrip("/")
        return f"http://{ip}"

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base + path

    def get_json(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        t = timeout if timeout is not None else self.device.timeout_sec
        try:
            r = self.session.get(self._url(path), timeout=t)
        except requests.RequestException as e:
            raise BitaxeError(f"GET {path} failed: {e}") from e
        if r.status_code != 200:
            raise BitaxeError(f"GET {path} → HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError as e:
            raise BitaxeError(f"GET {path} non-JSON: {r.text[:200]}") from e
        if not isinstance(data, dict):
            raise BitaxeError(f"GET {path} expected object, got {type(data)}")
        return data

    def patch_json(self, path: str, body: Dict[str, Any], timeout: Optional[float] = None) -> Tuple[int, str]:
        t = timeout if timeout is not None else self.device.timeout_sec
        try:
            r = self.session.patch(self._url(path), json=body, timeout=t)
            return r.status_code, r.text
        except requests.RequestException as e:
            raise BitaxeError(f"PATCH {path} failed: {e}") from e

    def post(self, path: str, timeout: Optional[float] = None) -> Tuple[int, str]:
        t = timeout if timeout is not None else min(self.device.timeout_sec, 3.0)
        try:
            r = self.session.post(self._url(path), timeout=t)
            return r.status_code, r.text
        except requests.RequestException as e:
            # restart often drops connection mid-response
            return 0, str(e)

    def system_info(self) -> Dict[str, Any]:
        # Primary AxeOS path; fall back to /api/system on older builds
        try:
            return self.get_json("/api/system/info")
        except BitaxeError:
            return self.get_json("/api/system")

    def system_settings(self) -> Dict[str, Any]:
        try:
            return self.get_json("/api/system")
        except BitaxeError:
            return self.system_info()

    def metrics(self, info: Optional[Dict[str, Any]] = None) -> NormalizedMetrics:
        info = info if info is not None else self.system_info()
        return normalize_axeos_info(info, self.cfg.qc.hashrate_unit)

    def detect_chip(self, info: Optional[Dict[str, Any]] = None) -> str:
        if self.device.chip:
            return self.device.chip.upper()
        info = info if info is not None else self.system_info()
        return chip_from_info(info)

    def ping(self) -> Tuple[bool, float, Optional[Dict[str, Any]], Optional[str]]:
        """Return (ok, rtt_ms, info, error)."""
        t0 = time.time()
        try:
            info = self.system_info()
            rtt = (time.time() - t0) * 1000.0
            return True, rtt, info, None
        except Exception as e:
            rtt = (time.time() - t0) * 1000.0
            return False, rtt, None, str(e)

    def clamp_profile(self, frequency: Optional[int] = None, core_voltage: Optional[int] = None) -> Dict[str, int]:
        b = self.bounds
        out: Dict[str, int] = {}
        if frequency is not None:
            f = int(frequency)
            f = max(b.min_freq_mhz, min(b.max_freq_mhz, f))
            out["frequency"] = f
        if core_voltage is not None:
            v = int(core_voltage)
            v = max(b.min_voltage_mv, min(b.max_voltage_mv, v))
            out["coreVoltage"] = v
        return out

    def gate_ok(self, m: NormalizedMetrics) -> Tuple[bool, str]:
        """Hard safety gates — must pass before and after apply."""
        b = self.bounds
        if m.temp_c >= b.max_temp_c:
            return False, f"temp {m.temp_c:.1f}°C ≥ max {b.max_temp_c}°C"
        if m.vr_temp_c is not None and m.vr_temp_c >= b.max_vr_temp_c:
            return False, f"VR temp {m.vr_temp_c:.1f}°C ≥ max {b.max_vr_temp_c}°C"
        if m.power_w > 0 and m.power_w >= b.max_power_w:
            return False, f"power {m.power_w:.1f}W ≥ max {b.max_power_w}W"
        # Vin: AxeOS reports ~5000 mV rail; only enforce if looks like rail not core
        if m.vin_mv is not None and m.vin_mv > 2000 and m.vin_mv < b.min_vin_mv:
            return False, f"Vin {m.vin_mv:.0f}mV < min {b.min_vin_mv:.0f}mV"
        if m.reject_pct is not None and m.reject_pct >= b.max_reject_pct:
            return False, f"reject {m.reject_pct:.2f}% ≥ max {b.max_reject_pct:.2f}%"
        return True, "ok"

    def apply_vf(
        self,
        *,
        frequency: Optional[int] = None,
        core_voltage: Optional[int] = None,
        dry_run: bool = True,
        force_restart: bool = False,
    ) -> ApplyResult:
        """
        PATCH frequency/coreVoltage within bounds.
        Never restarts unless allow_restart and force_restart.
        """
        before_info = None
        before_m = None
        try:
            before_info = self.system_info()
            before_m = self.metrics(before_info)
            ok, reason = self.gate_ok(before_m)
            if not ok:
                return ApplyResult(False, dry_run, {}, False, f"blocked pre-apply: {reason}", before_m, None)
        except BitaxeError as e:
            return ApplyResult(False, dry_run, {}, False, f"pre-read failed: {e}", None, None)

        payload = self.clamp_profile(frequency, core_voltage)
        if not payload:
            return ApplyResult(False, dry_run, {}, False, "empty profile", before_m, None)

        # AxeOS often wants overclock flag when above stock
        payload["overclockEnabled"] = 1

        if dry_run:
            return ApplyResult(
                True, True, payload, False,
                f"dry-run would PATCH {payload}",
                before_m, before_m,
            )

        try:
            code, text = self.patch_json("/api/system", payload)
        except BitaxeError as e:
            return ApplyResult(False, False, payload, False, str(e), before_m, None)

        if code not in (200, 201, 204):
            return ApplyResult(False, False, payload, False, f"PATCH HTTP {code}: {text[:200]}", before_m, None)

        restarted = False
        if force_restart and self.device.allow_restart:
            log.warning("Restarting device (allow_restart=true)…")
            self.post("/api/system/restart")
            restarted = True
            time.sleep(15)

        # re-read (best effort)
        after_m = None
        try:
            time.sleep(1.0)
            after_m = self.metrics()
        except BitaxeError:
            pass

        return ApplyResult(True, False, payload, restarted, "applied", before_m, after_m)

    def safe_profile(self) -> Dict[str, int]:
        """Conservative fallback profile for emergency rollback."""
        return self.clamp_profile(
            frequency=self.bounds.base_freq_mhz,
            core_voltage=None,
        )
