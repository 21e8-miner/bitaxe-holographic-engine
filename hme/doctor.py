"""Device health / configuration doctor."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from .client import BitaxeClient, BitaxeError
from .config import HMEConfig
from .units import efficiency_deviation_pct, score_efficiency


def run_doctor(cfg: HMEConfig, *, as_json: bool = False) -> int:
    """
    Probe Bitaxe, print units/chip/firmware/gates. Exit 0 if reachable and gates OK.
    """
    client = BitaxeClient(cfg)
    report: Dict[str, Any] = {
        "hme_version": __import__("hme").__version__,
        "config_source": cfg.source_path or "(defaults + env)",
        "device_ip": cfg.device.ip,
        "dry_run_default": cfg.tuner.dry_run,
        "bounds": {
            "freq_mhz": [cfg.bounds.min_freq_mhz, cfg.bounds.max_freq_mhz],
            "voltage_mv": [cfg.bounds.min_voltage_mv, cfg.bounds.max_voltage_mv],
            "max_temp_c": cfg.bounds.max_temp_c,
            "max_power_w": cfg.bounds.max_power_w,
        },
        "ok": False,
        "checks": [],
    }

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})

    ok, rtt, info, err = client.ping()
    if not ok or info is None:
        check("reachability", False, err or "unreachable")
        return _emit(report, as_json, 2)

    check("reachability", True, f"rtt={rtt:.0f}ms base={client.base}")
    report["rtt_ms"] = round(rtt, 1)
    report["raw_keys"] = sorted(info.keys())

    # Identify
    chip = client.detect_chip(info)
    report["chip"] = chip
    check("chip", chip != "unknown", f"detected={chip}" + (f" forced={cfg.device.chip}" if cfg.device.chip else ""))

    for k in ("version", "firmware", "axeOSVersion", "idfVersion", "boardVersion", "ASICModel", "hostname"):
        if k in info and info[k] is not None:
            report.setdefault("identity", {})[k] = info[k]

    try:
        m = client.metrics(info)
    except Exception as e:
        check("metrics", False, str(e))
        return _emit(report, as_json, 3)

    report["metrics"] = {
        "raw_hashrate": m.raw_hashrate,
        "hashrate_unit_assumed": m.hashrate_unit_assumed,
        "hashrate_ghs": round(m.hashrate_ghs, 3),
        "hashrate_ths": round(m.hashrate_ths, 6),
        "power_w": round(m.power_w, 3),
        "temp_c": round(m.temp_c, 2),
        "frequency_mhz": m.frequency_mhz,
        "voltage_mv": m.voltage_mv,
        "j_per_th": None if m.j_per_th is None else round(m.j_per_th, 3),
    }

    unit_note = (
        f"raw={m.raw_hashrate} treated as {m.hashrate_unit_assumed.upper()} "
        f"→ {m.hashrate_ghs:.1f} GH/s ({m.hashrate_ths:.3f} TH/s)"
    )
    check("hashrate_units", m.hashrate_ghs > 0, unit_note)

    ref = cfg.ref_jth_for_chip(chip if chip != "unknown" else "default")
    dev = efficiency_deviation_pct(m.j_per_th, ref)
    report["qc"] = {
        "ref_j_per_th": ref,
        "deviation_pct": None if dev is None else round(dev, 2),
        "score": score_efficiency(m, temp_soft=cfg.bounds.warn_temp_c),
    }
    if m.j_per_th is not None:
        check(
            "efficiency",
            True,
            f"{m.j_per_th:.2f} J/TH vs ref {ref:.1f} ({dev:+.1f}% dev)" if dev is not None else f"{m.j_per_th:.2f} J/TH",
        )
    else:
        check("efficiency", False, "cannot compute J/TH (need power + hashrate)")

    gate_ok, gate_reason = client.gate_ok(m)
    check("safety_gates", gate_ok, gate_reason)
    if m.temp_c >= cfg.bounds.warn_temp_c:
        check("temp_warn", False, f"{m.temp_c:.1f}°C ≥ warn {cfg.bounds.warn_temp_c}°C")
    else:
        check("temp_warn", True, f"{m.temp_c:.1f}°C < warn {cfg.bounds.warn_temp_c}°C")

    # Settings readability
    try:
        settings = client.system_settings()
        report["settings_sample"] = {
            k: settings.get(k)
            for k in ("frequency", "coreVoltage", "overclockEnabled", "stratumURL", "stratumUser")
            if k in settings
        }
        check("settings_api", True, f"{len(settings)} keys from /api/system")
    except BitaxeError as e:
        check("settings_api", False, str(e))

    report["ok"] = all(c["ok"] for c in report["checks"] if c["name"] in ("reachability", "metrics", "safety_gates"))
    # soft failures don't fail exit for doctor info mode, but gates/reachability do
    exit_code = 0 if report["ok"] else 1
    return _emit(report, as_json, exit_code)


def _emit(report: Dict[str, Any], as_json: bool, code: int) -> int:
    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return code

    print("=" * 60)
    print("HME doctor")
    print("=" * 60)
    print(f"config : {report.get('config_source')}")
    print(f"device : {report.get('device_ip')}")
    print(f"dry_run: {report.get('dry_run_default')} (default; override with --apply / HME_DRY_RUN=0)")
    print()
    for c in report.get("checks") or []:
        mark = "✓" if c["ok"] else "✗"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print()
    m = report.get("metrics") or {}
    if m:
        print("Metrics (normalized)")
        print(f"  hashrate : {m.get('hashrate_ghs')} GH/s  ({m.get('hashrate_ths')} TH/s)")
        print(f"  unit sniff: raw={m.get('raw_hashrate')} as {m.get('hashrate_unit_assumed')}")
        print(f"  power    : {m.get('power_w')} W")
        print(f"  temp     : {m.get('temp_c')} °C")
        print(f"  freq/V   : {m.get('frequency_mhz')} MHz / {m.get('voltage_mv')} mV")
        print(f"  J/TH     : {m.get('j_per_th')}")
    qc = report.get("qc") or {}
    if qc:
        print(f"  QC ref   : {qc.get('ref_j_per_th')} J/TH  dev={qc.get('deviation_pct')}%  score={qc.get('score')}")
    ident = report.get("identity") or {}
    if ident:
        print("Identity")
        for k, v in ident.items():
            print(f"  {k}: {v}")
    print()
    print(f"Overall: {'OK' if report.get('ok') else 'ATTENTION'} (exit {code})")
    print("=" * 60)
    return code
