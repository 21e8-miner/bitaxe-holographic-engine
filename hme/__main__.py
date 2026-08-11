#!/usr/bin/env python3
"""
HME CLI

  python -m hme doctor
  python -m hme status
  python -m hme tune              # dry-run proposals (default)
  python -m hme tune --apply      # live safe search (requires confirmation)
  python -m hme serve             # telemetry API on configured port
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .client import BitaxeClient, BitaxeError
from .config import load_config, write_example_if_missing
from .doctor import run_doctor
from .logger import TelemetryLog
from .tuner import SafeTuner, print_summary


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    return run_doctor(cfg, as_json=args.json)


def cmd_status(args: argparse.Namespace) -> int:
    from .pool_health import assess_pool_health, report_to_dict
    from .units import score_metrics

    cfg = load_config(args.config)
    client = BitaxeClient(cfg)
    try:
        info = client.system_info()
        m = client.metrics(info)
        chip = client.detect_chip(info)
    except BitaxeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    gate_ok, gate_reason = client.gate_ok(m)
    using_fb = bool(info.get("isUsingFallbackStratum"))
    stratum = (
        info.get("fallbackStratumURL") if using_fb else info.get("stratumURL")
    ) or info.get("stratumURL") or info.get("fallbackStratumURL")
    pool = None
    if cfg.pool.probe_enabled:
        pool = report_to_dict(assess_pool_health(info, tcp_timeout=cfg.pool.tcp_timeout_sec))
    score = score_metrics(
        m,
        objective=cfg.tuner.objective,
        temp_soft=cfg.bounds.warn_temp_c,
        vr_soft=cfg.bounds.warn_vr_temp_c,
        max_reject_pct=cfg.bounds.max_reject_pct,
    )
    payload = {
        "ip": cfg.device.ip,
        "chip": chip,
        "objective": cfg.tuner.objective,
        "free_power": cfg.qc.free_power,
        "hashrate_ghs": round(m.hashrate_ghs, 3),
        "hashrate_ths": round(m.hashrate_ths, 6),
        "power_w": round(m.power_w, 3),
        "temp_c": round(m.temp_c, 2),
        "vr_temp_c": m.vr_temp_c,
        "vin_mv": m.vin_mv,
        "frequency_mhz": m.frequency_mhz,
        "voltage_mv": m.voltage_mv,
        "j_per_th": None if m.j_per_th is None else round(m.j_per_th, 3),
        "hashrate_unit": m.hashrate_unit_assumed,
        "raw_hashrate": m.raw_hashrate,
        "shares_accepted": m.shares_accepted,
        "shares_rejected": m.shares_rejected,
        "reject_pct": None if m.reject_pct is None else round(m.reject_pct, 4),
        "fan_rpm": m.fan_rpm,
        "uptime_sec": info.get("uptimeSeconds"),
        "stratum": stratum,
        "stratum_primary": info.get("stratumURL"),
        "stratum_fallback": info.get("fallbackStratumURL"),
        "using_fallback_stratum": using_fb,
        "version": info.get("version"),
        "gates_ok": gate_ok,
        "gate_reason": gate_reason,
        "score": round(score, 2),
        "pool_health": pool,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"{payload['ip']}  {payload['chip']}  "
            f"{payload['hashrate_ghs']} GH/s  {payload['power_w']} W  "
            f"{payload['temp_c']}°C  VR={payload['vr_temp_c']}  "
            f"{payload['j_per_th']} J/TH  "
            f"{payload['frequency_mhz']} MHz / {payload['voltage_mv']} mV"
        )
        print(
            f"  objective={payload['objective']} free_power={payload['free_power']} "
            f"score={payload['score']} gates={'OK' if gate_ok else gate_reason}"
        )
        if m.shares_accepted is not None:
            print(
                f"  shares {m.shares_accepted} ok / {m.shares_rejected} rej "
                f"({payload['reject_pct']}%)  fan {payload['fan_rpm']} rpm  "
                f"stratum={'fallback ' if using_fb else ''}{stratum}"
            )
        if pool:
            print(f"  pool[{pool.get('severity')}]: {pool.get('recommendation')}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dry_run = not args.apply
    if args.dry_run:
        dry_run = True
    if args.apply and args.yes is False and not dry_run:
        # require --yes for live apply
        print(
            "Refusing live apply without --yes.\n"
            "  Dry-run:  python -m hme tune\n"
            "  Live:     python -m hme tune --apply --yes\n"
            f"  Target:   {cfg.device.ip}  max_temp={cfg.bounds.max_temp_c}°C  "
            f"freq={cfg.bounds.min_freq_mhz}–{cfg.bounds.max_freq_mhz} MHz",
            file=sys.stderr,
        )
        return 2

    # CLI overrides for soak length (useful for quick tests)
    if args.dwell is not None:
        cfg.tuner.dwell_sec = int(args.dwell)
    if args.steps is not None:
        cfg.tuner.max_steps = int(args.steps)
    if args.baseline is not None:
        baseline_sec = float(args.baseline)
    else:
        baseline_sec = None

    tlog = TelemetryLog(cfg)
    tuner = SafeTuner(cfg, tlog=tlog)
    try:
        summary = tuner.run(dry_run=dry_run, max_steps=args.steps, baseline_sec=baseline_sec)
    except BitaxeError as e:
        logging.getLogger("hme").error("tuner aborted: %s", e)
        tlog.event("tuner_abort", error=str(e))
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)

    # write results artifact
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "last_tune.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logging.getLogger("hme").info("Wrote %s", out_path)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Telemetry API + live dashboard UI (safe path — no auto overclock)."""
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS

    from .pool_health import assess_pool_health, report_to_dict
    from .share_truth import ShareTruthTracker
    from .units import score_metrics

    cfg = load_config(args.config)
    client = BitaxeClient(cfg)
    tlog = TelemetryLog(cfg)
    shares = ShareTruthTracker()
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask("hme", static_folder=str(static_dir), static_url_path="/static")
    CORS(app)

    @app.get("/")
    def index():
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            return jsonify({
                "error": "dashboard missing",
                "hint": "expected hme/static/index.html",
                "apis": ["/api/status", "/api/qc", "/api/health", "/api/raw"],
            }), 404
        return send_from_directory(static_dir, "index.html")

    @app.get("/dashboard")
    def dashboard_alias():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/health")
    def health():
        ok, rtt, info, err = client.ping()
        return jsonify({"ok": ok, "rtt_ms": rtt, "error": err, "version": __version__})

    @app.get("/api/status")
    def status():
        try:
            info = client.system_info()
            m = client.metrics(info)
            shares.push(m, info)
            tlog.sample(m, extra={
                "shares_accepted": m.shares_accepted,
                "shares_rejected": m.shares_rejected,
                "objective": cfg.tuner.objective,
            })
            gate_ok, gate_reason = client.gate_ok(m)
            pool = None
            if cfg.pool.probe_enabled:
                pool = report_to_dict(
                    assess_pool_health(info, tcp_timeout=cfg.pool.tcp_timeout_sec)
                )
            return jsonify({
                "chip": client.detect_chip(info),
                "version": info.get("version"),
                "hostname": info.get("hostname"),
                "objective": cfg.tuner.objective,
                "free_power": cfg.qc.free_power,
                "score": score_metrics(
                    m,
                    objective=cfg.tuner.objective,
                    temp_soft=cfg.bounds.warn_temp_c,
                    vr_soft=cfg.bounds.warn_vr_temp_c,
                    max_reject_pct=cfg.bounds.max_reject_pct,
                ),
                "metrics": {
                    "hashrate_ghs": m.hashrate_ghs,
                    "hashrate_ths": m.hashrate_ths,
                    "power_w": m.power_w,
                    "temp_c": m.temp_c,
                    "vr_temp_c": m.vr_temp_c,
                    "vin_mv": m.vin_mv,
                    "frequency_mhz": m.frequency_mhz,
                    "voltage_mv": m.voltage_mv,
                    "j_per_th": m.j_per_th,
                    "hashrate_unit": m.hashrate_unit_assumed,
                    "fan_rpm": m.fan_rpm,
                    "shares_accepted": m.shares_accepted,
                    "shares_rejected": m.shares_rejected,
                    "reject_pct": m.reject_pct or 0.0,
                },
                "share_truth": shares.window(300.0),
                "stratum": {
                    "url": info.get("stratumURL"),
                    "fallback": info.get("fallbackStratumURL"),
                    "using_fallback": bool(info.get("isUsingFallbackStratum")),
                    "user": info.get("stratumUser"),
                },
                "pool_health": pool,
                "gates_ok": gate_ok,
                "gate_reason": gate_reason,
                "ip": cfg.device.ip,
            })
        except BitaxeError as e:
            return jsonify({"error": str(e)}), 502

    @app.get("/api/pool")
    def pool_api():
        try:
            info = client.system_info()
            return jsonify(report_to_dict(
                assess_pool_health(info, tcp_timeout=cfg.pool.tcp_timeout_sec)
            ))
        except BitaxeError as e:
            return jsonify({"error": str(e)}), 502

    @app.get("/api/raw")
    def raw():
        """Proxy full AxeOS /api/system/info for dashboards."""
        try:
            return jsonify(client.system_info())
        except BitaxeError as e:
            return jsonify({"error": str(e)}), 502

    @app.get("/api/qc")
    def qc():
        try:
            info = client.system_info()
            m = client.metrics(info)
            chip = client.detect_chip(info)
            ref = cfg.ref_jth_for_chip(chip if chip != "unknown" else "default")
            from .units import efficiency_deviation_pct
            dev = efficiency_deviation_pct(m.j_per_th, ref)
            return jsonify({
                "chip": chip,
                "ref_j_per_th": ref,
                "actual_j_per_th": m.j_per_th,
                "deviation_pct": dev,
                "metrics": {
                    "hashrate_ghs": m.hashrate_ghs,
                    "power_w": m.power_w,
                    "temp_c": m.temp_c,
                },
            })
        except BitaxeError as e:
            return jsonify({"error": str(e)}), 502

    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"HME desk  http://{host}:{port}/")
    print(f"  device  {cfg.device.ip}  (monitor only, no auto-tune)")
    print(f"  api     http://{host}:{port}/api/status")
    app.run(host=host, port=port, debug=False)
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    dest = Path(args.output) if args.output else root / "config.toml"
    if dest.exists() and not args.force:
        print(f"exists: {dest} (use --force to overwrite)")
        return 1
    example = root / "config.example.toml"
    if not example.exists():
        print("config.example.toml missing", file=sys.stderr)
        return 1
    dest.write_text(example.read_text())
    print(f"Wrote {dest}")
    print("Edit device.ip, then: python -m hme doctor")
    return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    from .stratum_proxy import main as proxy_main
    return proxy_main([
        "--listen-host", args.listen_host,
        "--listen-port", str(args.listen_port),
        "--upstream-host", args.upstream_host,
        "--upstream-port", str(args.upstream_port),
    ])


def _detect_lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def cmd_repair_pool(args: argparse.Namespace) -> int:
    """PATCH Bitaxe primary stratum to local proxy; keep CKPool fallback."""
    cfg = load_config(args.config)
    bitaxe = args.bitaxe or cfg.device.ip
    primary = args.primary_host or _detect_lan_ip()
    port = int(args.primary_port)
    # verify proxy is listening
    import socket
    try:
        with socket.create_connection((primary if primary != "0.0.0.0" else "127.0.0.1", port), timeout=2):
            pass
    except OSError as e:
        print(
            f"error: nothing listening on {primary}:{port} ({e})\n"
            f"Start proxy first:  python -m hme proxy --listen-port {port}",
            file=sys.stderr,
        )
        return 2

    payload = {
        "stratumURL": primary,
        "stratumPort": port,
        "fallbackStratumURL": args.fallback_host,
        "fallbackStratumPort": int(args.fallback_port),
    }
    print(f"Bitaxe {bitaxe} → primary {primary}:{port}  fallback {args.fallback_host}:{args.fallback_port}")
    if not args.yes:
        print("Refusing without --yes (will PATCH AxeOS stratum settings).")
        return 2

    client = BitaxeClient(cfg)
    # override device ip if needed
    client.device.ip = bitaxe
    try:
        code, text = client.patch_json("/api/system", payload)
    except BitaxeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if code not in (200, 201, 204):
        print(f"error: PATCH HTTP {code}: {text[:300]}", file=sys.stderr)
        return 1
    print("✓ stratum settings updated")
    if args.restart:
        print("restarting Bitaxe mining service…")
        client.post("/api/system/restart")
        print("✓ restart issued (wait ~15–30s for reconnect)")
    else:
        print("Tip: add --restart if AxeOS does not re-subscribe immediately")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hme",
        description="Holographic Mining Engine — Bitaxe doctor & safe V/F tuner",
    )
    p.add_argument("--version", action="version", version=f"hme {__version__}")
    p.add_argument("-c", "--config", help="Path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Probe device, units, gates, firmware")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("status", help="One-shot normalized metrics")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    t = sub.add_parser("tune", help="Safe V/F search (dry-run by default)")
    t.add_argument("--apply", action="store_true", help="Actually PATCH device (requires --yes)")
    t.add_argument("--yes", action="store_true", help="Confirm live apply")
    t.add_argument("--dry-run", action="store_true", help="Force dry-run")
    t.add_argument("--dwell", type=int, help="Override dwell_sec")
    t.add_argument("--baseline", type=float, help="Baseline measure seconds")
    t.add_argument("--steps", type=int, help="Max accept steps")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_tune)

    sv = sub.add_parser("serve", help="Monitor-only HTTP API")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.set_defaults(func=cmd_serve)

    ic = sub.add_parser("init-config", help="Write config.toml from example")
    ic.add_argument("-o", "--output", help="Destination path")
    ic.add_argument("--force", action="store_true")
    ic.set_defaults(func=cmd_init_config)

    px = sub.add_parser("proxy", help="Local Stratum V1 proxy (repair primary when i7 is down)")
    px.add_argument("--listen-host", default="0.0.0.0")
    px.add_argument("--listen-port", type=int, default=3333)
    px.add_argument("--upstream-host", default="public-pool.io")
    px.add_argument("--upstream-port", type=int, default=21496)
    px.set_defaults(func=cmd_proxy)

    rp = sub.add_parser(
        "repair-pool",
        help="Point Bitaxe primary at this host's proxy IP (after proxy is running)",
    )
    rp.add_argument("--bitaxe", default=None, help="Bitaxe IP (default from config)")
    rp.add_argument("--primary-host", default=None, help="LAN IP of this machine (auto-detect)")
    rp.add_argument("--primary-port", type=int, default=3333)
    rp.add_argument("--fallback-host", default="solo.ckpool.org")
    rp.add_argument("--fallback-port", type=int, default=3333)
    rp.add_argument("--restart", action="store_true", help="Restart Bitaxe to re-lock stratum")
    rp.add_argument("--yes", action="store_true", help="Apply PATCH without prompt")
    rp.set_defaults(func=cmd_repair_pool)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
