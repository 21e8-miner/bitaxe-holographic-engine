"""Primary / fallback stratum health checks for free-power max-uptime ops."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


@dataclass
class HostProbe:
    host: str
    port: int
    ok: bool
    rtt_ms: Optional[float]
    error: Optional[str] = None


@dataclass
class PoolHealthReport:
    using_fallback: bool
    primary: Optional[HostProbe]
    fallback: Optional[HostProbe]
    primary_url: Optional[str]
    fallback_url: Optional[str]
    recommendation: str
    severity: str  # ok | warn | critical


def _parse_stratum_host_port(url: Optional[str], default_port: int = 3333) -> Optional[Tuple[str, int]]:
    if not url or not str(url).strip():
        return None
    s = str(url).strip()
    # bare host or host:port
    if "://" not in s:
        if ":" in s:
            host, _, p = s.rpartition(":")
            try:
                return host.strip(), int(p)
            except ValueError:
                return s, default_port
        return s, default_port
    # stratum+tcp://host:port
    s2 = s.replace("stratum+tcp://", "tcp://").replace("stratum+ssl://", "tcp://")
    u = urlparse(s2 if "://" in s2 else "tcp://" + s2)
    host = u.hostname or s
    port = u.port or default_port
    return host, int(port)


def tcp_probe(host: str, port: int, timeout: float = 2.5) -> HostProbe:
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            rtt = (time.time() - t0) * 1000.0
            return HostProbe(host, port, True, rtt, None)
    except OSError as e:
        rtt = (time.time() - t0) * 1000.0
        return HostProbe(host, port, False, rtt, str(e))


def http_probe(url: str, timeout: float = 2.5) -> HostProbe:
    """Best-effort HTTP GET for local pool UIs (e.g. public-pool dashboards)."""
    t0 = time.time()
    try:
        if not url.startswith("http"):
            url = "http://" + url
        r = requests.get(url, timeout=timeout)
        rtt = (time.time() - t0) * 1000.0
        ok = r.status_code < 500
        return HostProbe(url, 80, ok, rtt, None if ok else f"HTTP {r.status_code}")
    except requests.RequestException as e:
        rtt = (time.time() - t0) * 1000.0
        return HostProbe(url, 80, False, rtt, str(e))


def assess_pool_health(info: Dict[str, Any], *, tcp_timeout: float = 2.5) -> PoolHealthReport:
    primary_url = info.get("stratumURL")
    fallback_url = info.get("fallbackStratumURL")
    primary_port = int(info.get("stratumPort") or 3333)
    fallback_port = int(info.get("fallbackStratumPort") or 3333)
    using_fb = bool(info.get("isUsingFallbackStratum"))

    def probe_url(url: Optional[str], port_hint: int) -> Optional[HostProbe]:
        parsed = _parse_stratum_host_port(url, port_hint)
        if not parsed:
            return None
        host, port = parsed
        # LAN hosts: also try HTTP on 80 for dashboard liveness
        p = tcp_probe(host, port, timeout=tcp_timeout)
        return p

    primary = probe_url(primary_url, primary_port)
    fallback = probe_url(fallback_url, fallback_port)

    # Recommendations (free-power: maximize valid work path)
    if using_fb and primary and primary.ok:
        rec = (
            f"Miner is on FALLBACK but primary {primary.host}:{primary.port} accepts TCP "
            f"(rtt={primary.rtt_ms:.0f}ms). Check AxeOS stratum user/password/restart or pool job stream — "
            f"recovering primary may improve solo EV if that was the intent."
        )
        sev = "warn"
    elif using_fb and primary and not primary.ok:
        rec = (
            f"On fallback; primary {primary.host}:{primary.port} unreachable ({primary.error}). "
            f"Repair local pool/node or accept CKPool lottery."
        )
        sev = "warn"
    elif not using_fb and primary and not primary.ok:
        rec = f"Primary stratum TCP failed ({primary.error}) — expect rising stale rejects; failover should engage."
        sev = "critical"
    elif not using_fb and primary and primary.ok:
        rec = f"Primary path OK ({primary.host}:{primary.port}, rtt={primary.rtt_ms:.0f}ms)."
        sev = "ok"
    else:
        rec = "Insufficient stratum metadata to probe."
        sev = "warn"

    if fallback and not fallback.ok and using_fb:
        rec += f" Fallback also looks down ({fallback.error})."
        sev = "critical"

    return PoolHealthReport(
        using_fallback=using_fb,
        primary=primary,
        fallback=fallback,
        primary_url=str(primary_url) if primary_url else None,
        fallback_url=str(fallback_url) if fallback_url else None,
        recommendation=rec,
        severity=sev,
    )


def report_to_dict(r: PoolHealthReport) -> Dict[str, Any]:
    def hp(p: Optional[HostProbe]) -> Optional[Dict[str, Any]]:
        if not p:
            return None
        return {
            "host": p.host,
            "port": p.port,
            "ok": p.ok,
            "rtt_ms": None if p.rtt_ms is None else round(p.rtt_ms, 1),
            "error": p.error,
        }

    return {
        "using_fallback": r.using_fallback,
        "primary_url": r.primary_url,
        "fallback_url": r.fallback_url,
        "primary": hp(r.primary),
        "fallback": hp(r.fallback),
        "recommendation": r.recommendation,
        "severity": r.severity,
    }
