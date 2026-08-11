"""Pool health parsing / probes (no real network required for parse)."""

from hme.pool_health import _parse_stratum_host_port, assess_pool_health


def test_parse_bare_host():
    assert _parse_stratum_host_port("192.168.0.231", 3333) == ("192.168.0.231", 3333)


def test_parse_host_port():
    assert _parse_stratum_host_port("solo.ckpool.org:3333") == ("solo.ckpool.org", 3333)


def test_parse_stratum_url():
    h, p = _parse_stratum_host_port("stratum+tcp://pool.example:3334")
    assert h == "pool.example"
    assert p == 3334


def test_assess_fallback_when_primary_missing_meta():
    r = assess_pool_health({
        "stratumURL": "",
        "fallbackStratumURL": "solo.ckpool.org",
        "fallbackStratumPort": 3333,
        "isUsingFallbackStratum": 1,
    }, tcp_timeout=0.3)
    assert r.using_fallback is True
    assert r.severity in ("ok", "warn", "critical")
