"""Unit tests for hashrate / J/TH normalization."""

from hme.units import (
    j_per_th,
    normalize_axeos_info,
    score_efficiency,
    sniff_hashrate_unit,
    to_ghs,
)


def test_sniff_ghs():
    assert sniff_hashrate_unit(1136.0) == "ghs"
    assert sniff_hashrate_unit(500.0) == "ghs"


def test_sniff_ths():
    assert sniff_hashrate_unit(1.136) == "ths"
    assert sniff_hashrate_unit(0.9) == "ths"


def test_to_ghs():
    assert abs(to_ghs(1.0, "ths") - 1000.0) < 1e-6
    assert abs(to_ghs(1136.0, "ghs") - 1136.0) < 1e-6


def test_j_per_th():
    # 20.89 W @ 1136 GH/s = 20.89 / 1.136 ≈ 18.39 J/TH
    v = j_per_th(20.89, 1136.0)
    assert v is not None
    assert 18.0 < v < 19.0


def test_j_per_th_wrong_unit_trap():
    # If someone fed TH/s as GH/s: 1.136 GH/s & 20W → absurd ~18400 J/TH
    bad = j_per_th(20.89, 1.136)
    assert bad is not None and bad > 1000


def test_normalize_ghs_payload():
    m = normalize_axeos_info({"hashRate": 1136.0, "power": 20.89, "temp": 63.0, "frequency": 525})
    assert m.hashrate_unit_assumed == "ghs"
    assert abs(m.hashrate_ghs - 1136.0) < 0.1
    assert m.j_per_th is not None and 18.0 < m.j_per_th < 19.0


def test_normalize_ths_payload():
    m = normalize_axeos_info({"hashRate": 1.136, "power": 20.89, "temp": 63.0})
    assert m.hashrate_unit_assumed == "ths"
    assert abs(m.hashrate_ghs - 1136.0) < 0.1


def test_score_prefers_cooler_efficient():
    a = normalize_axeos_info({"hashRate": 1000, "power": 20, "temp": 60})
    b = normalize_axeos_info({"hashRate": 1000, "power": 22, "temp": 72})
    assert score_efficiency(a) > score_efficiency(b)


def test_score_max_hashrate_prefers_higher_hr():
    from hme.units import score_max_hashrate
    a = normalize_axeos_info({"hashRate": 1400, "power": 25, "temp": 66, "vrTemp": 72})
    b = normalize_axeos_info({"hashRate": 1200, "power": 22, "temp": 60, "vrTemp": 70})
    assert score_max_hashrate(a) > score_max_hashrate(b)


def test_normalize_includes_shares_and_vr():
    m = normalize_axeos_info({
        "hashRate": 1300, "power": 25, "temp": 67, "vrTemp": 74,
        "sharesAccepted": 100, "sharesRejected": 1, "frequency": 650,
        "coreVoltage": 1150, "isUsingFallbackStratum": 1,
    })
    assert m.vr_temp_c == 74
    assert m.shares_accepted == 100
    assert m.reject_pct is not None and m.reject_pct < 2
    assert m.using_fallback_stratum is True
