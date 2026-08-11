"""Tuner evaluate / candidate logic without hardware."""

from hme.config import HMEConfig, _from_raw
from hme.tuner import Candidate, SafeTuner
from hme.units import normalize_axeos_info


def _m(hr=1000.0, power=20.0, temp=60.0, freq=500, volt=1150):
    return normalize_axeos_info({
        "hashRate": hr, "power": power, "temp": temp,
        "frequency": freq, "coreVoltage": volt,
    })


def test_evaluate_accepts_improvement():
    # free-power default: need ≥2% HR gain
    cfg = _from_raw({"tuner": {"objective": "max_hashrate", "min_hashrate_gain": 0.02}})
    t = SafeTuner(cfg)
    base = _m(1000, 20, 60, 500)
    better = _m(1100, 20.5, 61, 525)
    ok, reason, bs, cs = t.evaluate(base, better)
    assert ok, reason
    assert cs > bs


def test_evaluate_rejects_hot():
    cfg = _from_raw({"bounds": {"max_temp_c": 70}, "tuner": {"objective": "max_hashrate"}})
    t = SafeTuner(cfg)
    base = _m(1000, 20, 60, 500)
    hot = _m(1200, 25, 75, 575)
    ok, reason, _, _ = t.evaluate(base, hot)
    assert not ok
    assert "temp" in reason.lower() or "gate" in reason.lower()


def test_evaluate_rejects_jth_regression():
    cfg = _from_raw({"tuner": {"objective": "efficiency", "max_jth_regression": 0.05}})
    t = SafeTuner(cfg)
    base = _m(1000, 18, 60, 500)   # 18 J/TH
    worse = _m(1000, 22, 60, 525)  # 22 J/TH — big regression
    ok, reason, _, _ = t.evaluate(base, worse)
    assert not ok
    assert "J/TH" in reason or "regression" in reason.lower()


def test_max_hashrate_rejects_tiny_gain():
    cfg = _from_raw({"tuner": {"objective": "max_hashrate", "min_hashrate_gain": 0.05}})
    t = SafeTuner(cfg)
    base = _m(1000, 20, 60, 500)
    tiny = _m(1020, 20.2, 60, 525)  # +2% only
    ok, reason, _, _ = t.evaluate(base, tiny)
    assert not ok
    assert "gain" in reason.lower() or "HR" in reason


def test_generate_candidates_climb():
    cfg = _from_raw({
        "bounds": {"min_freq_mhz": 425, "max_freq_mhz": 575, "base_freq_mhz": 500},
        "tuner": {"mode": "climb", "freq_step_mhz": 25},
    })
    t = SafeTuner(cfg)
    cur = _m(freq=500)
    cands = t.generate_candidates(cur)
    assert cands
    assert all(isinstance(c, Candidate) for c in cands)
    assert all(425 <= c.frequency_mhz <= 575 for c in cands)
    # should not include identity 500-only with no volt change as only candidate
    labels = [c.label() for c in cands]
    assert any("525" in x or "475" in x or "550" in x for x in labels)


def test_clamp_profile():
    from hme.client import BitaxeClient
    cfg = _from_raw({"bounds": {"min_freq_mhz": 425, "max_freq_mhz": 575, "min_voltage_mv": 1050, "max_voltage_mv": 1250}})
    c = BitaxeClient(cfg)
    p = c.clamp_profile(frequency=900, core_voltage=900)
    assert p["frequency"] == 575
    assert p["coreVoltage"] == 1050
