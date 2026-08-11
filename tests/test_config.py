"""Config loading tests."""

import os
from pathlib import Path

from hme.config import HMEConfig, load_config, _from_raw


def test_defaults_dry_run():
    cfg = load_config("/nonexistent/path.toml") if False else _from_raw({})
    assert cfg.tuner.dry_run is True
    assert cfg.bounds.max_temp_c == 70.0
    assert cfg.device.allow_restart is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("HME_BITAXE_IP", "10.0.0.9")
    monkeypatch.setenv("HME_DRY_RUN", "0")
    monkeypatch.setenv("HME_MAX_TEMP", "68")
    cfg = _from_raw({})
    assert cfg.device.ip == "10.0.0.9"
    assert cfg.tuner.dry_run is False
    assert cfg.bounds.max_temp_c == 68.0


def test_example_toml_loads():
    root = Path(__file__).resolve().parent.parent
    example = root / "config.example.toml"
    assert example.is_file()
    cfg = load_config(str(example))
    assert cfg.source_path
    assert cfg.bounds.min_freq_mhz < cfg.bounds.max_freq_mhz
    assert "BM1370" in cfg.qc.ref_j_per_th


def test_ref_jth():
    cfg = _from_raw({})
    assert cfg.ref_jth_for_chip("BM1370") == 15.5
    assert cfg.ref_jth_for_chip("unknown") == 17.0
