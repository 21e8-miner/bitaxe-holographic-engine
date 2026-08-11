"""Client apply dry-run with mocked HTTP."""

from unittest.mock import MagicMock, patch

from hme.client import BitaxeClient
from hme.config import _from_raw


def test_apply_vf_dry_run_no_http_patch():
    cfg = _from_raw({"device": {"ip": "127.0.0.1"}, "bounds": {"max_temp_c": 80, "max_power_w": 40}})
    client = BitaxeClient(cfg)
    fake_info = {
        "hashRate": 1000.0,
        "power": 18.0,
        "temp": 55.0,
        "frequency": 500,
        "coreVoltage": 1150,
    }
    with patch.object(client, "system_info", return_value=fake_info):
        with patch.object(client, "patch_json") as pj:
            res = client.apply_vf(frequency=525, dry_run=True)
            assert res.ok
            assert res.dry_run
            assert res.payload["frequency"] == 525
            pj.assert_not_called()


def test_apply_blocked_when_hot():
    cfg = _from_raw({"bounds": {"max_temp_c": 70}})
    client = BitaxeClient(cfg)
    hot = {"hashRate": 1000, "power": 20, "temp": 72, "frequency": 500}
    with patch.object(client, "system_info", return_value=hot):
        res = client.apply_vf(frequency=525, dry_run=False)
        assert not res.ok
        assert "temp" in res.message.lower()
