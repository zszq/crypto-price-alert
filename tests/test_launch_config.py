import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig
from price_alert.launch.config import LaunchConfig
from price_alert.universe import VolumeRange


def test_defaults_and_build_rule():
    config = LaunchConfig()
    rule = config.to_rule()

    assert (config.min_volume_24h_quote, config.max_volume_24h_quote) == (1_000_000, None)
    assert (rule.window_minutes, rule.baseline_minutes, rule.history_minutes) == (5, 360, 365)
    assert (rule.min_volume_ratio, rule.min_window_quote, rule.min_change_percent) == (3.0, 100_000, 3.0)
    assert (rule.require_breakout, rule.alert_drops) == (True, False)
    assert (rule.cooldown_minutes, rule.realert_step_percent) == (15, 5.0)
    assert rule.max_change_percent == 15.0
    assert LaunchConfig(max_change_percent=None).to_rule().max_change_percent is None


def test_launch_detector_joins_monitor_with_its_own_volume_range():
    config = AppConfig.model_validate({"launch": {"min_volume_24h_quote": 50_000, "max_volume_24h_quote": 5_000_000}})

    launch = build_detectors(config)[-1]

    assert launch.name == "放量启动"
    assert launch.volume == VolumeRange(50_000, 5_000_000)
    assert (launch.candle_interval, launch.warmup_candles) == ("1m", 366)
    disabled = AppConfig.model_validate({"launch": {"enabled": False}})
    assert "放量启动" not in [entry.name for entry in build_detectors(disabled)]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"window_minutes": 30, "baseline_minutes": 30}, "baseline_minutes 必须大于 window_minutes"),
        ({"window_minutes": 10, "cooldown_minutes": 5}, "cooldown_minutes 不能小于 window_minutes"),
        ({"min_volume_24h_quote": 100, "max_volume_24h_quote": 100}, "max_volume_24h_quote"),
        ({"min_volume_ratio": 1}, "min_volume_ratio"),
        ({"min_change_percent": 5, "max_change_percent": 5}, "max_change_percent 必须大于 min_change_percent"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_rejects_inconsistent_settings(values, message):
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate({"launch": values})
