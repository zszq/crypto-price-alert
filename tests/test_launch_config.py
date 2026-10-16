import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig
from price_alert.launch.config import LaunchConfig
from price_alert.launch.rule import Thresholds
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
    # 不写 drop 段：两个方向共用顶层门槛。
    assert rule.drop is None


def test_drop_section_overrides_only_what_it_sets():
    config = AppConfig.model_validate(
        {"launch": {"max_change_percent": 15, "drop": {"min_change_percent": 2, "min_volume_ratio": 4}}}
    ).launch

    assert config.to_rule().drop == Thresholds(4.0, 2.0, 15.0)
    # 显式 null 表示下跌不设上限；不写则沿用顶层。
    uncapped = LaunchConfig.model_validate({"drop": {"max_change_percent": None}})
    assert uncapped.to_rule().drop == Thresholds(3.0, 3.0, None)
    capped = LaunchConfig.model_validate({"max_change_percent": None, "drop": {"max_change_percent": 10}})
    assert capped.to_rule().drop == Thresholds(3.0, 3.0, 10.0)
    assert capped.to_rule().max_change_percent is None


def test_launch_detector_joins_monitor_with_its_own_volume_range():
    config = AppConfig.model_validate({"launch": {"min_volume_24h_quote": 50_000, "max_volume_24h_quote": 5_000_000}})

    launch = next(entry for entry in build_detectors(config) if entry.name == "放量启动")

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
        ({"drop": {"window_minutes": 10}}, "window_minutes"),
        ({"drop": {"min_volume_ratio": 1}}, "min_volume_ratio"),
        # 只写了下跌的下限，上限取自顶层 15：合并后冲突也要报出来。
        ({"drop": {"min_change_percent": 16}}, "放量下跌的 max_change_percent（15）必须大于 min_change_percent（16）"),
    ],
)
def test_rejects_inconsistent_settings(values, message):
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate({"launch": values})
