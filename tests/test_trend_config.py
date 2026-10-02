import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors, build_trend_detector
from price_alert.config import AppConfig, load_config
from price_alert.trend.config import TrendConfig


def test_defaults_enable_three_periods_with_period_specific_thresholds():
    trend = AppConfig().trend

    rules = trend.rules()
    assert [rule.period for rule in rules] == ["3m", "5m", "15m"]
    assert [rule.candles for rule in rules] == [5, 4, 3]
    assert [rule.min_change_percent for rule in rules] == [5.0, 6.0, 8.0]
    assert [rule.max_counter_candles for rule in rules] == [1, 1, 0]
    assert all(rule.max_rebound_ratio == 0.3 and rule.min_body_ratio == 0.5 for rule in rules)
    assert all(rule.max_single_candle_ratio == 0.5 for rule in rules)


def test_partial_period_override_keeps_that_periods_own_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
trend:
  periods:
    15m:
      candles: 8
    3m:
      enabled: false
""",
        encoding="utf-8",
    )

    trend = load_config(path).trend

    fifteen = {rule.period: rule for rule in trend.rules()}["15m"]
    assert (fifteen.candles, fifteen.min_change_percent, fifteen.max_counter_candles) == (8, 8.0, 0)
    assert [rule.period for rule in trend.rules()] == ["5m", "15m"]


def test_unknown_keys_and_periods_are_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"periods": {"2m": {}}}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"periods": {"3m": {"lookback": 5}}}})
    # 1 分钟周期已移除。
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"periods": {"1m": {}}}})
    # 已移除的配置项残留在旧 YAML 里时要报错，而不是被静默忽略。
    for removed in ({"exhaustion": {"enabled": False}}, {"cooldown_minutes": 30}, {"notify_end": True}):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"trend": removed})
    for removed_field in ("min_volume_ratio", "min_step_ratio", "baseline_candles", "min_trend_candle_ratio"):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"trend": {"periods": {"3m": {removed_field: 1}}}})


def test_enabled_trend_needs_at_least_one_period():
    periods = {name: {"enabled": False} for name in ("3m", "5m", "15m")}

    with pytest.raises(ValidationError, match="至少要启用一个周期"):
        AppConfig.model_validate({"trend": {"periods": periods}})
    # 整体关闭时不校验周期。
    assert AppConfig.model_validate({"trend": {"enabled": False, "periods": periods}}).trend.enabled is False


def test_counter_candles_must_be_fewer_than_candles():
    with pytest.raises(ValidationError, match="max_counter_candles 必须小于 candles"):
        AppConfig.model_validate({"trend": {"periods": {"3m": {"candles": 4, "max_counter_candles": 4}}}})


def test_disabled_trend_is_not_built_into_monitor():
    assert [entry.name for entry in build_detectors(AppConfig())] == ["ATR 异动", "K 线趋势"]
    disabled = AppConfig.model_validate({"trend": {"enabled": False}})
    assert [entry.name for entry in build_detectors(disabled)] == ["ATR 异动"]


def test_alert_continuing_defaults_on_and_is_passed_to_detector():
    assert TrendConfig().alert_continuing is True
    assert build_trend_detector(TrendConfig())._alert_continuing is True
    assert build_trend_detector(TrendConfig(alert_continuing=False))._alert_continuing is False
