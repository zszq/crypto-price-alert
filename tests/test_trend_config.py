import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig, load_config


def test_defaults_enable_four_periods_with_period_specific_thresholds():
    trend = AppConfig().trend

    rules = trend.rules()
    assert [rule.period for rule in rules] == ["1m", "3m", "5m", "15m"]
    assert [rule.min_change_percent for rule in rules] == [4.0, 5.0, 6.0, 8.0]
    assert all(rule.min_volume_ratio == 1.5 for rule in rules)
    assert trend.episode_settings().exhaustion_periods == frozenset({"5m", "15m"})


def test_partial_period_override_keeps_that_periods_own_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
trend:
  periods:
    15m:
      candles: 8
    1m:
      enabled: false
""",
        encoding="utf-8",
    )

    trend = load_config(path).trend

    fifteen = {rule.period: rule for rule in trend.rules()}["15m"]
    assert (fifteen.candles, fifteen.min_change_percent, fifteen.baseline_candles) == (8, 8.0, 16)
    assert "1m" not in {rule.period for rule in trend.rules()}


def test_unknown_keys_and_periods_are_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"periods": {"2m": {}}}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"periods": {"1m": {"lookback": 5}}}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"trend": {"exhaustion": {"periods": ["2m"]}}})


def test_enabled_trend_needs_at_least_one_period():
    periods = {name: {"enabled": False} for name in ("1m", "3m", "5m", "15m")}

    with pytest.raises(ValidationError, match="至少要启用一个周期"):
        AppConfig.model_validate({"trend": {"periods": periods}})
    # 整体关闭时不校验周期。
    assert AppConfig.model_validate({"trend": {"enabled": False, "periods": periods}}).trend.enabled is False


def test_history_must_fit_in_one_warmup_request():
    with pytest.raises(ValidationError, match="不能超过 1999"):
        AppConfig.model_validate({"trend": {"periods": {"15m": {"candles": 60, "baseline_candles": 100}}}})


def test_disabled_trend_is_not_built_into_monitor():
    assert [entry.name for entry in build_detectors(AppConfig())] == ["ATR 异动", "K 线趋势"]
    disabled = AppConfig.model_validate({"trend": {"enabled": False}})
    assert [entry.name for entry in build_detectors(disabled)] == ["ATR 异动"]


def test_exhaustion_periods_must_be_enabled():
    with pytest.raises(ValidationError, match="exhaustion.periods 中的周期 15m 未启用"):
        AppConfig.model_validate({"trend": {"periods": {"15m": {"enabled": False}}}})
    # 关闭衰竭提示时不受约束。
    AppConfig.model_validate({"trend": {"periods": {"15m": {"enabled": False}}, "exhaustion": {"enabled": False}}})
