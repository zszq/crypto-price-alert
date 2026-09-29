from pathlib import Path

import pytest
from pydantic import ValidationError

from price_alert.config import AppConfig, load_config


def test_load_config_and_webhook_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
gate:
  min_volume_24h_quote: 10000001
indicator:
  trigger_atr_multiple: 1.25
alerts:
  webhook_url: null
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRICE_ALERT_WEBHOOK_URL", "https://example.test/hook")

    config = load_config(path)

    assert config.gate.min_volume_24h_quote == 10_000_001
    assert config.indicator.trigger_atr_multiple == 1.25
    assert config.alerts.webhook_url == "https://example.test/hook"


def test_rejects_atr_configuration_without_enough_warmup_data():
    with pytest.raises(ValidationError, match="warmup_candles"):
        AppConfig.model_validate({"indicator": {"atr_period": 20, "warmup_candles": 20}})


def test_noise_protection_defaults_are_enabled():
    config = AppConfig()

    assert config.indicator.trigger_atr_multiple == 1.5
    assert config.indicator.min_change_percent == 1.0
    assert config.indicator.confirmation_seconds == 3


def test_default_yaml_passes_validation(monkeypatch):
    # default.yaml 的取值允许偏离代码默认值（运营随时调参），但仍须能通过校验，
    # 否则未知键或越界值要等到启动服务时才暴露。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    default_yaml = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

    load_config(default_yaml)


def test_rejects_reconnect_max_below_initial_delay():
    with pytest.raises(ValidationError, match="reconnect_max_seconds"):
        AppConfig.model_validate({"gate": {"reconnect_initial_seconds": 10, "reconnect_max_seconds": 5}})


def test_rejects_confirmation_longer_than_lookback():
    with pytest.raises(ValidationError, match="confirmation_seconds"):
        AppConfig.model_validate({"indicator": {"lookback_seconds": 5, "confirmation_seconds": 6}})


def test_atr_age_default_follows_candle_interval_but_explicit_value_is_validated():
    assert AppConfig.model_validate({"indicator": {"candle_interval": "5m"}}).indicator.max_atr_age_seconds == 900

    with pytest.raises(ValidationError, match="max_atr_age_seconds"):
        AppConfig.model_validate({"indicator": {"candle_interval": "5m", "max_atr_age_seconds": 180}})


def test_default_yaml_long_window_cooldown_follows_lookback(monkeypatch, tmp_path):
    # 运营只改长窗口长度时不应被冷却校验卡住，所以 default.yaml 刻意不写死冷却。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    default_yaml = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    edited = tmp_path / "config.yaml"
    edited.write_text(
        default_yaml.read_text(encoding="utf-8").replace("lookback_seconds: 180", "lookback_seconds: 300"),
        encoding="utf-8",
    )

    assert load_config(edited).indicator.long_window.cooldown_seconds == 300


def test_long_window_defaults_and_cooldown_follows_lookback():
    long_window = AppConfig().indicator.long_window
    assert long_window.enabled is True
    assert (long_window.lookback_seconds, long_window.cooldown_seconds) == (180, 180)

    # 只改观察窗口长度时冷却自动跟随，不会因为忘改冷却而重复提醒同一段行情。
    configured = AppConfig.model_validate({"indicator": {"long_window": {"lookback_seconds": 300}}})
    assert configured.indicator.long_window.cooldown_seconds == 300


def test_rejects_long_window_cooldown_shorter_than_its_lookback():
    with pytest.raises(ValidationError, match="cooldown_seconds"):
        AppConfig.model_validate({"indicator": {"long_window": {"lookback_seconds": 300, "cooldown_seconds": 120}}})


def test_rejects_long_window_not_longer_than_short_window():
    with pytest.raises(ValidationError, match="long_window.lookback_seconds"):
        AppConfig.model_validate({"indicator": {"lookback_seconds": 180}})

    # 关闭长窗口后不再要求它比短窗口长。
    AppConfig.model_validate({"indicator": {"lookback_seconds": 180, "long_window": {"enabled": False}}})


def test_cooldown_lives_under_indicator_not_alerts():
    assert AppConfig.model_validate({"indicator": {"cooldown_seconds": 45}}).indicator.cooldown_seconds == 45
    # 旧位置直接报错而不是静默忽略，避免以为改了冷却其实没生效。
    with pytest.raises(ValidationError, match="cooldown_seconds"):
        AppConfig.model_validate({"alerts": {"cooldown_seconds": 45}})


def test_rejects_unknown_long_window_keys():
    with pytest.raises(ValidationError, match="long_window"):
        AppConfig.model_validate({"indicator": {"long_window": {"lookback": 300}}})
