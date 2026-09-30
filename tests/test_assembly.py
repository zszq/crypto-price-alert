from price_alert.assembly import build_detector, build_windows
from price_alert.config import AppConfig


def test_build_windows_puts_short_window_first_and_respects_long_window_switch():
    app_config = AppConfig.model_validate(
        {"indicator": {"short_window": {"cooldown_seconds": 45}, "long_window": {"lookback_seconds": 300}}}
    )

    short, long = build_windows(app_config)
    assert (short.name, short.lookback_seconds, short.cooldown_seconds) == ("short", 30, 45)
    assert (long.name, long.lookback_seconds, long.cooldown_seconds) == ("long", 300, 300)
    # simulate 这类场景整体关掉冷却。
    assert {window.cooldown_seconds for window in build_windows(app_config, cooldown_seconds=0)} == {0}

    disabled = AppConfig.model_validate({"indicator": {"long_window": {"enabled": False}}})
    assert [window.name for window in build_windows(disabled)] == ["short"]
    assert build_detector(disabled).max_lookback_seconds == 30
