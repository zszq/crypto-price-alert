from price_alert.assembly import build_detector, build_windows
from price_alert.config import AppConfig
from price_alert.streak import StreakRule


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


def test_streak_gap_follows_short_cooldown_and_can_be_switched_off():
    assert build_detector(AppConfig()).streak_rule == StreakRule(min_alerts=3, max_gap_seconds=60)

    # 间隔上限 = 短窗口冷却 + 容差：只调冷却时连续提醒自动跟着放宽，不会永远连不成串。
    longer = AppConfig.model_validate(
        {"indicator": {"short_window": {"cooldown_seconds": 90}, "streak": {"gap_tolerance_seconds": 20}}}
    )
    assert build_detector(longer).streak_rule == StreakRule(min_alerts=3, max_gap_seconds=110)

    disabled = AppConfig.model_validate({"indicator": {"streak": {"enabled": False}}})
    assert build_detector(disabled).streak_rule is None
    assert build_detector(AppConfig(), streak=False).streak_rule is None
