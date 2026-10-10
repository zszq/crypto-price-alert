from datetime import UTC, datetime, timedelta

import pytest
from colorama import Style

from price_alert.models import PriceAlert
from price_alert.notifier import (
    LAUNCH_LABEL_COLOR,
    LONG_WINDOW_COLOR,
    STREAK_LABEL_COLOR,
    SURGE_COLOR,
    SYMBOL_COLOR,
    TREND_LABEL_COLOR,
    colorize_alert,
    format_alert,
)
from price_alert.streak import StreakAlert, StreakRule, StreakTracker, describe_duration

BASE = datetime(2026, 10, 9, 5, 54, 12, tzinfo=UTC)


def short_alert(offset: float, reference: float, price: float, direction: str = "surge", **overrides) -> PriceAlert:
    values = {
        "symbol": "RLC_USDT",
        "direction": direction,
        "price": price,
        "reference_price": reference,
        "change_percent": (price / reference - 1) * 100,
        "move_atr": 2.0,
        "atr": 0.01,
        "atr_period": 14,
        "lookback_seconds": 30,
        "trade_count": 100,
        "volume_24h_quote": 20_000_000,
        "timestamp": BASE + timedelta(seconds=offset),
        "price_decimals": 4,
    }
    return PriceAlert(**(values | overrides))


def observe_all(tracker: StreakTracker, alerts: list[PriceAlert]) -> list[StreakAlert | None]:
    return [tracker.observe(alert) for alert in alerts]


def test_third_same_direction_alert_reports_cumulative_move_from_first_window_start():
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))

    results = observe_all(
        tracker,
        [short_alert(0, 0.8745, 0.8950), short_alert(31.5, 0.8950, 0.9150), short_alert(61.5, 0.9150, 0.9386)],
    )

    assert results[:2] == [None, None]
    streak = results[2]
    assert streak is not None
    assert streak.alert_count == 3
    assert streak.direction == "surge"
    # 起点是第一条提醒的窗口起点，而不是第一条提醒本身：涨幅从那一刻开始算才是整段行情。
    assert streak.started_at == BASE - timedelta(seconds=30)
    assert streak.start_price == 0.8745
    assert streak.price == 0.9386
    assert streak.change_percent == pytest.approx((0.9386 / 0.8745 - 1) * 100)
    assert streak.duration_seconds == 92


def test_streak_is_reported_once_and_can_start_again_after_breaking():
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))
    chain = [short_alert(offset, 1.0, 1.02) for offset in (0, 30, 60, 90, 120)]

    # 同一串只在第 3 条报一次，之后的短窗口提醒本身就说明行情仍在延续。
    assert [result is not None for result in observe_all(tracker, chain)] == [False, False, True, False, False]

    # 间隔超过上限后重新计数，新的一串到第 3 条再报。
    again = [short_alert(offset, 1.0, 1.02) for offset in (181, 211, 241)]
    assert [result is not None for result in observe_all(tracker, again)] == [False, False, True]


def test_gap_at_the_limit_still_chains_but_longer_gap_restarts():
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))

    # 恰好等于上限仍算连续：冷却 30 秒加上冷却后重新确认的几秒，间隔本来就不会是整 30 秒。
    assert observe_all(tracker, [short_alert(0, 1, 1.02), short_alert(60, 1, 1.02)]) == [None, None]
    assert tracker.observe(short_alert(120, 1, 1.02)) is not None

    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))
    results = observe_all(tracker, [short_alert(0, 1, 1.02), short_alert(60.001, 1, 1.02), short_alert(90, 1, 1.02)])
    assert results == [None, None, None]


def test_opposite_alert_restarts_the_count():
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))

    results = observe_all(
        tracker,
        [
            short_alert(0, 1, 1.02),
            short_alert(30, 1, 1.02),
            short_alert(60, 1.02, 1.0, "drop"),
            short_alert(90, 1, 1.02),
            short_alert(120, 1, 1.02),
        ],
    )

    assert results == [None] * 5
    assert tracker.observe(short_alert(150, 1, 1.02)) is not None


def test_streak_is_held_back_while_the_whole_run_has_not_moved_in_the_alert_direction():
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))

    # 每条急涨提醒都只看自己的 30 秒，冷却期里急跌又拉回没有触发急跌提醒：三条急涨下来整段反而下跌。
    results = observe_all(
        tracker,
        [short_alert(0, 1.00, 1.02), short_alert(35, 0.96, 0.98), short_alert(70, 0.95, 0.97)],
    )
    assert results == [None, None, None]

    # 计数照常累加，后续提醒让整段确实上涨后再报，条数如实反映这一串的长度。
    streak = tracker.observe(short_alert(100, 0.99, 1.01))
    assert streak is not None
    assert streak.alert_count == 4
    assert streak.change_percent == pytest.approx(1.0)
    # 已经报过，同一串不再重复。
    assert tracker.observe(short_alert(130, 1.01, 1.03)) is None


def test_symbols_are_counted_independently():
    tracker = StreakTracker(StreakRule(min_alerts=2, max_gap_seconds=60))

    assert tracker.observe(short_alert(0, 1, 1.02)) is None
    assert tracker.observe(short_alert(30, 1, 1.02, symbol="QNT_USDT")) is None
    streak = tracker.observe(short_alert(31, 1, 1.03, symbol="QNT_USDT"))
    assert streak is not None and streak.symbol == "QNT_USDT"


def test_rejects_rules_that_could_never_mean_a_streak():
    with pytest.raises(ValueError, match="min_alerts"):
        StreakRule(min_alerts=1, max_gap_seconds=60)
    with pytest.raises(ValueError, match="max_gap_seconds"):
        StreakRule(min_alerts=3, max_gap_seconds=0)


def make_streak(direction: str = "surge") -> StreakAlert:
    tracker = StreakTracker(StreakRule(min_alerts=3, max_gap_seconds=60))
    if direction == "surge":
        prices = [(0.8745, 0.8950), (0.8950, 0.9150), (0.9150, 0.9386)]
    else:
        prices = [(0.9386, 0.9150), (0.9150, 0.8950), (0.8950, 0.8745)]
    results = observe_all(
        tracker, [short_alert(offset, *pair, direction) for offset, pair in zip((0, 31.5, 61.5), prices, strict=True)]
    )
    assert results[2] is not None
    return results[2]


def test_streak_text_and_structured_fields():
    surge = make_streak()
    assert format_alert(surge) == (
        "[连续急涨] 2026-10-09 13:55:13 | RLC_USDT | 30秒窗口连续 3 次急涨，近1分32秒累计上涨 7.33% | 0.8745 → 0.9386"
    )
    drop = make_streak("drop")
    assert "[连续急跌]" in format_alert(drop) and "累计下跌 6.83%" in format_alert(drop)

    payload = surge.to_dict()
    assert payload["kind"] == "streak"
    assert payload["alert_count"] == 3
    assert payload["started_at"] == "2026-10-09T05:53:42+00:00"
    assert payload["color"] == "green"
    assert drop.to_dict()["color"] == "red"


def test_duration_reads_in_minutes_and_seconds():
    assert describe_duration(45) == "45秒"
    assert describe_duration(120) == "2分钟"
    assert describe_duration(147) == "2分27秒"


def test_streak_label_has_its_own_color():
    alert = make_streak()
    text = colorize_alert(alert, format_alert(alert))

    assert text.startswith(f"{SURGE_COLOR}{STREAK_LABEL_COLOR}[连续急涨]{Style.RESET_ALL}{SURGE_COLOR}")
    assert f"{SYMBOL_COLOR}RLC_USDT" in text
    # 要和前面那条普通短窗口提醒、其他类型提醒的标签一眼区分开。
    others = {SURGE_COLOR, SYMBOL_COLOR, LONG_WINDOW_COLOR, TREND_LABEL_COLOR, LAUNCH_LABEL_COLOR}
    assert STREAK_LABEL_COLOR not in others
