from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from price_alert.models import Candle, PriceTick
from price_alert.trend.detector import EpisodeSettings, TrendDetector, TrendOutcome
from price_alert.trend.pattern import TrendRule

# 整点起步，1m/3m/5m/15m 的周期边界都对齐。
BASE = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "ARK_USDT"


def rule(period="1m", **overrides) -> TrendRule:
    values = {
        "period": period,
        "candles": 5,
        "min_change_percent": 4.0,
        "min_change_atr": None,
        "max_rebound_ratio": 0.3,
        "min_trend_candle_ratio": 0.7,
        "min_step_ratio": 0.7,
        "min_body_ratio": 0.5,
        "min_volume_ratio": None,
        "baseline_candles": 5,
    }
    return TrendRule(**(values | overrides))


SETTINGS = EpisodeSettings(
    cooldown_minutes=30,
    escalation_step_percent=5.0,
    end_rebound_ratio=0.5,
    stall_minutes=30,
    notify_end=True,
    exhaustion_enabled=False,
    exhaustion_shadow_ratio=0.6,
    exhaustion_candles=1,
    exhaustion_periods=frozenset({"1m"}),
)


def minute(index: int) -> datetime:
    return BASE + timedelta(minutes=index)


def bar(index: int, open_price: float, close: float, low: float | None = None, volume: float = 100.0) -> Candle:
    low_price = min(open_price, close) if low is None else low
    return Candle(minute(index), open_price, max(open_price, close), low_price, close, volume=volume)


def flat(count: int, start: int = 0, price: float = 100.0) -> list[Candle]:
    return [Candle(minute(start + i), price, price + 0.1, price - 0.1, price, volume=100.0) for i in range(count)]


def make(*rules: TrendRule, settings: EpisodeSettings = SETTINGS, seed: int = 10, observer=None) -> TrendDetector:
    detector = TrendDetector(list(rules) or [rule()], settings, observer)
    detector.add_symbol(SYMBOL, flat(seed), 1e9, price_decimals=4)
    return detector


def fall(detector: TrendDetector, start: int, count: int, step: float, price: float = 100.0):
    """从 start 分钟起逐分钟等幅下跌，返回每根 K 线收盘时产生的提醒。"""
    alerts = []
    for i in range(count):
        open_price = price - step * i
        alerts.append(detector.add_candle(SYMBOL, bar(start + i, open_price, open_price - step)))
    return alerts


def test_start_alert_after_window_is_fully_trending():
    detector = make()

    # 每分钟跌 0.9%：前 4 根（含一根平线）累计 3.6% 不够，第 5 根达到 4.5%。
    alerts = fall(detector, 10, 5, 0.9)

    assert [len(item) for item in alerts] == [0, 0, 0, 0, 1]
    alert = alerts[-1][0]
    assert (alert.stage, alert.direction, alert.period, alert.candles) == ("start", "drop", "1m", 5)
    assert alert.timestamp == minute(15)
    assert alert.started_at == minute(10)
    assert alert.anchor_price == 100.0
    assert alert.price == pytest.approx(95.5)
    assert alert.change_percent == pytest.approx(-4.5)
    assert alert.price_decimals == 4


def test_extend_alert_per_escalation_level_and_skips_crossed_levels():
    detector = make()
    fall(detector, 10, 5, 0.9)

    first = detector.add_candle(SYMBOL, bar(15, 95.5, 94.5))
    # 一根 K 线跨过 10% 与 15% 两档，只报一次，下一档从 20% 开始。
    second = detector.add_candle(SYMBOL, bar(16, 94.5, 84.0))
    third = detector.add_candle(SYMBOL, bar(17, 84.0, 81.0))

    assert [(a.stage, round(a.change_percent, 1)) for a in first] == [("extend", -5.5)]
    assert [(a.stage, round(a.change_percent, 1)) for a in second] == [("extend", -16.0)]
    assert third == []


def test_end_on_rebound_reports_extreme_and_allows_new_trend_after_cooldown():
    detector = make(settings=replace(SETTINGS, cooldown_minutes=0))
    fall(detector, 10, 5, 0.9)

    # 最低 95.5，从 100 起跌了 4.5；反弹一半（2.25）即结束。
    assert detector.add_candle(SYMBOL, bar(15, 95.5, 97.0)) == []
    ended = detector.add_candle(SYMBOL, bar(16, 97.0, 97.8))

    assert [(a.stage, a.end_reason, a.extreme_price) for a in ended] == [("end", "rebound", 95.5)]
    # 冷却为 0 时，新的一段下跌可以重新开始。
    restarted = [a for item in fall(detector, 17, 5, 0.9, price=97.8) for a in item]
    assert [a.stage for a in restarted] == ["start"]


def test_end_on_stall_without_new_extreme():
    detector = make(settings=replace(SETTINGS, stall_minutes=3))
    fall(detector, 10, 5, 0.9)

    alerts = [detector.add_candle(SYMBOL, bar(15 + i, 95.6, 95.6)) for i in range(3)]

    assert [len(item) for item in alerts] == [0, 0, 1]
    assert (alerts[-1][0].stage, alerts[-1][0].end_reason) == ("end", "stall")


def test_notify_end_false_ends_silently():
    detector = make(settings=replace(SETTINGS, stall_minutes=1, notify_end=False))
    fall(detector, 10, 5, 0.9)

    assert detector.add_candle(SYMBOL, bar(15, 95.6, 95.6)) == []
    # 趋势已结束，同方向再次成立时受冷却约束而不提醒，说明状态确实被清掉而不是仍在跟踪。
    evaluations = []
    detector._observer = evaluations.append
    fall(detector, 16, 5, 0.9, price=95.6)
    assert TrendOutcome.COOLDOWN in {item.outcome for item in evaluations}


def test_cooldown_blocks_restart_in_same_direction():
    evaluations = []
    detector = make(settings=replace(SETTINGS, stall_minutes=1), observer=evaluations.append)
    fall(detector, 10, 5, 0.9)
    detector.add_candle(SYMBOL, bar(15, 95.6, 95.6))

    again = [a for item in fall(detector, 16, 5, 0.9, price=95.6) for a in item]

    assert again == []
    assert evaluations[-1].outcome is TrendOutcome.COOLDOWN


def test_opposite_trend_ends_current_one_and_starts_new():
    detector = make()
    # 跌到 90（跌幅 10），随后反弹 4.5：未到结束所需的一半回撤，但反向趋势已经成立。
    fall(detector, 10, 10, 1.0)

    alerts = []
    for i in range(5):
        open_price = 90 + 0.9 * i
        alerts.extend(detector.add_candle(SYMBOL, bar(20 + i, open_price, open_price + 0.9)))

    assert [(a.stage, a.direction, a.end_reason) for a in alerts] == [
        ("end", "drop", "reversed"),
        ("start", "surge", None),
    ]


def test_same_minute_passes_are_arbitrated_shorter_period_first():
    evaluations = []
    detector = make(
        rule("1m", candles=6, min_change_percent=6.0),
        rule("3m", candles=2, min_change_percent=4.0),
        seed=12,
        observer=evaluations.append,
    )

    alerts = [a for item in fall(detector, 12, 6, 1.0) for a in item]

    # 两个周期在 00:18 同时成立：只发短周期，长周期记为被取代。
    assert [(a.stage, a.period) for a in alerts] == [("start", "1m")]
    at_boundary = {item.period: item.outcome for item in evaluations if item.timestamp == minute(18)}
    assert at_boundary == {"1m": TrendOutcome.ALERT, "3m": TrendOutcome.SUPERSEDED}


def test_longer_period_during_active_trend_does_not_alert_again():
    evaluations = []
    detector = make(
        rule("1m", candles=5, min_change_percent=4.0),
        rule("3m", candles=2, min_change_percent=4.0),
        seed=12,
        observer=evaluations.append,
    )

    alerts = [a for item in fall(detector, 12, 6, 1.0) for a in item if a.stage == "start"]

    assert [(a.stage, a.period) for a in alerts] == [("start", "1m")]
    three_minute = [item for item in evaluations if item.period == "3m"]
    # 3 分钟周期只在 3 的整数倍分钟收盘时判定。
    assert all(int(item.timestamp.timestamp()) % 180 == 0 for item in three_minute)
    assert three_minute[-1].outcome is TrendOutcome.EPISODE_ACTIVE


def test_insufficient_history_is_reported():
    evaluations = []
    detector = TrendDetector([rule(candles=5)], SETTINGS, evaluations.append)
    detector.add_symbol(SYMBOL, flat(2), 1e9)

    detector.add_candle(SYMBOL, bar(2, 100, 99))

    assert evaluations[-1].outcome is TrendOutcome.INSUFFICIENT_DATA


def test_exhaustion_hint_needs_long_shadow_at_new_extreme_and_rearms_after_extend():
    settings = replace(SETTINGS, exhaustion_enabled=True)
    detector = make(settings=settings)
    fall(detector, 10, 5, 0.9)

    # 探到新低后收回大半：下影线占振幅 88%。
    hint = detector.add_candle(SYMBOL, bar(15, 95.5, 95.2, low=93.0))
    repeated = detector.add_candle(SYMBOL, bar(16, 95.2, 95.1, low=93.0))
    extended = detector.add_candle(SYMBOL, bar(17, 95.1, 94.8))
    rearmed = detector.add_candle(SYMBOL, bar(18, 94.8, 94.7, low=93.0))

    assert [(a.stage, a.period, round(a.shadow_ratio, 2)) for a in hint] == [("exhaustion", "1m", 0.88)]
    assert repeated == []
    assert [a.stage for a in extended] == ["extend"]
    assert [a.stage for a in rearmed] == ["exhaustion"]


def test_exhaustion_ignores_shadow_away_from_extreme_and_unconfigured_periods():
    detector = make(settings=replace(SETTINGS, exhaustion_enabled=True))
    fall(detector, 10, 5, 0.9)
    # 长下影线但最低价没到趋势极值附近：只是途中的十字星。
    assert detector.add_candle(SYMBOL, bar(15, 97.0, 96.9, low=96.0)) == []

    other = make(settings=replace(SETTINGS, exhaustion_enabled=True, exhaustion_periods=frozenset({"5m"})))
    fall(other, 10, 5, 0.9)
    assert other.add_candle(SYMBOL, bar(15, 95.5, 95.2, low=93.0)) == []


def tick(second: float, price: float, size: float = 1.0) -> PriceTick:
    return PriceTick(SYMBOL, price, size, BASE + timedelta(seconds=second))


def test_ticks_build_minute_candles_and_fill_quiet_minutes():
    evaluations = []
    detector = TrendDetector([rule()], SETTINGS, evaluations.append)
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    start = 600

    alerts = []
    for i in range(5):
        # 每分钟两笔成交：开盘价与收盘价。
        price = 100 - 0.9 * i
        alerts += detector.add_tick(tick(start + 60 * i + 1, price, 2.0))
        alerts += detector.add_tick(tick(start + 60 * i + 50, price - 0.9, 3.0))
    # 空过三分钟后才有成交：最后一根趋势 K 线与中间的平线都要结算。
    alerts += detector.add_tick(tick(start + 60 * 8 + 5, 95.5))

    assert [a.stage for a in alerts] == ["start"]
    assert alerts[0].timestamp == minute(15)
    closed = [item.timestamp for item in evaluations]
    assert closed[-4:] == [minute(15), minute(16), minute(17), minute(18)]
    last_trend = detector._states[SYMBOL].minutes[-4]
    assert (last_trend.open, last_trend.close, last_trend.volume) == (pytest.approx(96.4), pytest.approx(95.5), 5.0)
    assert detector._states[SYMBOL].minutes[-1].volume == 0


def test_out_of_order_and_pre_live_ticks_are_ignored():
    detector = TrendDetector([rule()], SETTINGS)
    detector.add_symbol(SYMBOL, flat(10), 1e9, live_candle=Candle(minute(10), 100, 100, 100, 100))

    # 早于 REST 当前 K 线的成交已包含在其中。
    assert detector.add_tick(tick(9 * 60 + 30, 50.0)) == []
    detector.add_tick(tick(10 * 60 + 30, 99.0))
    detector.add_tick(tick(10 * 60 + 10, 10.0))

    live = detector._states[SYMBOL].live
    assert live is not None and (live.low, live.close) == (99.0, 99.0)


def test_stream_gap_pauses_until_resync_and_merges_live_minute():
    evaluations = []
    detector = TrendDetector([rule()], SETTINGS, evaluations.append)
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    detector.add_tick(tick(600 + 5, 100.0))

    detector.mark_stream_gap()
    assert detector.stale_symbols == [SYMBOL]
    # 失效期间跨分钟不结算、不判定。
    detector.add_tick(tick(12 * 60 + 5, 101.0))
    assert evaluations == []

    fetched_at = BASE + timedelta(minutes=12, seconds=10)
    rest_candles = flat(12)
    rest_live = Candle(minute(12), 100.5, 103.0, 99.0, 102.0, volume=50.0)
    assert detector.resync_symbol(SYMBOL, rest_candles, rest_live, fetched_at)
    assert detector.resync_symbol(SYMBOL, rest_candles, rest_live, fetched_at) is False
    assert detector.stale_symbols == []

    state = detector._states[SYMBOL]
    assert state.minutes[-1].timestamp == minute(11)
    assert state.live is not None
    # 本地只有一笔成交，高低点取两边并集；本地成交早于请求，收盘价用交易所的。
    assert (state.live.open, state.live.high, state.live.low, state.live.close) == (100.5, 103.0, 99.0, 102.0)
    detector.add_tick(tick(13 * 60 + 1, 102.0))
    assert evaluations[-1].timestamp == minute(13)


def test_resync_counts_exchange_live_candle_as_closed_when_local_moved_on():
    detector = TrendDetector([rule()], SETTINGS)
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    detector.mark_stream_gap()
    detector.add_tick(tick(13 * 60 + 5, 100.0))

    detector.resync_symbol(SYMBOL, flat(12), Candle(minute(12), 100, 101, 99, 100.5), BASE + timedelta(minutes=12))

    state = detector._states[SYMBOL]
    assert state.minutes[-1].timestamp == minute(12)
    assert state.live is not None and state.live.timestamp == minute(13)


def test_resync_keeps_active_trend_and_unknown_symbols_are_ignored():
    detector = make()
    fall(detector, 10, 5, 0.9)
    detector.mark_stream_gap()

    detector.resync_symbol(SYMBOL, flat(15), None, BASE + timedelta(minutes=15))

    assert detector._states[SYMBOL].episode is not None
    assert detector.resync_symbol("NOPE_USDT", [], None, BASE) is False
    assert detector.add_tick(PriceTick("NOPE_USDT", 1.0, 1.0, BASE)) == []
    assert detector.add_candle("NOPE_USDT", bar(0, 1, 1)) == []


def test_removed_symbol_keeps_cooldown_when_it_returns():
    detector = make()
    fall(detector, 10, 5, 0.9)
    detector.remove_symbols({SYMBOL})
    assert detector.symbols == []

    detector.add_symbol(SYMBOL, flat(10, start=15, price=95.5), 1e9)
    again = [a for item in fall(detector, 25, 5, 0.9, price=95.5) for a in item]

    assert again == []


def test_add_symbol_again_only_updates_metadata():
    detector = make()
    detector.add_symbol(SYMBOL, [], 5e9, price_decimals=2)

    state = detector._states[SYMBOL]
    assert (state.volume_24h_quote, state.price_decimals, len(state.minutes)) == (5e9, 2, 10)


def test_warmup_covers_longest_period_and_constructor_validation():
    detector = TrendDetector([rule("1m"), rule("15m", candles=6, baseline_candles=16)], SETTINGS)

    assert detector.warmup_candles == (6 + 16) * 15 + 1
    assert [item.period for item in detector.rules] == ["1m", "15m"]
    with pytest.raises(ValueError):
        TrendDetector([], SETTINGS)
    with pytest.raises(ValueError):
        TrendDetector([rule("1m"), rule("1m")], SETTINGS)
