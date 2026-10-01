from datetime import UTC, datetime, timedelta

import pytest

from price_alert.models import Candle, PriceTick
from price_alert.trend.detector import TrendDetector, TrendOutcome
from price_alert.trend.pattern import TrendRule

# 整点起步，1m/3m/5m/15m 的周期边界都对齐。
BASE = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "ARK_USDT"


def rule(period="1m", **overrides) -> TrendRule:
    values = {
        "period": period,
        "candles": 5,
        "min_change_percent": 4.0,
        "max_counter_candles": 1,
        "max_rebound_ratio": 0.3,
        "min_body_ratio": 0.5,
        # 默认不限制单根占比，各用例只验证自己关心的条件；单根占比由专门的用例覆盖。
        "max_single_candle_ratio": 1.0,
    }
    return TrendRule(**(values | overrides))


def minute(index: int) -> datetime:
    return BASE + timedelta(minutes=index)


def bar(index: int, open_price: float, close: float) -> Candle:
    return Candle(minute(index), open_price, max(open_price, close), min(open_price, close), close)


def flat(count: int, start: int = 0, price: float = 100.0) -> list[Candle]:
    return [Candle(minute(start + i), price, price + 0.1, price - 0.1, price) for i in range(count)]


def make(*rules: TrendRule, seed: int = 10, observer=None) -> TrendDetector:
    detector = TrendDetector(list(rules) or [rule()], observer)
    detector.add_symbol(SYMBOL, flat(seed), 1e9, price_decimals=4)
    return detector


def fall(detector: TrendDetector, start: int, count: int, step: float, price: float = 100.0):
    """从 start 分钟起逐分钟等幅下跌，返回每根 K 线收盘时产生的提醒。"""
    alerts = []
    for i in range(count):
        open_price = price - step * i
        alerts.append(detector.add_candle(SYMBOL, bar(start + i, open_price, open_price - step)))
    return alerts


def test_alert_when_window_forms_a_healthy_trend():
    detector = make()

    # 每分钟跌 0.9%：前 4 根（含一根平线）累计 3.6% 不够，第 5 根达到 4.5%。
    alerts = fall(detector, 10, 5, 0.9)

    assert [len(item) for item in alerts] == [0, 0, 0, 0, 1]
    alert = alerts[-1][0]
    assert (alert.direction, alert.period, alert.candles) == ("drop", "1m", 5)
    assert (alert.timestamp, alert.started_at) == (minute(15), minute(10))
    assert alert.metrics.change_percent == pytest.approx(-4.5)
    assert (alert.volume_24h_quote, alert.price_decimals) == (1e9, 4)


def test_no_repeat_while_pattern_keeps_holding():
    evaluations = []
    detector = make(observer=evaluations.append)

    alerts = [a for item in fall(detector, 10, 15, 0.9) for a in item]

    assert len(alerts) == 1
    assert evaluations[-1].outcome is TrendOutcome.CONTINUING


def test_alerts_again_after_pattern_breaks_and_opposite_direction_is_independent():
    detector = make()
    first = [a for item in fall(detector, 10, 5, 0.9) for a in item]

    # 反向的上涨形成：下跌形态随之被破坏，上涨作为新的一段提醒。
    rising = []
    for i in range(5):
        open_price = 95.5 + 1.0 * i
        rising.extend(detector.add_candle(SYMBOL, bar(15 + i, open_price, open_price + 1.0)))
    # 再次下跌：上一段下跌的形态早已被破坏，重新计算后再次提醒。
    again = [a for item in fall(detector, 20, 5, 1.0, price=100.5) for a in item]

    assert [a.direction for a in first + rising + again] == ["drop", "surge", "drop"]


def test_period_alerts_again_after_its_own_pattern_breaks():
    detector = make(rule("1m", candles=3, min_change_percent=2.0, max_counter_candles=0))

    first = [a for item in fall(detector, 10, 5, 0.9) for a in item]
    # 一根阳线破坏了 1 分钟形态，之后重新连跌三根：作为新的一段再次提醒。
    detector.add_candle(SYMBOL, bar(15, 95.5, 96.0))
    second = [a for item in fall(detector, 16, 3, 0.9, price=96.0) for a in item]

    assert len(first) == 1 and len(second) == 1
    assert second[0].started_at == minute(16)


def test_periods_are_independent_and_can_alert_in_the_same_minute():
    detector = make(
        rule("1m", candles=6, min_change_percent=6.0),
        rule("3m", candles=2, min_change_percent=4.0, max_counter_candles=0),
        seed=12,
    )

    alerts = [a for item in fall(detector, 12, 6, 1.0) for a in item]

    # 两个周期在 00:18 同时新形成：各报一条，短周期在前。
    assert [(a.period, a.timestamp) for a in alerts] == [("1m", minute(18)), ("3m", minute(18))]


def test_longer_period_alerts_even_while_shorter_period_is_holding():
    evaluations = []
    detector = make(
        rule("1m", candles=5, min_change_percent=4.0),
        rule("3m", candles=2, min_change_percent=4.0, max_counter_candles=0),
        seed=12,
        observer=evaluations.append,
    )

    alerts = [a for item in fall(detector, 12, 9, 1.0) for a in item]

    assert [a.period for a in alerts] == ["1m", "3m"]
    three_minute = [item for item in evaluations if item.period == "3m"]
    # 3 分钟周期只在 3 的整数倍分钟收盘时判定，成立后再判定为延续。
    assert all(int(item.timestamp.timestamp()) % 180 == 0 for item in three_minute)
    assert three_minute[-1].outcome is TrendOutcome.CONTINUING


def test_failed_and_insufficient_evaluations_are_reported():
    evaluations = []
    detector = TrendDetector([rule(candles=5)], evaluations.append)
    detector.add_symbol(SYMBOL, flat(2), 1e9)

    detector.add_candle(SYMBOL, bar(2, 100, 99))
    assert evaluations[-1].outcome is TrendOutcome.INSUFFICIENT_DATA

    detector.add_candle(SYMBOL, bar(3, 99, 99.5))
    detector.add_candle(SYMBOL, bar(4, 99.5, 99.2))
    assert evaluations[-1].outcome is TrendOutcome.FAILED
    assert evaluations[-1].metrics is not None


def tick(second: float, price: float) -> PriceTick:
    return PriceTick(SYMBOL, price, 1.0, BASE + timedelta(seconds=second))


def test_ticks_build_minute_candles_and_fill_quiet_minutes():
    evaluations = []
    detector = TrendDetector([rule()], evaluations.append)
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    start = 600

    alerts = []
    for i in range(5):
        # 每分钟两笔成交：开盘价与收盘价。
        price = 100 - 0.9 * i
        alerts += detector.add_tick(tick(start + 60 * i + 1, price))
        alerts += detector.add_tick(tick(start + 60 * i + 50, price - 0.9))
    # 空过三分钟后才有成交：最后一根趋势 K 线与中间的平线都要结算。
    alerts += detector.add_tick(tick(start + 60 * 8 + 5, 95.5))

    assert len(alerts) == 1 and alerts[0].timestamp == minute(15)
    assert [item.timestamp for item in evaluations][-4:] == [minute(15), minute(16), minute(17), minute(18)]
    last_trend = detector._states[SYMBOL].minutes[-4]
    assert (last_trend.open, last_trend.close) == (pytest.approx(96.4), pytest.approx(95.5))


def test_out_of_order_and_pre_live_ticks_are_ignored():
    detector = TrendDetector([rule()])
    detector.add_symbol(SYMBOL, flat(10), 1e9, live_candle=Candle(minute(10), 100, 100, 100, 100))

    # 早于 REST 当前 K 线的成交已包含在其中。
    assert detector.add_tick(tick(9 * 60 + 30, 50.0)) == []
    detector.add_tick(tick(10 * 60 + 30, 99.0))
    detector.add_tick(tick(10 * 60 + 10, 10.0))

    live = detector._states[SYMBOL].live
    assert live is not None and (live.low, live.close) == (99.0, 99.0)


def test_stream_gap_pauses_until_resync_and_merges_live_minute():
    evaluations = []
    detector = TrendDetector([rule()], evaluations.append)
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    detector.add_tick(tick(600 + 5, 100.0))

    detector.mark_stream_gap()
    assert detector.stale_symbols == [SYMBOL]
    # 失效期间跨分钟不结算、不判定。
    detector.add_tick(tick(12 * 60 + 5, 101.0))
    assert evaluations == []

    fetched_at = BASE + timedelta(minutes=12, seconds=10)
    rest_live = Candle(minute(12), 100.5, 103.0, 99.0, 102.0)
    assert detector.resync_symbol(SYMBOL, flat(12), rest_live, fetched_at)
    assert detector.resync_symbol(SYMBOL, flat(12), rest_live, fetched_at) is False
    assert detector.stale_symbols == []

    state = detector._states[SYMBOL]
    assert state.minutes[-1].timestamp == minute(11)
    assert state.live is not None
    # 本地只有一笔成交，高低点取两边并集；本地成交早于请求，收盘价用交易所的。
    assert (state.live.open, state.live.high, state.live.low, state.live.close) == (100.5, 103.0, 99.0, 102.0)
    detector.add_tick(tick(13 * 60 + 1, 102.0))
    assert evaluations[-1].timestamp == minute(13)


def test_resync_counts_exchange_live_candle_as_closed_when_local_moved_on():
    detector = TrendDetector([rule()])
    detector.add_symbol(SYMBOL, flat(10), 1e9)
    detector.mark_stream_gap()
    detector.add_tick(tick(13 * 60 + 5, 100.0))

    detector.resync_symbol(SYMBOL, flat(12), Candle(minute(12), 100, 101, 99, 100.5), BASE + timedelta(minutes=12))

    state = detector._states[SYMBOL]
    assert state.minutes[-1].timestamp == minute(12)
    assert state.live is not None and state.live.timestamp == minute(13)


def test_unknown_symbols_are_ignored():
    detector = make()

    assert detector.resync_symbol("NOPE_USDT", [], None, BASE) is False
    assert detector.add_tick(PriceTick("NOPE_USDT", 1.0, 1.0, BASE)) == []
    assert detector.add_candle("NOPE_USDT", bar(0, 1, 1)) == []


def test_removed_symbol_keeps_trend_state_when_it_returns():
    detector = make()
    fall(detector, 10, 5, 0.9)
    detector.remove_symbols({SYMBOL})
    assert detector.symbols == []

    # 很快重新入池，预热数据里走势仍在延续：同一段走势不再提醒。
    history = flat(10) + [bar(10 + i, 100 - 0.9 * i, 100 - 0.9 * (i + 1)) for i in range(5)]
    detector.add_symbol(SYMBOL, history, 1e9)
    again = [a for item in fall(detector, 15, 3, 0.9, price=95.5) for a in item]

    assert again == []


def test_add_symbol_again_only_updates_metadata():
    detector = make()
    detector.add_symbol(SYMBOL, [], 5e9, price_decimals=2)

    state = detector._states[SYMBOL]
    assert (state.volume_24h_quote, state.price_decimals, len(state.minutes)) == (5e9, 2, 5)


def test_warmup_covers_longest_period_and_constructor_validation():
    detector = TrendDetector([rule("1m"), rule("15m", candles=6)])

    assert detector.warmup_candles == 6 * 15 + 1
    assert [item.period for item in detector.rules] == ["1m", "15m"]
    with pytest.raises(ValueError):
        TrendDetector([])
    with pytest.raises(ValueError):
        TrendDetector([rule("1m"), rule("1m")])


def test_mark_symbol_gap_only_invalidates_that_symbol():
    detector = make()
    detector.add_symbol("OTHER_USDT", flat(10), 1e9)

    detector.mark_symbol_gap(SYMBOL.lower())
    detector.mark_symbol_gap("MISSING_USDT")

    assert detector.stale_symbols == [SYMBOL]
    assert detector.resync_symbol(SYMBOL, flat(10), None, minute(10))
    assert detector.stale_symbols == []
