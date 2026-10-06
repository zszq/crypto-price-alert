from datetime import UTC, datetime, timedelta

import pytest

from price_alert.launch.detector import LaunchDetector, LaunchOutcome
from price_alert.launch.rule import Condition, LaunchRule
from price_alert.models import Candle, PriceTick

BASE = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "RLC_USDT"


def rule(**overrides) -> LaunchRule:
    values = {
        "window_minutes": 3,
        "baseline_minutes": 10,
        "min_volume_ratio": 5.0,
        "min_window_quote": 3_000.0,
        "min_change_percent": 3.0,
        "cooldown_minutes": 5,
        "realert_step_percent": 5.0,
    }
    return LaunchRule(**(values | overrides))


def minute(index: int) -> datetime:
    return BASE + timedelta(minutes=index)


def bar(index: int, open_price: float, close: float, quote: float) -> Candle:
    return Candle(minute(index), open_price, max(open_price, close), min(open_price, close), close, quote)


def flat(count: int, start: int = 0, price: float = 100.0, quote: float = 100.0) -> list[Candle]:
    return [Candle(minute(start + i), price, price + 0.5, price - 0.5, price, quote) for i in range(count)]


def make(observer=None, seed: int = 10, **overrides) -> LaunchDetector:
    detector = LaunchDetector(rule(**overrides), observer)
    detector.add_symbol(SYMBOL, flat(seed), 1e5, price_decimals=4, quanto_multiplier=1.0)
    return detector


def rally(detector: LaunchDetector, start: int, closes: list[float], quote: float = 1000.0, price: float = 100.0):
    """从 start 分钟起逐分钟收在 closes，返回每根 K 线收盘时的提醒。"""
    alerts = []
    for offset, close in enumerate(closes):
        alerts.append(detector.add_candle(SYMBOL, bar(start + offset, price, close, quote)))
        price = close
    return alerts


def test_alerts_when_volume_bursts_and_price_breaks_out():
    detector = make()

    alerts = rally(detector, 10, [101.0, 102.5, 104.0])

    assert [len(item) for item in alerts] == [0, 0, 1]
    alert = alerts[-1][0]
    assert (alert.symbol, alert.direction, alert.wave) == (SYMBOL, "surge", 1)
    assert (alert.timestamp, alert.started_at) == (minute(13), minute(10))
    assert alert.metrics.change_percent == pytest.approx(4.0)
    assert (alert.volume_24h_quote, alert.price_decimals) == (1e5, 4)


def test_no_judgement_until_baseline_and_window_are_full():
    evaluations = []
    detector = LaunchDetector(rule(), evaluations.append)
    detector.add_symbol(SYMBOL, flat(5), 1e5, quanto_multiplier=1.0)

    assert rally(detector, 5, [101.0, 102.5, 104.0]) == [[], [], []]
    assert {item.outcome for item in evaluations} == {LaunchOutcome.INSUFFICIENT_DATA}


def test_cooldown_then_requires_price_to_extend():
    evaluations = []
    # 基准期放长到 30 分钟，整段都在提醒记忆的有效期内；不要求突破，便于构造回落后再放量。
    detector = make(evaluations.append, seed=30, baseline_minutes=30, require_breakout=False)
    first = rally(detector, 30, [101.0, 102.5, 104.0])
    assert [alert.wave for item in first for alert in item] == [1]

    # 冷却 5 分钟内继续放量拉升，条件满足也不再报。
    assert rally(detector, 33, [105.0], quote=5000.0, price=104.0) == [[]]
    assert evaluations[-1].outcome is LaunchOutcome.COOLDOWN

    # 回落后冷却已过，再次放量只拉回 104.5，没比上次提醒价 104 再推进 5%，原地放量不算新的一波。
    for index in range(34, 40):
        detector.add_candle(SYMBOL, Candle(minute(index), 100.0, 100.5, 99.5, 100.0, 100.0))
    assert not any(rally(detector, 40, [101.5, 103.0, 104.5], quote=5000.0))
    assert evaluations[-1].outcome is LaunchOutcome.NOT_EXTENDED

    # 继续放量推进到比 104 高 5% 以上，就是第 2 波。
    second = [alert for item in rally(detector, 43, [107.0, 109.5], quote=8000.0, price=104.5) for alert in item]
    assert [alert.wave for alert in second] == [2]
    assert second[0].label == "放量拉升·第2波"


def test_overshoot_is_silent_but_blocks_catch_up_alert_at_the_top():
    evaluations = []
    detector = make(evaluations.append, seed=30, baseline_minutes=30)
    # 3 分钟从 100 拉到 120（+20%），只因超过 15% 上限而不提醒。
    assert not any(rally(detector, 30, [105.0, 118.0, 120.0]))
    assert evaluations[-1].metrics.failures == (Condition.OVERSHOOT,)
    assert evaluations[-1].outcome is LaunchOutcome.OVERSHOOT

    # 窗口滑过插针后涨幅回到区间内（118 → 128，+8.5%）：插针已记入冷却，不在高位补报。
    assert not any(rally(detector, 33, [124.0, 128.0], price=120.0))
    assert evaluations[-1].outcome is LaunchOutcome.COOLDOWN


def test_after_overshoot_next_alert_must_extend_past_spike_close():
    evaluations = []
    detector = make(evaluations.append, seed=30, baseline_minutes=30, require_breakout=False)
    rally(detector, 30, [105.0, 118.0, 120.0])
    for index in range(33, 39):
        detector.add_candle(SYMBOL, Candle(minute(index), 122.0, 122.5, 121.5, 122.0, 100.0))

    # 冷却已过、再次放量：125.8 没比插针收盘价 120 再推进 5%，不算新的一波。
    assert not any(rally(detector, 39, [123.5, 125.0, 125.8], quote=2000.0, price=122.0))
    assert evaluations[-1].outcome is LaunchOutcome.NOT_EXTENDED

    # 推进到 128 才报；插针没有发出提醒，这仍是用户看到的第 1 波。
    alerts = rally(detector, 42, [128.0], quote=2000.0, price=125.8)
    assert [(alert.wave, alert.label) for item in alerts for alert in item] == [(1, "放量拉升")]


def test_overshoot_below_last_alert_does_not_lower_next_wave_bar():
    evaluations = []
    detector = make(evaluations.append, seed=30, baseline_minutes=30, require_breakout=False)
    assert [a.metrics.end_price for item in rally(detector, 30, [101.0, 102.5, 104.0]) for a in item] == [104.0]
    detector.add_candle(SYMBOL, bar(33, 104.0, 85.0, 100.0))
    for index in range(34, 40):
        detector.add_candle(SYMBOL, Candle(minute(index), 85.0, 85.5, 84.5, 85.0, 100.0))
    # 回落后插针 85 → 100（+17.6%），但没比上次提醒价 104 再推进 5%：真提醒也会被拦下，插针不能改写提醒记录。
    rally(detector, 40, [90.0, 97.0, 100.0], quote=2000.0, price=85.0)
    assert evaluations[-1].metrics.failures == (Condition.OVERSHOOT,)
    assert evaluations[-1].outcome is LaunchOutcome.FAILED
    for index in range(43, 48):
        detector.add_candle(SYMBOL, Candle(minute(index), 100.0, 100.5, 99.5, 100.0, 100.0))

    # 106 比插针价 100 高 6%，但比上次提醒价 104 只高 1.9%，仍不算新的一波。
    assert not any(rally(detector, 48, [102.0, 104.0, 106.0], quote=2000.0, price=100.0))
    assert evaluations[-1].outcome is LaunchOutcome.NOT_EXTENDED


def test_overshoot_alerts_directly_when_cap_disabled():
    alerts = rally(make(seed=30, baseline_minutes=30, max_change_percent=None), 30, [105.0, 118.0, 120.0])

    assert [alert.metrics.change_percent for item in alerts for alert in item] == [pytest.approx(20.0)]


def test_alert_memory_expires_after_one_baseline():
    detector = make()
    rally(detector, 10, [101.0, 102.5, 104.0])
    for index in range(13, 30):
        detector.add_candle(SYMBOL, Candle(minute(index), 104.0, 104.5, 103.5, 104.0, 100.0))

    # 上次提醒已超过一个基准期（10 分钟），再次放量突破就是新的启动，即使价格没比上次再推进 5%。
    alerts = rally(detector, 30, [105.0, 106.5, 108.0], price=104.0)

    assert [alert.wave for item in alerts for alert in item] == [1]


def test_cooldown_longer_than_baseline_is_still_honoured():
    evaluations = []
    detector = make(evaluations.append, cooldown_minutes=30)
    rally(detector, 10, [101.0, 102.5, 104.0])
    for index in range(13, 30):
        detector.add_candle(SYMBOL, Candle(minute(index), 104.0, 104.5, 103.5, 104.0, 100.0))

    # 已超过一个基准期（10 分钟），但冷却 30 分钟还没结束，再次放量突破也不提醒。
    assert not any(rally(detector, 30, [105.0, 106.5, 108.0], price=104.0))
    assert evaluations[-1].outcome is LaunchOutcome.COOLDOWN


def test_drop_alerts_only_when_enabled_and_tracked_separately():
    assert not any(rally(make(), 10, [99.0, 97.5, 96.0]))

    detector = make(alert_drops=True)
    drops = [alert for item in rally(detector, 10, [99.0, 97.5, 96.0]) for alert in item]
    assert [(alert.direction, alert.wave, alert.label) for alert in drops] == [("drop", 1, "放量下跌")]


def test_ticks_build_quote_volume_with_multiplier_and_settle_on_next_minute():
    detector = LaunchDetector(rule())
    detector.add_symbol(SYMBOL, flat(10), 1e5, quanto_multiplier=10.0)

    def tick(index: int, second: int, price: float, size: float) -> list:
        return detector.add_tick(PriceTick(SYMBOL, price, size, minute(index) + timedelta(seconds=second)))

    # 每分钟 1 张 × 面值 10 × 价格约 100 ≈ 1000 USDT；第三分钟的成交要等下一分钟第一笔到达才结算。
    assert tick(10, 5, 101.0, 1) == []
    assert tick(11, 5, 102.5, 1) == []
    assert tick(12, 5, 104.0, 1) == []
    alerts = tick(13, 1, 104.0, 1)

    assert len(alerts) == 1
    assert alerts[0].metrics.window_quote == pytest.approx(10 * (101.0 + 102.5 + 104.0))


def test_empty_minutes_are_filled_flat_and_judged():
    evaluations = []
    detector = make(evaluations.append)
    detector.add_tick(PriceTick(SYMBOL, 104.0, 30, minute(10) + timedelta(seconds=1)))

    # 10 分之后无人成交，13 分的成交把 10、11、12 三分钟一起结算，空分钟按平线补齐、每分钟都判定。
    # 10 分一笔 30 张即 3120 USDT，涨 4% 突破。
    alerts = detector.add_tick(PriceTick(SYMBOL, 104.0, 1, minute(13) + timedelta(seconds=1)))

    assert [item.timestamp for item in evaluations] == [minute(11), minute(12), minute(13)]
    # 12 分收盘时基准加窗口才凑满，此前不判定；补齐的平线分钟判定时看不到之后的 K 线。
    assert [item.outcome for item in evaluations] == [
        LaunchOutcome.INSUFFICIENT_DATA,
        LaunchOutcome.INSUFFICIENT_DATA,
        LaunchOutcome.ALERT,
    ]
    assert [alert.timestamp for alert in alerts] == [minute(13)]


@pytest.mark.parametrize(("later_price", "expected"), [(103.9, LaunchOutcome.ALERT), (102.0, LaunchOutcome.EXPIRED)])
def test_minute_settled_after_a_gap_is_rechecked_with_the_newer_price(later_price, expected):
    evaluations = []
    detector = make(evaluations.append)
    for index, price in ((10, 101.0), (11, 102.5), (12, 104.0)):
        detector.add_tick(PriceTick(SYMBOL, price, 1000 / price, minute(index) + timedelta(seconds=1)))

    # 12 分之后沉寂到 20 分才有成交：价格仍在 103.9（相对起点涨 3.9% 且仍高于基准高点）照常提醒；
    # 已回落到 102（只涨 2%），12 分的收盘价已过期，不再带着旧价格提醒。
    alerts = detector.add_tick(PriceTick(SYMBOL, later_price, 1, minute(20)))

    judged = [item for item in evaluations if item.timestamp == minute(13)]
    assert [item.outcome for item in judged] == [expected]
    assert len(alerts) == (1 if expected is LaunchOutcome.ALERT else 0)


def test_next_minute_settlement_is_not_rechecked():
    detector = make()
    for index, price in ((10, 101.0), (11, 102.5), (12, 104.0)):
        detector.add_tick(PriceTick(SYMBOL, price, 1000 / price, minute(index) + timedelta(seconds=1)))

    # 紧邻的下一分钟只晚几秒，新一分钟的单笔离群成交不能否掉本该发出的提醒。
    assert len(detector.add_tick(PriceTick(SYMBOL, 95.0, 1, minute(13)))) == 1


def test_symbol_without_multiplier_is_tracked_but_never_judged():
    evaluations = []
    detector = LaunchDetector(rule(), evaluations.append)
    detector.add_symbol(SYMBOL, flat(10), 1e5)

    for index in range(10, 15):
        detector.add_tick(PriceTick(SYMBOL, 110.0, 1000, minute(index)))

    assert detector.symbols == [SYMBOL]
    assert evaluations == []


def test_out_of_order_ticks_are_ignored():
    detector = make()
    detector.add_tick(PriceTick(SYMBOL, 101.0, 1000, minute(10) + timedelta(seconds=30)))
    detector.add_tick(PriceTick(SYMBOL, 150.0, 1000, minute(10) + timedelta(seconds=10)))

    detector.add_tick(PriceTick(SYMBOL, 101.0, 1, minute(11)))

    closed = detector._states[SYMBOL].minutes[-1]
    assert (closed.high, closed.close, closed.quote_volume) == (101.0, 101.0, 101_000.0)


def test_stream_gap_pauses_until_resync_and_merges_live_minute():
    evaluations = []
    detector = make(evaluations.append)
    detector.add_tick(PriceTick(SYMBOL, 101.0, 500, minute(10) + timedelta(seconds=5)))
    detector.mark_stream_gap()
    assert detector.stale_symbols == [SYMBOL]

    # 失效期间只维护当前分钟，不结算、不判定。
    detector.add_tick(PriceTick(SYMBOL, 102.0, 5, minute(11) + timedelta(seconds=5)))
    assert evaluations == []

    # 交易所的 11 分含断线期间的成交：高低点取并集，成交额取较大的一边。
    exchange_live = Candle(minute(11), 101.0, 103.0, 100.8, 102.5, 2000.0)
    history = flat(10) + [Candle(minute(10), 100.0, 101.5, 100.0, 101.0, 1500.0)]
    assert detector.resync_symbol(SYMBOL, history, exchange_live, minute(11) + timedelta(seconds=3))
    assert detector.stale_symbols == []
    assert not detector.resync_symbol(SYMBOL, history, exchange_live, minute(11))

    detector.add_tick(PriceTick(SYMBOL, 104.0, 1, minute(12)))
    closed = detector._states[SYMBOL].minutes[-1]
    assert (closed.timestamp, closed.high, closed.low, closed.close) == (minute(11), 103.0, 100.8, 102.0)
    assert closed.quote_volume == 2000.0
    assert evaluations[-1].timestamp == minute(12)


def test_resync_counts_exchange_live_candle_as_closed_when_local_moved_on():
    detector = make()
    detector.mark_stream_gap()
    detector.add_tick(PriceTick(SYMBOL, 101.0, 1, minute(12) + timedelta(seconds=1)))

    exchange_live = Candle(minute(11), 100.0, 101.0, 100.0, 101.0, 800.0)
    detector.resync_symbol(SYMBOL, flat(11), exchange_live, minute(11) + timedelta(seconds=59))

    state = detector._states[SYMBOL]
    assert state.minutes[-1] == exchange_live
    assert state.live is not None and state.live.timestamp == minute(12)


def test_removed_symbol_keeps_alert_memory_and_refresh_updates_metadata():
    detector = make()
    rally(detector, 10, [101.0, 102.5, 104.0])
    detector.remove_symbols({SYMBOL})
    assert detector.symbols == []

    # 很快重新入池时，同一波不再当作启动重报。
    detector.add_symbol(SYMBOL, flat(10, start=4) + [bar(14, 100.0, 104.0, 3000.0)], 2e5, quanto_multiplier=1.0)
    detector.add_symbol(SYMBOL, [], 3e5, price_decimals=2, quanto_multiplier=0.1)
    state = detector._states[SYMBOL]
    assert (state.volume_24h_quote, state.price_decimals, state.quanto_multiplier) == (3e5, 2, 0.1)
    alerts = rally(detector, 15, [104.5, 105.0], quote=5000.0, price=104.0)
    assert not any(alerts)


def test_warmup_covers_baseline_window_and_live_minute():
    assert LaunchDetector(rule()).warmup_candles == 14
