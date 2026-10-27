from datetime import UTC, datetime, timedelta

import pytest

from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.dormant.detector import DormantDetector, DormantOutcome
from price_alert.detectors.dormant.rule import (
    CALM_MINUTES,
    PRIOR_MINUTES,
    Condition,
    DormantGroup,
    DormantRule,
    minute_range_percent,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "IRYS_USDT"
WINDOW = 3
# 横盘段覆盖判定所需的全部历史，下标 SEED 起就是拉升。
SEED = PRIOR_MINUTES + WINDOW


def rule(**overrides) -> DormantRule:
    values = {
        "window_minutes": WINDOW,
        "max_prior_quote": 10_000.0,
        "min_window_quote": 1_000.0,
        "min_change_percent": 8.0,
        "min_active_minutes": 2,
        "cooldown_minutes": 30,
    }
    return DormantRule(**(values | overrides))


def minute(index: int) -> datetime:
    return BASE + timedelta(minutes=index)


def flat(count: int, start: int = 0, price: float = 1.0, quote: float = 5.0) -> list[Candle]:
    return [Candle(minute(start + i), price, price, price, price, quote) for i in range(count)]


def make(observer=None, seed: int = SEED, prior_quote: float = 5.0, **overrides) -> DormantDetector:
    detector = DormantDetector([rule(**overrides)], observer)
    detector.add_symbol(SYMBOL, flat(seed, quote=prior_quote), 2e5, price_decimals=4, quanto_multiplier=1.0)
    return detector


def rally(detector: DormantDetector, start: int, closes: list[float], quote: float = 500.0, price: float = 1.0):
    """从 start 分钟起逐分钟收在 closes，返回每根 K 线收盘时的提醒。"""
    alerts = []
    for offset, close in enumerate(closes):
        candle = Candle(minute(start + offset), price, max(price, close), min(price, close), close, quote)
        alerts.append(detector.add_candle(SYMBOL, candle))
        price = close
    return alerts


def test_alerts_when_dormant_coin_starts_trading_and_rallies():
    detector = make()

    alerts = rally(detector, SEED, [1.03, 1.06, 1.09])

    assert [len(item) for item in alerts] == [0, 0, 1]
    alert = alerts[-1][0]
    assert (alert.symbol, alert.direction, alert.label) == (SYMBOL, "surge", "冷门启动")
    assert (alert.timestamp, alert.started_at) == (minute(SEED + 3), minute(SEED))
    assert alert.metrics.change_percent == pytest.approx(9.0)
    assert (alert.metrics.window_quote, alert.metrics.active_minutes) == (1500, 3)
    assert (alert.volume_24h_quote, alert.price_decimals) == (2e5, 4)


def test_coin_that_was_already_trading_is_not_dormant():
    evaluations = []
    # 之前 24 小时成交额 10 × 1440 = 1.44 万 > 1 万。
    detector = make(evaluations.append, prior_quote=10.0)

    assert rally(detector, SEED, [1.03, 1.06, 1.09]) == [[], [], []]
    assert Condition.PRIOR in evaluations[-1].metrics.failures


def test_no_judgement_until_prior_day_and_window_are_full():
    evaluations = []
    detector = make(evaluations.append, seed=PRIOR_MINUTES)

    assert rally(detector, PRIOR_MINUTES, [1.03, 1.06]) == [[], []]
    assert {item.outcome for item in evaluations} == {DormantOutcome.INSUFFICIENT_DATA}


def test_cooldown_blocks_repeat_then_alerts_again():
    evaluations = []
    # 关掉补报，只看冷却本身。
    detector = make(evaluations.append, realert_step_percent=None)
    assert len(rally(detector, SEED, [1.03, 1.06, 1.09])[-1]) == 1
    # 冷却 30 分钟内继续上涨仍满足条件，只记为冷却中。
    assert rally(detector, SEED + 3, [1.12, 1.15]) == [[], []]
    assert evaluations[-1].outcome is DormantOutcome.COOLDOWN
    # 冷却过后横盘几分钟再拉起来，又报一次（之前 24 小时成交额仍在门槛内）。
    for candle in flat(28, start=SEED + 5, price=1.15):
        detector.add_candle(SYMBOL, candle)
    again = rally(detector, SEED + 33, [1.18, 1.22, 1.26], price=1.15)
    assert [len(item) for item in again] == [0, 0, 1]


def test_rally_that_keeps_going_is_reported_again_as_next_wave():
    evaluations = []
    # 之前 24 小时成交额压低一些，整段拉升都还算冷门。
    detector = make(evaluations.append, prior_quote=2.0)
    first = rally(detector, SEED, [1.03, 1.06, 1.09])[-1][0]
    assert (first.wave, first.label) == (1, "冷门启动")

    # 冷却期内每分钟都满足条件：比上次提醒价 1.09 只涨 2.75%（1.12）不报，涨 5.5%（1.15）报第 2 波；
    # 之后从 1.15 起算，1.18 只涨 2.6% 不报，1.22 报第 3 波。
    alerts = rally(detector, SEED + 3, [1.12, 1.15, 1.18, 1.22], price=1.09)
    assert [len(item) for item in alerts] == [0, 1, 0, 1]
    assert evaluations[3].outcome is DormantOutcome.COOLDOWN
    second, third = alerts[1][0], alerts[3][0]
    assert (second.wave, second.label, second.metrics.end_price) == (2, "冷门启动·第2波", 1.15)
    assert (third.wave, third.to_dict()["wave"]) == (3, 3)

    # 冷却从最近一次补报算起；过了冷却再拉起来就是新的一段，从第 1 波算。
    for candle in flat(60, start=SEED + 7, price=1.22, quote=2.0):
        detector.add_candle(SYMBOL, candle)
    again = rally(detector, SEED + 67, [1.26, 1.30, 1.34], price=1.22)
    assert again[-1][0].wave == 1


def test_wave_after_a_gap_is_rechecked_against_the_last_alert_price():
    evaluations = []
    detector = DormantDetector([rule(min_body_ratio=0.0)], evaluations.append)
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=1.0)
    for index, price in enumerate([1.03, 1.06, 1.09, 1.13]):
        detector.add_tick(PriceTick(SYMBOL, price, 500 / price, minute(SEED + index), f"t{index}"))
    # 1.13 那一分钟隔了空档才结算，这笔成交价 1.11 比上次提醒价 1.09 只高 1.8%，补报作废。
    detector.add_tick(PriceTick(SYMBOL, 1.11, 1.0, minute(SEED + 6), "late"))

    judged = [item for item in evaluations if item.timestamp == minute(SEED + 4)]
    assert judged[0].outcome is DormantOutcome.EXPIRED


def test_ticks_build_quote_volume_with_multiplier_and_settle_on_next_minute():
    # 每分钟的成交都是同一个价格，K 线没有实体；本用例只验证成交额聚合，实体条件关掉。
    detector = DormantDetector([rule(min_body_ratio=0.0)])
    detector.add_symbol(SYMBOL, flat(SEED - 1), 2e5, live_candle=flat(1, start=SEED - 1)[0], quanto_multiplier=10.0)
    alerts = []
    for index, price in enumerate([1.03, 1.06, 1.09]):
        # 2 张 × 面值 10 × 价格 ≈ 20 USDT 一笔，每分钟 20 笔。
        for second in range(20):
            stamp = minute(SEED + index) + timedelta(seconds=second)
            alerts += detector.add_tick(PriceTick(SYMBOL, price, 2.0, stamp, f"{index}-{second}"))
    assert alerts == []
    alerts += detector.add_tick(PriceTick(SYMBOL, 1.09, 1.0, minute(SEED + 3), "settle"))

    assert len(alerts) == 1
    assert alerts[0].metrics.window_quote == pytest.approx(20 * 20 * (1.03 + 1.06 + 1.09))


@pytest.mark.parametrize(("later_price", "expected"), [(1.085, DormantOutcome.ALERT), (1.05, DormantOutcome.EXPIRED)])
def test_minute_settled_after_a_gap_is_rechecked_with_the_newer_price(later_price, expected):
    evaluations = []
    # 每分钟只有一笔成交，K 线没有实体；本用例只验证复核，实体条件关掉。
    detector = DormantDetector([rule(min_body_ratio=0.0)], evaluations.append)
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=1.0)
    for index, price in enumerate([1.03, 1.06, 1.09]):
        detector.add_tick(PriceTick(SYMBOL, price, 500 / price, minute(SEED + index), f"t{index}"))
    # 冷门币之后沉寂几分钟才有下一笔：最后一分钟按这笔成交价复核。
    detector.add_tick(PriceTick(SYMBOL, later_price, 1.0, minute(SEED + 6), "late"))

    judged = [item for item in evaluations if item.timestamp == minute(SEED + 3)]
    assert judged[0].outcome is expected


def test_symbol_without_multiplier_is_never_judged():
    detector = DormantDetector([rule()])
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=None)

    for index, price in enumerate([1.03, 1.06, 1.09, 1.1]):
        assert detector.add_tick(PriceTick(SYMBOL, price, 500.0, minute(SEED + index), f"t{index}")) == []


def test_stream_gap_pauses_until_resync():
    detector = DormantDetector([rule()])
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=1.0)
    detector.mark_stream_gap()
    assert detector.stale_symbols == [SYMBOL]
    for index, price in enumerate([1.03, 1.06, 1.09, 1.1]):
        assert detector.add_tick(PriceTick(SYMBOL, price, 500 / price, minute(SEED + index), f"t{index}")) == []

    history = flat(SEED) + [
        Candle(minute(SEED + i), 1.0, close, 1.0, close, 500.0) for i, close in enumerate([1.03, 1.06, 1.09])
    ]
    # 交易所的当前分钟比本地多了断线期间的成交，收盘以交易所为准（本地最后一笔早于请求时间）。
    live = Candle(minute(SEED + 3), 1.09, 1.12, 1.09, 1.12, 400.0)
    assert detector.resync_symbol(SYMBOL, history, live, minute(SEED + 3) + timedelta(seconds=5))
    assert detector.stale_symbols == []
    assert not detector.resync_symbol(SYMBOL, history, live, minute(SEED + 3))
    # 回补后下一分钟的成交结算当前分钟，照常判定。
    alerts = detector.add_tick(PriceTick(SYMBOL, 1.1, 1.0, minute(SEED + 4), "after"))
    assert len(alerts) == 1


def test_removed_symbol_keeps_alert_memory():
    detector = make()
    assert len(rally(detector, SEED, [1.03, 1.06, 1.09])[-1]) == 1
    detector.remove_symbols({SYMBOL})
    assert detector.symbols == []

    detector.add_symbol(SYMBOL, flat(SEED + 3), 2e5, quanto_multiplier=1.0)
    # 重新入池后同一段拉升仍在冷却中。
    assert rally(detector, SEED + 3, [1.03, 1.06, 1.09]) == [[], [], []]


def test_warmup_covers_prior_day_window_and_live_minute():
    assert DormantDetector([rule(window_minutes=10, min_active_minutes=5)]).warmup_candles == PRIOR_MINUTES + 10 + 1


def test_prior_quote_and_pre_volatility_stay_exact_after_old_minutes_drop_out_and_resync():
    evaluations = []
    detector = DormantDetector([rule()], evaluations.append)
    # 每分钟成交额与价格都不同，挤出最早的分钟时减错一根就会看出来。
    candles = [
        Candle(minute(i), price, price, price, price, float(i % 7))
        for i, price in ((i, 1.0 + 0.01 * (i % 3)) for i in range(SEED + 50))
    ]
    detector.add_symbol(SYMBOL, candles[:SEED], 2e5, quanto_multiplier=1.0)
    for candle in candles[SEED:]:
        detector.add_candle(SYMBOL, candle)

    def expected(end: int) -> float:
        # end 是最后一根已收盘 K 线的下标：之前 24 小时 = 窗口之前的 1440 根。
        return sum(candle.quote_volume for candle in candles[end - WINDOW - PRIOR_MINUTES + 1 : end - WINDOW + 1])

    def calm(end: int) -> float:
        # 窗口开始前一根（end - WINDOW）及之前共 CALM_MINUTES 根的真实波幅平均。
        start = end - WINDOW
        return sum(
            minute_range_percent(candles[i], candles[i - 1].close) for i in range(start - CALM_MINUTES + 1, start + 1)
        ) / CALM_MINUTES

    assert evaluations[-1].metrics.prior_quote == pytest.approx(expected(SEED + 49))
    assert evaluations[-1].metrics.pre_volatility_percent == pytest.approx(calm(SEED + 49))
    assert evaluations[-1].metrics.pre_volatility_percent > 0

    detector.mark_stream_gap()
    assert detector.resync_symbol(SYMBOL, candles[:SEED + 30], None, minute(SEED + 30))
    detector.add_candle(SYMBOL, candles[SEED + 30])
    assert evaluations[-1].metrics.prior_quote == pytest.approx(expected(SEED + 30))
    assert evaluations[-1].metrics.pre_volatility_percent == pytest.approx(calm(SEED + 30))


def test_choppy_coin_and_single_minute_jump_do_not_alert():
    evaluations = []
    # 涨前在 1.0 与 1.03 之间来回，平均波幅约 3%，涨 9% 只有约 3 倍。
    choppy = [
        Candle(minute(i), price, price, price, price, 5.0)
        for i, price in ((i, 1.03 if (SEED - i) % 2 == 0 else 1.0) for i in range(SEED))
    ]
    detector = DormantDetector([rule()], evaluations.append)
    detector.add_symbol(SYMBOL, choppy, 2e5, quanto_multiplier=1.0)
    assert rally(detector, SEED, [1.03, 1.06, 1.09]) == [[], [], []]
    assert evaluations[-1].metrics.failures == (Condition.CALM, Condition.VOLATILE)

    # 一字横盘之后，最后一分钟一下跳 7%。
    detector = make(evaluations.append)
    assert rally(detector, SEED, [1.01, 1.02, 1.09]) == [[], [], []]
    assert evaluations[-1].metrics.failures == (Condition.JUMP,)


def test_slow_group_judges_alongside_main_with_its_own_window_and_cooldown():
    evaluations = []
    slow = rule(
        window_minutes=6, min_active_minutes=3, min_change_percent=12.0, realert_step_percent=None,
        group=DormantGroup.SLOW,
    )
    detector = DormantDetector([rule(), slow], evaluations.append)
    seed = PRIOR_MINUTES + 6
    detector.add_symbol(SYMBOL, flat(seed), 2e5, quanto_multiplier=1.0)

    alerts = rally(detector, seed, [1.03, 1.06, 1.09, 1.12, 1.15, 1.18])

    # 现行一组（3 分钟涨 9%）第 3 分钟报、1.15 比 1.09 再涨 5.5% 报第 2 波；慢启动一组（6 分钟涨 12%）第 4 分钟报，
    # 之后冷却内不补报。两组的冷却互不影响。
    labels = [[alert.label for alert in minute_alerts] for minute_alerts in alerts]
    assert labels == [[], [], ["冷门启动"], ["冷门慢启动"], ["冷门启动·第2波"], []]
    assert alerts[3][0].to_dict()["group"] == "slow"
    assert [item.outcome for item in evaluations if item.group is DormantGroup.SLOW][-2:] == [
        DormantOutcome.COOLDOWN,
        DormantOutcome.COOLDOWN,
    ]
    # 保留的 K 线按慢启动一组的窗口多留了 3 根，现行一组的之前 24 小时仍然只算窗口之前的 1440 根。
    first = next(item for item in evaluations if item.group is DormantGroup.MAIN)
    assert first.metrics.prior_quote == pytest.approx(5.0 * PRIOR_MINUTES)


def test_group_with_shorter_window_is_judged_before_the_longer_one_has_enough_history():
    evaluations = []
    slow = rule(window_minutes=6, min_active_minutes=3, group=DormantGroup.SLOW)
    detector = DormantDetector([rule(), slow], evaluations.append)
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=1.0)
    assert detector.warmup_candles == PRIOR_MINUTES + 6 + 1

    rally(detector, SEED, [1.03])

    outcomes = {item.group: item.outcome for item in evaluations}
    assert outcomes == {DormantGroup.MAIN: DormantOutcome.FAILED, DormantGroup.SLOW: DormantOutcome.INSUFFICIENT_DATA}


def test_rejects_missing_or_duplicate_groups():
    with pytest.raises(ValueError, match="至少要有一组"):
        DormantDetector([])
    with pytest.raises(ValueError, match="不能重复"):
        DormantDetector([rule(), rule(window_minutes=5)])


def test_follow_reports_each_further_rise_without_rechecking_conditions():
    evaluations = []
    detector = make(evaluations.append, realert_step_percent=None, follow_step_percent=5.0, follow_minutes=10)
    assert rally(detector, SEED, [1.03, 1.06, 1.09])[-1][0].wave == 1

    # 3 分钟只涨 5%~7%，条件不满足；1.145 比上次提醒价 1.09 再涨 5%，跟踪补报第 2 波。
    alerts = rally(detector, SEED + 3, [1.10, 1.12, 1.145], price=1.09)
    assert [len(item) for item in alerts] == [0, 0, 1]
    second = alerts[-1][0]
    assert (second.wave, second.label, second.metrics.end_price) == (2, "冷门启动·第2波", 1.145)
    assert evaluations[-1].outcome is DormantOutcome.FOLLOW
    assert Condition.CHANGE in evaluations[-1].metrics.failures

    # 之后 10 分钟没再涨够 5%（1.145 × 1.05 ≈ 1.202），跟踪结束：再涨上去也不补报。
    for candle in flat(11, start=SEED + 6, price=1.19):
        detector.add_candle(SYMBOL, candle)
    assert rally(detector, SEED + 17, [1.21], price=1.19) == [[]]


def test_follow_window_outlasts_cooldown_and_keeps_counting_waves():
    detector = make(realert_step_percent=None, follow_step_percent=5.0, follow_minutes=60)
    rally(detector, SEED, [1.03, 1.06, 1.09])
    # 冷却 30 分钟已过、跟踪 60 分钟还没到：慢慢涨够 5% 仍按跟踪补报第 2 波，不是新的第 1 波。
    for candle in flat(40, start=SEED + 3, price=1.09):
        detector.add_candle(SYMBOL, candle)

    alerts = rally(detector, SEED + 43, [1.11, 1.13, 1.145], price=1.09)

    assert [alert.wave for item in alerts for alert in item] == [2]


@pytest.mark.parametrize(("later_price", "expected"), [(1.15, DormantOutcome.FOLLOW), (1.13, DormantOutcome.FAILED)])
def test_follow_after_a_gap_is_rechecked_with_the_newer_price(later_price, expected):
    evaluations = []
    detector = DormantDetector(
        [rule(min_body_ratio=0.0, realert_step_percent=None, follow_step_percent=5.0)], evaluations.append
    )
    detector.add_symbol(SYMBOL, flat(SEED), 2e5, quanto_multiplier=1.0)
    for index, price in enumerate([1.03, 1.06, 1.09, 1.10, 1.12, 1.145]):
        detector.add_tick(PriceTick(SYMBOL, price, 500 / price, minute(SEED + index), f"t{index}"))
    # 1.145 那一分钟隔了空档才结算：这笔成交价也要比上次提醒价 1.09 高 5%（1.1445）才补报。
    detector.add_tick(PriceTick(SYMBOL, later_price, 1.0, minute(SEED + 8), "late"))

    judged = [item for item in evaluations if item.timestamp == minute(SEED + 6)]
    assert judged[0].outcome is expected
