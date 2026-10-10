from datetime import UTC, datetime, timedelta

import pytest

from price_alert.detector import AtrMoveDetector
from price_alert.models import Candle, PriceAlert, PriceTick
from price_alert.streak import StreakAlert, StreakRule
from price_alert.windows import MoveWindow, Outcome

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def history(range_size: float, period: int = 3) -> list[Candle]:
    return [
        Candle(
            BASE + timedelta(minutes=index),
            100.0,
            100.0 + range_size / 2,
            100.0 - range_size / 2,
            100.0,
        )
        for index in range(period)
    ]


def short_window(**overrides) -> MoveWindow:
    values = {
        "name": "short",
        "lookback_seconds": 10,
        "trigger_atr_multiple": 0.8,
        "min_change_percent": 1.0,
        "confirmation_seconds": 2,
        "min_window_trades": 2,
        "cooldown_seconds": 120,
    }
    return MoveWindow(**(values | overrides))


def detector(*extra_windows: MoveWindow, observer=None) -> AtrMoveDetector:
    return AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(), *extra_windows],
        observer=observer,
    )


def feed_window(instance: AtrMoveDetector, symbol: str, final_price: float):
    start = BASE + timedelta(minutes=3)
    alerts = []
    for second in range(11):
        instance.add_tick(PriceTick(symbol, 100.0, 1.0, start + timedelta(seconds=second), str(second)))
    for second in range(11, 14):
        alerts.extend(
            instance.add_tick(PriceTick(symbol, final_price, 1.0, start + timedelta(seconds=second), str(second)))
        )
    return alerts


def test_same_price_move_triggers_in_low_atr_but_not_high_atr_market():
    instance = detector()
    instance.add_symbol("LOW_USDT", history(1.0), 20_000_000)
    instance.add_symbol("HIGH_USDT", history(4.0), 30_000_000)

    low_alerts = feed_window(instance, "LOW_USDT", 101.0)
    high_alerts = feed_window(instance, "HIGH_USDT", 101.0)

    assert len(low_alerts) == 1
    assert low_alerts[0].move_atr == 1.0
    assert high_alerts == []


def test_rejects_move_below_minimum_percent_even_when_atr_multiple_is_high():
    instance = detector()
    instance.add_symbol("LOW_USDT", history(0.2), 20_000_000)

    alerts = feed_window(instance, "LOW_USDT", 100.9)

    assert alerts == []


def test_cooldown_is_shared_between_surge_and_drop():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    assert feed_window(instance, "BTC_USDT", 101.0)[0].direction == "surge"

    alerts = []
    start = BASE + timedelta(minutes=3)
    for second in range(14, 17):
        alerts.extend(instance.add_tick(PriceTick("BTC_USDT", 99.0, 1.0, start + timedelta(seconds=second))))

    assert alerts == []


def test_requires_consecutive_completed_seconds_before_alerting():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    assert instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, start + timedelta(seconds=11))) == []
    assert instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, start + timedelta(seconds=12))) == []
    alerts = instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, start + timedelta(seconds=13)))

    assert len(alerts) == 1


def test_small_outlier_trade_is_diluted_by_completed_second_vwap():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("BTC_USDT", 100.0, 10.0, start + timedelta(seconds=second)))

    outlier_time = start + timedelta(seconds=11)
    assert instance.add_tick(PriceTick("BTC_USDT", 110.0, 0.01, outlier_time)) == []
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 10.0, outlier_time + timedelta(milliseconds=100)))
    alerts = instance.add_tick(PriceTick("BTC_USDT", 100.0, 10.0, start + timedelta(seconds=12)))

    assert alerts == []


def test_ignores_unknown_and_out_of_order_ticks():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    current = BASE + timedelta(minutes=3)
    assert instance.add_tick(PriceTick("OTHER_USDT", 100, 1, current)) == []
    instance.add_tick(PriceTick("BTC_USDT", 100, 1, current))
    assert instance.add_tick(PriceTick("BTC_USDT", 80, 1, current - timedelta(seconds=1))) == []


def feed_window_at(instance: AtrMoveDetector, symbol: str, final_price: float, start):
    alerts = []
    for second in range(11):
        instance.add_tick(PriceTick(symbol, 100.0, 1.0, start + timedelta(seconds=second)))
    for second in range(11, 14):
        alerts.extend(instance.add_tick(PriceTick(symbol, final_price, 1.0, start + timedelta(seconds=second))))
    return alerts


def test_sparse_trades_are_confirmed_by_carrying_last_price_through_empty_seconds():
    instance = detector()
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    # 每隔一秒才有一笔成交：空秒不补齐的话“相邻秒”会被打断，永远无法确认。
    assert instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=11))) == []
    assert instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=13))) == []
    alerts = instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=15)))

    assert len(alerts) == 1
    assert alerts[0].direction == "surge"


def test_single_wick_followed_by_silence_does_not_alert_from_carried_seconds():
    instance = detector()
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    alerts = []
    alerts.extend(instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=11))))
    # 空秒 12、13 延续了确认进度，但价格在下一笔真实成交时已经回落，不能提醒。
    alerts.extend(instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=14))))
    alerts.extend(instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=15))))

    assert alerts == []


def stale_second_settled_after_gap(settling_price: float) -> list:
    """构造“异动秒之后出现空档，由一笔迟到的成交来结算它”的场景。

    确认进度在空档中由补齐桶凑满，因此提醒只能由空档前那个真实秒产生，
    而它的价格此时已经是 8 秒前的旧值。
    """
    instance = detector()
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    alerts = []
    for second in (11, 12):
        alerts.extend(instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=second))))
    assert alerts == []
    return instance.add_tick(PriceTick("THIN_USDT", settling_price, 1.0, start + timedelta(seconds=20)))


def test_stale_second_is_not_alerted_when_settling_trade_shows_the_move_is_over():
    # 结算这一秒的成交已经回到 100，异动在空档中就结束了，不能再按 101 发提醒。
    assert stale_second_settled_after_gap(100.0) == []


def test_stale_second_is_still_alerted_when_settling_trade_confirms_the_move():
    # 同一条路径上价格仍在 101：空档后结算是稀疏合约唯一的提醒时机，不能一并拦掉。
    alerts = stale_second_settled_after_gap(101.0)

    assert len(alerts) == 1
    assert alerts[0].direction == "surge"
    assert alerts[0].price == 101.0


def test_reversal_after_gap_does_not_alert_in_the_old_direction():
    # 复核必须看方向：砸穿基准价的成交在幅度上同样超标，只比幅度会发出方向相反的提醒。
    assert stale_second_settled_after_gap(98.0) == []


def test_gap_longer_than_lookback_is_not_filled():
    instance = detector()
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("THIN_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    alerts = []
    for second in (40, 41, 42):
        alerts.extend(instance.add_tick(PriceTick("THIN_USDT", 101.0, 1.0, start + timedelta(seconds=second))))

    assert alerts == []


def test_stream_gap_prevents_comparing_prices_across_disconnect():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, start + timedelta(seconds=second)))

    instance.mark_stream_gap()
    instance.resync_symbol("BTC_USDT", history(1.0), None, start + timedelta(seconds=10))
    alerts = []
    for second in range(11, 15):
        alerts.extend(instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, start + timedelta(seconds=second))))

    assert alerts == []


def test_atr_age_is_measured_from_candle_close():
    fresh = detector()
    fresh.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    # 最后一根种子 K 线 02:00 开盘、03:00 收盘；05:10 起的窗口距收盘不到 180 秒，仍应判定。
    assert len(feed_window_at(fresh, "BTC_USDT", 101.0, BASE + timedelta(minutes=5, seconds=10))) == 1

    stale = detector()
    stale.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    assert feed_window_at(stale, "BTC_USDT", 101.0, BASE + timedelta(minutes=6, seconds=10)) == []


def test_live_candle_from_warmup_is_included_in_next_atr_update():
    with_live = detector()
    live_candle = Candle(BASE + timedelta(minutes=3), 100.0, 106.0, 100.0, 100.0)
    with_live.add_symbol("BTC_USDT", history(1.0), 1_000_000_000, live_candle)
    without_live = detector()
    without_live.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    start = BASE + timedelta(minutes=4)
    # 订阅前的 6 点振幅计入 ATR 后，1% 位移不再足够异常。
    assert feed_window_at(with_live, "BTC_USDT", 101.0, start) == []
    assert len(feed_window_at(without_live, "BTC_USDT", 101.0, start)) == 1


def test_cooldown_survives_symbol_leaving_and_rejoining_universe():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    assert len(feed_window(instance, "BTC_USDT", 101.0)) == 1

    instance.remove_symbols({"BTC_USDT"})
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    assert feed_window_at(instance, "BTC_USDT", 99.0, BASE + timedelta(minutes=3, seconds=20)) == []


def atr_of(instance: AtrMoveDetector, symbol: str):
    # ATR 没有公开读取接口；K 线口径问题直接检查内部 ATR 最清楚，也不必借助整条提醒链路间接推断。
    return instance._states[symbol].atr


def test_stale_atr_pauses_detection_until_resynced():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    instance.mark_stream_gap()
    start = BASE + timedelta(minutes=3)

    assert instance.stale_symbols == ["BTC_USDT"]
    assert feed_window_at(instance, "BTC_USDT", 101.0, start) == []

    assert instance.resync_symbol("BTC_USDT", history(1.0), None, start + timedelta(seconds=14)) is True
    assert instance.stale_symbols == []
    assert len(feed_window_at(instance, "BTC_USDT", 101.0, start + timedelta(seconds=20))) == 1


def test_resync_ignores_unknown_or_already_fresh_symbols():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    assert instance.resync_symbol("BTC_USDT", history(4.0), None, BASE) is False
    assert instance.resync_symbol("ETH_USDT", history(4.0), None, BASE) is False
    assert atr_of(instance, "BTC_USDT").value == 1.0


def test_resync_merges_local_and_exchange_view_of_current_bar():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    bar = BASE + timedelta(minutes=3)
    # 断线前本地见到了 103 的高点，断线期间交易所记录了 96 的低点。
    instance.add_tick(PriceTick("BTC_USDT", 103.0, 1.0, bar + timedelta(seconds=5)))
    instance.mark_stream_gap()
    instance.add_tick(PriceTick("BTC_USDT", 98.0, 1.0, bar + timedelta(seconds=50)))

    exchange_bar = Candle(bar, 100.0, 101.0, 96.0, 97.0)
    assert instance.resync_symbol("BTC_USDT", history(1.0), exchange_bar, bar + timedelta(seconds=40))
    instance.add_tick(PriceTick("BTC_USDT", 98.0, 1.0, bar + timedelta(seconds=60)))

    atr = atr_of(instance, "BTC_USDT")
    # 合并 K 线高 103、低 96，TR=7，ATR=(1×2+7)/3。
    assert atr.last_timestamp == bar
    assert atr.value == pytest.approx(3.0)


def test_resync_counts_exchange_current_bar_as_closed_after_local_rollover():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    instance.mark_stream_gap()
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(seconds=61)))

    assert instance.resync_symbol(
        "BTC_USDT",
        history(1.0),
        Candle(bar, 100.0, 104.0, 100.0, 100.0),
        bar + timedelta(seconds=59),
    )

    atr = atr_of(instance, "BTC_USDT")
    assert atr.last_timestamp == bar
    assert atr.value == pytest.approx((1.0 * 2 + 4.0) / 3)


def test_bars_without_trades_are_filled_with_flat_candles_like_gate():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar))
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(minutes=3)))

    atr = atr_of(instance, "BTC_USDT")
    # 03:00 的实时 K 线与 04:00、05:00 两根平线 TR 均为 0，ATR 每根衰减为 2/3。
    assert atr.last_timestamp == BASE + timedelta(minutes=5)
    assert atr.value == pytest.approx((2 / 3) ** 3)


def test_flat_fill_is_capped_and_skipped_while_stale():
    bar = BASE + timedelta(minutes=3)
    capped = detector()
    capped.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    capped.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar))
    capped.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(minutes=1000)))
    assert atr_of(capped, "BTC_USDT").last_timestamp == bar + timedelta(minutes=999)

    stale = detector()
    stale.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    stale.mark_stream_gap()
    stale.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar))
    stale.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(minutes=3)))
    assert atr_of(stale, "BTC_USDT").last_timestamp == BASE + timedelta(minutes=2)


def evaluated_atr(evaluations, second) -> float | None:
    matched = [evaluation.atr for evaluation in evaluations if evaluation.second == second]
    assert matched, f"{second} 没有被评估"
    return matched[0]


def test_second_settled_after_long_silence_ignores_flat_bars_that_came_later():
    evaluations = []
    instance = detector(observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(seconds=5)))
    # 沉默到 08:00 才有下一笔，03:05 这一秒此时才被结算。
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(minutes=5)))

    # 评估 03:05 只能用 03:00 之前收盘的 K 线；若先计入 03:00 与之后的平线，ATR 会衰减到 (2/3)^5。
    assert evaluated_atr(evaluations, bar + timedelta(seconds=5)) == pytest.approx(1.0)
    # 处理完这笔成交后，03:00 的 K 线与 04:00~07:00 的平线照常计入。
    atr = atr_of(instance, "BTC_USDT")
    assert atr.last_timestamp == BASE + timedelta(minutes=7)
    assert atr.value == pytest.approx((2 / 3) ** 5)


def test_last_second_of_a_bar_is_judged_without_its_own_bar_even_when_settled_next_second():
    evaluations = []
    instance = detector(observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(seconds=58)))
    # 同一秒的后续成交同样要计入实时 K 线，否则 03:00 这根会漏掉 104 与 96 两个极值。
    for millis, price in ((0, 100.0), (100, 104.0), (200, 96.0)):
        instance.add_tick(PriceTick("BTC_USDT", price, 1.0, bar + timedelta(seconds=59, milliseconds=millis)))
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(minutes=1)))

    # 03:59 与 03:58 同属 03:00 这根 K 线，判定口径应一致：都不含本根 K 线。
    assert evaluated_atr(evaluations, bar + timedelta(seconds=58)) == pytest.approx(1.0)
    assert evaluated_atr(evaluations, bar + timedelta(seconds=59)) == pytest.approx(1.0)
    atr = atr_of(instance, "BTC_USDT")
    # 03:00 这根高 104、低 96，TR=8，ATR=(1×2+8)/3。
    assert atr.last_timestamp == bar
    assert atr.value == pytest.approx(10 / 3)


def test_carried_seconds_use_the_bars_closed_before_their_own_bar():
    evaluations = []
    instance = detector(observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(seconds=55)))
    # 03:56~04:02 由补齐桶覆盖，跨过了 04:00 的 K 线边界。
    instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, bar + timedelta(minutes=1, seconds=3)))

    carried = {evaluation.second: evaluation.atr for evaluation in evaluations if evaluation.carried}
    # 03:00 这根只有 100 一个价格，TR=0，收盘后 ATR 衰减为 2/3；只有 04:00 之后的补齐秒才能用上它。
    assert carried == {
        **{bar + timedelta(seconds=second): pytest.approx(1.0) for second in range(56, 60)},
        **{bar + timedelta(minutes=1, seconds=second): pytest.approx(2 / 3) for second in range(3)},
    }
    # 为补齐秒推演的 ATR 只是副本：实时状态与逐笔处理时一致，03:00 只计入一次，新 K 线由成交开出。
    atr = atr_of(instance, "BTC_USDT")
    assert atr.last_timestamp == bar
    assert atr.value == pytest.approx(2 / 3)
    live_bar = instance._states["BTC_USDT"].live_bar
    assert live_bar is not None
    assert live_bar.timestamp == bar + timedelta(minutes=1)
    assert (live_bar.open, live_bar.high, live_bar.low) == (101, 101, 101)


def test_second_left_pending_across_a_late_resync_is_not_judged_with_the_rebuilt_atr():
    evaluations = []
    instance = detector(observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)
    instance.mark_stream_gap()
    bar = BASE + timedelta(minutes=3)
    instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, bar + timedelta(seconds=5)))
    # 03:06 在回补完成前还没等到下一笔成交，一直留着待结算。
    instance.add_tick(PriceTick("BTC_USDT", 102.0, 1.0, bar + timedelta(seconds=6)))

    # 回补拖到 08:00 之后才成功，重建的 ATR 已含 03:00 之后的 K 线。
    late = BASE + timedelta(minutes=8)
    closed = [Candle(late - timedelta(minutes=offset), 102.0, 102.5, 101.5, 102.0) for offset in range(3, 0, -1)]
    live_candle = Candle(late, 102.0, 102.0, 102.0, 102.0)
    assert instance.resync_symbol("BTC_USDT", closed, live_candle, late + timedelta(seconds=5))
    instance.add_tick(PriceTick("BTC_USDT", 102.0, 1.0, late + timedelta(seconds=6)))
    instance.add_tick(PriceTick("BTC_USDT", 102.0, 1.0, late + timedelta(seconds=7)))

    # 03:06 对应周期的 ATR 已被替换，只能放弃判定，不能拿之后的 ATR 充数。
    pending = [evaluation for evaluation in evaluations if evaluation.second == bar + timedelta(seconds=6)]
    assert [evaluation.outcome for evaluation in pending] == [Outcome.ATR_UNAVAILABLE]
    # 回补之后产生的秒照常使用重建的 ATR。
    after = [evaluation for evaluation in evaluations if evaluation.second == late + timedelta(seconds=6)]
    assert [evaluation.atr for evaluation in after] == [pytest.approx(1.0)]


def long_window(**overrides) -> MoveWindow:
    values = {
        "name": "long",
        "lookback_seconds": 40,
        "trigger_atr_multiple": 1.5,
        "min_change_percent": 2.0,
        "confirmation_seconds": 2,
        "min_window_trades": 5,
        "cooldown_seconds": 40,
    }
    return MoveWindow(**(values | overrides))


def feed_prices(instance: AtrMoveDetector, symbol: str, prices: dict[int, float]):
    start = BASE + timedelta(minutes=3)
    alerts = []
    for second, price in sorted(prices.items()):
        alerts.extend(instance.add_tick(PriceTick(symbol, price, 1.0, start + timedelta(seconds=second), str(second))))
    return alerts


def test_slow_grind_is_caught_by_long_window_only():
    instance = detector(long_window())
    instance.add_symbol("QNT_USDT", history(1.0), 50_000_000)

    # 每秒涨 0.06：10 秒只涨 0.6%，短窗口不达标；40 秒涨 2.4%、2.4 ATR，长窗口达标。
    alerts = feed_prices(instance, "QNT_USDT", {second: 100.0 + 0.06 * second for second in range(56)})

    assert len(alerts) == 1
    assert alerts[0].window == "long"
    assert alerts[0].lookback_seconds == 40
    assert alerts[0].direction == "surge"
    # 第 40、41 秒连续达标，第 41 秒对比第 1 秒：102.46 / 100.06。
    assert alerts[0].change_percent == pytest.approx(2.4 / 100.06 * 100)


def test_both_windows_alert_in_the_same_second_independently():
    evaluations = []
    instance = detector(long_window(), observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    prices = {second: 100.0 for second in range(41)} | {second: 103.0 for second in range(41, 90)}
    alerts = feed_prices(instance, "BTC_USDT", prices)

    # 两个窗口互不影响：同一秒都满足就各报一条，短窗口在前；之后长窗口在自己的冷却内不再重复报这段行情。
    assert [alert.window for alert in alerts] == ["short", "long"]
    assert alerts[0].timestamp == alerts[1].timestamp
    assert Outcome.COOLDOWN in {item.outcome for item in evaluations if item.window == "long"}


def test_short_alerts_do_not_extend_long_cooldown():
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(cooldown_seconds=10), long_window()],
    )
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    # 第 41 秒跳到 103，两个窗口在第 43 秒各报一条；第 69~70 秒的尖刺让短窗口在第 71、81 秒又报两条，
    # 第 95 秒再涨到 106。长窗口的冷却只从它自己第 43 秒的提醒算起，第 97 秒照常提醒；
    # 若被短窗口第 81 秒的提醒续上，要到第 121 秒才能再报。
    prices = {second: 100.0 for second in range(41)} | {second: 103.0 for second in range(41, 95)}
    prices |= {69: 104.5, 70: 104.5} | {second: 106.0 for second in range(95, 100)}
    alerts = feed_prices(instance, "BTC_USDT", prices)

    start = BASE + timedelta(minutes=3)
    fired = [(alert.window, int((alert.timestamp - start).total_seconds())) for alert in alerts]
    assert fired == [("short", 43), ("long", 43), ("short", 71), ("short", 81), ("short", 97), ("long", 97)]


def test_long_cooldown_does_not_delay_a_new_short_move():
    instance = detector(long_window(cooldown_seconds=300))
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    # 先缓慢上涨触发长窗口，横盘后在第 170 秒急涨：此时已过短窗口的 120 秒冷却，
    # 但仍在长窗口的 300 秒冷却内，长冷却不能拖住短窗口对新一波行情的提醒。
    prices = {second: 100.0 + 0.06 * second for second in range(45)}
    prices |= {second: 102.64 for second in range(45, 170)}
    prices |= {second: 106.0 for second in range(170, 174)}
    alerts = feed_prices(instance, "BTC_USDT", prices)

    assert [alert.window for alert in alerts] == ["long", "short"]


def test_long_window_baseline_carries_price_through_gap_longer_than_fill_limit():
    evaluations = []
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        # 短窗口门槛设到不可能达到，只观察长窗口；补齐上限仍由短窗口的 10 秒决定。
        windows=[short_window(min_change_percent=50.0), long_window()],
        observer=evaluations.append,
    )
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000, price_decimals=2)

    # 第 5 秒到第 25 秒之间空了 19 秒，超过补齐上限，但短于长窗口的 40 秒。
    prices = {second: 100.0 for second in range(6)} | {second: 100.0 for second in range(25, 50)}
    prices |= {second: 103.0 for second in range(50, 54)}
    alerts = feed_prices(instance, "THIN_USDT", prices)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.window == "long"
    # 第 50 秒的窗口起点落在空档里，基准沿用空档前最后一笔成交价。
    assert alert.reference_price == 100.0
    assert alert.price_decimals == 2
    # 窗口笔数只统计基准桶（第 5 秒）之后的成交：第 25~51 秒共 27 笔。
    assert alert.trade_count == 27


def test_long_window_skips_when_gap_is_longer_than_the_window():
    evaluations = []
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(min_change_percent=50.0), long_window()],
        observer=evaluations.append,
    )
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)

    # 空档 44 秒，比长窗口还长：空档前的价格已失去参考意义，不能拿来当基准。
    prices = {0: 100.0} | {second: 103.0 for second in range(45, 60)}
    alerts = feed_prices(instance, "THIN_USDT", prices)

    assert alerts == []
    long_outcomes = [item.outcome for item in evaluations if item.window == "long" and item.second.second < 60]
    assert Outcome.NO_BASELINE in long_outcomes
    assert Outcome.ALERT not in long_outcomes


def test_observer_reports_why_a_second_did_not_alert():
    evaluations = []
    instance = detector(observer=evaluations.append)
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    feed_prices(instance, "BTC_USDT", {second: 100.0 for second in range(13)})

    outcomes = [item.outcome for item in evaluations]
    # 前 10 秒还没有 10 秒前的基准，之后价格不动，未达门槛。
    assert outcomes[0] is Outcome.NO_BASELINE
    assert outcomes[-1] is Outcome.BELOW_THRESHOLD
    assert evaluations[-1].change_percent == 0.0
    assert evaluations[-1].trade_count == 10


def test_price_decimals_follow_the_latest_universe_refresh():
    instance = detector()
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000, price_decimals=1)
    instance.add_symbol("BTC_USDT", [], 1_000_000_000, price_decimals=2)

    assert feed_window(instance, "BTC_USDT", 101.0)[0].price_decimals == 2


def test_rejects_invalid_window_sets():
    with pytest.raises(ValueError, match="至少需要一个"):
        AtrMoveDetector(atr_period=3, candle_interval_seconds=60, max_atr_age_seconds=180, windows=[])
    with pytest.raises(ValueError, match="不能重复"):
        AtrMoveDetector(
            atr_period=3,
            candle_interval_seconds=60,
            max_atr_age_seconds=180,
            windows=[short_window(), short_window()],
        )
    with pytest.raises(ValueError, match="confirmation_seconds"):
        short_window(confirmation_seconds=11)


def test_window_with_too_few_trades_does_not_alert_even_when_price_moved():
    evaluations = []
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(min_window_trades=20)],
        observer=evaluations.append,
    )
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)

    # 每秒只有一笔成交，10 秒窗口最多 10 笔，涨 3% 也不够 20 笔的门槛。
    prices = {second: 100.0 for second in range(11)} | {second: 103.0 for second in range(11, 14)}
    alerts = feed_prices(instance, "THIN_USDT", prices)

    assert alerts == []
    last = evaluations[-1]
    assert last.outcome is Outcome.FEW_TRADES
    assert (last.trade_count, last.change_percent) == (10, pytest.approx(3.0))


def test_short_window_does_not_fill_gaps_just_because_a_long_window_exists():
    instance = detector(long_window())
    instance.add_symbol("THIN_USDT", history(1.0), 20_000_000)

    # 空档 29 秒：长于短窗口的 10 秒、短于长窗口的 40 秒。补齐上限若误用最长窗口，
    # 短窗口会拿空档前的旧价补出基准，把 1% 的位移误报成 10 秒内的急涨。
    prices = {second: 100.0 for second in range(11)} | {second: 101.0 for second in range(40, 46)}

    assert feed_prices(instance, "THIN_USDT", prices) == []


def test_confirmation_restarts_after_each_alert():
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(cooldown_seconds=0)],
    )
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    start = BASE + timedelta(minutes=3)
    for second in range(11):
        instance.add_tick(PriceTick("BTC_USDT", 100.0, 1.0, start + timedelta(seconds=second)))
    fired = []
    for second in range(11, 18):
        if instance.add_tick(PriceTick("BTC_USDT", 101.0, 1.0, start + timedelta(seconds=second))):
            fired.append(second)

    # 没有冷却时，每次提醒后仍要重新连续确认 2 秒，而不是之后每秒都提醒。
    assert fired == [13, 15, 17]


def test_mark_symbol_gap_only_invalidates_that_symbol():
    instance = detector()
    instance.add_symbol("LOW_USDT", history(1.0), 20_000_000)
    instance.add_symbol("HIGH_USDT", history(4.0), 30_000_000)

    instance.mark_symbol_gap("low_usdt")
    instance.mark_symbol_gap("MISSING_USDT")

    assert instance.stale_symbols == ["LOW_USDT"]
    # 失效期间不判定；回补后恢复。
    assert feed_window(instance, "LOW_USDT", 101.0) == []
    assert instance.resync_symbol("LOW_USDT", history(1.0), None, BASE + timedelta(minutes=3))
    assert instance.stale_symbols == []


def test_consecutive_short_alerts_add_a_streak_alert_and_long_alerts_do_not_count():
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[
            short_window(cooldown_seconds=0),
            long_window(lookback_seconds=11, min_change_percent=0.5, trigger_atr_multiple=0.8),
        ],
        streak=StreakRule(min_alerts=3, max_gap_seconds=5),
    )
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    prices = {second: 100.0 for second in range(12)} | {second: 101.0 for second in range(12, 19)}
    alerts = feed_prices(instance, "BTC_USDT", prices)

    # 长窗口提醒夹在中间既不计数也不打断；第 3 条短窗口提醒之后紧跟一条连续提醒，同一串不再重复。
    kinds = [alert.window if isinstance(alert, PriceAlert) else "streak" for alert in alerts]
    assert kinds == ["short", "long", "short", "short", "streak"]
    streak = alerts[-1]
    assert isinstance(streak, StreakAlert)
    assert streak.alert_count == 3
    assert streak.timestamp == alerts[-2].timestamp


def test_detector_without_streak_rule_never_adds_streak_alerts():
    instance = AtrMoveDetector(
        atr_period=3,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[short_window(cooldown_seconds=0)],
    )
    instance.add_symbol("BTC_USDT", history(1.0), 1_000_000_000)

    prices = {second: 100.0 for second in range(11)} | {second: 101.0 for second in range(11, 30)}
    alerts = feed_prices(instance, "BTC_USDT", prices)

    assert len(alerts) > 3
    assert all(isinstance(alert, PriceAlert) for alert in alerts)
