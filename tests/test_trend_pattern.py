from datetime import UTC, datetime, timedelta

import pytest

from price_alert.models import Candle
from price_alert.trend.pattern import Condition, TrendRule, aggregate_candles, evaluate_pattern

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def rule(**overrides) -> TrendRule:
    values = {
        "period": "1m",
        "candles": 5,
        "min_change_percent": 4.0,
        "max_counter_candles": 1,
        "max_rebound_ratio": 0.3,
        "min_body_ratio": 0.5,
        # 默认不限制单根实体与单根占比，各用例只验证自己关心的条件；这两项由专门的用例覆盖。
        "min_candle_body_ratio": 0.0,
        "max_single_candle_ratio": 1.0,
    }
    return TrendRule(**(values | overrides))


def candles(*bars: tuple[float, float, float, float]) -> list[Candle]:
    """按 (开, 高, 低, 收) 生成连续的 1 分钟 K 线。"""
    return [Candle(BASE + timedelta(minutes=index), o, h, low, c) for index, (o, h, low, c) in enumerate(bars)]


def falling(start: float = 100.0, step: float = 1.0, count: int = 5) -> list[Candle]:
    # 无影线的等幅阴线：每一项条件都取到最理想的值。
    return candles(
        *((start - step * i, start - step * i, start - step * (i + 1), start - step * (i + 1)) for i in range(count))
    )


def test_clean_drop_passes():
    metrics = evaluate_pattern(falling(), rule())

    assert metrics is not None and metrics.passed
    assert metrics.direction == "drop"
    assert metrics.change_percent == pytest.approx(-5.0)
    assert (metrics.counter_candles, metrics.rebound_ratio, metrics.body_ratio) == (0, 0, 1)
    assert metrics.single_candle_ratio == pytest.approx(0.2)
    assert (metrics.start_price, metrics.end_price) == (100.0, 95.0)


def test_surge_is_mirror_of_drop():
    bars = [(100 + i, 101 + i, 100 + i, 101 + i) for i in range(5)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None and metrics.passed
    assert metrics.direction == "surge"


def test_one_small_counter_candle_is_tolerated():
    # 第三根小阳线，收盘没有收复多少跌幅。
    bars = [(100, 100, 98, 98), (98, 98, 96, 96), (96, 96.8, 96, 96.6), (96.6, 96.6, 94, 94), (94, 94, 92, 92)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None and metrics.passed
    assert metrics.counter_candles == 1
    assert metrics.rebound_ratio == pytest.approx(0.6 / 8)


# 振幅 1、门槛 0.25，取二进制能精确表示的数，边界比较不受浮点误差影响。
@pytest.mark.parametrize(("body", "counter"), [(0.125, True), (0.25, False)])
def test_small_body_candle_counts_as_counter(body, counter):
    # 第三根收阴但实体只占振幅的一小部分，价格在这根里来回拉扯，没有推进；恰好等于门槛不算。
    bars = [(100, 100, 98, 98), (98, 98, 96, 96), (96, 96.5, 95.5, 96 - body), (95.5, 95.5, 94, 94), (94, 94, 92, 92)]
    metrics = evaluate_pattern(candles(*bars), rule(min_candle_body_ratio=0.25))

    assert metrics is not None
    assert metrics.counter_candles == int(counter)


def test_stalled_tail_fails_although_every_candle_closes_down():
    # 三根大阴线跌了 15%，最后两根长下影线小阴线：整体实体占比仍过线，逐根看走势已经停滞。
    bars = [
        (100, 100.5, 95, 95.5),
        (95.5, 95.6, 90.5, 90.8),
        (90.8, 90.9, 86, 86.4),
        (86.4, 86.6, 82, 86),
        (86, 87.2, 85, 85.8),
    ]

    assert evaluate_pattern(candles(*bars), rule()).passed  # type: ignore[union-attr]
    metrics = evaluate_pattern(candles(*bars), rule(min_candle_body_ratio=0.3))
    assert metrics is not None
    assert metrics.counter_candles == 2
    assert metrics.failures == (Condition.COUNTER_CANDLES, Condition.COUNTER_EDGE)


def test_too_many_counter_candles_fail():
    bars = [(100, 100, 97, 97), (97, 97.5, 97, 97.4), (97.4, 97.4, 95, 95), (95, 95.3, 95, 95.2), (95.2, 95.2, 94, 94)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 2
    assert metrics.failures == (Condition.COUNTER_CANDLES,)
    assert evaluate_pattern(candles(*bars), rule(max_counter_candles=2)).passed  # type: ignore[union-attr]


def test_doji_counts_as_counter_candle():
    bars = [(100, 100, 98, 98), (98, 98.2, 97.8, 98), (98, 98, 96, 96), (96, 96.2, 95.8, 96), (96, 96, 94, 94)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 2
    assert Condition.COUNTER_CANDLES in metrics.failures


def test_large_counter_candle_fails_rebound_even_if_count_is_allowed():
    # 先跌 6%、一根大阳线反弹 4%、再跌到 -8%：反向 K 线只有一根，之后也创了新低，但反弹幅度太大。
    bars = [(100, 100, 97, 97), (97, 97, 94, 94), (94, 98, 94, 98), (98, 98, 93, 93), (93, 93, 92, 92)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 1
    assert metrics.failures == (Condition.REBOUND,)


@pytest.mark.parametrize(("next_close", "passed"), [(91.8, False), (91.0, False), (90.9, True)])
def test_candle_after_counter_must_break_close_before_it(next_close, passed):
    # 急跌两根后反弹一根：下一根只跌破反弹 K 线的收盘不够，要跌破反弹前的 91 才算重新推进，持平也不算。
    bars = [
        (100, 100, 95, 95),
        (95, 95, 91, 91),
        (91, 93, 91, 92.5),
        (92.5, 92.5, next_close, next_close),
        (next_close, next_close, 90.5, 90.6),
    ]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.failures == (() if passed else (Condition.COUNTER_RECOVERY,))


def test_consecutive_counter_candles_compare_with_close_before_the_run():
    # 连续两根反弹算一段，基准是这段之前的 96，而不是第二根反弹的收盘 96.5。
    def bars(next_close: float) -> list[Candle]:
        return candles(
            (100, 100, 96, 96),
            (96, 96.5, 96, 96.3),
            (96.3, 96.6, 96.3, 96.5),
            (96.5, 96.5, next_close, next_close),
            (next_close, next_close, 94, 94),
        )

    assert evaluate_pattern(bars(95.8), rule(max_counter_candles=2)).passed  # type: ignore[union-attr]
    metrics = evaluate_pattern(bars(96.1), rule(max_counter_candles=2))
    assert metrics is not None
    assert metrics.failures == (Condition.COUNTER_RECOVERY,)


def test_first_counter_candle_fails_edge():
    bars = [(100, 100.5, 100, 100.3), (100.3, 100.3, 98, 98), (98, 98, 96, 96), (96, 96, 94, 94), (94, 94, 93, 93)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 1
    assert metrics.failures == (Condition.COUNTER_EDGE,)


def test_last_doji_fails_edge():
    # 最后一根还在停顿，回调能否被收复未知，不算已形成的趋势。
    bars = [(100, 100, 98, 98), (98, 98, 96, 96), (96, 96, 94, 94), (94, 94, 93, 93), (93, 93.2, 92.8, 93)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 1
    assert metrics.failures == (Condition.COUNTER_EDGE,)


def test_rise_above_start_counts_as_rebound():
    # 起点之后先涨再跌：上涨的部分同样是逆向波动。
    bars = [(100, 103, 100, 103), (103, 103, 100, 100), (100, 100, 98, 98), (98, 98, 96, 96), (96, 96, 95, 95)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.rebound_ratio == pytest.approx(3 / 5)
    assert Condition.REBOUND in metrics.failures


def test_long_shadows_fail_body_ratio():
    bars = [(100 - i, 100 - i + 2, 100 - i - 3, 100 - i - 1) for i in range(5)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.body_ratio == pytest.approx(1 / 5)
    assert metrics.failures == (Condition.BODY,)


def test_one_dominant_candle_fails_single_candle_ratio():
    # 四根小阴线加一根大阴线：累计跌 6%，最后一根就占了 4 个点。
    bars = [(100, 100, 99.5, 99.5), (99.5, 99.5, 99, 99), (99, 99, 98.5, 98.5), (98.5, 98.5, 98, 98), (98, 98, 94, 94)]
    metrics = evaluate_pattern(candles(*bars), rule(max_single_candle_ratio=0.5))

    assert metrics is not None
    assert metrics.single_candle_ratio == pytest.approx(4 / 6)
    assert metrics.failures == (Condition.SINGLE_CANDLE,)
    assert evaluate_pattern(falling(), rule(max_single_candle_ratio=0.5)).passed  # type: ignore[union-attr]


def test_single_candle_ratio_ignores_counter_candles():
    # 反向 K 线不算“顺势推动”，只看顺势 K 线里最大的一根。
    bars = [(100, 100, 98, 98), (98, 98, 96, 96), (96, 96.8, 96, 96.6), (96.6, 96.6, 94, 94), (94, 94, 92, 92)]
    metrics = evaluate_pattern(candles(*bars), rule(max_single_candle_ratio=0.5))

    assert metrics is not None and metrics.passed
    assert metrics.single_candle_ratio == pytest.approx(2.6 / 8)


def test_small_move_fails_change_threshold():
    metrics = evaluate_pattern(falling(step=0.5), rule())

    assert metrics is not None
    assert metrics.failures == (Condition.CHANGE,)


def test_no_direction_or_wrong_length_returns_none():
    same = candles(*((100, 101, 99, 100) for _ in range(5)))

    assert evaluate_pattern(same, rule()) is None
    assert evaluate_pattern(falling(count=4), rule()) is None


def test_aggregate_candles_merges_aligned_groups():
    minutes = [Candle(BASE + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(6)]

    merged = aggregate_candles(minutes, 3)

    assert [(c.timestamp, c.open, c.high, c.low, c.close) for c in merged] == [
        (BASE, 100, 103, 99, 102.5),
        (BASE + timedelta(minutes=3), 103, 106, 102, 105.5),
    ]


def test_rule_rejects_invalid_values():
    with pytest.raises(ValueError):
        rule(candles=1)
    with pytest.raises(ValueError):
        rule(max_counter_candles=5)
    with pytest.raises(ValueError):
        rule(min_change_percent=0)
