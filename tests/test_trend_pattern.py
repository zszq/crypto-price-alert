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
    # 先跌 6%、一根大阳线反弹 4%、再跌到 -5%：反向 K 线只有一根，但它收复了大半跌幅。
    bars = [(100, 100, 97, 97), (97, 97, 94, 94), (94, 98, 94, 98), (98, 98, 96, 96), (96, 96, 95, 95)]
    metrics = evaluate_pattern(candles(*bars), rule())

    assert metrics is not None
    assert metrics.counter_candles == 1
    assert metrics.failures == (Condition.REBOUND,)


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
