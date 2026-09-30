from datetime import UTC, datetime, timedelta

import pytest

from price_alert.models import Candle
from price_alert.trend.pattern import (
    Condition,
    TrendRule,
    adverse_shadow_ratio,
    aggregate_candles,
    evaluate_pattern,
    format_multiple,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def rule(**overrides) -> TrendRule:
    values = {
        "period": "1m",
        "candles": 5,
        "min_change_percent": 4.0,
        "max_rebound_ratio": 0.3,
        "min_trend_candle_ratio": 0.7,
        "min_step_ratio": 0.7,
        "min_body_ratio": 0.5,
        "min_volume_ratio": None,
        "baseline_candles": 5,
    }
    return TrendRule(**(values | overrides))


def candles(*bars: tuple[float, float, float, float], volume: float = 100.0) -> list[Candle]:
    """按 (开, 高, 低, 收) 生成连续的 1 分钟 K 线。"""
    return [
        Candle(BASE + timedelta(minutes=index), o, h, low, c, volume=volume)
        for index, (o, h, low, c) in enumerate(bars)
    ]


def falling(start: float = 100.0, step: float = 1.0, count: int = 5, volume: float = 100.0) -> list[Candle]:
    # 无影线的等幅阴线：每一项比例都取到最理想的值。
    return candles(
        *((start - step * i, start - step * i, start - step * (i + 1), start - step * (i + 1)) for i in range(count)),
        volume=volume,
    )


def flat_baseline(count: int = 5, range_size: float = 0.2, volume: float = 100.0) -> list[Candle]:
    half = range_size / 2
    return candles(*((100.0, 100.0 + half, 100.0 - half, 100.0) for _ in range(count)), volume=volume)


def test_clean_drop_passes_with_ideal_ratios():
    metrics = evaluate_pattern(falling(), [], rule())

    assert metrics is not None and metrics.passed
    assert metrics.direction == "drop"
    assert metrics.change_percent == pytest.approx(-5.0)
    assert (metrics.trend_candle_ratio, metrics.step_ratio, metrics.rebound_ratio, metrics.body_ratio) == (1, 1, 0, 1)
    assert metrics.extreme_price == 95.0


def test_surge_is_mirror_of_drop():
    bars = [(100 + i, 101 + i, 100 + i, 101 + i) for i in range(5)]
    metrics = evaluate_pattern(candles(*bars), [], rule())

    assert metrics is not None and metrics.passed
    assert metrics.direction == "surge"
    assert metrics.extreme_price == 105


def test_one_small_counter_candle_is_tolerated():
    # 第三根小阳线，收盘没有收复多少跌幅。
    bars = [(100, 100, 98, 98), (98, 98, 96, 96), (96, 96.8, 96, 96.6), (96.6, 96.6, 94, 94), (94, 94, 92, 92)]
    metrics = evaluate_pattern(candles(*bars), [], rule())

    assert metrics is not None and metrics.passed
    assert metrics.trend_candle_ratio == pytest.approx(0.8)
    assert metrics.rebound_ratio == pytest.approx(0.6 / 8)


def test_large_rebound_fails_even_when_net_change_is_large():
    # 先跌 6%、反弹 4%、再跌到 -5%：净跌幅够，但中途反弹超过跌幅的 30%，说明不是持续趋势。
    bars = [(100, 100, 97, 97), (97, 97, 94, 94), (94, 98, 94, 98), (98, 98, 96, 96), (96, 96, 95, 95)]
    metrics = evaluate_pattern(candles(*bars), [], rule())

    assert metrics is not None
    assert Condition.REBOUND in metrics.failures


def test_rise_above_start_counts_as_rebound():
    # 起点之后先涨再跌：上涨的部分同样是逆向波动。
    bars = [(100, 103, 100, 103), (103, 103, 100, 100), (100, 100, 98, 98), (98, 98, 96, 96), (96, 96, 95, 95)]
    metrics = evaluate_pattern(candles(*bars), [], rule())

    assert metrics is not None
    assert metrics.rebound_ratio == pytest.approx(3 / 5)
    assert Condition.REBOUND in metrics.failures


def test_long_shadows_fail_body_ratio():
    bars = [(100 - i, 100 - i + 2, 100 - i - 3, 100 - i - 1) for i in range(5)]
    metrics = evaluate_pattern(candles(*bars), [], rule())

    assert metrics is not None
    assert metrics.body_ratio == pytest.approx(1 / 5)
    assert Condition.BODY in metrics.failures


def test_small_move_fails_change_threshold():
    metrics = evaluate_pattern(falling(step=0.5), [], rule())

    assert metrics is not None
    assert metrics.failures == (Condition.CHANGE,)


def test_mixed_candles_fail_ratio_conditions():
    bars = [(100, 100, 97, 97), (97, 98, 97, 98), (98, 98, 95, 95), (95, 96, 95, 96), (96, 96, 94, 94)]
    metrics = evaluate_pattern(candles(*bars), [], rule(max_rebound_ratio=1))

    assert metrics is not None
    assert metrics.trend_candle_ratio == pytest.approx(0.6)
    assert {Condition.TREND_CANDLES, Condition.STEPS} <= set(metrics.failures)


def test_no_direction_or_wrong_length_returns_none():
    same = candles(*((100, 101, 99, 100) for _ in range(5)))

    assert evaluate_pattern(same, [], rule()) is None
    assert evaluate_pattern(falling(count=4), [], rule()) is None


def test_volume_ratio_uses_baseline_median_so_one_spike_does_not_raise_the_bar():
    baseline = flat_baseline(volume=100.0)
    # 趋势前有一根巨量 K 线：用均值做基准会把门槛抬到 1000 以上。
    spiked = [*baseline[:-1], Candle(baseline[-1].timestamp, 100, 100.1, 99.9, 100, volume=5000)]
    metrics = evaluate_pattern(falling(volume=200.0), spiked, rule(min_volume_ratio=1.5))

    assert metrics is not None and metrics.passed
    assert metrics.volume_ratio == pytest.approx(2.0)


def test_low_volume_fails_volume_condition():
    metrics = evaluate_pattern(falling(volume=100.0), flat_baseline(volume=100.0), rule(min_volume_ratio=1.5))

    assert metrics is not None
    assert metrics.failures == (Condition.VOLUME,)


def test_missing_baseline_fails_only_when_volume_condition_is_enabled():
    short_baseline = flat_baseline(count=3)

    assert evaluate_pattern(falling(), short_baseline, rule()).passed  # type: ignore[union-attr]
    metrics = evaluate_pattern(falling(), short_baseline, rule(min_volume_ratio=1.0))
    assert metrics is not None
    assert metrics.failures == (Condition.BASELINE,)


def test_zero_baseline_volume_counts_as_passing():
    metrics = evaluate_pattern(falling(), flat_baseline(volume=0.0), rule(min_volume_ratio=1.5))

    assert metrics is not None and metrics.passed
    assert format_multiple(metrics.volume_ratio) == "∞"  # type: ignore[arg-type]
    assert format_multiple(1.25) == "1.2"


def test_sparse_baseline_falls_back_to_mean_volume():
    # 冷门合约过半的分钟无成交，中位数为 0；若直接用它，量能条件会恒成立。
    baseline = [
        Candle(c.timestamp, c.open, c.high, c.low, c.close, volume=500.0 if index == 2 else 0.0)
        for index, c in enumerate(flat_baseline())
    ]

    metrics = evaluate_pattern(falling(volume=150.0), baseline, rule(min_volume_ratio=1.5))

    # 基准成交量 [0, 0, 500, 0, 0]：中位数 0，均值 100。
    assert metrics is not None
    assert metrics.volume_ratio == pytest.approx(1.5)


def test_adverse_shadow_ratio_by_direction():
    candle = Candle(BASE, 100, 101, 95, 99)

    assert adverse_shadow_ratio(candle, "drop") == pytest.approx(4 / 6)
    assert adverse_shadow_ratio(candle, "surge") == pytest.approx(1 / 6)
    assert adverse_shadow_ratio(Candle(BASE, 100, 100, 100, 100), "drop") == 0


def test_aggregate_candles_merges_aligned_groups():
    minutes = [Candle(BASE + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 10, 1) for i in range(6)]

    merged = aggregate_candles(minutes, 3)

    assert [(c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in merged] == [
        (BASE, 100, 103, 99, 102.5, 3),
        (BASE + timedelta(minutes=3), 103, 106, 102, 105.5, 3),
    ]


def test_rule_rejects_invalid_values():
    with pytest.raises(ValueError):
        rule(candles=1)
    with pytest.raises(ValueError):
        rule(min_change_percent=0)
