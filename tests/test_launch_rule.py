from datetime import UTC, datetime, timedelta

import pytest

from price_alert.launch.rule import Condition, LaunchRule, evaluate_launch
from price_alert.models import Candle

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def rule(**overrides) -> LaunchRule:
    values = {
        "window_minutes": 3,
        "baseline_minutes": 10,
        "min_volume_ratio": 5.0,
        "min_window_quote": 3_000.0,
        "min_change_percent": 3.0,
    }
    return LaunchRule(**(values | overrides))


def history(window: list[tuple[float, float]], base_quote: float = 100.0, base_price: float = 100.0) -> list[Candle]:
    """10 根横盘基准（振幅 ±0.5）加窗口里逐分钟的（收盘价, 成交额）。"""
    candles = [
        Candle(BASE + timedelta(minutes=i), base_price, base_price + 0.5, base_price - 0.5, base_price, base_quote)
        for i in range(10)
    ]
    previous = base_price
    for offset, (close, quote) in enumerate(window):
        stamp = BASE + timedelta(minutes=10 + offset)
        candles.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return candles


def test_volume_burst_with_breakout_passes():
    # 基准每分钟 100 USDT，3 分钟均量 300；窗口 3000 USDT 是 10 倍，涨 4% 突破基准高点 100.5。
    metrics = evaluate_launch(history([(101.0, 1000), (102.5, 1000), (104.0, 1000)]), rule())

    assert metrics.passed
    assert metrics.direction == "surge"
    assert metrics.change_percent == pytest.approx(4.0)
    assert (metrics.window_quote, metrics.baseline_quote) == (3000, 300)
    assert metrics.volume_ratio == pytest.approx(10)
    assert metrics.breakout_price == 100.5
    assert metrics.broke_out


def test_each_threshold_is_reported_separately():
    # 成交额够、量比不够：基准太活跃。
    busy = evaluate_launch(history([(101.0, 1000), (102.5, 1000), (104.0, 1000)], base_quote=500), rule())
    assert busy.failures == (Condition.RATIO,)

    # 量比够、绝对成交额不够：冷门币几百 USDT 也能放大几十倍。
    thin = evaluate_launch(history([(101.0, 100), (102.5, 100), (104.0, 100)], base_quote=1), rule())
    assert thin.failures == (Condition.VOLUME,)

    small = evaluate_launch(history([(100.6, 1000), (100.8, 1000), (101.0, 1000)]), rule())
    assert small.failures == (Condition.CHANGE,)


def test_breakout_uses_close_and_can_be_disabled():
    candles = history([(101.0, 1000), (102.5, 1000), (104.0, 1000)])
    # 基准期里有一根冲到 110 的影线：收盘 104 没越过它，不算离开区间。
    candles[4] = Candle(candles[4].timestamp, 100, 110, 99.5, 100, 100)

    assert evaluate_launch(candles, rule()).failures == (Condition.BREAKOUT,)
    assert evaluate_launch(candles, rule(require_breakout=False)).passed


def test_drop_needs_alert_drops_and_breaks_the_low():
    candles = history([(99.0, 1000), (97.5, 1000), (96.0, 1000)])

    metrics = evaluate_launch(candles, rule())
    assert metrics.direction == "drop"
    assert metrics.failures == (Condition.DIRECTION,)

    allowed = evaluate_launch(candles, rule(alert_drops=True))
    assert allowed.passed
    assert allowed.breakout_price == 99.5


def test_silent_baseline_counts_as_ratio_satisfied():
    metrics = evaluate_launch(history([(101.0, 1000), (102.5, 1000), (104.0, 1000)], base_quote=0), rule())

    assert metrics.volume_ratio is None
    assert metrics.passed


def test_history_length_must_match_rule():
    with pytest.raises(ValueError, match="需要 13 根"):
        evaluate_launch(history([(101.0, 1000)]), rule())
    with pytest.raises(ValueError, match="baseline_minutes"):
        rule(baseline_minutes=3)
