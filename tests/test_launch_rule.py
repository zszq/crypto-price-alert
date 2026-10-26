from datetime import UTC, datetime, timedelta

import pytest

from price_alert.common.models import Candle
from price_alert.detectors.launch.rule import Condition, LaunchRule, evaluate_launch, still_launching

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def rule(**overrides) -> LaunchRule:
    values = {
        "direction": "surge",
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


def test_overshoot_is_rejected_unless_cap_disabled():
    # 3 分钟涨 20%：插针式拉盘，提醒时已在高点。
    candles = history([(105.0, 1000), (112.0, 1000), (120.0, 1000)])

    assert evaluate_launch(candles, rule()).failures == (Condition.OVERSHOOT,)
    assert evaluate_launch(candles, rule(max_change_percent=None)).passed
    # 不到上限照常通过。
    assert evaluate_launch(history([(105.0, 1000), (110.0, 1000), (114.0, 1000)]), rule()).passed


def test_each_direction_only_judges_its_own_moves_and_breaks_its_own_level():
    candles = history([(99.0, 1000), (97.5, 1000), (96.0, 1000)])

    # 上涨规则看到下跌：方向相反（自然也没有突破基准高点），判定里的方向仍是规则的方向。
    metrics = evaluate_launch(candles, rule())
    assert metrics.direction == "surge"
    assert metrics.failures == (Condition.DIRECTION, Condition.BREAKOUT)
    assert evaluate_launch(candles, rule(require_breakout=False)).failures == (Condition.DIRECTION,)

    falling = evaluate_launch(candles, rule(direction="drop"))
    assert falling.passed and falling.direction == "drop"
    assert falling.breakout_price == 99.5
    rising = history([(101.0, 1000), (102.5, 1000), (104.0, 1000)])
    assert evaluate_launch(rising, rule(direction="drop")).failures == (Condition.DIRECTION, Condition.BREAKOUT)


def test_silent_baseline_counts_as_ratio_satisfied():
    metrics = evaluate_launch(history([(101.0, 1000), (102.5, 1000), (104.0, 1000)], base_quote=0), rule())

    assert metrics.volume_ratio is None
    assert metrics.passed


def test_history_length_must_match_rule():
    with pytest.raises(ValueError, match="需要 13 根"):
        evaluate_launch(history([(101.0, 1000)]), rule())
    with pytest.raises(ValueError, match="baseline_minutes"):
        rule(baseline_minutes=3)
    with pytest.raises(ValueError, match="max_change_percent"):
        rule(max_change_percent=3.0)


def test_drop_overshoot_uses_the_absolute_change():
    # 3 分钟跌 20%：与上涨一样按绝对值算插针，不能因为涨跌幅为负就漏过上限。
    candles = history([(95.0, 1000), (88.0, 1000), (80.0, 1000)])

    assert evaluate_launch(candles, rule(direction="drop")).failures == (Condition.OVERSHOOT,)
    assert evaluate_launch(candles, rule(direction="drop", max_change_percent=None)).passed
    assert rule(direction="drop").overshoots(-20.0) and not rule(direction="drop").overshoots(-10.0)


def test_still_launching_rechecks_drops():
    drop = rule(direction="drop")
    metrics = evaluate_launch(history([(99.0, 1000), (97.5, 1000), (96.0, 1000)]), drop)

    # 起点 100：跌到 96.5 仍跌了 3.5%，且在基准低点 99.5 之下；回到 97.5 只跌 2.5%，已不算启动。
    assert still_launching(metrics, 96.5, drop)
    assert not still_launching(metrics, 97.5, drop)
    # 方向不能弄反：价格反而涨上去，对下跌提醒来说当然不再成立。
    assert not still_launching(metrics, 104.0, drop)
