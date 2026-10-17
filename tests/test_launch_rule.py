from datetime import UTC, datetime, timedelta

import pytest

from price_alert.launch.rule import Condition, LaunchRule, Thresholds, evaluate_launch, still_launching
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


def test_overshoot_is_rejected_unless_cap_disabled():
    # 3 分钟涨 20%：插针式拉盘，提醒时已在高点。
    candles = history([(105.0, 1000), (112.0, 1000), (120.0, 1000)])

    assert evaluate_launch(candles, rule()).failures == (Condition.OVERSHOOT,)
    assert evaluate_launch(candles, rule(max_change_percent=None)).passed
    # 不到上限照常通过。
    assert evaluate_launch(history([(105.0, 1000), (110.0, 1000), (114.0, 1000)]), rule()).passed


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
    with pytest.raises(ValueError, match="max_change_percent"):
        rule(max_change_percent=3.0)


def test_drop_thresholds_apply_only_to_drops():
    # 3 分钟跌 2.5%、量比 10：共用门槛（下限 3%）不够；下跌单独把下限降到 2% 才提醒，同样幅度的上涨仍不够。
    falling = history([(99.0, 1000), (98.0, 1000), (97.5, 1000)])
    rising = history([(101.0, 1000), (102.0, 1000), (102.5, 1000)])
    split = rule(alert_drops=True, drop=Thresholds(5.0, 2.0, 10.0))

    assert evaluate_launch(falling, rule(alert_drops=True)).failures == (Condition.CHANGE,)
    assert evaluate_launch(falling, split).passed
    assert evaluate_launch(rising, split).failures == (Condition.CHANGE,)
    # 量比按方向取：下跌要求 12 倍时 10 倍不够，上涨照旧按 5 倍判。
    strict = rule(alert_drops=True, drop=Thresholds(12.0, 2.0, 10.0))
    assert evaluate_launch(falling, strict).failures == (Condition.RATIO,)


def test_drop_cap_is_separate_from_surge_cap():
    # 跌 12% 超过下跌上限 10% 算插针；涨 12% 仍在上涨上限 15% 以内。
    split = rule(alert_drops=True, drop=Thresholds(5.0, 2.0, 10.0))
    falling = history([(96.0, 1000), (92.0, 1000), (88.0, 1000)])
    rising = history([(104.0, 1000), (108.0, 1000), (112.0, 1000)])

    assert evaluate_launch(falling, split).failures == (Condition.OVERSHOOT,)
    assert evaluate_launch(rising, split).passed
    assert split.overshoots(-12.0) and not split.overshoots(12.0)
    assert split.thresholds("surge") == Thresholds(5.0, 3.0, 15.0)
    assert rule().thresholds("drop") == rule().thresholds("surge")


def test_still_launching_uses_the_drop_threshold():
    split = rule(alert_drops=True, drop=Thresholds(5.0, 2.0, 10.0))
    metrics = evaluate_launch(history([(99.0, 1000), (98.0, 1000), (97.5, 1000)]), split)

    # 起点 100，跌到 97.9 仍超过下跌下限 2%；回到 98.5 只跌 1.5%，按下跌门槛已不算启动。
    assert still_launching(metrics, 97.9, split)
    assert not still_launching(metrics, 98.5, split)


def test_thresholds_validate_their_own_range():
    with pytest.raises(ValueError, match="max_change_percent"):
        Thresholds(3.0, 5.0, 5.0)
    assert not Thresholds(3.0, 2.0, None).overshoots(-50.0)


def test_drop_overshoot_uses_the_absolute_change_with_shared_thresholds():
    # 3 分钟跌 20%：与上涨一样按绝对值算插针，不能因为涨跌幅为负就漏过上限。
    candles = history([(95.0, 1000), (88.0, 1000), (80.0, 1000)])

    assert evaluate_launch(candles, rule(alert_drops=True)).failures == (Condition.OVERSHOOT,)
    assert evaluate_launch(candles, rule(alert_drops=True, max_change_percent=None)).passed
    assert rule().overshoots(-20.0) and not rule().overshoots(-10.0)


def test_still_launching_rechecks_drops_with_shared_thresholds():
    metrics = evaluate_launch(history([(99.0, 1000), (97.5, 1000), (96.0, 1000)]), rule(alert_drops=True))

    # 起点 100：跌到 96.5 仍跌了 3.5%，且在基准低点 99.5 之下；回到 97.5 只跌 2.5%，已不算启动。
    assert still_launching(metrics, 96.5, rule(alert_drops=True))
    assert not still_launching(metrics, 97.5, rule(alert_drops=True))
    # 方向不能弄反：价格反而涨上去，对下跌提醒来说当然不再成立。
    assert not still_launching(metrics, 104.0, rule(alert_drops=True))
