import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from price_alert.common.models import Candle
from price_alert.detectors.dormant.rule import (
    CALM_MINUTES,
    PRIOR_MINUTES,
    Condition,
    DormantMetrics,
    DormantRule,
    evaluate_dormant,
    minute_range_percent,
    still_rising,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
RULE_VALUES = {
    "window_minutes": 4,
    "max_prior_quote": 10_000,
    "min_window_quote": 1_000,
    "min_change_percent": 8,
    "min_active_minutes": 3,
}
RULE = DormantRule(**RULE_VALUES)


def evaluate(candles: list[Candle], rule: DormantRule = RULE) -> DormantMetrics:
    # 与检测器同样的取法：窗口之前那一根的收盘价是起点，之前 24 小时的成交额合计与涨前波动单独传入。
    window = candles[-rule.window_minutes :]
    prior = candles[-rule.window_minutes - PRIOR_MINUTES : -rule.window_minutes]
    calm = [
        minute_range_percent(candle, previous.close)
        for previous, candle in zip(prior[-CALM_MINUTES - 1 : -1], prior[-CALM_MINUTES:], strict=True)
    ]
    return evaluate_dormant(
        window, prior[-1].close, sum(candle.quote_volume for candle in prior), sum(calm) / CALM_MINUTES, rule
    )


def history(
    prior_quote: float = 5.0, window: list[tuple[float, float]] | None = None, swing: float = 0.0
) -> list[Candle]:
    """之前 24 小时每分钟成交 prior_quote，收盘在 1.0 与 1.0 + swing 之间来回（最后一根是 1.0）；
    窗口按 (收盘价, 成交额) 逐分钟给出。"""
    candles = []
    for i in range(PRIOR_MINUTES):
        price = 1.0 + swing if (PRIOR_MINUTES - i) % 2 == 0 else 1.0
        candles.append(Candle(BASE + timedelta(minutes=i), price, price, price, price, prior_quote))
    previous = 1.0
    for offset, (close, quote) in enumerate(window or [(1.03, 400), (1.06, 400), (1.09, 400), (1.10, 400)]):
        stamp = BASE + timedelta(minutes=PRIOR_MINUTES + offset)
        candles.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return candles


def test_passes_when_dormant_coin_rallies_with_steady_trading():
    metrics = evaluate(history(), RULE)

    assert metrics.passed
    assert (metrics.start_price, metrics.end_price) == (1.0, 1.10)
    assert metrics.change_percent == pytest.approx(10.0)
    assert (metrics.window_quote, metrics.active_minutes) == (1600, 4)
    assert metrics.prior_quote == pytest.approx(5.0 * PRIOR_MINUTES)
    # 涨前一字横盘：波动为 0，倍数无穷大；单分钟最大涨幅是第一分钟的 3%。
    assert (metrics.pre_volatility_percent, metrics.calm_multiple) == (0.0, math.inf)
    assert metrics.max_minute_change_percent == pytest.approx(3.0)
    # 逐分钟单边上涨、没有影线。
    assert metrics.body_ratio == pytest.approx(1.0)
    # 4 分钟窗口的后半是最后 2 分钟。
    assert (metrics.last_minute_quote, metrics.late_quote_share) == (400, 0.5)


def test_volume_fading_at_the_end_fails_last_minute_and_late_share():
    # 成交集中在前两分钟，拉到后面买盘跟不上：最后一分钟 150，后 2 分钟占 400 / 1600 = 25%。
    fading = [(1.03, 700), (1.06, 500), (1.09, 250), (1.10, 150)]
    strict = DormantRule(**{**RULE_VALUES, "min_last_minute_quote": 200, "min_late_quote_share": 0.4})
    metrics = evaluate(history(window=fading), strict)

    assert (metrics.last_minute_quote, metrics.late_quote_share) == (150, 0.25)
    assert metrics.failures == (Condition.LAST, Condition.LATE)
    # 代码默认两项都不限。
    assert evaluate(history(window=fading), RULE).passed
    # 奇数窗口的后半含中间那根：5 分钟窗口看最后 3 分钟。
    odd = DormantRule(**{**RULE_VALUES, "window_minutes": 5, "min_late_quote_share": 0.6})
    rising = [(1.02, 100), (1.04, 300), (1.06, 300), (1.08, 300), (1.10, 300)]
    assert evaluate(history(window=rising), odd).late_quote_share == pytest.approx(900 / 1300)


def test_each_threshold_reports_its_own_failure():
    # 之前 24 小时成交额 = 7 × 1440 > 1 万：已经不冷门。
    assert evaluate(history(prior_quote=7.0), RULE).failures == (Condition.PRIOR,)
    thin = [(1.03, 200), (1.06, 200), (1.09, 200), (1.10, 200)]
    assert evaluate(history(window=thin), RULE).failures == (Condition.VOLUME,)
    slow = [(1.02, 400), (1.04, 400), (1.06, 400), (1.07, 400)]
    assert evaluate(history(window=slow), RULE).failures == (Condition.CHANGE,)
    # 两笔大单撑起成交额与涨幅，只有 2 分钟有成交。
    burst = [(1.0, 0), (1.0, 0), (1.04, 800), (1.09, 800)]
    assert evaluate(history(window=burst), RULE).failures == (Condition.ACTIVE,)


def test_choppy_coin_fails_on_calm_multiple_and_volatility_cap():
    # 涨前每分钟在 1.0 与 1.02 之间来回，平均波幅约 1.98%，涨 10% 只有约 5 倍，也超过 0.5% 的上限。
    metrics = evaluate(history(swing=0.02), RULE)

    assert metrics.pre_volatility_percent == pytest.approx((2.0 + 2 / 1.02) / 2)
    assert metrics.calm_multiple == pytest.approx(10.0 / metrics.pre_volatility_percent)
    assert metrics.failures == (Condition.CALM, Condition.VOLATILE)
    relaxed = DormantRule(**{**RULE_VALUES, "min_calm_multiple": 5.0, "max_pre_volatility_percent": None})
    assert evaluate(history(swing=0.02), relaxed).passed


def test_volatility_cap_blocks_choppy_coin_even_when_multiple_is_enough():
    # 涨前在 1.0 与 1.006 之间来回，平均波幅约 0.6%：涨 10% 已有约 17 倍，只因涨前不够平稳不报。
    metrics = evaluate(history(swing=0.006), RULE)

    assert metrics.calm_multiple > RULE.min_calm_multiple
    assert metrics.failures == (Condition.VOLATILE,)
    assert evaluate(history(swing=0.006), DormantRule(**{**RULE_VALUES, "max_pre_volatility_percent": 0.7})).passed


def test_single_minute_jump_fails():
    # 前两分钟几乎不动，第三分钟一下跳 7.8%。
    jump = [(1.005, 400), (1.01, 400), (1.09, 400), (1.10, 400)]
    metrics = evaluate(history(window=jump), RULE)

    assert metrics.max_minute_change_percent == pytest.approx((1.09 / 1.01 - 1) * 100)
    assert metrics.failures == (Condition.JUMP,)


def test_long_upper_shadows_fail_on_body_ratio():
    # 收盘与上一个用例相同，只是每分钟都冲高 4% 再被砸回：实体 3%/3%/3%/1%，振幅各多 4%，总实体占比约 0.38。
    candles = history()
    for index in range(-4, 0):
        candle = candles[index]
        candles[index] = replace(candle, high=candle.close + 0.04)
    metrics = evaluate(candles, RULE)

    assert metrics.body_ratio == pytest.approx(0.10 / 0.26)
    assert metrics.failures == (Condition.BODY,)
    assert evaluate(candles, DormantRule(**{**RULE_VALUES, "min_body_ratio": 0.0})).passed


def test_true_range_counts_gaps_and_flat_minutes_as_zero():
    assert minute_range_percent(Candle(BASE, 1.0, 1.0, 1.0, 1.0), 1.0) == 0.0
    # 跳空：前收 1.0，这一分钟在 1.02~1.03 之间，波幅按 1.0~1.03 算。
    assert minute_range_percent(Candle(BASE, 1.02, 1.03, 1.02, 1.025), 1.0) == pytest.approx(3.0)


def test_falling_window_fails_on_change():
    dump = [(0.97, 400), (0.94, 400), (0.91, 400), (0.90, 400)]
    assert Condition.CHANGE in evaluate(history(window=dump), RULE).failures


def test_requires_exact_window_length():
    with pytest.raises(ValueError, match="窗口需要 4 根 K 线"):
        evaluate_dormant(history()[-3:], 1.0, 0.0, 0.0, RULE)


def test_rule_validates_active_minutes_against_window():
    with pytest.raises(ValueError, match="min_active_minutes"):
        DormantRule(window_minutes=3, min_active_minutes=4)
    assert DormantRule(window_minutes=10).history_minutes == PRIOR_MINUTES + 10


def test_still_rising_uses_the_same_change_threshold():
    metrics = evaluate(history(), RULE)
    assert still_rising(metrics, 1.08, RULE)
    assert not still_rising(metrics, 1.079, RULE)


def test_efficiency_volume_ratio_and_retrace_for_the_slow_group():
    # 第二分钟回落再拉起：路程 0.03 + 0.02 + 0.05 + 0.04 = 0.14，净涨 0.10。
    zigzag = [(1.03, 400), (1.01, 400), (1.06, 400), (1.10, 400)]
    metrics = evaluate(history(window=zigzag), RULE)

    assert metrics.efficiency == pytest.approx(0.10 / 0.14)
    # 之前 24 小时每分钟 5 USDT，4 分钟均值 20，窗口 1600 即 80 倍；收在最高价，没有回落。
    assert (metrics.volume_ratio, metrics.retrace) == (pytest.approx(80.0), 0.0)
    # 代码默认三项都不限。
    assert metrics.passed
    strict = DormantRule(**{**RULE_VALUES, "min_efficiency": 0.8, "min_volume_ratio": 100.0, "max_retrace": 0.3})
    assert evaluate(history(window=zigzag), strict).failures == (Condition.EFFICIENCY, Condition.RATIO)

    # 最后一分钟冲到 1.15 再回到 1.10：回落 0.05 占窗口最大涨幅 0.15 的三分之一。
    candles = history(window=zigzag)
    candles[-1] = replace(candles[-1], high=1.15)
    assert evaluate(candles, RULE).retrace == pytest.approx(1 / 3)
    assert evaluate(candles, strict).failures == (Condition.EFFICIENCY, Condition.RATIO, Condition.RETRACE)


def test_volume_ratio_is_infinite_without_prior_trading():
    metrics = evaluate(history(prior_quote=0.0), DormantRule(**{**RULE_VALUES, "min_volume_ratio": 1e9}))

    assert metrics.volume_ratio == math.inf
    assert metrics.passed
