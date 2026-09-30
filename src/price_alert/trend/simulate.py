"""合成行情验证趋势配置：每个启用的周期各跑一段标准的单边行情，正常时每个周期恰好产生一条“开始”提醒。

趋势段越走越深，开始之后还可能按档位升级；只统计“开始”提醒，验证的是门槛能否被满足。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_trend_detector
from price_alert.models import Candle
from price_alert.trend.alerts import TrendAlert
from price_alert.trend.config import TrendConfig
from price_alert.trend.detector import TrendDetector
from price_alert.trend.pattern import TrendRule

SYMBOL = "BTC_USDT"
_PRICE = 100.0
# 平稳段每根 K 线的振幅与成交量；趋势段放量到门槛的两倍，确保量能条件一定满足。
_FLAT_RANGE = 0.05
_FLAT_VOLUME = 100.0


def simulate_trend(config: TrendConfig) -> list[TrendAlert]:
    alerts: list[TrendAlert] = []
    for rule in config.rules():
        # 每个周期用只含它自己的检测器：要验证的是这个周期的门槛能否被满足，不能让别的周期先提醒掩盖问题。
        detector = build_trend_detector(config, rules=[rule])
        alerts.extend(alert for alert in _simulate_rule(detector, rule) if alert.stage == "start")
    return alerts


def _simulate_rule(detector: TrendDetector, rule: TrendRule) -> list[TrendAlert]:
    period_seconds = rule.minutes * 60
    now = int(datetime.now(UTC).timestamp())
    # 趋势段恰好在最近的周期边界结束，周期 K 线全部对齐。
    trend_end = datetime.fromtimestamp(now - now % period_seconds, tz=UTC)
    trend_minutes = rule.candles * rule.minutes
    trend_start = trend_end - timedelta(minutes=trend_minutes)
    baseline_start = trend_start - timedelta(minutes=rule.baseline_candles * rule.minutes)

    flat = [
        Candle(
            baseline_start + timedelta(minutes=index),
            _PRICE,
            _PRICE + _FLAT_RANGE / 2,
            _PRICE - _FLAT_RANGE / 2,
            _PRICE,
            volume=_FLAT_VOLUME,
        )
        for index in range(rule.baseline_candles * rule.minutes)
    ]
    detector.add_symbol(SYMBOL, flat, 1_000_000_000)

    # 按门槛的 1.5 倍单边下跌，每分钟等幅走低且无影线：各项比例都取到最理想的值。
    target = max(rule.min_change_percent, (rule.min_change_atr or 0) * _FLAT_RANGE / _PRICE * 100) * 1.5
    step = _PRICE * target / 100 / trend_minutes
    volume = _FLAT_VOLUME * max(2.0, (rule.min_volume_ratio or 1) * 2)
    alerts: list[TrendAlert] = []
    for index in range(trend_minutes):
        open_price = _PRICE - step * index
        close_price = open_price - step
        timestamp = trend_start + timedelta(minutes=index)
        candle = Candle(timestamp, open_price, open_price, close_price, close_price, volume=volume)
        alerts.extend(detector.add_candle(SYMBOL, candle))
    return alerts
