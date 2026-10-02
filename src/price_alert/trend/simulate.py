"""合成行情验证趋势配置：每个启用的周期各跑一段标准的单边行情，正常时每个周期恰好产生一条提醒。"""

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
# 平稳段每根 K 线的振幅。
_FLAT_RANGE = 0.05


def simulate_trend(config: TrendConfig) -> list[TrendAlert]:
    alerts: list[TrendAlert] = []
    # 合成行情在趋势段结束前形态就可能已成立，延续提醒会让数量随周期参数浮动；这里只验证形态能否形成。
    formation_only = config.model_copy(update={"alert_continuing": False})
    for rule in config.rules():
        # 每个周期用只含它自己的检测器：要验证的是这个周期的门槛能否被满足，不能让别的周期先提醒掩盖问题。
        detector = build_trend_detector(formation_only, rules=[rule])
        alerts.extend(_simulate_rule(detector, rule))
    return alerts


def _simulate_rule(detector: TrendDetector, rule: TrendRule) -> list[TrendAlert]:
    period_seconds = rule.minutes * 60
    now = int(datetime.now(UTC).timestamp())
    # 趋势段恰好在最近的周期边界结束，周期 K 线全部对齐。
    trend_end = datetime.fromtimestamp(now - now % period_seconds, tz=UTC)
    trend_minutes = rule.candles * rule.minutes
    trend_start = trend_end - timedelta(minutes=trend_minutes)
    flat_start = trend_start - timedelta(minutes=trend_minutes)

    flat = [
        Candle(
            flat_start + timedelta(minutes=index), _PRICE, _PRICE + _FLAT_RANGE / 2, _PRICE - _FLAT_RANGE / 2, _PRICE
        )
        for index in range(trend_minutes)
    ]
    detector.add_symbol(SYMBOL, flat, 1_000_000_000)

    # 按门槛的 1.5 倍单边下跌，每分钟等幅走低且无影线：各项条件都取到最理想的值。
    step = _PRICE * rule.min_change_percent * 1.5 / 100 / trend_minutes
    alerts: list[TrendAlert] = []
    for index in range(trend_minutes):
        open_price = _PRICE - step * index
        close_price = open_price - step
        candle = Candle(trend_start + timedelta(minutes=index), open_price, open_price, close_price, close_price)
        alerts.extend(detector.add_candle(SYMBOL, candle))
    return alerts
