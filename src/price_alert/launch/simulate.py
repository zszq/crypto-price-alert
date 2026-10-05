"""合成行情验证放量启动配置：平稳的基准期之后放量拉升一段，正常时恰好产生一条提醒。"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_launch_detector
from price_alert.launch.alerts import LaunchAlert
from price_alert.launch.config import LaunchConfig
from price_alert.models import Candle

SYMBOL = "BTC_USDT"
_PRICE = 100.0
# 基准期每分钟的成交额；基准期价格一字横盘，再小的涨幅门槛也能突破基准期高点。
_BASE_QUOTE = 1_000.0


def simulate_launch(config: LaunchConfig) -> list[LaunchAlert]:
    # 拉升过程中条件可能提前满足，冷却覆盖整段行情，只验证门槛能否被满足，数量不随参数浮动。
    rule = dataclasses.replace(config.to_rule(), cooldown_minutes=config.baseline_minutes)
    detector = build_launch_detector(config, rule=rule)
    now = int(datetime.now(UTC).timestamp())
    launch_start = datetime.fromtimestamp(now - now % 60, tz=UTC) - timedelta(minutes=rule.window_minutes)
    baseline_start = launch_start - timedelta(minutes=rule.baseline_minutes)
    flat = [
        Candle(baseline_start + timedelta(minutes=index), _PRICE, _PRICE, _PRICE, _PRICE, _BASE_QUOTE)
        for index in range(rule.baseline_minutes)
    ]
    detector.add_symbol(SYMBOL, flat, 1_000_000_000, quanto_multiplier=1.0)

    # 成交额与涨幅都取门槛的 1.5 倍，平均分到窗口的每一分钟，逐分钟走高且无影线。
    window_quote = max(rule.min_window_quote, rule.min_volume_ratio * _BASE_QUOTE * rule.window_minutes) * 1.5
    step = _PRICE * rule.min_change_percent * 1.5 / 100 / rule.window_minutes
    alerts: list[LaunchAlert] = []
    for index in range(rule.window_minutes):
        open_price = _PRICE + step * index
        close_price = open_price + step
        candle = Candle(
            launch_start + timedelta(minutes=index),
            open_price,
            close_price,
            open_price,
            close_price,
            window_quote / rule.window_minutes,
        )
        alerts.extend(detector.add_candle(SYMBOL, candle))
    return alerts
