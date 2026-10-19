"""合成行情验证放量启动配置：每个启用的方向各跑一段平稳基准期之后放量上涨（下跌）的行情，正常时各恰好产生一条提醒。"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_launch_detector
from price_alert.launch.alerts import LaunchAlert
from price_alert.launch.config import LaunchConfig
from price_alert.launch.rule import LaunchRule
from price_alert.models import Candle

SYMBOL = "BTC_USDT"
_PRICE = 100.0
# 基准期每分钟的成交额；基准期价格一字横盘，再小的涨跌幅门槛也能突破基准期高低点。
_BASE_QUOTE = 1_000.0


def simulate_launch(config: LaunchConfig) -> list[LaunchAlert]:
    alerts: list[LaunchAlert] = []
    for rule in config.rules():
        alerts.extend(_simulate_side(rule))
    return alerts


def _simulate_side(configured: LaunchRule) -> list[LaunchAlert]:
    # 拉升过程中条件可能提前满足，冷却覆盖整段行情，只验证门槛能否被满足，数量不随参数浮动。
    rule = dataclasses.replace(configured, cooldown_minutes=configured.baseline_minutes)
    detector = build_launch_detector(rule)
    now = int(datetime.now(UTC).timestamp())
    launch_start = datetime.fromtimestamp(now - now % 60, tz=UTC) - timedelta(minutes=rule.window_minutes)
    baseline_start = launch_start - timedelta(minutes=rule.baseline_minutes)
    flat = [
        Candle(baseline_start + timedelta(minutes=index), _PRICE, _PRICE, _PRICE, _PRICE, _BASE_QUOTE)
        for index in range(rule.baseline_minutes)
    ]
    detector.add_symbol(SYMBOL, flat, 1_000_000_000, quanto_multiplier=1.0)

    # 成交额与涨跌幅都取门槛的 1.5 倍（不超过上下限的中点），平均分到窗口的每一分钟，逐分钟单边移动且无影线。
    window_quote = max(rule.min_window_quote, rule.min_volume_ratio * _BASE_QUOTE * rule.window_minutes) * 1.5
    change = rule.min_change_percent * 1.5
    if rule.max_change_percent is not None:
        change = min(change, (rule.min_change_percent + rule.max_change_percent) / 2)
    if rule.sign < 0:
        # 跌幅不能到 100%，价格会变成 0 或负数。
        change = min(change, (rule.min_change_percent + 100) / 2)
    step = rule.sign * _PRICE * change / 100 / rule.window_minutes
    alerts: list[LaunchAlert] = []
    for index in range(rule.window_minutes):
        open_price = _PRICE + step * index
        close_price = open_price + step
        candle = Candle(
            launch_start + timedelta(minutes=index),
            open_price,
            max(open_price, close_price),
            min(open_price, close_price),
            close_price,
            window_quote / rule.window_minutes,
        )
        alerts.extend(detector.add_candle(SYMBOL, candle))
    return alerts
