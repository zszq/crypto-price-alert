"""合成行情验证秒级异动配置：每个启用的观察窗口各跑一段平稳后跳变的行情，正常时每个窗口恰好产生一条提醒。"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_detector, build_windows
from price_alert.common.models import INTERVAL_SECONDS, Candle, PriceTick
from price_alert.detectors.move.alerts import PriceAlert
from price_alert.detectors.move.config import MoveConfig
from price_alert.detectors.move.streak import StreakAlert
from price_alert.detectors.move.windows import MoveWindow

SYMBOL = "BTC_USDT"
DEMO_ATR = 2.0


def simulate_move(config: MoveConfig) -> list[PriceAlert | StreakAlert]:
    alerts: list[PriceAlert | StreakAlert] = []
    for window in build_windows(config, cooldown_seconds=0):
        alerts.extend(_simulate_window(config, window))
    return alerts


def _build_demo_candles(end: datetime, count: int, interval_seconds: int) -> list[Candle]:
    # 每根 K 线高低差恰好为 DEMO_ATR 且收盘不跳空，种子 ATR 因此确定等于 DEMO_ATR。
    half_range = DEMO_ATR / 2
    return [
        Candle(
            end - timedelta(seconds=interval_seconds * (count - index)),
            100.0,
            100.0 + half_range,
            100.0 - half_range,
            100.0,
        )
        for index in range(count)
    ]


def _simulate_window(config: MoveConfig, window: MoveWindow) -> list[PriceAlert | StreakAlert]:
    interval_seconds = INTERVAL_SECONDS[config.candle_interval]
    # 每个窗口用只含它自己的检测器：要验证的是这个窗口的门槛能否被满足，不能让另一个窗口先提醒掩盖问题。
    detector = build_detector(config, windows=[window], streak=False)

    # 让异动恰好从 K 线边界开始：异动秒全部落在同一根未收盘 K 线里，不会被提前计入 ATR 而削弱倍数，
    # 同时之前的平稳 K 线按时收盘，ATR 始终新鲜，结果不随 lookback 等参数变化而失效。
    now = datetime.now(UTC)
    move_start = datetime.fromtimestamp(int(now.timestamp()) // interval_seconds * interval_seconds, tz=UTC)
    flat_start = move_start - timedelta(seconds=window.lookback_seconds + 1)
    history_end = datetime.fromtimestamp(
        int(flat_start.timestamp()) // interval_seconds * interval_seconds,
        tz=UTC,
    )
    detector.add_symbol(
        SYMBOL,
        _build_demo_candles(history_end, config.atr_period + 1, interval_seconds),
        1_000_000_000,
    )

    # 每秒成交笔数要足以让一个观察窗口满足 min_window_trades。
    trades_per_second = max(1, math.ceil(window.min_window_trades / window.lookback_seconds))

    def ticks_at(second: datetime, price: float, label: str) -> list[PriceTick]:
        return [
            PriceTick(SYMBOL, price, 1.0, second + timedelta(milliseconds=index), f"{label}-{index}")
            for index in range(trades_per_second)
        ]

    for offset in range(window.lookback_seconds + 1):
        for tick in ticks_at(flat_start + timedelta(seconds=offset), 100.0, f"flat-{offset}"):
            detector.add_tick(tick)

    # 平稳段跨过的 K 线只会让 ATR 衰减，按 DEMO_ATR 计算的位移因此总能满足 ATR 倍数。
    moved_price = 100.0 + max(DEMO_ATR * window.trigger_atr_multiple, window.min_change_percent) * 1.1
    alerts: list[PriceAlert | StreakAlert] = []
    # 多送一秒用于结算最后一个确认桶，与真实行情的完整秒检测保持一致。
    for offset in range(window.confirmation_seconds + 1):
        for tick in ticks_at(move_start + timedelta(seconds=offset), moved_price, f"demo-{offset}"):
            alerts.extend(detector.add_tick(tick))
    return alerts
