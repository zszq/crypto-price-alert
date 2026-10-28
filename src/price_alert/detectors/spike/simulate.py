"""合成行情验证瞬时异动、短时异动配置：每个启用的观察窗口各跑一段平稳后跳变的行情，正常时每个窗口恰好产生一条提醒。

开了前置走势或成交暴增的窗口改跑「平稳 1 小时 → 一段下跌 → 成交涌入 → 急拉」，与反弹回落的模拟同一形态。
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_detector, build_windows
from price_alert.common.models import INTERVAL_SECONDS, Candle, PriceTick
from price_alert.detectors.spike.alerts import PriceAlert
from price_alert.detectors.spike.config import AtrSpikeConfig
from price_alert.detectors.spike.detector import AtrSpikeDetector
from price_alert.detectors.spike.streak import StreakAlert
from price_alert.detectors.spike.windows import BURST_BASE_SECONDS, BURST_SECONDS, MAX_BURST_RATIO, SpikeWindow

SYMBOL = "BTC_USDT"
DEMO_ATR = 2.0
_PRICE = 100.0
# 急拉前成交开始涌入的秒数；加上最多 10 秒的确认仍在 1 分钟之内。
BUSY_LEAD_SECONDS = 50


def burst_ratio(busy: int, busy_seconds: int) -> float:
    """最近 1 分钟有 busy_seconds 秒每秒 busy 笔、其余每秒 1 笔时，成交暴增倍数。"""
    minute = busy_seconds * busy + (BURST_SECONDS - busy_seconds)
    hour = busy_seconds * busy + (BURST_BASE_SECONDS - busy_seconds)
    return minute / hour * MAX_BURST_RATIO


def simulate_spike(config: AtrSpikeConfig) -> list[PriceAlert | StreakAlert]:
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


def _simulate_window(config: AtrSpikeConfig, window: SpikeWindow) -> list[PriceAlert | StreakAlert]:
    interval_seconds = INTERVAL_SECONDS[config.candle_interval]
    # 每个窗口用只含它自己的检测器：要验证的是这个窗口的门槛能否被满足，不能让另一个窗口先提醒掩盖问题。
    detector = build_detector(config, windows=[window], streak=False)
    if window.history_seconds > 0:
        return _simulate_gated_window(config, window, detector)

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


def _simulate_gated_window(
    config: AtrSpikeConfig, window: SpikeWindow, detector: AtrSpikeDetector
) -> list[PriceAlert | StreakAlert]:
    interval_seconds = INTERVAL_SECONDS[config.candle_interval]
    now = int(datetime.now(UTC).timestamp())
    # 急拉从 K 线边界开始：急拉的秒都落在同一根未收盘 K 线里，不会提前计入 ATR 削弱倍数。
    move_start = datetime.fromtimestamp(now - now % interval_seconds, tz=UTC)
    # 下跌在窗口起点前跌完，窗口内停在低位，基准价就是低位。
    low_start = move_start - timedelta(seconds=window.lookback_seconds + 2)
    busy_start = move_start - timedelta(seconds=BUSY_LEAD_SECONDS)
    # 前置走势的起点是提醒那一秒往前推窗口加回看秒数；下跌从最晚的起点开始，量到的就是整段跌幅（门槛的 1.5 倍）。
    decline_start = move_start - timedelta(
        seconds=window.lookback_seconds + window.prior_seconds - window.confirmation_seconds
    )
    drop = min((window.min_prior_move_percent or 0.0) * 1.5, 90.0)
    low = _PRICE * (1 - drop / 100)
    # 平稳段保证 1 小时的成交笔数基准。
    flat_start = decline_start - timedelta(seconds=BURST_BASE_SECONDS + 120)
    history_end = datetime.fromtimestamp(
        int(flat_start.timestamp()) // interval_seconds * interval_seconds, tz=UTC
    )
    count = config.atr_period + 1
    candles = [
        Candle(
            history_end - timedelta(seconds=interval_seconds * (count - index)),
            _PRICE,
            _PRICE + 0.05,
            _PRICE - 0.05,
            _PRICE,
        )
        for index in range(count)
    ]
    detector.add_symbol(SYMBOL, candles, 1_000_000_000)

    # 提醒那一秒的倍数达到门槛的 1.2 倍（门槛小于理论上限由配置保证，取二者之间的值）。
    ratio = window.min_trade_burst_ratio or 1.0
    target = min(ratio * 1.2, (ratio + MAX_BURST_RATIO) / 2)
    busy_seconds = BUSY_LEAD_SECONDS + window.confirmation_seconds
    busy = 1
    while burst_ratio(busy, busy_seconds) < target:
        busy *= 2
    # 窗口里落在涌入段的秒数有限，每秒笔数还要让窗口笔数够。
    busy = max(busy, window.min_window_trades // min(window.lookback_seconds, busy_seconds) + 1)
    # 下跌段每根 K 线的跌幅就是这段时间 ATR 的上限；急拉幅度取涨幅门槛与 ATR 门槛中较大者的 1.5 倍。
    decline_seconds = (low_start - decline_start).total_seconds()
    bar_range = (_PRICE - low) / decline_seconds * interval_seconds
    jump = max(low * window.min_change_percent / 100, bar_range * window.trigger_atr_multiple, 0.1) * 1.5

    def feed(second: datetime, price: float, count: int) -> list[PriceAlert | StreakAlert]:
        out: list[PriceAlert | StreakAlert] = []
        for index in range(count):
            # 同一秒的成交均匀摊在这一秒里，笔数再多也不会溢出到下一秒。
            moment = second + timedelta(seconds=index / count)
            out.extend(detector.add_tick(PriceTick(SYMBOL, price, 1.0, moment, f"{second.timestamp()}-{index}")))
        return out

    alerts: list[PriceAlert | StreakAlert] = []
    moment = flat_start
    while moment < move_start:
        if moment < decline_start:
            price = _PRICE
        elif moment < low_start:
            price = _PRICE - (_PRICE - low) * (moment - decline_start).total_seconds() / decline_seconds
        else:
            price = low
        alerts.extend(feed(moment, price, busy if moment >= busy_start else 1))
        moment += timedelta(seconds=1)
    # 多送一秒结算最后一个确认秒，与真实行情的整秒判定一致。
    for offset in range(window.confirmation_seconds + 1):
        alerts.extend(feed(move_start + timedelta(seconds=offset), low + jump, busy))
    return alerts
