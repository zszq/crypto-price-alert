"""合成行情验证反弹回落配置：平稳 1 小时 → 一段下跌 → 最后一段成交涌入 → 急拉，正常时恰好产生一条「下跌反弹」。

配置了反向极值距离时，预热 K 线铺满 24 小时（都在起始价附近），急拉后的价格离下跌段的最低价不超过上限。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_reversal_detector
from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.reversal.alerts import ReversalAlert
from price_alert.detectors.reversal.config import ReversalConfig
from price_alert.detectors.reversal.rule import ATR_PERIOD, BURST_BASE_SECONDS, MAX_BURST_RATIO

# 急拉前成交涌入的秒数与暴增倍数的算法与短时异动附加条件的模拟共用。
from price_alert.detectors.spike.simulate import BUSY_LEAD_SECONDS as _BUSY_LEAD
from price_alert.detectors.spike.simulate import burst_ratio as _burst_ratio

SYMBOL = "BTC_USDT"
_PRICE = 100.0


def simulate_reversal(config: ReversalConfig) -> list[ReversalAlert]:
    rule = config.to_rule()
    detector = build_reversal_detector(rule)
    now = int(datetime.now(UTC).timestamp())
    # 急拉从分钟边界开始：急拉的秒都落在同一根未收盘 K 线里，不会提前计入 ATR 削弱倍数。
    move_start = datetime.fromtimestamp(now - now % 60, tz=UTC)
    # 提醒落在急拉的第 confirmation_seconds 秒；下跌在窗口起点前跌完，窗口内停在低位，基准价就是低位。
    low_start = move_start - timedelta(seconds=rule.lookback_seconds + 2)
    # 成交涌入只放在急拉前 BUSY_LEAD 秒和急拉本身：提醒那一秒回看 1 分钟，涌入的秒数 BUSY_LEAD + 确认秒数 ≤ 60。
    busy_start = move_start - timedelta(seconds=_BUSY_LEAD)
    # 前置走势的起点是提醒那一秒往前推窗口加回看秒数；下跌从最晚的起点开始，量到的就是整段跌幅。
    # 幅度取门槛的 1.5 倍（不超过 90%）。回看至少 60 秒、确认至多 10 秒，下跌段总有几十秒。
    decline_start = move_start - timedelta(
        seconds=rule.lookback_seconds + rule.prior_seconds - rule.confirmation_seconds
    )
    drop = min(rule.min_prior_move_percent * 1.5, 90.0)
    low = _PRICE * (1 - drop / 100)
    # 平稳段保证 1 小时的成交笔数基准。
    flat_start = decline_start - timedelta(seconds=BURST_BASE_SECONDS + 120)
    history_end = datetime.fromtimestamp(int(flat_start.timestamp()) // 60 * 60, tz=UTC)
    count = max(ATR_PERIOD + 1, rule.warmup_candles)
    candles = [
        Candle(history_end - timedelta(minutes=count - index), _PRICE, _PRICE + 0.05, _PRICE - 0.05, _PRICE)
        for index in range(count)
    ]
    detector.add_symbol(SYMBOL, candles, 1_000_000_000)

    # 提醒那一秒的倍数达到门槛的 1.2 倍（门槛小于理论上限 60 由配置保证，取二者之间的值）。
    target = min(rule.min_trade_burst_ratio * 1.2, (rule.min_trade_burst_ratio + MAX_BURST_RATIO) / 2)
    busy_seconds = _BUSY_LEAD + rule.confirmation_seconds
    busy = 1
    while _burst_ratio(busy, busy_seconds) < target:
        busy *= 2
    # 窗口里落在涌入段的秒数有限，每秒笔数还要让窗口笔数够。
    busy = max(busy, rule.min_window_trades // min(rule.lookback_seconds, busy_seconds) + 1)
    # 下跌段每分钟跌幅就是这段时间 ATR 的上限；急拉幅度取涨幅门槛与 ATR 门槛中较大者的 1.5 倍。
    decline_seconds = (low_start - decline_start).total_seconds()
    minute_range = (_PRICE - low) / decline_seconds * 60
    need = max(low * rule.min_change_percent / 100, minute_range * rule.trigger_atr_multiple)
    jump = need * 1.5
    if rule.max_extreme_distance_percent is not None:
        # 低位段落在已收盘的 K 线里，急拉后的价格离 24 小时最低价正好是急拉幅度，不能超过上限。
        # ATR 门槛要求的幅度已超过上限时（如 5 ATR 配 2%）这里造不出提醒、模拟会报数量不符；
        # 实盘里急跌、急拉都落在同一分钟仍可能报，只是合成行情不构造这种形态。
        jump = min(jump, (need + low * rule.max_extreme_distance_percent / 100) / 2)

    def feed(second: datetime, price: float, count: int) -> list[ReversalAlert]:
        out: list[ReversalAlert] = []
        for index in range(count):
            # 同一秒的成交均匀摊在这一秒里，笔数再多也不会溢出到下一秒。
            moment = second + timedelta(seconds=index / count)
            tick = PriceTick(SYMBOL, price, 1.0, moment, f"{second.timestamp()}-{index}")
            out.extend(detector.add_tick(tick))
        return out

    alerts: list[ReversalAlert] = []
    moment = flat_start
    while moment < move_start:
        if moment < decline_start:
            price = _PRICE
        elif moment < low_start:
            price = _PRICE - (_PRICE - low) * (moment - decline_start).total_seconds() / decline_seconds
        else:
            price = low
        # 涌入段可能从下跌段中途开始（窗口短时低位段只有几秒），笔数只看是否已进入涌入段。
        alerts.extend(feed(moment, price, busy if moment >= busy_start else 1))
        moment += timedelta(seconds=1)
    # 多送一秒结算最后一个确认秒，与真实行情的整秒判定一致。
    for offset in range(rule.confirmation_seconds + 1):
        alerts.extend(feed(move_start + timedelta(seconds=offset), low + jump, busy))
    return alerts
