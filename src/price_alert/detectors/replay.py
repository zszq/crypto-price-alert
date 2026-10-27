"""各检测器历史回放的公共部分：时间与合约名解析、报价精度，以及两类回放的区间与取数。

- 成交逐秒回放（spike、reversal）：ReplayPlan / ReplayData 与 fetch_replay_data（取成交与预热 K 线）；
- 1 分钟 K 线回放（trend、launch、dormant）：MinuteReplayPlan 与 plan_minute_replay（对齐与 Gate 保留期检查），
  K 线由 exchange.gate.fetch_minute_candles 分段取。
各自的驱动检测器与排版放在自己的包里。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from price_alert.common.formatting import BEIJING_TIME, beijing_time
from price_alert.common.minute_bars import MINUTE
from price_alert.common.models import INTERVAL_SECONDS, Candle, PriceTick
from price_alert.exchange.gate import GATE_MINUTE_HISTORY, GateRestClient, price_decimals

LOGGER = logging.getLogger(__name__)

# 一秒要等下一秒的第一笔成交才会结算，多取一点成交，区间最后几秒才不会漏评。
SETTLE_MARGIN = timedelta(seconds=10)
# 余量内仍没有成交时，向后找结算用的那一笔，最远找这么久；冷门合约停摆几小时不必一直翻，超过后标注结果不完整。
SETTLE_SEARCH_LIMIT = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 成交从这里开始喂入：留足判定所需的历史，并对齐 K 线边界，让第一根实时 K 线是完整的。
    feed_start: datetime
    # ATR 用 [warmup_start, feed_start) 内已收盘的 K 线预热，与实时监控启动时的口径一致。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class MinuteReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 预热从这里开始：判定一次所需的全部历史。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class ReplayData:
    candles: list[Candle]
    trades: list[PriceTick]
    price_decimals: int | None = None


def normalize_symbol(text: str) -> str:
    # 用户习惯只输入币名，补上本项目唯一支持的 USDT 永续后缀。
    symbol = text.strip().upper()
    return symbol if "_" in symbol else f"{symbol}_USDT"


def parse_time(text: str, today: date | None = None) -> datetime:
    """解析回放时间；不带时区的一律按北京时间，只给时分秒时取北京时间的今天。"""
    value = text.strip()
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        try:
            clock = time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"无法识别的时间：{text}（示例：2026-09-29 09:27 或 09:27）") from exc
        day = today or datetime.now(BEIJING_TIME).date()
        moment = datetime.combine(day, clock)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=BEIJING_TIME)


def fetch_price_decimals(rest: GateRestClient, symbol: str) -> int | None:
    try:
        return price_decimals(rest.fetch_contract(symbol).get("order_price_round"))
    except Exception as exc:
        # 精度只影响价格显示，取不到就按有效数字显示，不值得让整个回放失败。
        LOGGER.warning("%s 合约信息获取失败，价格按有效数字显示：%s", symbol, exc)
        return None


def plan_minute_replay(
    symbol: str, start: datetime, end: datetime, warmup_minutes: int, now: datetime | None = None
) -> MinuteReplayPlan:
    """warmup_minutes 为判定一次所需的已收盘 1 分钟 K 线根数；now 供测试固定当前时间。"""
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    aligned = datetime.fromtimestamp(int(start.timestamp()) // 60 * 60, tz=start.tzinfo)
    warmup = MINUTE * warmup_minutes
    warmup_start = aligned - warmup
    # 预先检查，而不是等交易所返回含义不明的 400。10000 根含当前未收盘的一根，再留一分钟余量，
    # 防止检查之后、请求之前跨过一分钟。
    earliest = (now or datetime.now(UTC)) - MINUTE * (GATE_MINUTE_HISTORY - 2)
    if warmup_start < earliest:
        raise ValueError(
            f"Gate 只保留最近 {GATE_MINUTE_HISTORY} 根 1 分钟 K 线（约 6.9 天），"
            f"开始时间最早为 {beijing_time(earliest + warmup + MINUTE, '%Y-%m-%d %H:%M')}（北京时间）"
        )
    return MinuteReplayPlan(symbol, start, end, warmup_start)


def fetch_replay_data(rest: GateRestClient, plan: ReplayPlan, candle_interval: str) -> ReplayData:
    """candle_interval 为预热 K 线周期：短时异动取 spike.candle_interval，反弹回落固定用 1 分钟。"""
    decimals = fetch_price_decimals(rest, plan.symbol)
    interval = timedelta(seconds=INTERVAL_SECONDS[candle_interval])
    candles = rest.fetch_candles_between(
        plan.symbol,
        candle_interval,
        plan.warmup_start,
        plan.feed_start - interval,
    )
    LOGGER.info(
        "正在获取 %s 自 %s 起的历史成交，区间越长越慢",
        plan.symbol,
        beijing_time(plan.feed_start),
    )
    trades = rest.fetch_trades(plan.symbol, plan.feed_start, plan.end + SETTLE_MARGIN)
    if trades and second_of(trades[-1]) <= plan.end:
        settling = _find_settling_trade(rest, plan)
        if settling is not None:
            trades = [*trades, settling]
    LOGGER.info("共获取 %s 笔成交、%d 根预热 K 线", f"{len(trades):,}", len(candles))
    return ReplayData(candles, trades, decimals)


def second_of(tick: PriceTick) -> datetime:
    # 与检测器按秒分桶的口径一致。
    return tick.timestamp.replace(microsecond=0)


def _find_settling_trade(rest: GateRestClient, plan: ReplayPlan) -> PriceTick | None:
    """余量内没有成交时，向后找第一笔成交来结算区间内最后一个有成交的秒；找不到返回 None。

    必须用真实的下一笔成交：空档后结算要用它的价格复核，才能还原实时监控当时的判定。
    查找区间逐段翻倍，停摆不久时只需一两次小请求。
    """
    start = plan.end + SETTLE_MARGIN
    limit = start + SETTLE_SEARCH_LIMIT
    span = SETTLE_MARGIN
    margin = int(SETTLE_MARGIN.total_seconds())
    LOGGER.info("%s 区间结束后 %d 秒内没有成交，继续向后查找用于结算最后一秒的成交", plan.symbol, margin)
    while start < limit:
        end = min(start + span, limit)
        found = rest.fetch_trades(plan.symbol, start, end)
        if found:
            return found[0]
        start = end
        span *= 2
    return None
