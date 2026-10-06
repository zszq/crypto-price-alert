"""历史成交回放：用与实时监控完全相同的检测器逐秒重放一段行情，解释它为什么提醒或没有提醒。

分四步，便于单独测试：plan_replay 只做时间推算，fetch_replay_data 只负责联网取数，
run_replay 离线驱动检测器，render_replay 把结果排成文本。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from price_alert.assembly import build_detector, build_windows
from price_alert.config import INTERVAL_SECONDS, AppConfig
from price_alert.detector import BASELINE_TOLERANCE_SECONDS
from price_alert.formatting import BEIJING_TIME, beijing_time, format_price
from price_alert.gate import GateRestClient, price_decimals
from price_alert.models import Candle, PriceAlert, PriceTick
from price_alert.notifier import format_alert
from price_alert.windows import (
    OUTCOME_LABELS,
    WINDOW_LABELS,
    MoveWindow,
    Outcome,
    WindowEvaluation,
    describe_rule,
)

LOGGER = logging.getLogger(__name__)

# 一秒要等下一秒的第一笔成交才会结算，多取一点成交，区间最后几秒才不会因为没人“结算”而漏评。
SETTLE_MARGIN = timedelta(seconds=10)
# 余量内仍没有成交时，继续向后找结算用的那一笔，最远找这么久。实时监控里多晚的成交都会结算，
# 但冷门合约停摆几小时也没必要一直翻下去，超过后如实标注结果不完整。
SETTLE_SEARCH_LIMIT = timedelta(hours=1)
# 这些结果说明窗口已经进入或越过门槛判定，默认输出只列它们，其余秒只计入统计。
_NOTABLE_OUTCOMES = frozenset(
    {Outcome.CONFIRMING, Outcome.CARRIED, Outcome.MOVE_FADED, Outcome.COOLDOWN, Outcome.ALERT}
)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 成交从这里开始喂入：留足最长窗口的历史，并对齐 K 线边界，让第一根实时 K 线是完整的。
    feed_start: datetime
    # ATR 用 [warmup_start, feed_start) 内已收盘的 K 线预热，与实时监控启动时的口径一致。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class ReplayData:
    candles: list[Candle]
    trades: list[PriceTick]
    price_decimals: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    plan: ReplayPlan
    windows: tuple[MoveWindow, ...]
    evaluations: list[WindowEvaluation]
    alerts: list[PriceAlert]
    trades_fed: int
    warmup_candles: int
    price_decimals: int | None = None
    # 区间内最后一个有成交的秒，没有后续成交来结算、因而缺少判定时才有值。
    unsettled_second: datetime | None = None


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


def plan_replay(symbol: str, start: datetime, end: datetime, config: AppConfig) -> ReplayPlan:
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    interval = INTERVAL_SECONDS[config.indicator.candle_interval]
    max_lookback = max(window.lookback_seconds for window in build_windows(config))
    earliest = int(start.timestamp()) - max_lookback - BASELINE_TOLERANCE_SECONDS
    feed_start = datetime.fromtimestamp(earliest // interval * interval, tz=BEIJING_TIME)
    warmup_start = feed_start - timedelta(seconds=interval * config.indicator.warmup_candles)
    return ReplayPlan(symbol, start, end, feed_start, warmup_start)


def fetch_replay_data(rest: GateRestClient, plan: ReplayPlan, config: AppConfig) -> ReplayData:
    decimals: int | None = None
    try:
        decimals = price_decimals(rest.fetch_contract(plan.symbol).get("order_price_round"))
    except Exception as exc:
        # 精度只影响价格显示，取不到就按有效数字显示，不值得让整个回放失败。
        LOGGER.warning("%s 合约信息获取失败，价格按有效数字显示：%s", plan.symbol, exc)
    interval = timedelta(seconds=INTERVAL_SECONDS[config.indicator.candle_interval])
    candles = rest.fetch_candles_between(
        plan.symbol,
        config.indicator.candle_interval,
        plan.warmup_start,
        plan.feed_start - interval,
    )
    LOGGER.info(
        "正在获取 %s 自 %s 起的历史成交，区间越长越慢",
        plan.symbol,
        beijing_time(plan.feed_start),
    )
    trades = rest.fetch_trades(plan.symbol, plan.feed_start, plan.end + SETTLE_MARGIN)
    if trades and _second_of(trades[-1]) <= plan.end:
        settling = _find_settling_trade(rest, plan)
        if settling is not None:
            trades = [*trades, settling]
    LOGGER.info("共获取 %s 笔成交、%d 根预热 K 线", f"{len(trades):,}", len(candles))
    return ReplayData(candles, trades, decimals)


def _second_of(tick: PriceTick) -> datetime:
    # 与检测器按秒分桶的口径一致。
    return tick.timestamp.replace(microsecond=0)


def _find_settling_trade(rest: GateRestClient, plan: ReplayPlan) -> PriceTick | None:
    """余量内没有成交时，向后找第一笔成交来结算区间内最后一个有成交的秒；找不到返回 None。

    必须用真实的下一笔成交而不是人为补一笔：空档后结算时要用它的价格复核异动是否仍在，
    只有它才能还原实时监控当时的判定。查找区间逐段翻倍，停摆不久时只需一两次小请求。
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


def run_replay(config: AppConfig, plan: ReplayPlan, data: ReplayData) -> ReplayResult:
    """离线驱动检测器；冷却从空白开始，与实时进程当时的冷却状态可能不同。"""
    evaluations: list[WindowEvaluation] = []
    detector = build_detector(config, observer=evaluations.append)
    # 只用 feed_start 之前已收盘的 K 线：之后的行情全部由成交重建，才与实时监控的 ATR 口径一致。
    seed = [candle for candle in data.candles if candle.timestamp < plan.feed_start]
    # 历史 24h 成交额无从得知，回放也不依赖它，填 0 即可。
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=data.price_decimals)

    alerts: list[PriceAlert] = []
    fed = 0
    last_tick: PriceTick | None = None
    for tick in data.trades:
        if tick.symbol.upper() != plan.symbol or tick.timestamp < plan.feed_start:
            continue
        fed += 1
        last_tick = tick
        alerts.extend(detector.add_tick(tick))

    def in_range(second: datetime) -> bool:
        return plan.start <= second <= plan.end

    # 最后一笔成交所在的秒要等下一笔成交才会结算：它落在区间内，说明这一秒没有判定，
    # 恰好完成确认的提醒会被漏掉，不能让结果看起来像“没有提醒”。
    unsettled = _second_of(last_tick) if last_tick is not None else None

    # 提醒与 ALERT 判定一一对应且顺序相同；按判定所属的秒筛选，而不是按结算它的那笔成交的时间，
    # 否则区间末尾之后才结算的秒会被算进来，区间开头之前的秒也可能漏掉。
    fired = [item for item in evaluations if item.outcome is Outcome.ALERT]
    return ReplayResult(
        plan=plan,
        windows=detector.windows,
        evaluations=[item for item in evaluations if in_range(item.second)],
        alerts=[alert for alert, item in zip(alerts, fired, strict=True) if in_range(item.second)],
        trades_fed=fed,
        warmup_candles=len(seed),
        price_decimals=data.price_decimals,
        unsettled_second=unsettled if unsettled is not None and in_range(unsettled) else None,
    )


def _closeness(evaluation: WindowEvaluation, window: MoveWindow) -> float:
    # 两道门槛是“且”的关系，离触发有多近取决于较弱的那一道，所以取两者达标比例的较小值。
    if evaluation.change_percent is None or evaluation.move_atr is None:
        return 0.0
    return min(
        abs(evaluation.change_percent) / window.min_change_percent,
        evaluation.move_atr / window.trigger_atr_multiple,
    )


def _is_notable(evaluation: WindowEvaluation, window: MoveWindow) -> bool:
    if evaluation.outcome in _NOTABLE_OUTCOMES:
        return True
    # 笔数检查排在门槛之前，笔数不足的秒即使涨跌已经达标也不会走到确认，单独捞出来才看得到“差在笔数”。
    return evaluation.outcome is Outcome.FEW_TRADES and _closeness(evaluation, window) >= 1.0


def _format_row(evaluation: WindowEvaluation, window: MoveWindow, decimals: int | None) -> str:
    parts = [beijing_time(evaluation.second, "%H:%M:%S"), WINDOW_LABELS[window.name]]
    parts.append(format_price(evaluation.price, decimals) + ("（补齐）" if evaluation.carried else ""))
    if evaluation.change_percent is not None and evaluation.baseline_price is not None:
        assert evaluation.move_atr is not None
        parts.append(f"基准 {format_price(evaluation.baseline_price, decimals)}")
        parts.append(f"{evaluation.change_percent:+.2f}%")
        parts.append(f"{evaluation.move_atr:.2f} ATR")
        parts.append(f"{evaluation.trade_count} 笔")
    if evaluation.confirmed_seconds:
        parts.append(f"确认 {evaluation.confirmed_seconds}/{window.confirmation_seconds}")
    parts.append(OUTCOME_LABELS[evaluation.outcome])
    return " | ".join(parts)


def _window_summary(
    window: MoveWindow,
    evaluations: Sequence[WindowEvaluation],
    decimals: int | None,
) -> list[str]:
    label = WINDOW_LABELS[window.name]
    if not evaluations:
        return [f"{label}：区间内没有可评估的秒（没有成交）"]
    counts = Counter(item.outcome for item in evaluations)
    distribution = "、".join(
        f"{OUTCOME_LABELS[outcome]} {counts[outcome]} 秒" for outcome in Outcome if counts[outcome]
    )
    lines = [f"{label}：{distribution}"]
    measured = [item for item in evaluations if item.change_percent is not None]
    if not measured:
        return lines
    closest = max(measured, key=lambda item: _closeness(item, window))
    assert closest.change_percent is not None and closest.move_atr is not None
    assert closest.baseline_price is not None
    lines.append(
        f"  最接近触发：{beijing_time(closest.second, '%H:%M:%S')} "
        f"{format_price(closest.baseline_price, decimals)} → {format_price(closest.price, decimals)}，"
        f"{closest.change_percent:+.2f}%（门槛 {window.min_change_percent:g}%），"
        f"{closest.move_atr:.2f} ATR（门槛 {window.trigger_atr_multiple:g}），"
        f"窗口 {closest.trade_count} 笔（门槛 {window.min_window_trades}），"
        f"达标程度 {_closeness(closest, window):.0%}"
    )
    return lines


def render_replay(result: ReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    decimals = result.price_decimals
    lines = [
        f"回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 K 线 {result.warmup_candles} 根，成交 {result.trades_fed:,} 笔"
        f"（自 {beijing_time(plan.feed_start, '%H:%M:%S')} 起喂入，为最长窗口预留历史）",
        *(describe_rule(window) for window in result.windows),
        "说明：回放从空白冷却开始，也不含断线；实时进程当时若未运行、断线或处于冷却，结果会与回放不同。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有提醒。")
    if result.unsettled_second is not None:
        # 紧跟在提醒结论之后，避免“没有提醒”被当成完整结论。
        lines.append(
            f"注意：结果不完整。最后一笔成交在 {beijing_time(result.unsettled_second, '%H:%M:%S')}，"
            "之后取不到用于结算的成交，这一秒没有判定；它若恰好完成确认，提醒会被漏掉。"
        )
    lines.append("")

    by_window: dict[str, list[WindowEvaluation]] = {window.name: [] for window in result.windows}
    for evaluation in result.evaluations:
        by_window[evaluation.window].append(evaluation)
    lines.append("各窗口判定统计：")
    for window in result.windows:
        lines.extend(f"  {line}" for line in _window_summary(window, by_window[window.name], decimals))
    lines.append("")

    windows = {window.name: window for window in result.windows}
    rows = [item for item in result.evaluations if show_all or _is_notable(item, windows[item.window])]
    if show_all:
        lines.append("逐秒判定：")
    elif rows:
        lines.append("门槛达标的秒（加 --all 查看每一秒）：")
    else:
        lines.append("没有任何一秒同时达到涨跌幅与 ATR 门槛（加 --all 查看每一秒）。")
    lines.extend(f"  {_format_row(item, windows[item.window], decimals)}" for item in rows)
    return lines

