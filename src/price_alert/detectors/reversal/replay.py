"""反弹回落回放：用 REST 历史成交逐秒重放同一个检测器，解释为什么提醒或没有提醒。

与短时异动的回放相同分三步（取数复用 detectors.replay.fetch_replay_data）：plan_reversal_replay 只做时间推算，
run_reversal_replay 离线驱动检测器，render_reversal_replay 排成文本。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from price_alert.common.formatting import BEIJING_TIME, beijing_time, describe_window, format_price
from price_alert.detectors.replay import ReplayData, ReplayPlan
from price_alert.detectors.reversal.alerts import ReversalAlert, describe_reversal_rule, format_reversal_alert
from price_alert.detectors.reversal.detector import (
    OUTCOME_LABELS,
    ReversalDetector,
    ReversalEvaluation,
    ReversalOutcome,
)
from price_alert.detectors.reversal.rule import CANDLE_INTERVAL_SECONDS, ReversalRule
from price_alert.detectors.spike.detector import BASELINE_TOLERANCE_SECONDS
from price_alert.detectors.spike.windows import OUTCOME_LABELS as WINDOW_OUTCOME_LABELS
from price_alert.detectors.spike.windows import Outcome


@dataclass(frozen=True, slots=True)
class ReversalReplayResult:
    plan: ReplayPlan
    rule: ReversalRule
    evaluations: list[ReversalEvaluation]
    alerts: list[ReversalAlert]
    trades_fed: int
    warmup_candles: int
    price_decimals: int | None = None
    # 区间内最后一个有成交的秒，没有后续成交来结算、因而缺少判定时才有值。
    unsettled_second: datetime | None = None


def plan_reversal_replay(symbol: str, start: datetime, end: datetime, rule: ReversalRule) -> ReplayPlan:
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    # 区间第一秒就要有前置走势的起点与 1 小时成交基准，成交从更早喂起，并对齐 K 线边界让第一根实时 K 线完整。
    earliest = int(start.timestamp()) - rule.history_seconds - BASELINE_TOLERANCE_SECONDS
    interval = CANDLE_INTERVAL_SECONDS
    feed_start = datetime.fromtimestamp(earliest // interval * interval, tz=BEIJING_TIME)
    warmup_start = feed_start - timedelta(seconds=interval * rule.warmup_candles)
    return ReplayPlan(symbol, start, end, feed_start, warmup_start)


def run_reversal_replay(rule: ReversalRule, plan: ReplayPlan, data: ReplayData) -> ReversalReplayResult:
    """离线驱动检测器；冷却从空白开始，与实时进程当时的冷却状态可能不同。"""
    evaluations: list[ReversalEvaluation] = []
    detector = ReversalDetector(rule, observer=evaluations.append)
    # 只用 feed_start 之前已收盘的 K 线：之后的行情全部由成交重建，与实时监控的 ATR 口径一致。
    seed = [candle for candle in data.candles if candle.timestamp < plan.feed_start]
    # 历史 24h 成交额无从得知，回放也不依赖它，填 0 即可。
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=data.price_decimals)
    fed = 0
    last_second: datetime | None = None
    for tick in data.trades:
        if tick.symbol.upper() != plan.symbol or tick.timestamp < plan.feed_start:
            continue
        fed += 1
        last_second = tick.timestamp.replace(microsecond=0)
        detector.add_tick(tick)

    def in_range(second: datetime) -> bool:
        return plan.start <= second <= plan.end

    kept = [item for item in evaluations if in_range(item.window.second)]
    # 按判定所属的秒取舍提醒，而不是按结算它的那笔成交的时间，与短时异动回放一致。
    alerts = [item.alert for item in kept if item.alert is not None]
    return ReversalReplayResult(
        plan=plan,
        rule=rule,
        evaluations=kept,
        alerts=alerts,
        trades_fed=fed,
        warmup_candles=len(seed),
        price_decimals=data.price_decimals,
        # 最后一笔成交所在的秒要等下一笔成交才会结算：它落在区间内说明这一秒没有判定。
        unsettled_second=last_second if last_second is not None and in_range(last_second) else None,
    )


def _window_label(outcome: Outcome) -> str:
    # 急动窗口的 ALERT 只表示急动本身可以提醒，还要过前置走势与成交暴增，不能写成「提醒」。
    return "急动达标、可以提醒" if outcome is Outcome.ALERT else WINDOW_OUTCOME_LABELS[outcome]


def _closeness(item: ReversalEvaluation, rule: ReversalRule) -> float:
    # 两道门槛是「且」的关系，离触发有多近取决于较弱的那一道。
    window = item.window
    if window.change_percent is None or window.move_atr is None:
        return 0.0
    return min(abs(window.change_percent) / rule.min_change_percent, window.move_atr / rule.trigger_atr_multiple)


# 急动达到涨跌幅与 ATR 门槛的秒（确认中、补齐、空档复核失败或已可提醒），默认只列这些。
_NOTABLE = frozenset({Outcome.CONFIRMING, Outcome.CARRIED, Outcome.MOVE_FADED, Outcome.ALERT})


def _is_notable(item: ReversalEvaluation) -> bool:
    return item.window.outcome in _NOTABLE


def _format_row(item: ReversalEvaluation, rule: ReversalRule, decimals: int | None) -> str:
    window = item.window
    parts = [beijing_time(window.second, "%H:%M:%S")]
    parts.append(format_price(window.price, decimals) + ("（补齐）" if window.carried else ""))
    if window.change_percent is not None and window.baseline_price is not None:
        assert window.move_atr is not None
        parts.append(f"基准 {format_price(window.baseline_price, decimals)}")
        parts.append(f"{window.change_percent:+.2f}%")
        parts.append(f"{window.move_atr:.2f} ATR")
        parts.append(f"{window.trade_count} 笔")
    if window.confirmed_seconds:
        parts.append(f"确认 {window.confirmed_seconds}/{rule.confirmation_seconds}")
    if item.prior_change_percent is not None:
        parts.append(f"此前{describe_window(rule.prior_seconds)} {item.prior_change_percent:+.2f}%")
    elif item.outcome is ReversalOutcome.PRIOR_TREND:
        parts.append("前置走势历史不足")
    if item.trade_burst_ratio is not None:
        parts.append(f"成交 ×{item.trade_burst_ratio:.1f}")
    elif item.outcome is ReversalOutcome.NO_TRADE_BURST:
        parts.append("不足 1 小时成交")
    if item.extreme_distance_percent is not None:
        side = "低点" if (window.change_percent or 0) > 0 else "高点"
        parts.append(f"距24小时{side} {item.extreme_distance_percent:+.2f}%")
    elif item.outcome is ReversalOutcome.EXTREME:
        parts.append("K 线不足 24 小时")
    if item.outcome is ReversalOutcome.MOVE:
        parts.append(WINDOW_OUTCOME_LABELS[window.outcome])
    else:
        parts.append(OUTCOME_LABELS[item.outcome])
    return " | ".join(parts)


def render_reversal_replay(result: ReversalReplayResult, show_all: bool = False) -> list[str]:
    plan, rule, decimals = result.plan, result.rule, result.price_decimals
    lines = [
        f"反弹回落回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 K 线 {result.warmup_candles} 根，成交 {result.trades_fed:,} 笔"
        f"（自 {beijing_time(plan.feed_start, '%H:%M:%S')} 起喂入，为前置走势与 1 小时成交基准预留历史）",
        describe_reversal_rule(rule),
        "说明：回放从空白冷却开始，也不含断线；实时进程当时若未运行、断线或处于冷却，结果会与回放不同。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_reversal_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有提醒。")
    if result.unsettled_second is not None:
        # 紧跟在提醒结论之后，避免「没有提醒」被当成完整结论。
        lines.append(
            f"注意：结果不完整。最后一笔成交在 {beijing_time(result.unsettled_second, '%H:%M:%S')}，"
            "之后取不到用于结算的成交，这一秒没有判定；它若恰好完成确认，提醒会被漏掉。"
        )
    lines.append("")
    if not result.evaluations:
        lines.append("区间内没有可评估的秒（没有成交）。")
        return lines

    window_counts = Counter(item.window.outcome for item in result.evaluations)
    lines.append(
        "急动窗口判定："
        + "、".join(f"{_window_label(o)} {window_counts[o]} 秒" for o in Outcome if window_counts[o])
    )
    gate_counts = Counter(item.outcome for item in result.evaluations if item.window.outcome is Outcome.ALERT)
    if gate_counts:
        lines.append(
            "急动可以提醒之后："
            + "、".join(f"{OUTCOME_LABELS[o]} {gate_counts[o]} 秒" for o in ReversalOutcome if gate_counts[o])
        )
    measured = [item for item in result.evaluations if item.window.change_percent is not None]
    if measured:
        closest = max(measured, key=lambda item: _closeness(item, rule))
        window = closest.window
        assert window.change_percent is not None and window.move_atr is not None and window.baseline_price is not None
        lines.append(
            f"急动最接近门槛：{beijing_time(window.second, '%H:%M:%S')} "
            f"{format_price(window.baseline_price, decimals)} → {format_price(window.price, decimals)}，"
            f"{window.change_percent:+.2f}%（门槛 {rule.min_change_percent:g}%），"
            f"{window.move_atr:.2f} ATR（门槛 {rule.trigger_atr_multiple:g}），"
            f"窗口 {window.trade_count} 笔（门槛 {rule.min_window_trades}），达标程度 {_closeness(closest, rule):.0%}"
        )
    lines.append("")

    rows = [item for item in result.evaluations if show_all or _is_notable(item)]
    if show_all:
        lines.append("逐秒判定：")
    elif rows:
        lines.append("急动达到涨跌幅与 ATR 门槛的秒（加 --all 查看每一秒）：")
    else:
        lines.append("没有任何一秒的急动同时达到涨跌幅与 ATR 门槛（加 --all 查看每一秒）。")
    lines.extend(f"  {_format_row(item, rule, decimals)}" for item in rows)
    return lines
