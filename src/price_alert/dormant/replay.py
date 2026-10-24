"""冷门启动回放：用 Gate 历史 1 分钟 K 线驱动与实时监控相同的检测器，逐分钟解释为什么提醒或没有提醒。

交易所 1 分钟 K 线带计价成交额（sum），与实时监控由成交聚合的口径相同，不必逐笔拉成交。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_dormant_detector
from price_alert.dormant.alerts import DormantAlert, describe_dormant_rule, format_calm, format_dormant_alert
from price_alert.dormant.detector import OUTCOME_LABELS, DormantEvaluation, DormantOutcome
from price_alert.dormant.rule import CONDITION_LABELS, Condition, DormantRule, late_minutes
from price_alert.formatting import beijing_time, format_amount, format_price
from price_alert.gate import GATE_MINUTE_HISTORY
from price_alert.models import Candle

MINUTE = timedelta(minutes=1)
# 默认输出只列满足条件的分钟，其余只计入统计。
_QUIET_OUTCOMES = frozenset({DormantOutcome.INSUFFICIENT_DATA, DormantOutcome.FAILED})


@dataclass(frozen=True, slots=True)
class DormantReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 预热从这里开始：判定一次所需的全部历史。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class DormantReplayResult:
    plan: DormantReplayPlan
    rule: DormantRule
    evaluations: list[DormantEvaluation]
    alerts: list[DormantAlert]
    candles_fed: int
    warmup_candles: int
    price_decimals: int | None = None


def plan_dormant_replay(
    symbol: str, start: datetime, end: datetime, rule: DormantRule, now: datetime | None = None
) -> DormantReplayPlan:
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    aligned = datetime.fromtimestamp(int(start.timestamp()) // 60 * 60, tz=start.tzinfo)
    warmup = MINUTE * rule.history_minutes
    warmup_start = aligned - warmup
    # 预先检查而不是等交易所返回含义不明的 400；与放量回放同样留一分钟余量。
    earliest = (now or datetime.now(UTC)) - MINUTE * (GATE_MINUTE_HISTORY - 2)
    if warmup_start < earliest:
        raise ValueError(
            f"Gate 只保留最近 {GATE_MINUTE_HISTORY} 根 1 分钟 K 线（约 6.9 天），"
            f"开始时间最早为 {beijing_time(earliest + warmup + MINUTE, '%Y-%m-%d %H:%M')}（北京时间）"
        )
    return DormantReplayPlan(symbol, start, end, warmup_start)


def run_dormant_replay(
    rule: DormantRule,
    plan: DormantReplayPlan,
    candles: Sequence[Candle],
    price_decimals: int | None = None,
) -> DormantReplayResult:
    """离线驱动检测器；从空白提醒记录开始，区间开始前已经提醒过的行情会在区间内再提醒一次。"""
    start = datetime.fromtimestamp(int(plan.start.timestamp()) // 60 * 60, tz=plan.start.tzinfo)
    seed = [candle for candle in candles if candle.timestamp < start]
    replayed = [candle for candle in candles if candle.timestamp >= start and candle.timestamp + MINUTE <= plan.end]
    evaluations: list[DormantEvaluation] = []
    detector = build_dormant_detector(rule, observer=evaluations.append)
    # K 线自带计价成交额，不需要合约面值换算；面值只用于实时成交。
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=price_decimals, quanto_multiplier=1.0)
    alerts: list[DormantAlert] = []
    # 只喂收盘不晚于结束时间的 K 线，判定时间点才全部落在区间内。
    for candle in replayed:
        alerts.extend(detector.add_candle(plan.symbol, candle))
    return DormantReplayResult(plan, rule, evaluations, alerts, len(replayed), len(seed), price_decimals)


def _format_row(evaluation: DormantEvaluation, rule: DormantRule, decimals: int | None) -> str:
    parts = [beijing_time(evaluation.timestamp, "%H:%M")]
    metrics = evaluation.metrics
    if metrics is not None:
        parts.append(
            f"{format_price(metrics.start_price, decimals)} → {format_price(metrics.end_price, decimals)} "
            f"{metrics.change_percent:+.2f}%"
        )
        parts.append(
            f"{rule.window_minutes}分钟成交额 {format_amount(metrics.window_quote)} "
            f"有成交 {metrics.active_minutes} 分钟 之前24小时 {format_amount(metrics.prior_quote)}"
        )
        parts.append(
            f"{format_calm(metrics)} 单分钟最大 {metrics.max_minute_change_percent:+.2f}% "
            f"实体占比 {metrics.body_ratio:.0%}"
        )
        parts.append(
            f"最后一分钟成交额 {format_amount(metrics.last_minute_quote)} "
            f"后{late_minutes(rule.window_minutes)}分钟占比 {metrics.late_quote_share:.0%}"
        )
    label = OUTCOME_LABELS[evaluation.outcome]
    if evaluation.outcome is DormantOutcome.FAILED and metrics is not None:
        label += "：" + "、".join(CONDITION_LABELS[condition] for condition in metrics.failures)
    parts.append(label)
    return " | ".join(parts)


def _closeness(evaluation: DormantEvaluation, rule: DormantRule) -> tuple[int, float]:
    # 先比未满足的条件数，再比各道数值门槛中最弱的一项达到门槛的程度（上限类按门槛 ÷ 实际值算）。
    metrics = evaluation.metrics
    assert metrics is not None
    quote = metrics.window_quote / rule.min_window_quote if rule.min_window_quote > 0 else 1.0
    change = metrics.change_percent / rule.min_change_percent
    active = metrics.active_minutes / rule.min_active_minutes
    calm = metrics.calm_multiple / rule.min_calm_multiple if rule.min_calm_multiple > 0 else 1.0
    jump = rule.max_minute_change_percent / max(metrics.max_minute_change_percent, 1e-9)
    body = metrics.body_ratio / rule.min_body_ratio if rule.min_body_ratio > 0 else 1.0
    cap = rule.max_pre_volatility_percent
    volatile = cap / max(metrics.pre_volatility_percent, 1e-9) if cap is not None else 1.0
    last = metrics.last_minute_quote / rule.min_last_minute_quote if rule.min_last_minute_quote > 0 else 1.0
    late = metrics.late_quote_share / rule.min_late_quote_share if rule.min_late_quote_share > 0 else 1.0
    # 达标的项不参与比较：倍数可能是无穷大，跳空小的分钟比值也会很大，都与「差多少」无关。
    shortfalls = [min(value, 1.0) for value in (quote, change, active, calm, jump, body, volatile, last, late)]
    return (-len(metrics.failures), min(shortfalls))


def render_dormant_replay(result: DormantReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    rule = result.rule
    decimals = result.price_decimals
    lines = [
        f"冷门启动回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 1 分钟 K 线 {result.warmup_candles} 根，回放 {result.candles_fed} 根",
        describe_dormant_rule(rule),
        "说明：回放从空白提醒记录开始，也不按合约池筛选；实时监控由成交聚合 K 线，与交易所 K 线可能有细微差别。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_dormant_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有冷门启动提醒。")
    lines.append("")
    in_range = [item for item in result.evaluations if plan.start <= item.timestamp <= plan.end]
    counts = Counter(item.outcome for item in in_range)
    lines.append(
        "判定统计："
        + "、".join(f"{OUTCOME_LABELS[outcome]} {counts[outcome]} 次" for outcome in DormantOutcome if counts[outcome])
    )
    failures = Counter(
        condition for item in in_range if item.metrics is not None for condition in item.metrics.failures
    )
    if failures:
        lines.append(
            "  未满足的条件："
            + "、".join(f"{CONDITION_LABELS[condition]} {count} 次" for condition, count in failures.most_common())
        )
    # 不冷门的分钟离触发最远（再怎么拉升也不会报），不参与「最接近触发」的比较。
    failed = [
        item
        for item in in_range
        if item.outcome is DormantOutcome.FAILED and item.metrics is not None
        and Condition.PRIOR not in item.metrics.failures
    ]
    if failed:
        closest = max(failed, key=lambda item: _closeness(item, rule))
        lines.append(f"  最接近触发：{_format_row(closest, rule, decimals)}")
    rows = [item for item in in_range if show_all or item.outcome not in _QUIET_OUTCOMES]
    if show_all:
        lines.append("逐分钟判定：")
    elif rows:
        lines.append("满足条件的分钟（加 --all 查看每一分钟的判定）：")
    else:
        lines.append("没有满足条件的分钟（加 --all 查看每一分钟的判定）。")
    lines.extend(f"  {_format_row(item, rule, decimals)}" for item in rows)
    return lines
