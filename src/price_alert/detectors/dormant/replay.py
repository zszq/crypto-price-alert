"""冷门启动回放：用 Gate 历史 1 分钟 K 线驱动与实时监控相同的检测器，逐分钟解释为什么提醒或没有提醒。

交易所 1 分钟 K 线带计价成交额（sum），与实时监控由成交聚合的口径相同，不必逐笔拉成交。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from price_alert.assembly import build_dormant_detector
from price_alert.common.formatting import beijing_time, format_amount, format_price
from price_alert.common.minute_bars import MINUTE
from price_alert.common.models import Candle
from price_alert.detectors.dormant.alerts import (
    LABELS,
    DormantAlert,
    describe_dormant_rule,
    format_calm,
    format_dormant_alert,
)
from price_alert.detectors.dormant.detector import OUTCOME_LABELS, DormantEvaluation, DormantOutcome
from price_alert.detectors.dormant.rule import CONDITION_LABELS, Condition, DormantRule, late_minutes
from price_alert.detectors.replay import MinuteReplayPlan, plan_minute_replay

# 默认输出只列满足条件的分钟，其余只计入统计。
_QUIET_OUTCOMES = frozenset({DormantOutcome.INSUFFICIENT_DATA, DormantOutcome.FAILED})


@dataclass(frozen=True, slots=True)
class DormantReplayResult:
    plan: MinuteReplayPlan
    rules: tuple[DormantRule, ...]
    evaluations: list[DormantEvaluation]
    alerts: list[DormantAlert]
    candles_fed: int
    warmup_candles: int
    price_decimals: int | None = None


def plan_dormant_replay(
    symbol: str, start: datetime, end: datetime, rules: Sequence[DormantRule], now: datetime | None = None
) -> MinuteReplayPlan:
    # 各组共用一个检测器，预热按窗口最长的一组。
    return plan_minute_replay(symbol, start, end, max(rule.history_minutes for rule in rules), now)


def run_dormant_replay(
    rules: Sequence[DormantRule],
    plan: MinuteReplayPlan,
    candles: Sequence[Candle],
    price_decimals: int | None = None,
) -> DormantReplayResult:
    """离线驱动检测器；从空白提醒记录开始，区间开始前已经提醒过的行情会在区间内再提醒一次。"""
    start = datetime.fromtimestamp(int(plan.start.timestamp()) // 60 * 60, tz=plan.start.tzinfo)
    seed = [candle for candle in candles if candle.timestamp < start]
    replayed = [candle for candle in candles if candle.timestamp >= start and candle.timestamp + MINUTE <= plan.end]
    evaluations: list[DormantEvaluation] = []
    detector = build_dormant_detector(rules, observer=evaluations.append)
    # K 线自带计价成交额，不需要合约面值换算；面值只用于实时成交。
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=price_decimals, quanto_multiplier=1.0)
    alerts: list[DormantAlert] = []
    # 只喂收盘不晚于结束时间的 K 线，判定时间点才全部落在区间内。
    for candle in replayed:
        alerts.extend(detector.add_candle(plan.symbol, candle))
    return DormantReplayResult(plan, tuple(rules), evaluations, alerts, len(replayed), len(seed), price_decimals)


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
        # 只有开了这三项的组（慢启动）才列，现行一组的输出不变。
        if rule.min_efficiency > 0 or rule.min_volume_ratio > 0 or rule.max_retrace is not None:
            parts.append(
                f"效率 {metrics.efficiency:.2f} 量比 {_format_ratio(metrics.volume_ratio)} "
                f"从最高价回落 {metrics.retrace:.0%}"
            )
    label = OUTCOME_LABELS[evaluation.outcome]
    if evaluation.outcome is DormantOutcome.FAILED and metrics is not None:
        label += "：" + "、".join(CONDITION_LABELS[condition] for condition in metrics.failures)
    parts.append(label)
    return " | ".join(parts)


def _format_ratio(ratio: float) -> str:
    # 之前 24 小时没有成交时量比为无穷大。
    return f"{ratio:.1f} 倍" if math.isfinite(ratio) else "无穷大"


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
    efficiency = metrics.efficiency / rule.min_efficiency if rule.min_efficiency > 0 else 1.0
    ratio = metrics.volume_ratio / rule.min_volume_ratio if rule.min_volume_ratio > 0 else 1.0
    retrace = rule.max_retrace / max(metrics.retrace, 1e-9) if rule.max_retrace is not None else 1.0
    # 达标的项不参与比较：倍数可能是无穷大，跳空小的分钟比值也会很大，都与「差多少」无关。
    values = (quote, change, active, calm, jump, body, volatile, last, late, efficiency, ratio, retrace)
    shortfalls = [min(value, 1.0) for value in values]
    return (-len(metrics.failures), min(shortfalls))


def render_dormant_replay(result: DormantReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    lines = [
        f"冷门启动回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 1 分钟 K 线 {result.warmup_candles} 根，回放 {result.candles_fed} 根",
        *(describe_dormant_rule(rule) for rule in result.rules),
        "说明：回放从空白提醒记录开始，也不按合约池筛选；实时监控由成交聚合 K 线，与交易所 K 线可能有细微差别。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_dormant_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有冷门启动提醒。")
    for rule in result.rules:
        lines.append("")
        if len(result.rules) > 1:
            lines.append(f"【{LABELS[rule.group]}】")
        lines.extend(_render_group(result, rule, show_all))
    return lines


def _render_group(result: DormantReplayResult, rule: DormantRule, show_all: bool) -> list[str]:
    plan = result.plan
    decimals = result.price_decimals
    lines: list[str] = []
    in_range = [
        item for item in result.evaluations if item.group is rule.group and plan.start <= item.timestamp <= plan.end
    ]
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
    # 跟踪补报的分钟条件多半不满足，也列出来。
    subject = "满足条件" if rule.follow_step_percent is None else "满足条件或跟踪补报"
    if show_all:
        lines.append("逐分钟判定：")
    elif rows:
        lines.append(f"{subject}的分钟（加 --all 查看每一分钟的判定）：")
    else:
        lines.append(f"没有{subject}的分钟（加 --all 查看每一分钟的判定）。")
    lines.extend(f"  {_format_row(item, rule, decimals)}" for item in rows)
    return lines
