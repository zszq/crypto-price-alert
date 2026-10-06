"""放量启动回放：用 Gate 历史 1 分钟 K 线驱动与实时监控相同的检测器，逐分钟解释为什么提醒或没有提醒。

交易所 1 分钟 K 线带计价成交额（sum），与实时监控由成交聚合的口径相同（都排除内部成交），
所以不必逐笔拉成交，几个小时的区间也只需几次请求。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_launch_detector
from price_alert.formatting import beijing_time, format_amount, format_price
from price_alert.gate import GATE_MINUTE_HISTORY
from price_alert.launch.alerts import LaunchAlert, describe_launch_rule, format_launch_alert
from price_alert.launch.config import LaunchConfig
from price_alert.launch.detector import OUTCOME_LABELS, LaunchEvaluation, LaunchOutcome
from price_alert.launch.rule import CONDITION_LABELS, LaunchRule
from price_alert.models import Candle

MINUTE = timedelta(minutes=1)
# 默认输出只列满足条件或被插针拦下（影响之后冷却）的分钟，其余只计入统计。
_QUIET_OUTCOMES = frozenset({LaunchOutcome.INSUFFICIENT_DATA, LaunchOutcome.FAILED})


@dataclass(frozen=True, slots=True)
class LaunchReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 预热从这里开始：判定一次所需的全部历史。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class LaunchReplayResult:
    plan: LaunchReplayPlan
    rule: LaunchRule
    evaluations: list[LaunchEvaluation]
    alerts: list[LaunchAlert]
    candles_fed: int
    warmup_candles: int
    price_decimals: int | None = None


def plan_launch_replay(
    symbol: str,
    start: datetime,
    end: datetime,
    rule: LaunchRule,
    now: datetime | None = None,
) -> LaunchReplayPlan:
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    aligned = datetime.fromtimestamp(int(start.timestamp()) // 60 * 60, tz=start.tzinfo)
    warmup = MINUTE * rule.history_minutes
    warmup_start = aligned - warmup
    # 预先检查而不是等交易所返回含义不明的 400；与趋势回放同样留一分钟余量。
    earliest = (now or datetime.now(UTC)) - MINUTE * (GATE_MINUTE_HISTORY - 2)
    if warmup_start < earliest:
        raise ValueError(
            f"Gate 只保留最近 {GATE_MINUTE_HISTORY} 根 1 分钟 K 线（约 6.9 天），"
            f"开始时间最早为 {beijing_time(earliest + warmup + MINUTE, '%Y-%m-%d %H:%M')}（北京时间）"
        )
    return LaunchReplayPlan(symbol, start, end, warmup_start)


def run_launch_replay(
    config: LaunchConfig,
    plan: LaunchReplayPlan,
    candles: Sequence[Candle],
    price_decimals: int | None = None,
) -> LaunchReplayResult:
    """离线驱动检测器；从空白提醒记录开始，区间开始前已经提醒过的行情会在区间内再当作启动提醒。"""
    evaluations: list[LaunchEvaluation] = []
    detector = build_launch_detector(config, observer=evaluations.append)
    start = datetime.fromtimestamp(int(plan.start.timestamp()) // 60 * 60, tz=plan.start.tzinfo)
    seed = [candle for candle in candles if candle.timestamp < start]
    # K 线自带计价成交额，不需要合约面值换算；面值只用于实时成交。
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=price_decimals, quanto_multiplier=1.0)
    alerts: list[LaunchAlert] = []
    fed = 0
    for candle in candles:
        # 只喂收盘不晚于结束时间的 K 线，判定时间点才全部落在区间内。
        if candle.timestamp < start or candle.timestamp + MINUTE > plan.end:
            continue
        fed += 1
        alerts.extend(detector.add_candle(plan.symbol, candle))
    return LaunchReplayResult(plan, detector.rule, evaluations, alerts, fed, len(seed), price_decimals)


def _format_row(evaluation: LaunchEvaluation, rule: LaunchRule, decimals: int | None) -> str:
    parts = [beijing_time(evaluation.timestamp, "%H:%M")]
    metrics = evaluation.metrics
    if metrics is not None:
        ratio = "基准无成交" if metrics.volume_ratio is None else f"量比 {metrics.volume_ratio:.1f}"
        parts.append(
            f"{format_price(metrics.start_price, decimals)} → {format_price(metrics.end_price, decimals)} "
            f"{metrics.change_percent:+.2f}%"
        )
        parts.append(f"{rule.window_minutes}分钟成交额 {format_amount(metrics.window_quote)} {ratio}")
    label = OUTCOME_LABELS[evaluation.outcome]
    if evaluation.outcome is LaunchOutcome.FAILED and metrics is not None:
        label += "：" + "、".join(CONDITION_LABELS[condition] for condition in metrics.failures)
    parts.append(label)
    return " | ".join(parts)


def _closeness(evaluation: LaunchEvaluation, rule: LaunchRule) -> tuple[int, float]:
    # 先比未满足的条件数，再比三道数值门槛中最弱的一项达到门槛的程度。
    metrics = evaluation.metrics
    assert metrics is not None
    ratio = metrics.volume_ratio / rule.min_volume_ratio if metrics.volume_ratio is not None else 1.0
    quote = metrics.window_quote / rule.min_window_quote if rule.min_window_quote > 0 else 1.0
    change = abs(metrics.change_percent) / rule.min_change_percent
    return (-len(metrics.failures), min(ratio, quote, change))


def render_launch_replay(result: LaunchReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    rule = result.rule
    decimals = result.price_decimals
    lines = [
        f"放量启动回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 1 分钟 K 线 {result.warmup_candles} 根，回放 {result.candles_fed} 根",
        describe_launch_rule(rule),
        "说明：回放从空白提醒记录开始；实时监控由成交聚合 K 线，与交易所 K 线可能有细微差别。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_launch_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有放量启动提醒。")
    lines.append("")

    in_range = [item for item in result.evaluations if plan.start <= item.timestamp <= plan.end]
    counts = Counter(item.outcome for item in in_range)
    lines.append(
        "判定统计："
        + "、".join(f"{OUTCOME_LABELS[outcome]} {counts[outcome]} 次" for outcome in LaunchOutcome if counts[outcome])
    )
    failures = Counter(
        condition for item in in_range if item.metrics is not None for condition in item.metrics.failures
    )
    if failures:
        lines.append(
            "  未满足的条件："
            + "、".join(f"{CONDITION_LABELS[condition]} {count} 次" for condition, count in failures.most_common())
        )
    failed = [item for item in in_range if item.outcome is LaunchOutcome.FAILED]
    if failed:
        closest = max(failed, key=lambda item: _closeness(item, rule))
        lines.append(f"  最接近触发：{_format_row(closest, rule, decimals)}")
    lines.append("")

    rows = [item for item in in_range if show_all or item.outcome not in _QUIET_OUTCOMES]
    if show_all:
        lines.append("逐分钟判定：")
    elif rows:
        lines.append("满足条件或被插针拦下的分钟（加 --all 查看每一分钟的判定）：")
    else:
        lines.append("没有满足条件或被插针拦下的分钟（加 --all 查看每一分钟的判定）。")
    lines.extend(f"  {_format_row(item, rule, decimals)}" for item in rows)
    return lines
