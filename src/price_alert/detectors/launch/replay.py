"""放量启动回放：用 Gate 历史 1 分钟 K 线驱动与实时监控相同的检测器，逐分钟解释为什么提醒或没有提醒。

交易所 1 分钟 K 线带计价成交额（sum），与实时监控由成交聚合的口径相同（都排除内部成交），
所以不必逐笔拉成交，几个小时的区间也只需几次请求。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from price_alert.assembly import build_launch_detector
from price_alert.common.formatting import beijing_time, format_amount, format_price
from price_alert.common.minute_bars import MINUTE
from price_alert.common.models import Candle
from price_alert.detectors.launch.alerts import LABELS, LaunchAlert, describe_launch_rule, format_launch_alert
from price_alert.detectors.launch.config import LaunchConfig
from price_alert.detectors.launch.detector import OUTCOME_LABELS, LaunchEvaluation, LaunchOutcome
from price_alert.detectors.launch.rule import CONDITION_LABELS, Condition, LaunchRule
from price_alert.detectors.replay import MinuteReplayPlan, plan_minute_replay

# 默认输出只列满足条件或被插针拦下（影响之后冷却）的分钟，其余只计入统计。
_QUIET_OUTCOMES = frozenset({LaunchOutcome.INSUFFICIENT_DATA, LaunchOutcome.FAILED})


@dataclass(frozen=True, slots=True)
class LaunchSideReplay:
    """一个方向（一个检测器）的回放结果。"""

    rule: LaunchRule
    evaluations: list[LaunchEvaluation]
    alerts: list[LaunchAlert]


@dataclass(frozen=True, slots=True)
class LaunchReplayResult:
    plan: MinuteReplayPlan
    sides: list[LaunchSideReplay]
    candles_fed: int
    warmup_candles: int
    price_decimals: int | None = None

    @property
    def alerts(self) -> list[LaunchAlert]:
        # 两个方向的提醒按时间合在一起，与实时监控看到的顺序一致。
        return sorted((alert for side in self.sides for alert in side.alerts), key=lambda alert: alert.timestamp)


def plan_launch_replay(
    symbol: str,
    start: datetime,
    end: datetime,
    rules: Sequence[LaunchRule],
    now: datetime | None = None,
) -> MinuteReplayPlan:
    if not rules:
        raise ValueError("放量上涨与放量下跌都已关闭，没有可回放的规则")
    # 两个方向的基准期可以不同，按较长的一边预热，两个检测器都能从区间开始就判定。
    return plan_minute_replay(symbol, start, end, max(rule.history_minutes for rule in rules), now)


def run_launch_replay(
    config: LaunchConfig,
    plan: MinuteReplayPlan,
    candles: Sequence[Candle],
    price_decimals: int | None = None,
) -> LaunchReplayResult:
    """离线驱动每个启用方向的检测器；从空白提醒记录开始，区间开始前已经提醒过的行情会在区间内再当作启动提醒。"""
    start = datetime.fromtimestamp(int(plan.start.timestamp()) // 60 * 60, tz=plan.start.tzinfo)
    seed = [candle for candle in candles if candle.timestamp < start]
    replayed = [candle for candle in candles if candle.timestamp >= start and candle.timestamp + MINUTE <= plan.end]
    sides = []
    for rule in config.rules():
        evaluations: list[LaunchEvaluation] = []
        detector = build_launch_detector(rule, observer=evaluations.append)
        # K 线自带计价成交额，不需要合约面值换算；面值只用于实时成交。
        detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=price_decimals, quanto_multiplier=1.0)
        alerts: list[LaunchAlert] = []
        # 只喂收盘不晚于结束时间的 K 线，判定时间点才全部落在区间内。
        for candle in replayed:
            alerts.extend(detector.add_candle(plan.symbol, candle))
        sides.append(LaunchSideReplay(rule, evaluations, alerts))
    return LaunchReplayResult(plan, sides, len(replayed), len(seed), price_decimals)


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


def _render_side(side: LaunchSideReplay, plan: MinuteReplayPlan, decimals: int | None, show_all: bool) -> list[str]:
    rule = side.rule
    lines = [f"【{LABELS[rule.direction]}】{describe_launch_rule(rule)}"]
    in_range = [item for item in side.evaluations if plan.start <= item.timestamp <= plan.end]
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
    # 方向相反的分钟离触发最远，不参与「最接近触发」的比较。
    failed = [
        item
        for item in in_range
        if item.outcome is LaunchOutcome.FAILED and item.metrics is not None
        and Condition.DIRECTION not in item.metrics.failures
    ]
    if failed:
        closest = max(failed, key=lambda item: _closeness(item, rule))
        lines.append(f"  最接近触发：{_format_row(closest, rule, decimals)}")
    rows = [item for item in in_range if show_all or item.outcome not in _QUIET_OUTCOMES]
    if show_all:
        lines.append("逐分钟判定：")
    elif rows:
        lines.append("满足条件或被插针拦下的分钟（加 --all 查看每一分钟的判定）：")
    else:
        lines.append("没有满足条件或被插针拦下的分钟（加 --all 查看每一分钟的判定）。")
    lines.extend(f"  {_format_row(item, rule, decimals)}" for item in rows)
    return lines


def render_launch_replay(result: LaunchReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    decimals = result.price_decimals
    lines = [
        f"放量启动回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 1 分钟 K 线 {result.warmup_candles} 根，回放 {result.candles_fed} 根",
        "说明：回放从空白提醒记录开始；实时监控由成交聚合 K 线，与交易所 K 线可能有细微差别。",
        "",
    ]
    alerts = result.alerts
    if alerts:
        lines.append(f"提醒 {len(alerts)} 条：")
        lines.extend(f"  {format_launch_alert(alert)}" for alert in alerts)
    else:
        lines.append("区间内没有放量启动提醒。")
    for side in result.sides:
        lines.append("")
        lines.extend(_render_side(side, plan, decimals, show_all))
    return lines
