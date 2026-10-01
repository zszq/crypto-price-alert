"""趋势提醒回放：用 Gate 历史 1 分钟 K 线驱动与实时监控相同的趋势检测器，解释为什么提醒或没有提醒。

实时监控由成交聚合 1 分钟 K 线，这里直接用交易所的 1 分钟 K 线（同样由成交生成，且都排除内部成交），
所以不必逐笔拉成交，几个小时的区间也只需几次请求。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from price_alert.assembly import build_trend_detector
from price_alert.formatting import beijing_time, format_price
from price_alert.gate import GateRestClient
from price_alert.models import Candle
from price_alert.trend.alerts import TrendAlert, describe_trend_rule, format_trend_alert
from price_alert.trend.config import TrendConfig
from price_alert.trend.detector import OUTCOME_LABELS, TrendDetector, TrendEvaluation, TrendOutcome
from price_alert.trend.pattern import CONDITION_LABELS, TrendRule

MINUTE = timedelta(minutes=1)
# 单次请求的 K 线根数，低于 Gate 的 2000 上限并留出余量。
_CHUNK_CANDLES = 1000
# 默认输出只列形态成立的判定，其余只计入统计。
_QUIET_OUTCOMES = frozenset({TrendOutcome.INSUFFICIENT_DATA, TrendOutcome.NO_DIRECTION, TrendOutcome.FAILED})


@dataclass(frozen=True, slots=True)
class TrendReplayPlan:
    symbol: str
    start: datetime
    end: datetime
    # 预热从这里开始：最长周期判定一次所需的全部历史。
    warmup_start: datetime


@dataclass(frozen=True, slots=True)
class TrendReplayResult:
    plan: TrendReplayPlan
    rules: tuple[TrendRule, ...]
    evaluations: list[TrendEvaluation]
    alerts: list[TrendAlert]
    candles_fed: int
    warmup_candles: int
    price_decimals: int | None = None


def plan_trend_replay(symbol: str, start: datetime, end: datetime, detector: TrendDetector) -> TrendReplayPlan:
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    aligned = datetime.fromtimestamp(int(start.timestamp()) // 60 * 60, tz=start.tzinfo)
    # warmup_candles 含一根未收盘 K 线，回放只需已收盘的部分。
    return TrendReplayPlan(symbol, start, end, aligned - MINUTE * (detector.warmup_candles - 1))


def fetch_minute_candles(rest: GateRestClient, symbol: str, start: datetime, end: datetime) -> list[Candle]:
    """取开盘时间在 [start, end) 内的 1 分钟 K 线，按单次上限分段请求。"""
    candles: dict[datetime, Candle] = {}
    cursor = start
    while cursor < end:
        # Gate 的 from/to 两端都包含，分段的 to 取下一段起点前一分钟，避免重复。
        stop = min(cursor + MINUTE * _CHUNK_CANDLES, end)
        for candle in rest.fetch_candles_between(symbol, "1m", cursor, stop - MINUTE):
            candles[candle.timestamp] = candle
        cursor = stop
    return [candles[key] for key in sorted(candles) if start <= key < end]


def run_trend_replay(
    config: TrendConfig,
    plan: TrendReplayPlan,
    candles: Sequence[Candle],
    price_decimals: int | None = None,
) -> TrendReplayResult:
    """离线驱动趋势检测器；从空白状态开始：区间开始时已在延续的走势，会在第一次判定时当作新的一段提醒。"""
    evaluations: list[TrendEvaluation] = []
    detector = build_trend_detector(config, observer=evaluations.append)
    start = datetime.fromtimestamp(int(plan.start.timestamp()) // 60 * 60, tz=plan.start.tzinfo)
    seed = [candle for candle in candles if candle.timestamp < start]
    detector.add_symbol(plan.symbol, seed, 0.0, price_decimals=price_decimals)
    alerts: list[TrendAlert] = []
    fed = 0
    for candle in candles:
        # 只喂收盘不晚于结束时间的 K 线，判定时间点才全部落在区间内。
        if candle.timestamp < start or candle.timestamp + MINUTE > plan.end:
            continue
        fed += 1
        alerts.extend(detector.add_candle(plan.symbol, candle))
    return TrendReplayResult(plan, detector.rules, evaluations, alerts, fed, len(seed), price_decimals)


def _format_row(evaluation: TrendEvaluation, decimals: int | None) -> str:
    parts = [beijing_time(evaluation.timestamp, "%H:%M"), evaluation.period]
    metrics = evaluation.metrics
    if metrics is not None:
        parts.append(
            f"{format_price(metrics.start_price, decimals)} → {format_price(metrics.end_price, decimals)} "
            f"{metrics.change_percent:+.2f}%"
        )
        parts.append(
            f"反向 {metrics.counter_candles} 根 反弹 {metrics.rebound_ratio:.0%} 实体 {metrics.body_ratio:.0%}"
            f" 单根 {metrics.single_candle_ratio:.0%}"
        )
    label = OUTCOME_LABELS[evaluation.outcome]
    if evaluation.outcome is TrendOutcome.FAILED and metrics is not None:
        label += "：" + "、".join(CONDITION_LABELS[condition] for condition in metrics.failures)
    parts.append(label)
    return " | ".join(parts)


def _closeness(evaluation: TrendEvaluation, rule: TrendRule) -> tuple[int, float]:
    # 先比未满足的条件数，再比累计幅度达标的程度。
    metrics = evaluation.metrics
    assert metrics is not None
    return (-len(metrics.failures), abs(metrics.change_percent) / rule.min_change_percent)


def _period_summary(rule: TrendRule, evaluations: Sequence[TrendEvaluation], decimals: int | None) -> list[str]:
    if not evaluations:
        return [f"{rule.period}：区间内没有收盘的 K 线"]
    counts = Counter(item.outcome for item in evaluations)
    distribution = "、".join(
        f"{OUTCOME_LABELS[outcome]} {counts[outcome]} 次" for outcome in TrendOutcome if counts[outcome]
    )
    lines = [f"{rule.period}：{distribution}"]
    failures = Counter(
        condition for item in evaluations if item.metrics is not None for condition in item.metrics.failures
    )
    if failures:
        lines.append(
            "  未满足的条件："
            + "、".join(f"{CONDITION_LABELS[condition]} {count} 次" for condition, count in failures.most_common())
        )
    failed = [item for item in evaluations if item.outcome is TrendOutcome.FAILED]
    if failed:
        closest = max(failed, key=lambda item: _closeness(item, rule))
        lines.append(f"  最接近成立：{_format_row(closest, decimals)}")
    return lines


def render_trend_replay(result: TrendReplayResult, show_all: bool = False) -> list[str]:
    plan = result.plan
    decimals = result.price_decimals
    lines = [
        f"趋势回放 {plan.symbol}：{beijing_time(plan.start)} ～ {beijing_time(plan.end)}（北京时间）",
        f"数据：预热 1 分钟 K 线 {result.warmup_candles} 根，回放 {result.candles_fed} 根",
        *(describe_trend_rule(rule) for rule in result.rules),
        "说明：回放从空白状态开始，区间开始时已在延续的走势也会提醒；"
        "实时监控由成交聚合 K 线，与交易所 K 线可能有细微差别。",
        "",
    ]
    if result.alerts:
        lines.append(f"提醒 {len(result.alerts)} 条：")
        lines.extend(f"  {format_trend_alert(alert)}" for alert in result.alerts)
    else:
        lines.append("区间内没有趋势提醒。")
    lines.append("")

    in_range = [item for item in result.evaluations if plan.start <= item.timestamp <= plan.end]
    lines.append("各周期判定统计：")
    for rule in result.rules:
        own = [item for item in in_range if item.period == rule.period]
        lines.extend(f"  {line}" for line in _period_summary(rule, own, decimals))
    lines.append("")

    rows = [item for item in in_range if show_all or item.outcome not in _QUIET_OUTCOMES]
    if show_all:
        lines.append("逐根判定：")
    elif rows:
        lines.append("形态成立的时刻（加 --all 查看每一根 K 线的判定）：")
    else:
        lines.append("没有任何周期的形态成立（加 --all 查看每一根 K 线的判定）。")
    lines.extend(f"  {_format_row(item, decimals)}" for item in rows)
    return lines
