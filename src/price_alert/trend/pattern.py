"""K 线形态趋势判定：对某个周期最近 N 根已收盘 K 线整体打分，而不是逐根套硬规则。

四个核心条件（全部满足才算趋势）：
- 最大反弹比例：从滚动极值算起的最大反向回撤不超过累计位移的一定比例，容忍小阳线和十字星，排除真正的反转；
- 顺势 K 线与收盘递进比例：方向一致；
- 实体占比：整体上影线/下影线不长，允许个别长影线；
- 累计幅度：百分比门槛，另可要求不低于趋势前基准 ATR 的若干倍。
附加的量能条件要求窗口平均成交量高于趋势前基准；长影线不参与开始判定，由检测器用作衰竭提示。

本模块只有纯函数，实时检测、回放与参数调研共用同一份判定。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from price_alert.models import Candle

PeriodName = Literal["1m", "3m", "5m", "15m"]
# 趋势周期全部由 1 分钟 K 线聚合而来，这里是周期到分钟数的唯一映射。
PERIOD_MINUTES: dict[PeriodName, int] = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}
Direction = Literal["surge", "drop"]


@dataclass(frozen=True, slots=True)
class TrendRule:
    """一个周期上的趋势门槛。可选条件为 None 表示不启用。"""

    period: PeriodName
    candles: int
    min_change_percent: float
    min_change_atr: float | None
    max_rebound_ratio: float
    min_trend_candle_ratio: float
    min_step_ratio: float
    min_body_ratio: float
    min_volume_ratio: float | None
    baseline_candles: int

    def __post_init__(self) -> None:
        # 检测器直接信任这些值，非法组合在构造时拦下，比运行中静默不提醒更容易发现。
        if self.candles < 2 or self.baseline_candles < 1:
            raise ValueError("candles 至少为 2，baseline_candles 至少为 1")
        if self.min_change_percent <= 0:
            raise ValueError("min_change_percent 必须大于 0")

    @property
    def minutes(self) -> int:
        return PERIOD_MINUTES[self.period]

    @property
    def needs_baseline(self) -> bool:
        return self.min_change_atr is not None or self.min_volume_ratio is not None

    @property
    def history_minutes(self) -> int:
        """判定一次所需的 1 分钟 K 线数：窗口加基准。"""
        return (self.candles + self.baseline_candles) * self.minutes


class Condition(StrEnum):
    """未满足的条件，按判定展示顺序排列。"""

    CHANGE = "change"
    CHANGE_ATR = "change_atr"
    REBOUND = "rebound"
    TREND_CANDLES = "trend_candles"
    STEPS = "steps"
    BODY = "body"
    VOLUME = "volume"
    BASELINE = "baseline"


CONDITION_LABELS: dict[Condition, str] = {
    Condition.CHANGE: "累计幅度不足",
    Condition.CHANGE_ATR: "累计幅度不足基准 ATR 倍数",
    Condition.REBOUND: "反弹过大",
    Condition.TREND_CANDLES: "顺势 K 线占比不足",
    Condition.STEPS: "收盘递进占比不足",
    Condition.BODY: "实体占比不足（影线过长）",
    Condition.VOLUME: "量能不足",
    Condition.BASELINE: "趋势前基准数据不足",
}


@dataclass(frozen=True, slots=True)
class PatternMetrics:
    """一次判定的全部指标；比例都按趋势方向计算，便于与门槛直接对比。"""

    direction: Direction
    start_price: float
    end_price: float
    extreme_price: float
    change_percent: float
    rebound_ratio: float
    trend_candle_ratio: float
    step_ratio: float
    body_ratio: float
    volume_ratio: float | None
    change_atr: float | None
    failures: tuple[Condition, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_pattern(
    window: Sequence[Candle],
    baseline: Sequence[Candle],
    rule: TrendRule,
) -> PatternMetrics | None:
    """按规则评估窗口 K 线；窗口根数不对或首尾价格相同（没有方向）时返回 None。

    baseline 是紧邻窗口之前的 K 线，用来衡量“趋势开始前”的波动与量能：
    趋势本身会抬高近期波动率和成交量，拿含趋势的数据当基准会让门槛随行情水涨船高。
    """
    if len(window) != rule.candles:
        return None
    start = window[0].open
    end = window[-1].close
    if end == start:
        return None
    direction: Direction = "surge" if end > start else "drop"
    # sign 把下跌翻成上涨来计算，两个方向共用同一套公式。
    sign = 1.0 if direction == "surge" else -1.0
    net_move = abs(end - start)

    trend_candles = sum(1 for candle in window if sign * (candle.close - candle.open) > 0)
    steps = 0
    previous = start
    extreme = start
    max_rebound = 0.0
    for candle in window:
        if sign * (candle.close - previous) > 0:
            steps += 1
        previous = candle.close
        # 反弹只看收盘价：盘中影线已由实体占比约束，收盘才代表这一根的定价。
        max_rebound = max(max_rebound, sign * (extreme - candle.close))
        if sign * (candle.close - extreme) > 0:
            extreme = candle.close

    total_range = sum(candle.high - candle.low for candle in window)
    body_ratio = sum(abs(candle.close - candle.open) for candle in window) / total_range if total_range > 0 else 0.0
    change_percent = (end - start) / start * 100.0

    failures: list[Condition] = []
    if abs(change_percent) < rule.min_change_percent:
        failures.append(Condition.CHANGE)
    change_atr: float | None = None
    volume_ratio: float | None = None
    baseline_ready = len(baseline) >= rule.baseline_candles
    if rule.min_change_atr is not None and baseline_ready:
        change_atr = _ratio(net_move, _mean_true_range(baseline))
        if change_atr < rule.min_change_atr:
            failures.append(Condition.CHANGE_ATR)
    rebound_ratio = max_rebound / net_move
    if rebound_ratio > rule.max_rebound_ratio:
        failures.append(Condition.REBOUND)
    trend_candle_ratio = trend_candles / len(window)
    if trend_candle_ratio < rule.min_trend_candle_ratio:
        failures.append(Condition.TREND_CANDLES)
    step_ratio = steps / len(window)
    if step_ratio < rule.min_step_ratio:
        failures.append(Condition.STEPS)
    if body_ratio < rule.min_body_ratio:
        failures.append(Condition.BODY)
    if rule.min_volume_ratio is not None and baseline_ready:
        volume_ratio = _ratio(_mean(candle.volume for candle in window), _volume_baseline(baseline))
        if volume_ratio < rule.min_volume_ratio:
            failures.append(Condition.VOLUME)
    if rule.needs_baseline and not baseline_ready:
        # 新上市或刚恢复数据的合约没有足够的趋势前数据，宁可不报，也不拿不完整的基准放宽门槛。
        failures.append(Condition.BASELINE)

    return PatternMetrics(
        direction=direction,
        start_price=start,
        end_price=end,
        extreme_price=extreme,
        change_percent=change_percent,
        rebound_ratio=rebound_ratio,
        trend_candle_ratio=trend_candle_ratio,
        step_ratio=step_ratio,
        body_ratio=body_ratio,
        volume_ratio=volume_ratio,
        change_atr=change_atr,
        failures=tuple(failures),
    )


def format_multiple(value: float) -> str:
    # 基准全为 0 时比值为无穷大，显示成“inf”不直观。
    return "∞" if math.isinf(value) else f"{value:.1f}"


def adverse_shadow_ratio(candle: Candle, direction: Direction) -> float:
    """逆趋势一侧的影线占振幅比例：下跌看下影线（买盘承接），上涨看上影线（卖压）。"""
    total = candle.high - candle.low
    if total <= 0:
        return 0.0
    if direction == "drop":
        return (min(candle.open, candle.close) - candle.low) / total
    return (candle.high - max(candle.open, candle.close)) / total


def aggregate_candles(minutes: Sequence[Candle], period_minutes: int) -> list[Candle]:
    """把连续、按周期边界对齐的 1 分钟 K 线合并为周期 K 线；调用方保证根数是周期的整数倍。"""
    candles: list[Candle] = []
    for index in range(0, len(minutes) - period_minutes + 1, period_minutes):
        group = minutes[index : index + period_minutes]
        candles.append(
            Candle(
                group[0].timestamp,
                group[0].open,
                max(candle.high for candle in group),
                min(candle.low for candle in group),
                group[-1].close,
                sum(candle.quote_volume for candle in group),
                sum(candle.volume for candle in group),
            )
        )
    return candles


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _volume_baseline(baseline: Sequence[Candle]) -> float:
    # 取中位数：趋势前若有一波放量急拉急砸，均值会被少数几根巨量 K 线抬高，随后的趋势反而显得“缩量”。
    # 冷门合约过半的分钟无成交时中位数为 0，量能条件会恒成立，此时退回均值。
    volumes = [candle.volume for candle in baseline]
    median = _median(volumes)
    return median if median > 0 else _mean(volumes)


def _median(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    middle = len(items) // 2
    return items[middle] if len(items) % 2 else (items[middle - 1] + items[middle]) / 2


def _mean_true_range(candles: Sequence[Candle]) -> float:
    total = 0.0
    previous_close: float | None = None
    for candle in candles:
        true_range = candle.high - candle.low
        if previous_close is not None:
            true_range = max(true_range, abs(candle.high - previous_close), abs(candle.low - previous_close))
        total += true_range
        previous_close = candle.close
    return total / len(candles) if candles else 0.0


def _ratio(value: float, base: float) -> float:
    # 基准为 0（长期无成交、价格不动）时任何变化都是相对无穷大，按通过处理。
    return value / base if base > 0 else math.inf
