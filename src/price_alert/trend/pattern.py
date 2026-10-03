"""K 线形态趋势判定：某个周期最近 N 根已收盘 K 线是否构成一段“健康”的单边走势。

七个条件全部满足才算趋势：
- 累计涨跌幅：第一根开盘到最后一根收盘达到门槛，过滤横盘里幅度很小的连阴连阳；
- 反向 K 线根数：允许夹杂一两根反向 K 线，其余都要顺着趋势方向；实体占本根振幅不足门槛的
  小实体 K 线（十字星、长影线）没有推进价格，同样算反向；
- 首尾顺势：反向 K 线只能夹在中间，走势要以顺势 K 线开始、以顺势 K 线收尾；
- 反向后突破：每段反向 K 线之后的下一根，收盘要突破反向之前那根的收盘价，回调后重新推进才算趋势延续；
- 最大反弹比例：夹杂的反向 K 线只能是小幅回调，从滚动极值算起的反向回撤不能超过累计位移的一定比例；
- 总实体占比：整体上下影线不长，价格是一路推进而不是来回拉扯；
- 最大实体占比：最大的一根顺势 K 线实体不能占累计位移太多，涨跌是几根 K 线共同推出来的，而不是一根急拉急砸。

本模块只有纯函数，实时检测、回放与模拟共用同一份判定。
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """一个周期上的趋势门槛。"""

    period: PeriodName
    candles: int
    min_change_percent: float
    max_counter_candles: int
    max_rebound_ratio: float
    min_body_ratio: float
    min_candle_body_ratio: float
    max_single_candle_ratio: float

    def __post_init__(self) -> None:
        # 检测器直接信任这些值，非法组合在构造时拦下，比运行中静默不提醒更容易发现。
        if self.candles < 2:
            raise ValueError("candles 至少为 2")
        if not 0 <= self.max_counter_candles < self.candles:
            raise ValueError("max_counter_candles 必须小于 candles")
        if self.min_change_percent <= 0:
            raise ValueError("min_change_percent 必须大于 0")

    @property
    def minutes(self) -> int:
        return PERIOD_MINUTES[self.period]

    @property
    def history_minutes(self) -> int:
        """判定一次所需的 1 分钟 K 线数。"""
        return self.candles * self.minutes


class Condition(StrEnum):
    """未满足的条件，按判定展示顺序排列。"""

    CHANGE = "change"
    COUNTER_CANDLES = "counter_candles"
    COUNTER_EDGE = "counter_edge"
    COUNTER_RECOVERY = "counter_recovery"
    REBOUND = "rebound"
    BODY = "body"
    SINGLE_CANDLE = "single_candle"


CONDITION_LABELS: dict[Condition, str] = {
    Condition.CHANGE: "累计幅度不足",
    Condition.COUNTER_CANDLES: "反向 K 线过多",
    Condition.COUNTER_EDGE: "首尾 K 线反向",
    Condition.COUNTER_RECOVERY: "反向 K 线后未突破前收盘",
    Condition.REBOUND: "反弹过大",
    Condition.BODY: "总实体占比不足（影线过长）",
    Condition.SINGLE_CANDLE: "最大实体占比过大",
}


@dataclass(frozen=True, slots=True)
class PatternMetrics:
    """一次判定的全部指标，都按趋势方向计算，便于与门槛直接对比。"""

    direction: Direction
    start_price: float
    end_price: float
    change_percent: float
    counter_candles: int
    rebound_ratio: float
    body_ratio: float
    single_candle_ratio: float
    failures: tuple[Condition, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_pattern(window: Sequence[Candle], rule: TrendRule) -> PatternMetrics | None:
    """按规则评估窗口 K 线；窗口根数不对或首尾价格相同（没有方向）时返回 None。"""
    if len(window) != rule.candles:
        return None
    start = window[0].open
    end = window[-1].close
    if end == start:
        return None
    direction: Direction = "surge" if end > start else "drop"
    # sign 把下跌翻成上涨来计算，两个方向共用同一套公式。
    sign = 1.0 if direction == "surge" else -1.0

    counter_flags = [_is_counter(candle, sign, rule.min_candle_body_ratio) for candle in window]
    counter_candles = sum(counter_flags)
    extreme = start
    max_rebound = 0.0
    for candle in window:
        # 反弹只看收盘价：盘中影线已由总实体占比约束，收盘才代表这一根的定价。
        max_rebound = max(max_rebound, sign * (extreme - candle.close))
        if sign * (candle.close - extreme) > 0:
            extreme = candle.close
    net_move = abs(end - start)
    rebound_ratio = max_rebound / net_move
    single_candle_ratio = max(max(sign * (candle.close - candle.open) for candle in window), 0.0) / net_move
    total_range = sum(candle.high - candle.low for candle in window)
    body_ratio = sum(abs(candle.close - candle.open) for candle in window) / total_range if total_range > 0 else 0.0
    change_percent = (end - start) / start * 100.0

    failures: list[Condition] = []
    if abs(change_percent) < rule.min_change_percent:
        failures.append(Condition.CHANGE)
    if counter_candles > rule.max_counter_candles:
        failures.append(Condition.COUNTER_CANDLES)
    # 第一根反向说明走势还没开始，最后一根反向说明正在回调、能否收复未知，都不算已形成的趋势。
    if counter_flags[0] or counter_flags[-1]:
        failures.append(Condition.COUNTER_EDGE)
    if not _recovered_after_counter(window, counter_flags, sign):
        failures.append(Condition.COUNTER_RECOVERY)
    if rebound_ratio > rule.max_rebound_ratio:
        failures.append(Condition.REBOUND)
    if body_ratio < rule.min_body_ratio:
        failures.append(Condition.BODY)
    if single_candle_ratio > rule.max_single_candle_ratio:
        failures.append(Condition.SINGLE_CANDLE)

    return PatternMetrics(
        direction=direction,
        start_price=start,
        end_price=end,
        change_percent=change_percent,
        counter_candles=counter_candles,
        rebound_ratio=rebound_ratio,
        body_ratio=body_ratio,
        single_candle_ratio=single_candle_ratio,
        failures=tuple(failures),
    )


def _is_counter(candle: Candle, sign: float, min_candle_body_ratio: float) -> bool:
    """反向或没有推进价格的 K 线。

    健康的趋势应当每一根都在推进：十字星和长影线小实体 K 线虽然收在顺势一侧，但价格在这根里来回拉扯，
    总实体占比按窗口合计，会被其他大实体 K 线摊薄，所以逐根判定。
    """
    body = sign * (candle.close - candle.open)
    # body > 0 时振幅必然大于 0，用乘法比较免去除零判断。
    return body <= 0 or body < min_candle_body_ratio * (candle.high - candle.low)


def _recovered_after_counter(window: Sequence[Candle], counter_flags: Sequence[bool], sign: float) -> bool:
    """每段反向 K 线之后的下一根，收盘是否都突破了这段反向之前的收盘价。

    只比反向 K 线自己的收盘太宽松：先急跌两根、反弹一根、再小跌一根也能过，实际是急跌后的横盘。
    要求越过反向之前的收盘价，才说明回调之后价格重新推进。连续几根反向 K 线算一段，
    下一根指这段之后的第一根顺势 K 线。首尾的反向 K 线由 COUNTER_EDGE 单独判定，这里只看有前有后的。
    """
    # 当前这段反向 K 线之前的收盘价；None 表示不在反向段中。
    reference: float | None = None
    for index in range(1, len(window)):
        if counter_flags[index]:
            if reference is None:
                reference = window[index - 1].close
        elif reference is not None:
            if sign * (window[index].close - reference) <= 0:
                return False
            reference = None
    return True


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
            )
        )
    return candles
