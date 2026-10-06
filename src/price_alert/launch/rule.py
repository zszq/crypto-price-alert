"""放量启动的判定规则：纯函数，实时检测、回放与测试共用。

思路取自量化里常见的「相对成交量（RVOL）+ 价格突破」：最近一小段时间的成交额相对自身基准期显著放大，
同时价格朝一个方向明显移动、并越过基准期的高点（或低点），才算从沉寂中启动。只看量会被对倒、大单
误导，只看价会被冷门币的单笔插针误导，两者同时出现才说明有真金白银推动价格离开原来的区间。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from price_alert.models import Candle

Direction = Literal["surge", "drop"]


class Condition(StrEnum):
    VOLUME = "volume"
    RATIO = "ratio"
    CHANGE = "change"
    BREAKOUT = "breakout"
    DIRECTION = "direction"
    OVERSHOOT = "overshoot"


CONDITION_LABELS: dict[Condition, str] = {
    Condition.VOLUME: "窗口成交额不足",
    Condition.RATIO: "量比不足",
    Condition.CHANGE: "涨跌幅不足",
    Condition.BREAKOUT: "未突破基准期高低点",
    Condition.DIRECTION: "下跌（未开启下跌提醒）",
    Condition.OVERSHOOT: "涨跌幅过大（插针）",
}


@dataclass(frozen=True, slots=True)
class LaunchRule:
    window_minutes: int
    baseline_minutes: int
    min_volume_ratio: float
    min_window_quote: float
    min_change_percent: float
    require_breakout: bool = True
    alert_drops: bool = False
    cooldown_minutes: int = 15
    realert_step_percent: float = 5.0
    max_change_percent: float | None = 15.0

    def __post_init__(self) -> None:
        if self.window_minutes < 1:
            raise ValueError("window_minutes 必须至少为 1")
        if self.baseline_minutes <= self.window_minutes:
            raise ValueError("baseline_minutes 必须大于 window_minutes")
        if self.max_change_percent is not None and self.max_change_percent <= self.min_change_percent:
            raise ValueError("max_change_percent 必须大于 min_change_percent")

    @property
    def history_minutes(self) -> int:
        """判定一次所需的已收盘 1 分钟 K 线：基准期加观察窗口。"""
        return self.baseline_minutes + self.window_minutes

    def overshoots(self, change_percent: float) -> bool:
        """涨跌幅（取绝对值）超过上限：这么短时间走完这么多，提醒时已在插针的高点。"""
        return self.max_change_percent is not None and abs(change_percent) > self.max_change_percent


@dataclass(frozen=True, slots=True)
class LaunchMetrics:
    direction: Direction
    # 窗口开始前最后一根 K 线的收盘价，即这一小段行情的起点。
    start_price: float
    end_price: float
    change_percent: float
    window_quote: float
    # 基准期内与窗口等长的平均成交额。
    baseline_quote: float
    # 量比 = 窗口成交额 ÷ 基准平均，即最近一个窗口（默认 5 分钟）的成交额是它平时同样长度成交额的多少倍；
    # 基准期完全无成交时为 None，视为满足。
    volume_ratio: float | None
    # 基准期最高价（上涨）或最低价（下跌）：收盘越过它才算离开原来的区间。
    breakout_price: float
    failures: tuple[Condition, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def broke_out(self) -> bool:
        return breaks_out(self.direction, self.end_price, self.breakout_price)


def still_launching(metrics: LaunchMetrics, price: float, rule: LaunchRule) -> bool:
    """用更新的成交价按同样的涨跌幅与突破门槛复核：空档之后才结算的分钟，收盘价已经过期，
    价格已经回落的不应再带着旧价格提醒。"""
    sign = 1 if metrics.direction == "surge" else -1
    if (price / metrics.start_price - 1) * 100 * sign < rule.min_change_percent:
        return False
    return not rule.require_breakout or breaks_out(metrics.direction, price, metrics.breakout_price)


def breaks_out(direction: Direction, price: float, breakout_price: float) -> bool:
    """按收盘价判断是否离开基准期区间：只靠影线刺破、收盘又回到区间内的不算。"""
    return price > breakout_price if direction == "surge" else price < breakout_price


def evaluate_launch(history: Sequence[Candle], rule: LaunchRule) -> LaunchMetrics:
    """history 为最近 history_minutes 根连续的已收盘 1 分钟 K 线（旧到新），最后 window_minutes 根是观察窗口。"""
    if len(history) != rule.history_minutes:
        raise ValueError(f"需要 {rule.history_minutes} 根 K 线，实际 {len(history)} 根")
    baseline = history[: rule.baseline_minutes]
    window = history[rule.baseline_minutes :]
    start_price = baseline[-1].close
    end_price = window[-1].close
    change_percent = (end_price / start_price - 1) * 100
    direction: Direction = "surge" if change_percent >= 0 else "drop"

    window_quote = sum(candle.quote_volume for candle in window)
    # 用均值而不是中位数：冷门币多数分钟零成交，中位数常为 0，任何一点成交都会显得「放量」。
    baseline_quote = sum(candle.quote_volume for candle in baseline) / rule.baseline_minutes * rule.window_minutes
    volume_ratio = window_quote / baseline_quote if baseline_quote > 0 else None
    if direction == "surge":
        breakout_price = max(candle.high for candle in baseline)
    else:
        breakout_price = min(candle.low for candle in baseline)

    failures: list[Condition] = []
    if direction == "drop" and not rule.alert_drops:
        failures.append(Condition.DIRECTION)
    # 量比只衡量相对放大，基准接近零的冷门币一点成交就是几十倍，必须同时有绝对成交额兜底。
    if window_quote < rule.min_window_quote:
        failures.append(Condition.VOLUME)
    if volume_ratio is not None and volume_ratio < rule.min_volume_ratio:
        failures.append(Condition.RATIO)
    if abs(change_percent) < rule.min_change_percent:
        failures.append(Condition.CHANGE)
    if rule.overshoots(change_percent):
        failures.append(Condition.OVERSHOOT)
    if rule.require_breakout and not breaks_out(direction, end_price, breakout_price):
        failures.append(Condition.BREAKOUT)
    return LaunchMetrics(
        direction=direction,
        start_price=start_price,
        end_price=end_price,
        change_percent=change_percent,
        window_quote=window_quote,
        baseline_quote=baseline_quote,
        volume_ratio=volume_ratio,
        breakout_price=breakout_price,
        failures=tuple(failures),
    )
