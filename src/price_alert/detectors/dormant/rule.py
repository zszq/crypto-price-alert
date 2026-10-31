"""冷门启动的判定规则：纯函数，实时检测、回放与测试共用。

冷门币一两笔单子就能撑起涨幅，所以除了涨幅与成交额，还要求多分钟有成交（有人在持续买）、涨前平稳、
不靠一两分钟跳涨、实体饱满。最后一分钟成交额、后半窗口成交额占比代码默认不限；
效率、量比、回吐只给慢启动一组用，代码默认不限。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from price_alert.common.models import Candle

# 「之前 24 小时」：判断是否冷门的区间。不开放配置：冷门门槛 max_prior_quote 是 24 小时的成交额，
# 换了长度门槛就失去含义。
PRIOR_MINUTES = 1440
# 「涨前波动」的统计区间（窗口之前的分钟数），不开放配置：涨前波动上限与倍数门槛都是按这个长度定的。
# 须短于 PRIOR_MINUTES，保留的 K 线才够用。
CALM_MINUTES = 240


class DormantGroup(StrEnum):
    """冷门启动的参数组：同一个检测器、同一份 K 线，各组独立判定、独立冷却。"""

    MAIN = "main"
    SLOW = "slow"


class Condition(StrEnum):
    PRIOR = "prior"
    VOLUME = "volume"
    CHANGE = "change"
    ACTIVE = "active"
    CALM = "calm"
    VOLATILE = "volatile"
    JUMP = "jump"
    BODY = "body"
    LAST = "last"
    LATE = "late"
    EFFICIENCY = "efficiency"
    RATIO = "ratio"
    RETRACE = "retrace"


CONDITION_LABELS: dict[Condition, str] = {
    Condition.PRIOR: "之前 24 小时成交额过多（不冷门）",
    Condition.VOLUME: "窗口成交额不足",
    Condition.CHANGE: "涨幅不足",
    Condition.ACTIVE: "有成交的分钟数不足",
    Condition.CALM: "涨幅相对涨前波动的倍数不足",
    Condition.VOLATILE: "涨前波动过大（不够平稳）",
    Condition.JUMP: "单分钟涨幅过大",
    Condition.BODY: "总实体占比不足（影线过多）",
    Condition.LAST: "最后一分钟成交额不足",
    Condition.LATE: "后半窗口成交额占比不足（买盘没跟上）",
    Condition.EFFICIENCY: "拉升不够顺滑（来回拉扯）",
    Condition.RATIO: "量比不足",
    Condition.RETRACE: "从窗口最高价回落过多",
}


def late_minutes(window_minutes: int) -> int:
    """「后半窗口」的分钟数：最后 ⌈W/2⌉ 根（奇数窗口时中间那根算后半）。"""
    return (window_minutes + 1) // 2


@dataclass(frozen=True, slots=True)
class DormantRule:
    window_minutes: int = 10
    max_prior_quote: float = 1_000_000.0
    min_window_quote: float = 5_000.0
    min_active_minutes: int = 5
    min_change_percent: float = 8.0
    # 窗口内单根 K 线收盘相对前一根收盘的涨幅上限：一两分钟跳上去的多是几笔单子拉的，之后容易回落。
    max_minute_change_percent: float = 5.0
    # 窗口总实体占比（实体和 ÷ 振幅和）下限：上影线多说明冲高被砸回，拉升没站稳。
    min_body_ratio: float = 0.6
    # 窗口涨幅 ÷ 涨前波动（窗口之前 CALM_MINUTES 分钟平均每分钟真实波幅）的下限。
    min_calm_multiple: float = 10.0
    # 涨前波动（%/分钟）上限，None 不限：倍数只看相对大小，挡不住涨前乱跳但涨得够多的币。
    max_pre_volatility_percent: float | None = 0.5
    # 窗口最后一分钟成交额下限（USDT），0 即不限：最后一分钟已没人买的，多是几笔单子拉完就停了。
    min_last_minute_quote: float = 0.0
    # 后半窗口（最后 late_minutes 根）成交额占比下限，0 即不限：成交集中在前半段、后面跟不上的容易回落。
    min_late_quote_share: float = 0.0
    cooldown_minutes: int = 60
    # 冷却期内价格比上次提醒价再涨这么多（%）就再报「第 N 波」，None 即冷却内不再报：
    # 一段拉升中再涨一截同样值得提醒，冷却只挡原地不动的重复。
    realert_step_percent: float | None = 3.0
    # 效率（窗口净涨幅价差 ÷ 逐分钟收盘价差绝对值之和）下限，0 即不限：慢启动要一路往上，不是来回拉扯。
    min_efficiency: float = 0.0
    # 量比（窗口成交额 ÷ 之前 24 小时同长度的平均成交额）下限，0 即不限：慢启动涨幅小，要求成交确实放大。
    min_volume_ratio: float = 0.0
    # 从窗口最高价回落的比例上限（回落 ÷ 窗口起点到最高价的涨幅），None 不限：提醒时还要在高位附近。
    max_retrace: float | None = None
    # 跟踪补报：提醒（含补报）后收盘价比上次提醒价再涨这么多（%）就补报下一波，不看其他条件；None 即不跟踪。
    # 不同于 realert_step_percent：慢启动启动后涨前波动、效率很快不再满足，要求条件仍满足的第 N 波几乎报不出来。
    follow_step_percent: float | None = None
    # 距上次提醒（含补报）超过这么久（分钟）还没再涨够，就停止跟踪：一路涨就一路跟，停涨了就结束。
    follow_minutes: int = 240
    group: DormantGroup = DormantGroup.MAIN

    def __post_init__(self) -> None:
        if self.realert_step_percent is not None and self.realert_step_percent <= 0:
            # 0 会让冷却期内每分钟都补报，等于没有冷却。
            raise ValueError("realert_step_percent 必须大于 0")
        if self.follow_step_percent is not None and self.follow_step_percent <= 0:
            # 0 会让跟踪期内每分钟都补报。
            raise ValueError("follow_step_percent 必须大于 0")
        if self.follow_minutes < 1:
            raise ValueError("follow_minutes 必须至少为 1")
        if self.window_minutes < 1:
            raise ValueError("window_minutes 必须至少为 1")
        if not 1 <= self.min_active_minutes <= self.window_minutes:
            raise ValueError("min_active_minutes 必须在 1 到 window_minutes 之间")

    @property
    def history_minutes(self) -> int:
        """判定一次所需的已收盘 1 分钟 K 线：之前 24 小时加观察窗口。"""
        return PRIOR_MINUTES + self.window_minutes


@dataclass(frozen=True, slots=True)
class DormantMetrics:
    # 窗口开始前最后一根 K 线的收盘价，即这一段拉升的起点。
    start_price: float
    end_price: float
    change_percent: float
    window_quote: float
    # 窗口内有成交的分钟数。
    active_minutes: int
    # 窗口之前 24 小时的成交额。
    prior_quote: float
    # 涨前波动：窗口之前 CALM_MINUTES 分钟平均每分钟的真实波幅（%），一字横盘为 0。
    pre_volatility_percent: float
    # 窗口涨幅 ÷ 涨前波动；涨前波动为 0 时为无穷大。
    calm_multiple: float
    # 窗口内单分钟收盘涨幅的最大值（%）。
    max_minute_change_percent: float
    # 窗口内各根 K 线实体之和 ÷ 振幅之和；窗口里全是没有振幅的平线时为 0。
    body_ratio: float
    # 窗口最后一根 K 线的成交额。
    last_minute_quote: float
    # 最后 late_minutes 根的成交额 ÷ 窗口成交额；窗口没有成交时为 0。
    late_quote_share: float
    # 窗口净涨幅的价差 ÷ 逐分钟收盘价差绝对值之和（含窗口前一根到第一根），收盘价一直不动时为 0。
    efficiency: float
    # 窗口成交额 ÷（之前 24 小时成交额 ÷ 1440 × 窗口分钟数）；之前 24 小时没有成交时为无穷大。
    volume_ratio: float
    # （窗口最高价 − 最后收盘）÷（窗口最高价 − 起点）；最高价不高于起点时为 0。
    retrace: float
    failures: tuple[Condition, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def minute_range_percent(candle: Candle, previous_close: float) -> float:
    """一分钟的真实波幅（%）：跳空也算波动，没有成交的平线为 0。"""
    high = max(candle.high, previous_close)
    low = min(candle.low, previous_close)
    return (high - low) / previous_close * 100


def still_rising(metrics: DormantMetrics, price: float, rule: DormantRule) -> bool:
    """用更新的成交价按同样的涨幅门槛复核：空档之后才结算的分钟，收盘价已经过期，价格已回落的不应再提醒。"""
    return (price / metrics.start_price - 1) * 100 >= rule.min_change_percent


def evaluate_dormant(
    window: Sequence[Candle],
    start_price: float,
    prior_quote: float,
    pre_volatility_percent: float,
    rule: DormantRule,
) -> DormantMetrics:
    """window 为最近 window_minutes 根连续的已收盘 1 分钟 K 线（旧到新），start_price 为窗口前一根的收盘价，
    prior_quote 为窗口之前 24 小时的成交额，pre_volatility_percent 为窗口之前 CALM_MINUTES 分钟
    minute_range_percent 的平均值。

    这两项由调用方滚动维护后传入：沉寂很久的币一笔成交补出上千根平线、逐根判定时，每分钟重新求和会卡住成交循环。"""
    if len(window) != rule.window_minutes:
        raise ValueError(f"窗口需要 {rule.window_minutes} 根 K 线，实际 {len(window)} 根")
    end_price = window[-1].close
    change_percent = (end_price / start_price - 1) * 100
    window_quote = sum(candle.quote_volume for candle in window)
    active_minutes = sum(1 for candle in window if candle.quote_volume > 0)
    calm_multiple = change_percent / pre_volatility_percent if pre_volatility_percent > 0 else math.inf
    closes = [start_price, *(candle.close for candle in window)]
    max_minute_change = max((current / previous - 1) * 100 for previous, current in pairwise(closes))
    # 与趋势形态的总实体占比口径相同；影线不分上下，冷门币拉升中的长下影线同样说明盘口来回被扫。
    total_range = sum(candle.high - candle.low for candle in window)
    body_ratio = sum(abs(candle.close - candle.open) for candle in window) / total_range if total_range > 0 else 0.0
    last_minute_quote = window[-1].quote_volume
    late_quote = sum(candle.quote_volume for candle in window[-late_minutes(rule.window_minutes) :])
    late_quote_share = late_quote / window_quote if window_quote > 0 else 0.0
    path = sum(abs(current - previous) for previous, current in pairwise(closes))
    efficiency = abs(end_price - start_price) / path if path > 0 else 0.0
    average = prior_quote / PRIOR_MINUTES * rule.window_minutes
    volume_ratio = window_quote / average if average > 0 else math.inf
    high = max(candle.high for candle in window)
    retrace = (high - end_price) / (high - start_price) if high > start_price else 0.0

    failures: list[Condition] = []
    if prior_quote > rule.max_prior_quote:
        failures.append(Condition.PRIOR)
    if window_quote < rule.min_window_quote:
        failures.append(Condition.VOLUME)
    if change_percent < rule.min_change_percent:
        failures.append(Condition.CHANGE)
    if active_minutes < rule.min_active_minutes:
        failures.append(Condition.ACTIVE)
    if calm_multiple < rule.min_calm_multiple:
        failures.append(Condition.CALM)
    if rule.max_pre_volatility_percent is not None and pre_volatility_percent > rule.max_pre_volatility_percent:
        failures.append(Condition.VOLATILE)
    if max_minute_change > rule.max_minute_change_percent:
        failures.append(Condition.JUMP)
    if body_ratio < rule.min_body_ratio:
        failures.append(Condition.BODY)
    if last_minute_quote < rule.min_last_minute_quote:
        failures.append(Condition.LAST)
    if late_quote_share < rule.min_late_quote_share:
        failures.append(Condition.LATE)
    if efficiency < rule.min_efficiency:
        failures.append(Condition.EFFICIENCY)
    if volume_ratio < rule.min_volume_ratio:
        failures.append(Condition.RATIO)
    if rule.max_retrace is not None and retrace > rule.max_retrace:
        failures.append(Condition.RETRACE)
    return DormantMetrics(
        start_price=start_price,
        end_price=end_price,
        change_percent=change_percent,
        window_quote=window_quote,
        active_minutes=active_minutes,
        prior_quote=prior_quote,
        pre_volatility_percent=pre_volatility_percent,
        calm_multiple=calm_multiple,
        max_minute_change_percent=max_minute_change,
        body_ratio=body_ratio,
        last_minute_quote=last_minute_quote,
        late_quote_share=late_quote_share,
        efficiency=efficiency,
        volume_ratio=volume_ratio,
        retrace=retrace,
        failures=tuple(failures),
    )
