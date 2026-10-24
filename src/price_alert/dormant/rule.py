"""冷门启动的判定规则：纯函数，实时检测、回放与测试共用。

门槛来自 research/launch_trade/dormant_solo.md 与 dormant_calm.md：只用开发期按「提醒之后先涨 3%~5%、没先跌回同样幅度」
的准确率与冷门大行情的召回率选出。量比、突破、回吐这几项都没有让提醒更准，所以不设；
冷门币一两笔单子就能撑起涨幅，有成交的分钟数是「有人在持续买」的条件。涨幅相对涨前波动的倍数与单分钟涨幅上限是用户
要求的形态（涨前平稳、拉升不靠一两分钟跳涨），样本外比不带这两项略准但未达显著，是用户知情后的例外。
总实体占比下限（上影线多的拉升不报）同样是用户指定的形态，按历史数据算准确率持平、提醒少约三成，
见 dormant_calm.md 第四节。涨前波动上限（涨前近乎一条直线）也是用户要的形态：在现行规则上加 0.5% 准确率不变、
提醒少约两成，见 dormant_calm.md 第六节。最后一分钟成交额与后半窗口成交额占比两项来自 dormant_early.md 第三轮
（假启动更少优先），判据未通过（PBO 0.58），是用户知情后的例外；代码默认值为不限，只由 YAML 开启。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from price_alert.models import Candle

# 「之前 24 小时」：判断是否冷门的区间，回测只验证过这一个长度，不开放配置。
PRIOR_MINUTES = 1440
# 「涨前波动」的统计区间：窗口开始之前这么多分钟。回测在 60 与 240 中选了 240，不开放配置；必须短于 PRIOR_MINUTES，
# 检测器保留的 K 线才够用。
CALM_MINUTES = 240


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
}


def late_minutes(window_minutes: int) -> int:
    """「后半窗口」的分钟数：最后 ⌈W/2⌉ 根，与回测口径相同（奇数窗口时中间那根算后半）。"""
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
    # 涨前波动（%/分钟）的绝对上限，None 不限：倍数只看相对大小，涨前来回乱跳的币涨得够多也能过，这一项挡掉它们。
    max_pre_volatility_percent: float | None = 0.5
    # 窗口最后一分钟成交额下限（USDT），0 即不限：拉升到最后一分钟已经没人买的，多是前面几笔单子拉完就停了。
    min_last_minute_quote: float = 0.0
    # 后半窗口（最后 late_minutes 根）成交额占窗口成交额的比例下限，0 即不限：成交集中在前半段、后面跟不上的容易回落。
    min_late_quote_share: float = 0.0
    cooldown_minutes: int = 60
    # 冷却期内价格比上次提醒价再涨这么多（%）就再报「第 N 波」，第一次没注意到也不至于整段错过；None 即冷却内不再报。
    # 回测中这些补报的准确率不低于第一次提醒（dormant_calm.md 第五节，判据全部通过）。
    realert_step_percent: float | None = 3.0

    def __post_init__(self) -> None:
        if self.realert_step_percent is not None and self.realert_step_percent <= 0:
            # 0 会让冷却期内每分钟都补报，等于没有冷却。
            raise ValueError("realert_step_percent 必须大于 0")
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

    这两项都由调用方滚动维护后传入：每分钟对上千根重新求和，冷门币沉寂很久后一笔成交补出上千根平线、逐根判定时，
    会在成交循环里同步算上百万次。"""
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
        failures=tuple(failures),
    )
