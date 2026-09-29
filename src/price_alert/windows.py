"""观察窗口规则与逐秒判定记录。

检测器、服务组装和回放共用这一套词汇：窗口规则描述“多长时间内涨跌多少才算异动”，
判定记录描述“某一秒在某个窗口上走到了哪一步”，回放据此解释为什么提醒或没有提醒。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from price_alert.formatting import describe_window

WindowName = Literal["short", "long"]

WINDOW_LABELS: dict[WindowName, str] = {"short": "短窗口", "long": "长窗口"}


@dataclass(frozen=True, slots=True)
class MoveWindow:
    """一个观察窗口及其独立门槛；多个窗口共享秒级数据与 ATR，任意一个满足即可提醒。"""

    name: WindowName
    lookback_seconds: int
    trigger_atr_multiple: float
    min_change_percent: float
    confirmation_seconds: int
    min_window_trades: int
    cooldown_seconds: int

    def __post_init__(self) -> None:
        # 检测器直接信任这些值做时间运算，非法组合在构造时拦下，比运行中静默不提醒更容易发现。
        if self.lookback_seconds <= 0:
            raise ValueError("lookback_seconds 必须大于 0")
        if not 1 <= self.confirmation_seconds <= self.lookback_seconds:
            raise ValueError("confirmation_seconds 必须在 1 到 lookback_seconds 之间")
        if self.trigger_atr_multiple <= 0 or self.min_change_percent <= 0:
            raise ValueError("触发门槛必须大于 0")
        if self.min_window_trades < 1 or self.cooldown_seconds < 0:
            raise ValueError("min_window_trades 至少为 1，cooldown_seconds 不能为负")

    def exceeds_thresholds(self, price_move: float, baseline_price: float, atr: float) -> bool:
        # 两道门槛必须同时满足：百分比保证肉眼可感知，ATR 倍数适配不同市场波动率。
        # 判定与空档后的复核共用这一份规则，避免两处阈值逐渐走偏。
        return (
            abs(price_move / baseline_price * 100.0) >= self.min_change_percent
            and abs(price_move) / atr >= self.trigger_atr_multiple
        )


def describe_rule(window: MoveWindow) -> str:
    return (
        f"{WINDOW_LABELS[window.name]}：{describe_window(window.lookback_seconds)}内涨跌 ≥ "
        f"{window.min_change_percent:g}% 且 ≥ {window.trigger_atr_multiple:g} ATR，"
        f"窗口成交 ≥ {window.min_window_trades} 笔，连续 {window.confirmation_seconds} 秒，"
        f"冷却 {window.cooldown_seconds} 秒"
    )


class Outcome(StrEnum):
    """一秒在一个窗口上的判定结果，按判定流程的先后排列。"""

    ATR_UNAVAILABLE = "atr_unavailable"
    ATR_EXPIRED = "atr_expired"
    NO_BASELINE = "no_baseline"
    FEW_TRADES = "few_trades"
    BELOW_THRESHOLD = "below_threshold"
    CONFIRMING = "confirming"
    CARRIED = "carried"
    MOVE_FADED = "move_faded"
    COOLDOWN = "cooldown"
    SUPERSEDED = "superseded"
    ALERT = "alert"


OUTCOME_LABELS: dict[Outcome, str] = {
    Outcome.ATR_UNAVAILABLE: "ATR 未就绪或断线待回补",
    Outcome.ATR_EXPIRED: "ATR 已过期",
    Outcome.NO_BASELINE: "缺少基准价（数据不足或空档过长）",
    Outcome.FEW_TRADES: "窗口成交笔数不足",
    Outcome.BELOW_THRESHOLD: "未达涨跌幅或 ATR 门槛",
    Outcome.CONFIRMING: "达标，连续确认中",
    Outcome.CARRIED: "确认已满但为无成交补齐秒",
    Outcome.MOVE_FADED: "空档后复核：异动已消退",
    Outcome.COOLDOWN: "冷却中",
    Outcome.SUPERSEDED: "同一秒已由其他窗口提醒",
    Outcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    """一秒在一个窗口上的完整判定上下文；门槛前的步骤失败时，后续字段为 None。"""

    symbol: str
    window: WindowName
    second: datetime
    outcome: Outcome
    price: float
    carried: bool
    atr: float | None = None
    baseline_price: float | None = None
    change_percent: float | None = None
    move_atr: float | None = None
    trade_count: int | None = None
    confirmed_seconds: int = 0
