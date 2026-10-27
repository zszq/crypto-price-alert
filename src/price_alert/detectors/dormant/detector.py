"""冷门启动检测器：由实时成交聚合带成交额的 1 分钟 K 线，每分钟收盘判定一次。

判定见 rule.evaluate_dormant。同一合约距上次提醒（含补报）不足 cooldown_minutes 时，只有价格比上次提醒价
再涨 realert_step_percent 才再报「第 N 波」：冷门币一段拉升通常十几到几十分钟，窗口滑过期间不会每分钟报一次，
但第一次没注意到时还有补报；超过冷却再满足条件就当作新的启动，从第 1 波算。拉升之后之前 24 小时的成交额
很快超过门槛，自然不再满足冷门条件。

设了 follow_step_percent 的组（慢启动）另有跟踪补报：距上次提醒（含补报）不超过 follow_minutes 时，收盘价比上次提醒价
再涨够步长就补报下一波，不再检查其他条件（dormant_waves.md）。条件满足且已过冷却时仍按新的一段从第 1 波算。

可以同时带几组参数（现行一组与可选的慢启动一组）：共用同一份 K 线与滚动统计量，每分钟各组独立判定、按（合约, 组）
各自冷却，同一分钟两组都满足就各报一条。不另建一个检测器，是为了不把约 550 个合约的 1500 根 K 线再存一份、
每笔成交再聚合一遍。

K 线聚合、空分钟补平线、断线回补与服务约定的接口都在 common.minute_bars.MinuteBarDetector，这里只写每分钟的判定
与判定要用的滚动统计量。冷门币成交稀疏，隔了空档才结算很常见，结算时用触发它的成交价复核。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from price_alert.common.minute_bars import MINUTE, MinuteBarDetector, MinuteState
from price_alert.common.models import Candle
from price_alert.detectors.dormant.alerts import DormantAlert
from price_alert.detectors.dormant.rule import (
    CALM_MINUTES,
    PRIOR_MINUTES,
    DormantGroup,
    DormantMetrics,
    DormantRule,
    evaluate_dormant,
    minute_range_percent,
    still_rising,
)


class DormantOutcome(StrEnum):
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"
    COOLDOWN = "cooldown"
    EXPIRED = "expired"
    FOLLOW = "follow"
    ALERT = "alert"


OUTCOME_LABELS: dict[DormantOutcome, str] = {
    DormantOutcome.INSUFFICIENT_DATA: "K 线不足",
    DormantOutcome.FAILED: "未满足条件",
    DormantOutcome.COOLDOWN: "满足条件，冷却中且价格未比上次提醒再涨够",
    DormantOutcome.EXPIRED: "满足条件，但空档后结算时价格已回落",
    DormantOutcome.FOLLOW: "跟踪补报（价格比上次提醒再涨够，不看其他条件）",
    DormantOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class DormantEvaluation:
    """一根 1 分钟 K 线收盘时的判定，回放据此解释为什么提醒或没有提醒。"""

    symbol: str
    timestamp: datetime
    outcome: DormantOutcome
    metrics: DormantMetrics | None = None
    group: DormantGroup = DormantGroup.MAIN


@dataclass(frozen=True, slots=True)
class _LastAlert:
    timestamp: datetime
    price: float
    wave: int


@dataclass(slots=True)
class _SymbolState(MinuteState):
    # 与 minutes 一一对应的成交额、真实波幅累计和（从这只合约开始累计），任意一段之和两次取值相减即得：
    # 每分钟对 1440 根重新求和太慢；各组窗口长度不同，之前 24 小时的区间也不同，只滚动维护一个合计不够用。
    quote_totals: deque[float] = field(default_factory=deque)
    range_totals: deque[float] = field(default_factory=deque)
    # 最近挤出的那一根对应的累计和：区间从保留的第一根开始时，用它作减数。
    quote_base: float = 0.0
    range_base: float = 0.0

    def append(self, candle: Candle) -> None:
        minutes = self.minutes
        # 第一根没有前收，用自己的开盘价代替；它在保留的最前面，不会落进涨前波动的统计区间。
        previous = minutes[-1].close if minutes else candle.open
        if len(minutes) == minutes.maxlen:
            self.quote_base = self.quote_totals.popleft()
            self.range_base = self.range_totals.popleft()
        minutes.append(candle)
        last_quote = self.quote_totals[-1] if self.quote_totals else self.quote_base
        self.quote_totals.append(last_quote + candle.quote_volume)
        last_range = self.range_totals[-1] if self.range_totals else self.range_base
        self.range_totals.append(last_range + minute_range_percent(candle, previous))

    def clear(self) -> None:
        self.minutes.clear()
        self.quote_totals.clear()
        self.range_totals.clear()
        self.quote_base = 0.0
        self.range_base = 0.0

    def quote_sum(self, first: int, last: int) -> float:
        """下标 first..last（含两端）那几根的成交额之和。"""
        return self.quote_totals[last] - (self.quote_totals[first - 1] if first > 0 else self.quote_base)

    def range_sum(self, first: int, last: int) -> float:
        return self.range_totals[last] - (self.range_totals[first - 1] if first > 0 else self.range_base)


class DormantDetector(MinuteBarDetector[_SymbolState, DormantAlert]):
    state_class = _SymbolState
    requires_quote = True

    def __init__(
        self, rules: Sequence[DormantRule], observer: Callable[[DormantEvaluation], None] | None = None
    ) -> None:
        if not rules:
            raise ValueError("至少要有一组冷门启动参数")
        if len({rule.group for rule in rules}) != len(rules):
            # 冷却按（合约, 组）记录，同组两份参数会互相顶掉冷却。
            raise ValueError("冷门启动的参数组不能重复")
        super().__init__(max(rule.history_minutes for rule in rules))
        self.rules = tuple(rules)
        self._observer = observer
        # 不随合约移出而清除：合约很快重新入池时，同一段拉升不应再提醒一次。
        self._last_alerts: dict[tuple[str, DormantGroup], _LastAlert] = {}

    def _evaluate(
        self, symbol: str, state: _SymbolState, closed_at: datetime, latest_price: float | None
    ) -> list[DormantAlert]:
        alerts: list[DormantAlert] = []
        for rule in self.rules:
            alert = self._evaluate_rule(rule, symbol, state, closed_at, latest_price)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _evaluate_rule(
        self, rule: DormantRule, symbol: str, state: _SymbolState, closed_at: datetime, latest_price: float | None
    ) -> DormantAlert | None:
        alert: DormantAlert | None = None
        # 保留长度按窗口最长的一组算，窗口短的组 K 线够了就先判定。
        if len(state.minutes) < rule.history_minutes:
            evaluation = DormantEvaluation(symbol, closed_at, DormantOutcome.INSUFFICIENT_DATA, group=rule.group)
        else:
            minutes = state.minutes
            size = len(minutes)
            # deque 按下标取靠近两端的元素很快，窗口只有最后几根。
            window = [minutes[index] for index in range(size - rule.window_minutes, size)]
            start = size - rule.window_minutes - 1
            start_price = minutes[start].close
            prior_quote = state.quote_sum(start - PRIOR_MINUTES + 1, start)
            pre_volatility = state.range_sum(start - CALM_MINUTES + 1, start) / CALM_MINUTES
            metrics = evaluate_dormant(window, start_price, prior_quote, pre_volatility, rule)
            key = (symbol, rule.group)
            last = self._last_alerts.get(key)
            # 冷却期内的补报要相对上次提醒价再涨够；冷却过后就是新的一段，从第 1 波算。
            repeat = last if last is not None and closed_at - last.timestamp < MINUTE * rule.cooldown_minutes else None
            wave = 1 if repeat is None else repeat.wave + 1
            if not metrics.passed:
                outcome = DormantOutcome.FAILED
            elif repeat is not None and not _extended(rule, repeat, metrics.end_price):
                outcome = DormantOutcome.COOLDOWN
            elif latest_price is not None and not (
                still_rising(metrics, latest_price, rule) and (repeat is None or _extended(rule, repeat, latest_price))
            ):
                # 不记入冷却：价格回落后再次拉起来满足条件，仍应提醒。
                outcome = DormantOutcome.EXPIRED
            else:
                outcome = DormantOutcome.ALERT
            if (
                outcome is not DormantOutcome.ALERT
                and last is not None
                and _followed(rule, last, closed_at, metrics, latest_price)
            ):
                outcome = DormantOutcome.FOLLOW
                # 跟踪期可能长于冷却，冷却过了 repeat 为 None，波次仍接着上次提醒数。
                wave = last.wave + 1
            evaluation = DormantEvaluation(symbol, closed_at, outcome, metrics, rule.group)
            if outcome in (DormantOutcome.ALERT, DormantOutcome.FOLLOW):
                self._last_alerts[key] = _LastAlert(closed_at, metrics.end_price, wave)
                alert = DormantAlert(
                    symbol=symbol,
                    timestamp=closed_at,
                    started_at=closed_at - MINUTE * rule.window_minutes,
                    window_minutes=rule.window_minutes,
                    metrics=metrics,
                    volume_24h_quote=state.volume_24h_quote,
                    price_decimals=state.price_decimals,
                    wave=wave,
                    group=rule.group,
                )
        if self._observer is not None:
            self._observer(evaluation)
        return alert


def _extended(rule: DormantRule, last: _LastAlert, price: float) -> bool:
    step = rule.realert_step_percent
    return step is not None and (price / last.price - 1) * 100 >= step


def _followed(
    rule: DormantRule, last: _LastAlert, closed_at: datetime, metrics: DormantMetrics, latest_price: float | None
) -> bool:
    step = rule.follow_step_percent
    if step is None or closed_at - last.timestamp > MINUTE * rule.follow_minutes:
        return False
    # 隔了空档才结算时，触发结算的成交价也要比上次提醒价高够步长，否则是在报一个已经回落的价格。
    prices = (metrics.end_price,) if latest_price is None else (metrics.end_price, latest_price)
    return all((price / last.price - 1) * 100 >= step for price in prices)
