"""冷门启动检测器：由实时成交聚合带成交额的 1 分钟 K 线，每分钟收盘判定一次。

判定见 rule.evaluate_dormant。同一合约距上次提醒（含补报）不足 cooldown_minutes 时，只有价格比上次提醒价
再涨 realert_step_percent 才再报「第 N 波」：冷门币一段拉升通常十几到几十分钟，窗口滑过期间不会每分钟报一次，
但第一次没注意到时还有补报；超过冷却再满足条件就当作新的启动，从第 1 波算。拉升之后之前 24 小时的成交额
很快超过门槛，自然不再满足冷门条件。

K 线聚合、空分钟补平线、断线回补与服务约定的接口都在 common.minute_bars.MinuteBarDetector，这里只写每分钟的判定
与判定要用的滚动统计量。冷门币成交稀疏，隔了空档才结算很常见，结算时用触发它的成交价复核。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from price_alert.common.minute_bars import MINUTE, MinuteBarDetector, MinuteState
from price_alert.common.models import Candle
from price_alert.detectors.dormant.alerts import DormantAlert
from price_alert.detectors.dormant.rule import (
    CALM_MINUTES,
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
    ALERT = "alert"


OUTCOME_LABELS: dict[DormantOutcome, str] = {
    DormantOutcome.INSUFFICIENT_DATA: "K 线不足",
    DormantOutcome.FAILED: "未满足条件",
    DormantOutcome.COOLDOWN: "满足条件，冷却中且价格未比上次提醒再涨够",
    DormantOutcome.EXPIRED: "满足条件，但空档后结算时价格已回落",
    DormantOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class DormantEvaluation:
    """一根 1 分钟 K 线收盘时的判定，回放据此解释为什么提醒或没有提醒。"""

    symbol: str
    timestamp: datetime
    outcome: DormantOutcome
    metrics: DormantMetrics | None = None


@dataclass(frozen=True, slots=True)
class _LastAlert:
    timestamp: datetime
    price: float
    wave: int


@dataclass(slots=True)
class _SymbolState(MinuteState):
    # minutes 中全部 K 线的成交额合计，随追加、挤出滚动更新，判定时不必每分钟对 1450 根重新求和。
    total_quote: float = 0.0
    # 与 minutes 一一对应的真实波幅累计和（从这只合约开始累计），任意一段的涨前波动两次取值相减即得，
    # 同样是为了不在每分钟判定时对 240 根逐根重算。
    range_totals: deque[float] = field(default_factory=deque)

    def append(self, candle: Candle) -> None:
        minutes = self.minutes
        # 第一根没有前收，用自己的开盘价代替；它在保留的最前面，不会落进涨前波动的统计区间。
        previous = minutes[-1].close if minutes else candle.open
        if len(minutes) == minutes.maxlen:
            self.total_quote -= minutes[0].quote_volume
            self.range_totals.popleft()
        minutes.append(candle)
        self.total_quote += candle.quote_volume
        last = self.range_totals[-1] if self.range_totals else 0.0
        self.range_totals.append(last + minute_range_percent(candle, previous))

    def clear(self) -> None:
        self.minutes.clear()
        self.total_quote = 0.0
        self.range_totals.clear()


class DormantDetector(MinuteBarDetector[_SymbolState, DormantAlert]):
    state_class = _SymbolState
    requires_quote = True

    def __init__(self, rule: DormantRule, observer: Callable[[DormantEvaluation], None] | None = None) -> None:
        super().__init__(rule.history_minutes)
        self.rule = rule
        self._observer = observer
        # 不随合约移出而清除：合约很快重新入池时，同一段拉升不应再提醒一次。
        self._last_alerts: dict[str, _LastAlert] = {}

    def _evaluate(
        self, symbol: str, state: _SymbolState, closed_at: datetime, latest_price: float | None
    ) -> list[DormantAlert]:
        rule = self.rule
        alert: DormantAlert | None = None
        if len(state.minutes) < self._retention:
            evaluation = DormantEvaluation(symbol, closed_at, DormantOutcome.INSUFFICIENT_DATA)
        else:
            minutes = state.minutes
            size = len(minutes)
            # deque 按下标取靠近两端的元素很快，窗口只有最后几根。
            window = [minutes[index] for index in range(size - rule.window_minutes, size)]
            start = size - rule.window_minutes - 1
            start_price = minutes[start].close
            prior_quote = state.total_quote - sum(candle.quote_volume for candle in window)
            totals = state.range_totals
            pre_volatility = (totals[start] - totals[start - CALM_MINUTES]) / CALM_MINUTES
            metrics = evaluate_dormant(window, start_price, prior_quote, pre_volatility, rule)
            last = self._last_alerts.get(symbol)
            # 冷却期内的补报要相对上次提醒价再涨够；冷却过后就是新的一段，从第 1 波算。
            repeat = last if last is not None and closed_at - last.timestamp < MINUTE * rule.cooldown_minutes else None
            wave = 1 if repeat is None else repeat.wave + 1
            if not metrics.passed:
                outcome = DormantOutcome.FAILED
            elif repeat is not None and not self._extended(repeat, metrics.end_price):
                outcome = DormantOutcome.COOLDOWN
            elif latest_price is not None and not (
                still_rising(metrics, latest_price, rule) and (repeat is None or self._extended(repeat, latest_price))
            ):
                # 不记入冷却：价格回落后再次拉起来满足条件，仍应提醒。
                outcome = DormantOutcome.EXPIRED
            else:
                outcome = DormantOutcome.ALERT
            evaluation = DormantEvaluation(symbol, closed_at, outcome, metrics)
            if outcome is DormantOutcome.ALERT:
                self._last_alerts[symbol] = _LastAlert(closed_at, metrics.end_price, wave)
                alert = DormantAlert(
                    symbol=symbol,
                    timestamp=closed_at,
                    started_at=closed_at - MINUTE * rule.window_minutes,
                    window_minutes=rule.window_minutes,
                    metrics=metrics,
                    volume_24h_quote=state.volume_24h_quote,
                    price_decimals=state.price_decimals,
                    wave=wave,
                )
        if self._observer is not None:
            self._observer(evaluation)
        return [] if alert is None else [alert]

    def _extended(self, last: _LastAlert, price: float) -> bool:
        step = self.rule.realert_step_percent
        return step is not None and (price / last.price - 1) * 100 >= step
