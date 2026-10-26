"""K 线形态趋势检测器：由实时成交聚合 1 分钟 K 线，再合成各周期 K 线，形态成立即提醒。

每个周期各自独立：形态从不成立变为成立时提醒，持续成立期间每根周期 K 线收盘再报一条「延续」提醒
（可关闭，关闭后同一段只报一次），形态被破坏后重新计算，再次成立就是新的一段。不同周期之间互不影响。

K 线聚合、空分钟补平线、断线回补与服务约定的接口都在 common.minute_bars.MinuteBarDetector，这里只写每分钟的判定。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import islice

from price_alert.common.minute_bars import MinuteBarDetector, MinuteState
from price_alert.common.models import Candle
from price_alert.detectors.trend.alerts import TrendAlert
from price_alert.detectors.trend.pattern import (
    Direction,
    PatternMetrics,
    PeriodName,
    TrendRule,
    aggregate_candles,
    evaluate_pattern,
)


class TrendOutcome(StrEnum):
    INSUFFICIENT_DATA = "insufficient_data"
    NO_DIRECTION = "no_direction"
    FAILED = "failed"
    CONTINUING = "continuing"
    CONTINUING_ALERT = "continuing_alert"
    ALERT = "alert"


OUTCOME_LABELS: dict[TrendOutcome, str] = {
    TrendOutcome.INSUFFICIENT_DATA: "K 线不足",
    TrendOutcome.NO_DIRECTION: "首尾价格相同",
    TrendOutcome.FAILED: "未满足形态条件",
    TrendOutcome.CONTINUING: "形态延续，本段已提醒",
    TrendOutcome.CONTINUING_ALERT: "形态延续，延续提醒已发",
    TrendOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class TrendEvaluation:
    """某个周期的一根 K 线收盘时的判定，回放据此解释为什么提醒或没有提醒。"""

    symbol: str
    period: PeriodName
    timestamp: datetime
    outcome: TrendOutcome
    metrics: PatternMetrics | None = None


class TrendDetector(MinuteBarDetector[MinuteState, TrendAlert]):
    # 各周期都由 1 分钟 K 线合成，预热只需要这一种 K 线（基类的 warmup_interval 即 1m）。

    def __init__(
        self,
        rules: Sequence[TrendRule],
        observer: Callable[[TrendEvaluation], None] | None = None,
        *,
        alert_continuing: bool = True,
    ) -> None:
        if not rules:
            raise ValueError("至少需要一个趋势周期")
        if len({rule.period for rule in rules}) != len(rules):
            raise ValueError("趋势周期不能重复")
        super().__init__(max(rule.history_minutes for rule in rules))
        # 按周期从短到长排列，同一分钟多个周期都提醒时短周期在前。
        self.rules = tuple(sorted(rules, key=lambda rule: rule.minutes))
        self._observer = observer
        self._alert_continuing = alert_continuing
        # 每个合约各周期最近一次判定时形态成立的方向（不成立为 None）。不随合约移出而清除：
        # 合约很快重新入池时，仍在延续的同一段走势不应再提醒一次。
        self._holding: dict[str, dict[PeriodName, Direction | None]] = {}

    def _evaluate(
        self, symbol: str, state: MinuteState, closed_at: datetime, latest_price: float | None
    ) -> list[TrendAlert]:
        """对收盘时刻落在周期边界的各周期判定；某个周期的形态新形成时提醒。

        不用 latest_price 复核：形态描述的是已经形成的走势，结算晚了也不改变它曾经成立。
        """
        evaluations: list[TrendEvaluation] = []
        alerts: list[TrendAlert] = []
        holding = self._holding.setdefault(symbol, {})
        for rule in self.rules:
            if int(closed_at.timestamp()) % (rule.minutes * 60):
                continue
            previous = holding.get(rule.period)
            holding[rule.period] = None
            window = self._window(state, rule)
            if len(window) < rule.candles:
                evaluations.append(TrendEvaluation(symbol, rule.period, closed_at, TrendOutcome.INSUFFICIENT_DATA))
                continue
            metrics = evaluate_pattern(window, rule)
            if metrics is None:
                evaluations.append(TrendEvaluation(symbol, rule.period, closed_at, TrendOutcome.NO_DIRECTION))
                continue
            if not metrics.passed:
                evaluations.append(TrendEvaluation(symbol, rule.period, closed_at, TrendOutcome.FAILED, metrics))
                continue
            holding[rule.period] = metrics.direction
            # 上一根周期 K 线收盘时同方向形态就已成立，属于同一段走势的延续。
            continuing = previous == metrics.direction
            if not continuing:
                outcome = TrendOutcome.ALERT
            elif self._alert_continuing:
                outcome = TrendOutcome.CONTINUING_ALERT
            else:
                outcome = TrendOutcome.CONTINUING
            evaluation = TrendEvaluation(symbol, rule.period, closed_at, outcome, metrics)
            if outcome is not TrendOutcome.CONTINUING:
                alerts.append(
                    TrendAlert(
                        symbol=symbol,
                        direction=metrics.direction,
                        period=rule.period,
                        candles=rule.candles,
                        timestamp=closed_at,
                        started_at=window[0].timestamp,
                        metrics=metrics,
                        volume_24h_quote=state.volume_24h_quote,
                        price_decimals=state.price_decimals,
                        continuing=continuing,
                    )
                )
            evaluations.append(evaluation)
        if self._observer is not None:
            for evaluation in evaluations:
                self._observer(evaluation)
        return alerts

    def _window(self, state: MinuteState, rule: TrendRule) -> list[Candle]:
        # 1 分钟 K 线连续且以周期边界结尾，取周期整数倍的尾部即按边界对齐。
        available = min(len(state.minutes), rule.history_minutes)
        count = available - available % rule.minutes
        tail = list(islice(state.minutes, len(state.minutes) - count, None))
        return aggregate_candles(tail, rule.minutes)
