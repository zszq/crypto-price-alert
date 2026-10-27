"""放量启动检测器：由实时成交聚合带成交额的 1 分钟 K 线，每分钟收盘判定一次。

一个检测器只判定 rule.direction 一个方向，放量上涨、放量下跌是两个互不影响的实例。判定见 rule.evaluate_launch。
重复提醒按（合约, 方向）记录上一次提醒：
- 距上次提醒不足 cooldown_minutes 不再提醒，窗口滑过期间不会每分钟报一次；
- 冷却过后，价格还要比上次提醒价再推进 realert_step_percent 才报下一波，原地放量震荡不重复提醒；
- 上次提醒已超过一个基准期，再次满足条件就是新的启动，从第 1 波计数；
- 只因插针（涨跌幅超上限）不提醒的分钟也按提醒记录（波次不变），否则窗口滑过插针后会在高位补报。

K 线聚合、补平线、断线回补都在 MinuteBarDetector。没有合约面值的合约换算不出成交额，保留在池里但不判定。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from price_alert.common.minute_bars import MINUTE, MinuteBarDetector, MinuteState
from price_alert.detectors.launch.alerts import LaunchAlert
from price_alert.detectors.launch.rule import (
    Condition,
    Direction,
    LaunchMetrics,
    LaunchRule,
    evaluate_launch,
    still_launching,
)


class LaunchOutcome(StrEnum):
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"
    COOLDOWN = "cooldown"
    NOT_EXTENDED = "not_extended"
    EXPIRED = "expired"
    OVERSHOOT = "overshoot"
    ALERT = "alert"


OUTCOME_LABELS: dict[LaunchOutcome, str] = {
    LaunchOutcome.INSUFFICIENT_DATA: "K 线不足",
    LaunchOutcome.FAILED: "未满足条件",
    LaunchOutcome.COOLDOWN: "满足条件，冷却中",
    LaunchOutcome.NOT_EXTENDED: "满足条件，价格未比上次提醒再推进",
    LaunchOutcome.EXPIRED: "满足条件，但空档后结算时价格已回落",
    LaunchOutcome.OVERSHOOT: "只因涨跌幅过大（插针）不提醒，按提醒记入冷却与波次",
    LaunchOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class LaunchEvaluation:
    """一根 1 分钟 K 线收盘时的判定，回放据此解释为什么提醒或没有提醒。"""

    symbol: str
    timestamp: datetime
    outcome: LaunchOutcome
    metrics: LaunchMetrics | None = None


@dataclass(frozen=True, slots=True)
class _LastAlert:
    timestamp: datetime
    price: float
    wave: int


class LaunchDetector(MinuteBarDetector[MinuteState, LaunchAlert]):
    requires_quote = True

    def __init__(self, rule: LaunchRule, observer: Callable[[LaunchEvaluation], None] | None = None) -> None:
        super().__init__(rule.history_minutes)
        self.rule = rule
        self._observer = observer
        # 不随合约移出而清除：合约很快重新入池时，仍在延续的同一波不应再当作启动提醒一次。
        # 也记录因插针而静默的分钟（见 _record_overshoot）。
        self._last_alerts: dict[tuple[str, Direction], _LastAlert] = {}

    def _evaluate(
        self, symbol: str, state: MinuteState, closed_at: datetime, latest_price: float | None
    ) -> list[LaunchAlert]:
        rule = self.rule
        alert: LaunchAlert | None = None
        if len(state.minutes) < self._retention:
            evaluation = LaunchEvaluation(symbol, closed_at, LaunchOutcome.INSUFFICIENT_DATA)
        else:
            metrics = evaluate_launch(list(state.minutes), rule)
            if not metrics.passed:
                outcome, wave = LaunchOutcome.FAILED, 0
                if metrics.failures == (Condition.OVERSHOOT,) and self._record_overshoot(symbol, metrics, closed_at):
                    # 单独成一类判定：回放据此解释之后为什么「冷却中」或要求再推进。
                    outcome = LaunchOutcome.OVERSHOOT
            elif latest_price is not None and not still_launching(metrics, latest_price, rule):
                outcome, wave = LaunchOutcome.EXPIRED, 0
            else:
                outcome, wave = self._repeat_outcome(symbol, metrics, closed_at)
            evaluation = LaunchEvaluation(symbol, closed_at, outcome, metrics)
            if outcome is LaunchOutcome.ALERT:
                self._last_alerts[(symbol, metrics.direction)] = _LastAlert(closed_at, metrics.end_price, wave)
                alert = LaunchAlert(
                    symbol=symbol,
                    timestamp=closed_at,
                    started_at=closed_at - MINUTE * rule.window_minutes,
                    window_minutes=rule.window_minutes,
                    baseline_minutes=rule.baseline_minutes,
                    metrics=metrics,
                    volume_24h_quote=state.volume_24h_quote,
                    price_decimals=state.price_decimals,
                    wave=wave,
                )
        if self._observer is not None:
            self._observer(evaluation)
        return [] if alert is None else [alert]

    def _record_overshoot(self, symbol: str, metrics: LaunchMetrics, closed_at: datetime) -> bool:
        """其他条件都满足、只是涨跌幅超过上限：不提醒，但当作行情已启动，返回是否写入了提醒记录。
        之后要过冷却并比插针收盘价再推进才提醒，免得窗口滑过插针后在高位补报。"""
        # 与真提醒走同一套冷却与再推进判断：真提醒也会被拦下的插针不改写记录，免得拉低下一波的门槛。
        outcome, wave = self._repeat_outcome(symbol, metrics, closed_at)
        if outcome is not LaunchOutcome.ALERT:
            return False
        # 没有发出提醒，波次不加：之后发出的第一条仍是第 1 波，或接着上一条发出的提醒计数。
        self._last_alerts[(symbol, metrics.direction)] = _LastAlert(closed_at, metrics.end_price, wave - 1)
        return True

    def _repeat_outcome(self, symbol: str, metrics: LaunchMetrics, closed_at: datetime) -> tuple[LaunchOutcome, int]:
        """条件已满足时，按同方向上一次提醒决定是否再提醒，返回判定与这次是第几波。"""
        rule = self.rule
        last = self._last_alerts.get((symbol, metrics.direction))
        # 冷却先于基准期重置判断：冷却配得比基准期还长时，也必须等冷却结束。
        if last is not None and closed_at - last.timestamp < MINUTE * rule.cooldown_minutes:
            return LaunchOutcome.COOLDOWN, 0
        if last is None or closed_at - last.timestamp >= MINUTE * rule.baseline_minutes:
            return LaunchOutcome.ALERT, 1
        sign = 1 if metrics.direction == "surge" else -1
        progress = (metrics.end_price / last.price - 1) * 100 * sign
        if progress < rule.realert_step_percent:
            return LaunchOutcome.NOT_EXTENDED, 0
        return LaunchOutcome.ALERT, last.wave + 1
