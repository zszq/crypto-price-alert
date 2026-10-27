"""反弹回落检测器：在秒级异动的单窗口判定之上，再要求前置走势反向、成交笔数暴增。

急动判定与 AtrMoveDetector 完全相同（研究也按这套判定选参），所以继承它，只覆盖两处：
- 结算真实秒时另记逐秒价格与累计成交笔数（基类的秒桶只按窗口保留，看不到前置走势与 1 小时成交）；
- 可以提醒的那一秒再判两个条件：不满足只拦这一秒、不清零确认，与基类对补齐秒、空档复核失败的处理一致。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from price_alert.detectors.move.detector import AtrMoveDetector, _SecondBucket, _SymbolState
from price_alert.detectors.move.windows import MoveWindow, Outcome, WindowEvaluation
from price_alert.detectors.reversal.alerts import ReversalAlert
from price_alert.detectors.reversal.rule import (
    ATR_PERIOD,
    BURST_BASE_SECONDS,
    BURST_SECONDS,
    CANDLE_INTERVAL_SECONDS,
    MAX_ATR_AGE_SECONDS,
    ReversalRule,
)

# 每个有成交的秒：时间、最后成交价、到这一秒为止的累计成交笔数。
_Point = tuple[datetime, float, int]


def _point_time(point: _Point) -> datetime:
    return point[0]


class ReversalOutcome(StrEnum):
    """急动窗口之后的判定结果，按判定流程的先后排列。"""

    MOVE = "move"
    PRIOR_TREND = "prior_trend"
    NO_TRADE_BURST = "no_trade_burst"
    COOLDOWN = "cooldown"
    ALERT = "alert"


OUTCOME_LABELS: dict[ReversalOutcome, str] = {
    ReversalOutcome.MOVE: "急动未达标或未确认",
    ReversalOutcome.PRIOR_TREND: "前置走势反向不够（或历史不足）",
    ReversalOutcome.NO_TRADE_BURST: "成交笔数未暴增（或不足 1 小时历史）",
    ReversalOutcome.COOLDOWN: "冷却中",
    ReversalOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class ReversalEvaluation:
    """一秒的完整判定：急动窗口本身的判定，加上急动达标后两个条件的取值与结论，回放靠它解释原因。"""

    window: WindowEvaluation
    outcome: ReversalOutcome
    # 窗口之前那段的实际涨跌幅（带符号）与成交暴增倍数；急动没达到涨跌幅与 ATR 门槛时不计算，为 None。
    prior_change_percent: float | None = None
    trade_burst_ratio: float | None = None
    alert: ReversalAlert | None = None


# 急动已达到涨跌幅与 ATR 门槛（还在确认、补齐秒、空档复核失败或已可提醒），观察者需要看两个条件的取值。
_MEASURED = frozenset({Outcome.CONFIRMING, Outcome.CARRIED, Outcome.MOVE_FADED, Outcome.ALERT})


class ReversalDetector(AtrMoveDetector):
    def __init__(self, rule: ReversalRule, observer: Callable[[ReversalEvaluation], None] | None = None) -> None:
        window = MoveWindow(
            name="short",
            lookback_seconds=rule.lookback_seconds,
            trigger_atr_multiple=rule.trigger_atr_multiple,
            min_change_percent=rule.min_change_percent,
            confirmation_seconds=rule.confirmation_seconds,
            min_window_trades=rule.min_window_trades,
            cooldown_seconds=rule.cooldown_seconds,
        )
        super().__init__(
            atr_period=ATR_PERIOD,
            candle_interval_seconds=CANDLE_INTERVAL_SECONDS,
            max_atr_age_seconds=MAX_ATR_AGE_SECONDS,
            windows=[window],
        )
        self.rule = rule
        # 每秒回调一条 ReversalEvaluation，回放用；实时监控不传。
        self._reversal_observer = observer
        self._history: dict[str, deque[_Point]] = {}
        self._history_span = timedelta(seconds=rule.history_seconds)

    def remove_symbols(self, symbols: set[str]) -> None:
        super().remove_symbols(symbols)
        for symbol in symbols:
            self._history.pop(symbol, None)

    # 断线时一并清空逐秒记录：断线期间成交缺失，1 小时笔数基准已不可靠；基类清空秒桶后累计笔数从 0 重新计数，
    # 留着旧记录会算出负数。重连后重新攒满 1 小时才判定，宁可漏报不误报。
    def mark_stream_gap(self) -> None:
        super().mark_stream_gap()
        self._history.clear()

    def mark_symbol_gap(self, symbol: str) -> None:
        super().mark_symbol_gap(symbol)
        self._history.pop(symbol.upper(), None)

    def _settle_through(
        self,
        symbol: str,
        state: _SymbolState,
        second: datetime,
        price: float,
        timestamp: datetime,
        alerts: list,
    ) -> None:
        if state.buckets:
            # 进入结算时最后一个桶就是刚结束的真实秒（补齐桶在结算过程中才追加）。
            completed = state.buckets[-1]
            assert completed.last_price is not None
            history = self._history.setdefault(symbol, deque())
            history.append((completed.timestamp, completed.last_price, completed.trades_through))
            # 与秒桶相同：留一个不晚于最早所需时刻的点，那一刻的价格与累计笔数要靠它确定。
            oldest_required = completed.timestamp - self._history_span
            while len(history) > 1 and history[1][0] <= oldest_required:
                history.popleft()
        super()._settle_through(symbol, state, second, price, timestamp, alerts)

    def _evaluate(
        self,
        symbol: str,
        state: _SymbolState,
        current: _SecondBucket,
        timestamp: datetime,
        latest_price: float | None,
        atr_state,
    ) -> list:
        (window,) = self.windows
        (candidate,) = state.candidates
        evaluation = self._evaluate_window(symbol, state, window, candidate, current, latest_price, atr_state)
        observe = self._reversal_observer
        # 实时监控只在可以提醒时算两个条件；回放还要看确认途中的取值，才能解释「差在哪」。
        if evaluation.outcome is not Outcome.ALERT and not (observe is not None and evaluation.outcome in _MEASURED):
            if observe is not None:
                observe(ReversalEvaluation(evaluation, ReversalOutcome.MOVE))
            return []
        assert evaluation.baseline_price is not None and evaluation.change_percent is not None
        direction: Literal["surge", "drop"] = "surge" if evaluation.change_percent > 0 else "drop"
        history = self._history.get(symbol, deque())
        prior = self._prior_change(history, current.timestamp, evaluation.baseline_price, direction)
        burst = self._trade_burst(history, current)
        # 内部按「顺急动方向为正」判定，记录与提醒里换回带符号的实际涨跌幅：急拉之前是负的（下跌）。
        actual_prior = None if prior is None else (prior if direction == "surge" else -prior)

        def report(outcome: ReversalOutcome, alert: ReversalAlert | None = None) -> None:
            if observe is not None:
                observe(ReversalEvaluation(evaluation, outcome, actual_prior, burst, alert))

        if evaluation.outcome is not Outcome.ALERT:
            report(ReversalOutcome.MOVE)
            return []
        if prior is None or prior > -self.rule.min_prior_move_percent:
            report(ReversalOutcome.PRIOR_TREND)
            return []
        if burst is None or burst < self.rule.min_trade_burst_ratio:
            report(ReversalOutcome.NO_TRADE_BURST)
            return []
        # 以下与基类相同：冷却按窗口记，提醒或冷却拦下后清零确认。
        key = (symbol, window.name)
        previous = self._last_alert.get(key)
        candidate.reset()
        if previous is not None and timestamp - previous < self._cooldowns[0]:
            report(ReversalOutcome.COOLDOWN)
            return []
        self._last_alert[key] = timestamp
        assert actual_prior is not None
        alert = self._reversal_alert(symbol, state, evaluation, timestamp, actual_prior, burst)
        report(ReversalOutcome.ALERT, alert)
        return [alert]

    def _prior_change(
        self,
        history: deque[_Point],
        second: datetime,
        baseline_price: float,
        direction: Literal["surge", "drop"],
    ) -> float | None:
        """窗口开始前 prior_seconds 秒到基准价的涨跌幅，顺急动方向为正；没有那么早的成交时返回 None。"""
        target = second - timedelta(seconds=self.rule.lookback_seconds + self.rule.prior_seconds)
        index = bisect_right(history, target, key=_point_time) - 1
        if index < 0:
            return None
        change = (baseline_price / history[index][1] - 1.0) * 100.0
        return change if direction == "surge" else -change

    @staticmethod
    def _trades_at(history: deque[_Point], moment: datetime) -> int | None:
        index = bisect_right(history, moment, key=_point_time) - 1
        return history[index][2] if index >= 0 else None

    @classmethod
    def _trade_burst(cls, history: deque[_Point], current: _SecondBucket) -> float | None:
        """最近 1 分钟成交笔数 ÷ 最近 1 小时每分钟平均笔数；不足 1 小时的成交历史时返回 None。"""
        hour_start = cls._trades_at(history, current.timestamp - timedelta(seconds=BURST_BASE_SECONDS))
        if hour_start is None:
            return None
        hour = current.trades_through - hour_start
        if hour <= 0:
            return None
        # 1 小时前有记录，1 分钟前必然也有。
        minute_start = cls._trades_at(history, current.timestamp - timedelta(seconds=BURST_SECONDS))
        assert minute_start is not None
        return (current.trades_through - minute_start) / hour * (BURST_BASE_SECONDS / BURST_SECONDS)

    def _reversal_alert(
        self,
        symbol: str,
        state: _SymbolState,
        evaluation: WindowEvaluation,
        timestamp: datetime,
        prior: float,
        burst: float,
    ) -> ReversalAlert:
        assert evaluation.baseline_price is not None and evaluation.change_percent is not None
        assert evaluation.move_atr is not None and evaluation.trade_count is not None
        direction: Literal["surge", "drop"] = "surge" if evaluation.change_percent > 0 else "drop"
        return ReversalAlert(
            symbol=symbol,
            direction=direction,
            timestamp=timestamp,
            price=evaluation.price,
            reference_price=evaluation.baseline_price,
            change_percent=evaluation.change_percent,
            lookback_seconds=self.rule.lookback_seconds,
            move_atr=evaluation.move_atr,
            trade_count=evaluation.trade_count,
            prior_change_percent=prior,
            prior_seconds=self.rule.prior_seconds,
            trade_burst_ratio=burst,
            volume_24h_quote=state.volume_24h_quote,
            price_decimals=state.price_decimals,
        )
