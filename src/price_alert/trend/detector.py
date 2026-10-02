"""K 线形态趋势检测器：由实时成交聚合 1 分钟 K 线，再合成各周期 K 线，形态成立即提醒。

每个周期各自独立：形态从不成立变为成立时提醒，持续成立期间每根周期 K 线收盘再报一条「延续」提醒
（可关闭，关闭后同一段只报一次），形态被破坏后重新计算，再次成立就是新的一段。不同周期之间互不影响。

对外接口与 ATR 异动检测器保持一致（add_symbol / add_tick / mark_stream_gap / resync_symbol 等），
服务主循环因此不必知道它的存在；回放和模拟则直接喂已收盘的 1 分钟 K 线（add_candle）。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import islice

from price_alert.models import Candle, PriceTick
from price_alert.trend.alerts import TrendAlert
from price_alert.trend.pattern import (
    Direction,
    PatternMetrics,
    PeriodName,
    TrendRule,
    aggregate_candles,
    evaluate_pattern,
)

MINUTE = timedelta(minutes=1)


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


@dataclass(slots=True)
class _LiveMinute:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_candle(cls, candle: Candle) -> _LiveMinute:
        return cls(candle.timestamp, candle.open, candle.high, candle.low, candle.close)

    def update(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price

    def to_candle(self) -> Candle:
        return Candle(self.timestamp, self.open, self.high, self.low, self.close)


@dataclass(slots=True)
class _SymbolState:
    minutes: deque[Candle]
    volume_24h_quote: float
    price_decimals: int | None = None
    live: _LiveMinute | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，本地 K 线不完整，必须等 REST 回补后才能继续判定。
    stale: bool = False


def _floor_minute(moment: datetime) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def _flat(timestamp: datetime, price: float) -> Candle:
    # 与 Gate 的口径一致：无成交的周期是开高低收都等于上一收盘价的平线 K 线。
    return Candle(timestamp, price, price, price, price)


class TrendDetector:
    # 各周期都由 1 分钟 K 线合成，预热只需要这一种 K 线。
    warmup_interval = "1m"

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
        # 按周期从短到长排列，同一分钟多个周期都提醒时短周期在前。
        self.rules = tuple(sorted(rules, key=lambda rule: rule.minutes))
        self._observer = observer
        self._alert_continuing = alert_continuing
        self._retention = max(rule.history_minutes for rule in self.rules)
        self._states: dict[str, _SymbolState] = {}
        # 每个合约各周期最近一次判定时形态成立的方向（不成立为 None）。不随合约移出而清除：
        # 合约很快重新入池时，仍在延续的同一段走势不应再提醒一次。
        self._holding: dict[str, dict[PeriodName, Direction | None]] = {}

    @property
    def warmup_candles(self) -> int:
        # 最长周期判定一次所需的 1 分钟 K 线，加一根未收盘的当前 K 线。
        return self._retention + 1

    @property
    def symbols(self) -> list[str]:
        return sorted(self._states)

    @property
    def stale_symbols(self) -> list[str]:
        return sorted(symbol for symbol, state in self._states.items() if state.stale)

    def add_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        volume_24h_quote: float,
        live_candle: Candle | None = None,
        price_decimals: int | None = None,
    ) -> None:
        normalized = symbol.upper()
        existing = self._states.get(normalized)
        if existing is not None:
            existing.volume_24h_quote = volume_24h_quote
            existing.price_decimals = price_decimals
            return
        state = _SymbolState(deque(maxlen=self._retention), volume_24h_quote, price_decimals)
        self._seed(state, candles, live_candle)
        self._states[normalized] = state

    def remove_symbols(self, symbols: set[str]) -> None:
        for symbol in symbols:
            self._states.pop(symbol, None)

    def mark_stream_gap(self) -> None:
        for state in self._states.values():
            state.stale = True

    def mark_symbol_gap(self, symbol: str) -> None:
        """单个合约的 K 线可能已不一致时调用（如处理成交中途出错）：与断线同样处理，等 resync_symbol 回补。"""
        state = self._states.get(symbol.upper())
        if state is not None:
            state.stale = True

    def resync_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        live_candle: Candle | None,
        fetched_at: datetime,
    ) -> bool:
        """用 REST 1 分钟 K 线重建历史；合约已移除或已不需要回补时返回 False。各周期的形态状态保留。"""
        state = self._states.get(symbol.upper())
        if state is None or not state.stale:
            return False
        seed = sorted(candles, key=lambda item: item.timestamp)
        live = state.live
        if live_candle is not None:
            if live is None or live.timestamp < live_candle.timestamp:
                live = _LiveMinute.from_candle(live_candle)
            elif live.timestamp == live_candle.timestamp:
                # 本地含请求返回后才到的成交，交易所含断线期间的成交：高低点取并集。
                is_local_newer = state.last_tick_time is not None and state.last_tick_time > fetched_at
                live = _LiveMinute(
                    live.timestamp,
                    live_candle.open,
                    max(live.high, live_candle.high),
                    min(live.low, live_candle.low),
                    live.close if is_local_newer else live_candle.close,
                )
            else:
                # 请求返回后本地已跨入新的一分钟，交易所返回的“当前 K 线”其实已收盘。
                seed.append(live_candle)
        state.minutes.clear()
        state.live = None
        self._seed(state, seed, live.to_candle() if live is not None else None)
        state.stale = False
        return True

    def add_tick(self, tick: PriceTick) -> list[TrendAlert]:
        symbol = tick.symbol.upper()
        state = self._states.get(symbol)
        if state is None:
            return []
        timestamp = tick.timestamp.astimezone(UTC)
        if state.last_tick_time is not None and timestamp < state.last_tick_time:
            return []
        state.last_tick_time = timestamp
        minute = _floor_minute(timestamp)
        live = state.live
        if live is not None and minute == live.timestamp:
            live.update(tick.price)
            return []
        if live is not None and minute < live.timestamp:
            # REST 给的当前 K 线可能比首笔实时成交还新，更早的成交已包含在其中。
            return []
        alerts: list[TrendAlert] = []
        if live is not None and not state.stale:
            closed = [live.to_candle()]
            # 上一分钟之后无人成交的分钟按平线补齐，周期 K 线才能按时收盘。
            if minute - live.timestamp > MINUTE:
                closed.append(_flat(minute - MINUTE, live.close))
            for candle in closed:
                alerts.extend(self._close_minute(symbol, state, candle))
        # 失效期间只维护当前这一分钟，已收盘的分钟等 REST 回补。
        state.live = _LiveMinute(minute, tick.price, tick.price, tick.price, tick.price)
        return alerts

    def add_candle(self, symbol: str, candle: Candle) -> list[TrendAlert]:
        """直接喂入一根已收盘的 1 分钟 K 线，供回放与模拟使用；不能与 add_tick 混用。"""
        state = self._states.get(symbol.upper())
        if state is None:
            return []
        return self._close_minute(symbol.upper(), state, candle)

    def _seed(self, state: _SymbolState, candles: Sequence[Candle], live_candle: Candle | None) -> None:
        for candle in sorted(candles, key=lambda item: item.timestamp):
            self._append(state, candle)
        last = state.minutes[-1].timestamp if state.minutes else None
        if live_candle is not None and (last is None or live_candle.timestamp > last):
            state.live = _LiveMinute.from_candle(live_candle)

    def _append(self, state: _SymbolState, candle: Candle) -> list[Candle]:
        """按时间追加一根 1 分钟 K 线，中间缺的分钟补平线；返回实际追加的 K 线。"""
        minutes = state.minutes
        if minutes and candle.timestamp <= minutes[-1].timestamp:
            return []
        appended: list[Candle] = []
        if minutes:
            previous = minutes[-1]
            missing = int((candle.timestamp - previous.timestamp) / MINUTE) - 1
            # 超出保留长度的平线进来也会被挤出，只补需要的部分。
            for offset in range(min(missing, self._retention), 0, -1):
                appended.append(_flat(candle.timestamp - MINUTE * offset, previous.close))
        appended.append(candle)
        minutes.extend(appended)
        return appended

    def _close_minute(self, symbol: str, state: _SymbolState, candle: Candle) -> list[TrendAlert]:
        alerts: list[TrendAlert] = []
        for minute in self._append(state, candle):
            alerts.extend(self._evaluate(symbol, state, minute.timestamp + MINUTE))
        return alerts

    def _evaluate(self, symbol: str, state: _SymbolState, closed_at: datetime) -> list[TrendAlert]:
        """对收盘时刻落在周期边界的各周期判定；某个周期的形态新形成时提醒。"""
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

    def _window(self, state: _SymbolState, rule: TrendRule) -> list[Candle]:
        # 1 分钟 K 线连续且以周期边界结尾，取周期整数倍的尾部即按边界对齐。
        available = min(len(state.minutes), rule.history_minutes)
        count = available - available % rule.minutes
        tail = list(islice(state.minutes, len(state.minutes) - count, None))
        return aggregate_candles(tail, rule.minutes)
