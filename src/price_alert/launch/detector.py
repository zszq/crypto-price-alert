"""放量启动检测器：由实时成交聚合带成交额的 1 分钟 K 线，每分钟收盘判定一次。

判定见 rule.evaluate_launch。重复提醒按（合约, 方向）记录上一次提醒：
- 距上次提醒不足 cooldown_minutes 不再提醒，同一段放量在窗口滑过期间不会每分钟报一次；
- 冷却过后，价格还要比上次提醒价再推进 realert_step_percent 才报下一波，原地放量震荡不重复提醒；
- 上次提醒已超过一个基准期，它的放量和高点都已移出基准期，再次满足条件就是新的启动，从第 1 波重新计数。

对外接口与其他检测器保持一致（add_symbol / add_tick / mark_stream_gap / resync_symbol 等），
服务主循环因此不必知道它的存在；回放和模拟则直接喂已收盘的 1 分钟 K 线（add_candle）。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from price_alert.launch.alerts import LaunchAlert
from price_alert.launch.rule import Direction, LaunchMetrics, LaunchRule, evaluate_launch, still_launching
from price_alert.models import Candle, PriceTick

MINUTE = timedelta(minutes=1)


class LaunchOutcome(StrEnum):
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"
    COOLDOWN = "cooldown"
    NOT_EXTENDED = "not_extended"
    EXPIRED = "expired"
    ALERT = "alert"


OUTCOME_LABELS: dict[LaunchOutcome, str] = {
    LaunchOutcome.INSUFFICIENT_DATA: "K 线不足",
    LaunchOutcome.FAILED: "未满足条件",
    LaunchOutcome.COOLDOWN: "满足条件，冷却中",
    LaunchOutcome.NOT_EXTENDED: "满足条件，价格未比上次提醒再推进",
    LaunchOutcome.EXPIRED: "满足条件，但空档后结算时价格已回落",
    LaunchOutcome.ALERT: "提醒",
}


@dataclass(frozen=True, slots=True)
class LaunchEvaluation:
    """一根 1 分钟 K 线收盘时的判定，回放据此解释为什么提醒或没有提醒。"""

    symbol: str
    timestamp: datetime
    outcome: LaunchOutcome
    metrics: LaunchMetrics | None = None


@dataclass(slots=True)
class _LiveMinute:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float = 0.0

    @classmethod
    def from_candle(cls, candle: Candle) -> _LiveMinute:
        return cls(candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.quote_volume)

    def update(self, price: float, quote: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.quote_volume += quote

    def to_candle(self) -> Candle:
        return Candle(self.timestamp, self.open, self.high, self.low, self.close, self.quote_volume)


@dataclass(frozen=True, slots=True)
class _LastAlert:
    timestamp: datetime
    price: float
    wave: int


@dataclass(slots=True)
class _SymbolState:
    minutes: deque[Candle]
    volume_24h_quote: float
    price_decimals: int | None = None
    quanto_multiplier: float | None = None
    live: _LiveMinute | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，本地 K 线的成交额偏小，必须等 REST 回补后才能继续判定。
    stale: bool = False


def _floor_minute(moment: datetime) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def _flat(timestamp: datetime, price: float) -> Candle:
    # 与 Gate 的口径一致：无成交的分钟是开高低收都等于上一收盘价、成交额为 0 的平线 K 线。
    return Candle(timestamp, price, price, price, price)


class LaunchDetector:
    warmup_interval = "1m"

    def __init__(self, rule: LaunchRule, observer: Callable[[LaunchEvaluation], None] | None = None) -> None:
        self.rule = rule
        self._observer = observer
        self._retention = rule.history_minutes
        self._states: dict[str, _SymbolState] = {}
        # 不随合约移出而清除：合约很快重新入池时，仍在延续的同一波不应再当作启动提醒一次。
        self._last_alerts: dict[tuple[str, Direction], _LastAlert] = {}

    @property
    def warmup_candles(self) -> int:
        # 判定一次所需的 1 分钟 K 线，加一根未收盘的当前 K 线。
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
        quanto_multiplier: float | None = None,
    ) -> None:
        normalized = symbol.upper()
        existing = self._states.get(normalized)
        if existing is not None:
            existing.volume_24h_quote = volume_24h_quote
            existing.price_decimals = price_decimals
            existing.quanto_multiplier = quanto_multiplier
            return
        state = _SymbolState(deque(maxlen=self._retention), volume_24h_quote, price_decimals, quanto_multiplier)
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
        """用 REST 1 分钟 K 线重建历史；合约已移除或已不需要回补时返回 False。提醒记录保留。"""
        state = self._states.get(symbol.upper())
        if state is None or not state.stale:
            return False
        seed = sorted(candles, key=lambda item: item.timestamp)
        live = state.live
        if live_candle is not None:
            if live is None or live.timestamp < live_candle.timestamp:
                live = _LiveMinute.from_candle(live_candle)
            elif live.timestamp == live_candle.timestamp:
                # 本地含请求返回后才到的成交，交易所含断线期间的成交：高低点取并集。两边的成交有重叠、
                # 无法相加，成交额取较大的一边，宁可略低估也不重复计入而凭空放量。
                is_local_newer = state.last_tick_time is not None and state.last_tick_time > fetched_at
                live = _LiveMinute(
                    live.timestamp,
                    live_candle.open,
                    max(live.high, live_candle.high),
                    min(live.low, live_candle.low),
                    live.close if is_local_newer else live_candle.close,
                    max(live.quote_volume, live_candle.quote_volume),
                )
            else:
                # 请求返回后本地已跨入新的一分钟，交易所返回的“当前 K 线”其实已收盘。
                seed.append(live_candle)
        state.minutes.clear()
        state.live = None
        self._seed(state, seed, live.to_candle() if live is not None else None)
        state.stale = False
        return True

    def add_tick(self, tick: PriceTick) -> list[LaunchAlert]:
        symbol = tick.symbol.upper()
        state = self._states.get(symbol)
        # 没有合约面值就无法把张数换算成成交额，量比无从谈起，干脆不跟踪。
        if state is None or state.quanto_multiplier is None:
            return []
        timestamp = tick.timestamp.astimezone(UTC)
        if state.last_tick_time is not None and timestamp < state.last_tick_time:
            return []
        state.last_tick_time = timestamp
        quote = tick.size * state.quanto_multiplier * tick.price
        minute = _floor_minute(timestamp)
        live = state.live
        if live is not None and minute == live.timestamp:
            live.update(tick.price, quote)
            return []
        if live is not None and minute < live.timestamp:
            # REST 给的当前 K 线可能比首笔实时成交还新，更早的成交已包含在其中。
            return []
        alerts: list[LaunchAlert] = []
        if live is not None and not state.stale:
            closed = [live.to_candle()]
            # 上一分钟之后无人成交的分钟按平线补齐（_filled 补中间的），与交易所 K 线一样每分钟都按时判定。
            late = minute - live.timestamp > MINUTE
            if late:
                closed.append(_flat(minute - MINUTE, live.close))
            # 隔了空档才结算时，这些分钟的收盘价已过期，用这笔成交价复核；紧邻的下一分钟结算只晚几秒，不复核，
            # 免得新一分钟的单笔离群成交否掉本该发出的提醒（与 ATR 异动的口径一致）。
            for candle in closed:
                alerts.extend(self._close_minute(symbol, state, candle, tick.price if late else None))
        # 失效期间只维护当前这一分钟，已收盘的分钟等 REST 回补。
        state.live = _LiveMinute(minute, tick.price, tick.price, tick.price, tick.price, quote)
        return alerts

    def add_candle(self, symbol: str, candle: Candle) -> list[LaunchAlert]:
        """直接喂入一根已收盘的 1 分钟 K 线，供回放与模拟使用；不能与 add_tick 混用。"""
        state = self._states.get(symbol.upper())
        if state is None:
            return []
        return self._close_minute(symbol.upper(), state, candle)

    def _seed(self, state: _SymbolState, candles: list[Candle], live_candle: Candle | None) -> None:
        for candle in sorted(candles, key=lambda item: item.timestamp):
            state.minutes.extend(self._filled(state, candle))
        last = state.minutes[-1].timestamp if state.minutes else None
        if live_candle is not None and (last is None or live_candle.timestamp > last):
            state.live = _LiveMinute.from_candle(live_candle)

    def _filled(self, state: _SymbolState, candle: Candle) -> list[Candle]:
        """接在已有 K 线之后需要追加的 K 线：中间缺的分钟补平线；不晚于最后一根的返回空列表。"""
        minutes = state.minutes
        if minutes and candle.timestamp <= minutes[-1].timestamp:
            return []
        filled: list[Candle] = []
        if minutes:
            previous = minutes[-1]
            missing = int((candle.timestamp - previous.timestamp) / MINUTE) - 1
            # 超出保留长度的平线进来也会被挤出，只补需要的部分。
            for offset in range(min(missing, self._retention), 0, -1):
                filled.append(_flat(candle.timestamp - MINUTE * offset, previous.close))
        filled.append(candle)
        return filled

    def _close_minute(
        self, symbol: str, state: _SymbolState, candle: Candle, latest_price: float | None = None
    ) -> list[LaunchAlert]:
        alerts: list[LaunchAlert] = []
        # 逐根追加、逐根判定：补齐的平线分钟判定时，不能看到它之后才收盘的 K 线。
        for minute in self._filled(state, candle):
            state.minutes.append(minute)
            alert = self._evaluate(symbol, state, minute.timestamp + MINUTE, latest_price)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _evaluate(
        self, symbol: str, state: _SymbolState, closed_at: datetime, latest_price: float | None
    ) -> LaunchAlert | None:
        rule = self.rule
        alert: LaunchAlert | None = None
        if len(state.minutes) < self._retention:
            evaluation = LaunchEvaluation(symbol, closed_at, LaunchOutcome.INSUFFICIENT_DATA)
        else:
            metrics = evaluate_launch(list(state.minutes), rule)
            if not metrics.passed:
                outcome, wave = LaunchOutcome.FAILED, 0
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
        return alert

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
