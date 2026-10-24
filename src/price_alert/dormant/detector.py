"""冷门启动检测器：由实时成交聚合带成交额的 1 分钟 K 线，每分钟收盘判定一次。

判定见 rule.evaluate_dormant。同一合约距上次提醒（含补报）不足 cooldown_minutes 时，只有价格比上次提醒价
再涨 realert_step_percent 才再报「第 N 波」：冷门币一段拉升通常十几到几十分钟，窗口滑过期间不会每分钟报一次，
但第一次没注意到时还有补报；超过冷却再满足条件就当作新的启动，从第 1 波算。拉升之后之前 24 小时的成交额
很快超过门槛，自然不再满足冷门条件。

K 线聚合、空分钟补平线、断线回补的做法与放量启动检测器相同；对外接口与其他检测器一致，服务主循环不必知道它的存在；
回放和模拟直接喂已收盘的 1 分钟 K 线（add_candle）。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from price_alert.dormant.alerts import DormantAlert
from price_alert.dormant.rule import (
    CALM_MINUTES,
    DormantMetrics,
    DormantRule,
    evaluate_dormant,
    minute_range_percent,
    still_rising,
)
from price_alert.models import Candle, PriceTick

MINUTE = timedelta(minutes=1)


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


@dataclass(slots=True)
class _SymbolState:
    minutes: deque[Candle]
    volume_24h_quote: float
    price_decimals: int | None = None
    quanto_multiplier: float | None = None
    live: _LiveMinute | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，本地 K 线的成交额偏小、有成交的分钟数也会少，必须等 REST 回补后才能继续判定。
    stale: bool = False
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


def _floor_minute(moment: datetime) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def _flat(timestamp: datetime, price: float) -> Candle:
    # 与 Gate 的口径一致：无成交的分钟是开高低收都等于上一收盘价、成交额为 0 的平线 K 线。
    return Candle(timestamp, price, price, price, price)


class DormantDetector:
    warmup_interval = "1m"

    def __init__(self, rule: DormantRule, observer: Callable[[DormantEvaluation], None] | None = None) -> None:
        self.rule = rule
        self._observer = observer
        self._retention = rule.history_minutes
        self._states: dict[str, _SymbolState] = {}
        # 不随合约移出而清除：合约很快重新入池时，同一段拉升不应再提醒一次。
        self._last_alerts: dict[str, _LastAlert] = {}

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
                # 与放量启动相同：高低点取并集；两边的成交有重叠、无法相加，成交额取较大的一边。
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
        state.clear()
        state.live = None
        self._seed(state, seed, live.to_candle() if live is not None else None)
        state.stale = False
        return True

    def add_tick(self, tick: PriceTick) -> list[DormantAlert]:
        symbol = tick.symbol.upper()
        state = self._states.get(symbol)
        # 没有合约面值就无法把张数换算成成交额，冷门与否都判断不了，干脆不跟踪。
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
        alerts: list[DormantAlert] = []
        if live is not None and not state.stale:
            closed = [live.to_candle()]
            # 上一分钟之后无人成交的分钟按平线补齐（_filled 补中间的），每分钟都按时判定。
            late = minute - live.timestamp > MINUTE
            if late:
                closed.append(_flat(minute - MINUTE, live.close))
            # 冷门币成交稀疏，隔了空档才结算很常见：这些分钟的收盘价已过期，用这笔成交价复核；
            # 紧邻的下一分钟结算只晚几秒，不复核（与放量启动的口径一致）。
            for candle in closed:
                alerts.extend(self._close_minute(symbol, state, candle, tick.price if late else None))
        # 失效期间只维护当前这一分钟，已收盘的分钟等 REST 回补。
        state.live = _LiveMinute(minute, tick.price, tick.price, tick.price, tick.price, quote)
        return alerts

    def add_candle(self, symbol: str, candle: Candle) -> list[DormantAlert]:
        """直接喂入一根已收盘的 1 分钟 K 线，供回放与模拟使用；不能与 add_tick 混用。"""
        state = self._states.get(symbol.upper())
        if state is None:
            return []
        return self._close_minute(symbol.upper(), state, candle)

    def _seed(self, state: _SymbolState, candles: list[Candle], live_candle: Candle | None) -> None:
        for candle in sorted(candles, key=lambda item: item.timestamp):
            for minute in self._filled(state, candle):
                state.append(minute)
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
    ) -> list[DormantAlert]:
        alerts: list[DormantAlert] = []
        # 逐根追加、逐根判定：补齐的平线分钟判定时，不能看到它之后才收盘的 K 线。
        for minute in self._filled(state, candle):
            state.append(minute)
            alert = self._evaluate(symbol, state, minute.timestamp + MINUTE, latest_price)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _evaluate(
        self, symbol: str, state: _SymbolState, closed_at: datetime, latest_price: float | None
    ) -> DormantAlert | None:
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
        return alert

    def _extended(self, last: _LastAlert, price: float) -> bool:
        step = self.rule.realert_step_percent
        return step is not None and (price / last.price - 1) * 100 >= step
