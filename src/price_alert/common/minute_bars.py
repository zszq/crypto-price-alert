"""由实时成交聚合 1 分钟 K 线、每分钟收盘判定一次的检测器骨架，K 线趋势、放量启动、冷门启动共用。

这里只管 K 线：成交聚合、空分钟补平线、断线失效与 REST 回补，以及服务约定的合约增删接口；
每根 1 分钟 K 线收盘后怎么判定由子类的 _evaluate 决定。三者原先各写一份，回补合并、补平线这类细节
改一处漏两处就会出现「同一段断线三个检测器结果不一样」，所以收在一处。

- 下一分钟第一笔成交到达时才结算上一分钟；中间无人成交的分钟按上一收盘价补平线，与 Gate 的 K 线口径一致。
- 补出的多根 K 线逐根追加、逐根判定：判定某一分钟时，看不到它之后才收盘的 K 线。
- 断线或处理出错后合约标记为失效：只维护当前这一分钟，不结算、不判定，等 resync_symbol 用 REST K 线重建。
- add_candle 直接喂已收盘的 1 分钟 K 线，供回放与模拟使用，不能与 add_tick 混用。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar, Generic, TypeVar

from price_alert.common.models import Candle, PriceTick

MINUTE = timedelta(minutes=1)


@dataclass(slots=True)
class LiveMinute:
    """尚未收盘的当前一分钟。"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float = 0.0

    @classmethod
    def from_candle(cls, candle: Candle) -> LiveMinute:
        return cls(candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.quote_volume)

    def update(self, price: float, quote: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.quote_volume += quote

    def to_candle(self) -> Candle:
        return Candle(self.timestamp, self.open, self.high, self.low, self.close, self.quote_volume)


@dataclass(slots=True)
class MinuteState:
    """一个合约已收盘的 1 分钟 K 线与当前一分钟；需要随 K 线滚动维护统计量的检测器继承它并覆盖 append/clear。"""

    minutes: deque[Candle]
    volume_24h_quote: float
    price_decimals: int | None = None
    quanto_multiplier: float | None = None
    live: LiveMinute | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，本地 K 线不完整（成交额偏小、有成交的分钟数偏少），必须等 REST 回补后才能继续判定。
    stale: bool = False

    def append(self, candle: Candle) -> None:
        self.minutes.append(candle)

    def clear(self) -> None:
        self.minutes.clear()


def floor_minute(moment: datetime) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def flat_candle(timestamp: datetime, price: float) -> Candle:
    # 与 Gate 的口径一致：无成交的分钟是开高低收都等于上一收盘价、成交额为 0 的平线 K 线。
    return Candle(timestamp, price, price, price, price)


StateT = TypeVar("StateT", bound=MinuteState)
AlertT = TypeVar("AlertT")


class MinuteBarDetector(Generic[StateT, AlertT]):
    warmup_interval = "1m"
    # 子类按需换成带滚动统计量的状态类型。
    state_class: ClassVar[type[MinuteState]] = MinuteState
    # 判定要用计价成交额（张数 × 面值 × 价格）时为 True：没有面值的合约无从换算，保留在池里但不聚合、不判定。
    requires_quote: ClassVar[bool] = False

    def __init__(self, retention: int) -> None:
        # 判定所需的已收盘 1 分钟 K 线根数，更早的随追加挤出。
        self._retention = retention
        self._states: dict[str, StateT] = {}

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
        candles: Sequence[Candle],
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
        state = self.state_class(deque(maxlen=self._retention), volume_24h_quote, price_decimals, quanto_multiplier)
        self._seed(state, candles, live_candle)
        self._states[normalized] = state

    def remove_symbols(self, symbols: set[str]) -> None:
        # 子类的提醒记录不在这里，不随合约移出清除：合约很快重新入池时，同一段行情不应再提醒一次。
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
        candles: Sequence[Candle],
        live_candle: Candle | None,
        fetched_at: datetime,
    ) -> bool:
        """用 REST 1 分钟 K 线重建历史；合约已移除或已不需要回补时返回 False。子类的提醒记录保留。"""
        state = self._states.get(symbol.upper())
        if state is None or not state.stale:
            return False
        seed = sorted(candles, key=lambda item: item.timestamp)
        live = state.live
        if live_candle is not None:
            if live is None or live.timestamp < live_candle.timestamp:
                live = LiveMinute.from_candle(live_candle)
            elif live.timestamp == live_candle.timestamp:
                # 本地含请求返回后才到的成交，交易所含断线期间的成交：高低点取并集。两边的成交有重叠、
                # 无法相加，成交额取较大的一边，宁可略低估也不重复计入而凭空放量。
                is_local_newer = state.last_tick_time is not None and state.last_tick_time > fetched_at
                live = LiveMinute(
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

    def add_tick(self, tick: PriceTick) -> list[AlertT]:
        symbol = tick.symbol.upper()
        state = self._states.get(symbol)
        if state is None or (self.requires_quote and state.quanto_multiplier is None):
            return []
        timestamp = tick.timestamp.astimezone(UTC)
        if state.last_tick_time is not None and timestamp < state.last_tick_time:
            return []
        state.last_tick_time = timestamp
        multiplier = state.quanto_multiplier
        quote = tick.size * multiplier * tick.price if multiplier is not None else 0.0
        minute = floor_minute(timestamp)
        live = state.live
        if live is not None and minute == live.timestamp:
            live.update(tick.price, quote)
            return []
        if live is not None and minute < live.timestamp:
            # REST 给的当前 K 线可能比首笔实时成交还新，更早的成交已包含在其中。
            return []
        alerts: list[AlertT] = []
        if live is not None and not state.stale:
            closed = [live.to_candle()]
            # 上一分钟之后无人成交的分钟按平线补齐（_filled 补中间的），与交易所 K 线一样每分钟都按时判定。
            late = minute - live.timestamp > MINUTE
            if late:
                closed.append(flat_candle(minute - MINUTE, live.close))
            # 隔了空档才结算时，这些分钟的收盘价已过期，把这笔成交价交给判定复核；紧邻的下一分钟结算只晚几秒，
            # 不复核，免得新一分钟的单笔离群成交否掉本该发出的提醒（与 ATR 异动的口径一致）。
            for candle in closed:
                alerts.extend(self._close_minute(symbol, state, candle, tick.price if late else None))
        # 失效期间只维护当前这一分钟，已收盘的分钟等 REST 回补。
        state.live = LiveMinute(minute, tick.price, tick.price, tick.price, tick.price, quote)
        return alerts

    def add_candle(self, symbol: str, candle: Candle) -> list[AlertT]:
        """直接喂入一根已收盘的 1 分钟 K 线，供回放与模拟使用；不能与 add_tick 混用。"""
        state = self._states.get(symbol.upper())
        if state is None:
            return []
        return self._close_minute(symbol.upper(), state, candle)

    def _evaluate(self, symbol: str, state: StateT, closed_at: datetime, latest_price: float | None) -> list[AlertT]:
        """closed_at 收盘的那一分钟已追加到 state.minutes；latest_price 不为 None 时是空档后触发结算的成交价。"""
        raise NotImplementedError

    def _seed(self, state: MinuteState, candles: Sequence[Candle], live_candle: Candle | None) -> None:
        for candle in sorted(candles, key=lambda item: item.timestamp):
            for minute in self._filled(state, candle):
                state.append(minute)
        last = state.minutes[-1].timestamp if state.minutes else None
        if live_candle is not None and (last is None or live_candle.timestamp > last):
            state.live = LiveMinute.from_candle(live_candle)

    def _filled(self, state: MinuteState, candle: Candle) -> list[Candle]:
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
                filled.append(flat_candle(candle.timestamp - MINUTE * offset, previous.close))
        filled.append(candle)
        return filled

    def _close_minute(
        self, symbol: str, state: StateT, candle: Candle, latest_price: float | None = None
    ) -> list[AlertT]:
        alerts: list[AlertT] = []
        # 逐根追加、逐根判定：判定窗口取的是最近的 K 线，一次补出多根平线时若先全部追加，
        # 判定靠前的分钟会把之后的平线算进窗口，周期 K 线也不再对齐周期边界。
        for minute in self._filled(state, candle):
            state.append(minute)
            alerts.extend(self._evaluate(symbol, state, minute.timestamp + MINUTE, latest_price))
        return alerts
