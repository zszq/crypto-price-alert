"""使用 ATR 标准化短时价格位移，识别动态异动。"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Literal

from price_alert.common.indicators import WilderAtr
from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.spike.alerts import PriceAlert, SpikeKind
from price_alert.detectors.spike.streak import StreakAlert, StreakRule, StreakTracker
from price_alert.detectors.spike.windows import (
    BURST_BASE_SECONDS,
    BURST_SECONDS,
    Outcome,
    SpikeWindow,
    WindowEvaluation,
    WindowName,
)

# 基准桶允许比目标时刻早的秒数：成交稀疏时恰好落在目标秒上的桶未必存在，差一两秒不影响位移的含义。
BASELINE_TOLERANCE_SECONDS = 2

class _SecondHistory:
    """每个有成交的秒：时间（Unix 秒）、最后成交价、到这一秒为止的累计成交笔数，按时间递增。

    整个合约池每个合约都要留约 1 小时（3600 个点），三列分开存进 array 每点约 25 字节；
    用 tuple 加 datetime 存约 176 字节，两个检测器（短时异动、反弹回落）各一份时要多占几百 MB。
    """

    __slots__ = ("times", "prices", "trades")

    def __init__(self) -> None:
        self.times = array("q")
        self.prices = array("d")
        self.trades = array("q")

    def __len__(self) -> int:
        return len(self.times)

    def append(self, second: int, price: float, trades: int, oldest_required: int) -> None:
        self.times.append(second)
        self.prices.append(price)
        self.trades.append(trades)
        self.trim(oldest_required)

    def trim(self, oldest_required: int) -> None:
        # 与秒桶相同：留一个不晚于最早所需时刻的点，那一刻的价格与累计笔数要靠它确定。
        stale = bisect_right(self.times, oldest_required) - 1
        if stale > 0:
            del self.times[:stale]
            del self.prices[:stale]
            del self.trades[:stale]

    def index_at(self, moment: int) -> int:
        """moment 时刻最后一个有成交的秒的下标；没有那么早的记录时为 -1。"""
        return bisect_right(self.times, moment) - 1


_EMPTY_HISTORY = _SecondHistory()


@dataclass(slots=True)
class _SecondBucket:
    timestamp: datetime
    weighted_price_sum: float = 0.0
    total_size: float = 0.0
    price_sum: float = 0.0
    trade_count: int = 0
    last_price: float | None = None
    # 本桶之前累计的成交笔数。窗口笔数取两个桶的累计值之差，长窗口不必每秒逐桶求和。
    trades_before: int = 0

    @classmethod
    def carried(cls, timestamp: datetime, price: float, trades_before: int) -> _SecondBucket:
        return cls(timestamp, last_price=price, trades_before=trades_before)

    @property
    def has_trades(self) -> bool:
        return self.trade_count > 0

    @property
    def trades_through(self) -> int:
        return self.trades_before + self.trade_count

    def add(self, price: float, size: float) -> None:
        self.price_sum += price
        self.trade_count += 1
        self.last_price = price
        if size > 0:
            self.weighted_price_sum += price * size
            self.total_size += size

    @property
    def price(self) -> float:
        # 成交量加权能降低小额离群成交的影响；零成交量数据退化为普通均价。
        if self.total_size > 0:
            return self.weighted_price_sum / self.total_size
        if self.trade_count > 0:
            return self.price_sum / self.trade_count
        # 无成交的秒没有新的价格发现，市场价格仍停留在最后一笔成交。
        assert self.last_price is not None
        return self.last_price


def _bucket_time(bucket: _SecondBucket) -> datetime:
    return bucket.timestamp


@dataclass(slots=True)
class _LiveBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_candle(cls, candle: Candle) -> _LiveBar:
        return cls(candle.timestamp, candle.open, candle.high, candle.low, candle.close)

    def update(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price

    def to_candle(self) -> Candle:
        return Candle(self.timestamp, self.open, self.high, self.low, self.close)


@dataclass(slots=True)
class _Candidate:
    """单个窗口的连续确认进度；各窗口独立计数，互不打断。"""

    direction: Literal["surge", "drop"] | None = None
    seconds: int = 0
    last_second: datetime | None = None

    def reset(self) -> None:
        self.direction = None
        self.seconds = 0
        self.last_second = None

    def confirm(self, direction: Literal["surge", "drop"], second: datetime) -> int:
        is_consecutive = (
            self.direction == direction
            and self.last_second is not None
            and second - self.last_second == timedelta(seconds=1)
        )
        self.seconds = self.seconds + 1 if is_consecutive else 1
        self.direction = direction
        self.last_second = second
        return self.seconds


@dataclass(slots=True)
class _SymbolState:
    atr: WilderAtr
    volume_24h_quote: float
    candidates: list[_Candidate]
    price_decimals: int | None = None
    buckets: deque[_SecondBucket] = field(default_factory=deque)
    live_bar: _LiveBar | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，实时 K 线无法自行修复，必须等 REST 回补后才能再用 ATR 判定。
    atr_stale: bool = False


class AtrSpikeDetector:
    def __init__(
        self,
        *,
        atr_period: int,
        candle_interval_seconds: int,
        max_atr_age_seconds: int,
        windows: Sequence[SpikeWindow],
        observer: Callable[[WindowEvaluation], None] | None = None,
        streak: StreakRule | None = None,
        history_seconds: int = 0,
        kind: SpikeKind = "spike",
    ) -> None:
        """history_seconds 是子类另外需要的逐秒成交历史（秒），与各窗口附加条件所需的取较长者；
        kind 区分按瞬时异动还是短时异动的参数在跑，只影响提醒的标签与 kind 字段，不影响判定。"""
        if not windows:
            raise ValueError("至少需要一个观察窗口")
        if len({window.name for window in windows}) != len(windows):
            raise ValueError("观察窗口名称不能重复")
        self.atr_period = atr_period
        self.candle_interval_seconds = candle_interval_seconds
        self.max_atr_age = timedelta(seconds=max_atr_age_seconds)
        # 各窗口判定与冷却互不影响；顺序只决定同一秒多个窗口都提醒时的输出先后。
        self.windows = tuple(windows)
        self._observer = observer
        self._cooldowns = tuple(timedelta(seconds=window.cooldown_seconds) for window in self.windows)
        # 秒级数据按最长窗口保留、各窗口共用；空秒补齐以最短窗口为界，保证短窗口行为与单窗口时一致。
        self.max_lookback_seconds = max(window.lookback_seconds for window in self.windows)
        self._retention = timedelta(seconds=self.max_lookback_seconds)
        self._fill_limit = min(window.lookback_seconds for window in self.windows)
        self._candle_interval = timedelta(seconds=candle_interval_seconds)
        self._states: dict[str, _SymbolState] = {}
        # 按（合约, 窗口）记录上次提醒：短窗口频繁提醒不能把长窗口的冷却一直续上。
        self._last_alert: dict[tuple[str, WindowName], datetime] = {}
        # 只串联短窗口提醒：长窗口冷却等于窗口长度，本身就在描述持续行情，不需要再串。
        self.kind = kind
        self.streak_rule = streak
        self._streaks = StreakTracker(streak) if streak is not None else None
        # 前置走势与成交暴增要看窗口之前、乃至 1 小时的成交，秒桶只按窗口保留，所以另记一份逐秒价格与累计笔数。
        # 没有窗口用到它们时为 0，不记，实时监控不多花内存。
        span = max([history_seconds, *(window.history_seconds for window in self.windows)])
        self._history_span = span if span > 0 else None
        self._history: dict[str, _SecondHistory] = {}
        # 等待用 REST 补历史的合约：新加入或断线后历史不足，靠实时成交攒满要等 1 小时，期间附加条件判不了。
        # 自然攒满或补上后移除。补的历史只能接在实时成交之前，所以要记下实时成交从哪一秒开始（秒桶为空时的第一笔）。
        self._history_pending: set[str] = set()
        self._live_start: dict[str, int] = {}

    @property
    def history_seconds(self) -> int:
        """逐秒成交历史需要保留的秒数；为 0 时不记历史，也不需要补。"""
        return self._history_span or 0

    def history_backfill(self) -> dict[str, datetime | None]:
        """等待补历史的合约 → 实时成交开始的那一秒；还没收到实时成交的为 None，此时补了也不知道接在哪里。"""
        return {
            symbol: (datetime.fromtimestamp(self._live_start[symbol], UTC) if symbol in self._live_start else None)
            for symbol in sorted(self._history_pending)
        }

    def seed_history(self, symbol: str, seconds: Sequence[tuple[int, float, int]], live_start: datetime) -> bool:
        """把 REST 取回的逐秒成交（Unix 秒、最后成交价、笔数，升序）接在实时成交之前，返回是否采用。

        live_start 是请求前从 history_backfill 读到的实时成交起点，seconds 要含它之前的全部成交。
        只有起点没变才首尾相接：取数途中断线重连时实时成交重新开始，中间缺了一段，1 小时笔数会算少、
        暴增倍数算大，宁可不用。拿起点比对而不是比时间，本地时钟与交易所有偏差也不影响。
        """
        symbol = symbol.upper()
        if (
            self._history_span is None
            or symbol not in self._history_pending
            or self._live_start.get(symbol) != int(live_start.timestamp())
        ):
            return False
        live_second = int(live_start.timestamp())
        kept = [point for point in seconds if point[0] < live_second]
        seeded = _SecondHistory()
        # 实时成交第一个秒桶的累计笔数从 0 起算，REST 部分倒着排成负数，两段相减的笔数照样对。
        running = -sum(count for _, _, count in kept)
        for second, price, count in kept:
            running += count
            seeded.times.append(second)
            seeded.prices.append(price)
            seeded.trades.append(running)
        live = self._history.get(symbol)
        if live is not None:
            seeded.times.extend(live.times)
            seeded.prices.extend(live.prices)
            seeded.trades.extend(live.trades)
        if seeded:
            seeded.trim(seeded.times[-1] - self._history_span)
        self._history[symbol] = seeded
        self._history_pending.discard(symbol)
        return True

    @property
    def symbols(self) -> list[str]:
        return sorted(self._states)

    @property
    def stale_symbols(self) -> list[str]:
        return sorted(symbol for symbol, state in self._states.items() if state.atr_stale)

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
            return
        atr = WilderAtr(self.atr_period)
        atr.seed(candles)
        state = _SymbolState(
            atr=atr,
            volume_24h_quote=volume_24h_quote,
            candidates=[_Candidate() for _ in self.windows],
            price_decimals=price_decimals,
        )
        if live_candle is not None and (atr.last_timestamp is None or live_candle.timestamp > atr.last_timestamp):
            # 订阅前当前 K 线已走完一段，用 REST 的未收盘 K 线起步，避免只含订阅后成交的残缺 K 线低估 ATR。
            state.live_bar = _LiveBar.from_candle(live_candle)
        self._states[normalized] = state
        self._reset_history(normalized, backfill=True)

    def remove_symbols(self, symbols: set[str]) -> None:
        # 冷却记录刻意保留：合约被移出后很快重新入池时，不应绕过冷却再次提醒同一段行情。
        for symbol in symbols:
            self._states.pop(symbol, None)
            self._reset_history(symbol, backfill=False)

    # 断线时一并清空逐秒历史：断线期间成交缺失，1 小时笔数基准已不可靠；秒桶清空后累计笔数从 0 重新计数，
    # 留着旧记录会算出负数。重连后由服务用 REST 补上，补不上就重新攒满才判定，宁可漏报不误报。
    def mark_stream_gap(self) -> None:
        """行情中断后调用：丢弃秒级窗口与确认进度，并暂停 ATR 判定直到 resync_symbol 回补 K 线。"""
        for state in self._states.values():
            self._invalidate(state)
        for symbol in list(self._states):
            self._reset_history(symbol, backfill=True)

    def mark_symbol_gap(self, symbol: str) -> None:
        """单个合约的状态可能已不一致时调用（如处理成交中途出错）：与断线同样处理，等 resync_symbol 回补。"""
        state = self._states.get(symbol.upper())
        if state is not None:
            self._invalidate(state)
            self._reset_history(symbol.upper(), backfill=True)

    def _reset_history(self, symbol: str, backfill: bool) -> None:
        self._history.pop(symbol, None)
        self._live_start.pop(symbol, None)
        if backfill and self._history_span is not None:
            self._history_pending.add(symbol)
        else:
            self._history_pending.discard(symbol)

    @staticmethod
    def _invalidate(state: _SymbolState) -> None:
        state.buckets.clear()
        for candidate in state.candidates:
            candidate.reset()
        state.atr_stale = True

    def resync_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        live_candle: Candle | None,
        fetched_at: datetime,
    ) -> bool:
        """用 REST K 线重建 ATR 与实时 K 线；合约已移除或已不需要回补时返回 False。"""
        state = self._states.get(symbol.upper())
        if state is None or not state.atr_stale:
            # 回补期间合约可能被移出又以全新预热状态重新入池，此时不能用这批数据覆盖。
            return False

        seed = sorted(candles, key=lambda item: item.timestamp)
        live_bar = state.live_bar
        if live_candle is not None:
            if live_bar is None or live_bar.timestamp < live_candle.timestamp:
                # 本地实时 K 线停留在断线前的周期，整体换成交易所的当前 K 线。
                live_bar = _LiveBar.from_candle(live_candle)
            elif live_bar.timestamp == live_candle.timestamp:
                # 本地 K 线含有请求返回后才到达的成交，交易所 K 线含有断线期间的成交，两者取并集才完整。
                is_local_newer = state.last_tick_time is not None and state.last_tick_time > fetched_at
                live_bar = _LiveBar(
                    live_candle.timestamp,
                    live_candle.open,
                    max(live_bar.high, live_candle.high),
                    min(live_bar.low, live_candle.low),
                    live_bar.close if is_local_newer else live_candle.close,
                )
            else:
                # 请求返回后本地已跨入新周期，交易所返回的“当前 K 线”其实已经收盘，应计入 ATR。
                seed.append(live_candle)

        atr = WilderAtr(self.atr_period)
        atr.seed(candle for candle in seed if live_bar is None or candle.timestamp < live_bar.timestamp)
        state.atr = atr
        state.live_bar = live_bar
        state.atr_stale = False
        return True

    def add_tick(self, tick: PriceTick) -> list[PriceAlert | StreakAlert]:
        symbol = tick.symbol.upper()
        state = self._states.get(symbol)
        if state is None:
            return []

        timestamp = tick.timestamp.astimezone(UTC)
        if state.last_tick_time is not None and timestamp < state.last_tick_time:
            # 乱序旧价容易制造虚假的瞬时位移，因此不回填实时检测窗口。
            return []
        state.last_tick_time = timestamp

        second = timestamp.replace(microsecond=0)
        alerts: list[PriceAlert | StreakAlert] = []
        if state.buckets and state.buckets[-1].timestamp == second:
            state.buckets[-1].add(tick.price, tick.size)
        else:
            self._settle_through(symbol, state, second, tick.price, timestamp, alerts)
            trades_before = state.buckets[-1].trades_through if state.buckets else 0
            if not state.buckets and self._history_span is not None:
                # 秒桶为空（新加入或断线后）时的第一笔，累计笔数从这里的 0 起算，补的历史要接在这一秒之前。
                self._live_start[symbol] = int(second.timestamp())
            current = _SecondBucket(second, trades_before=trades_before)
            current.add(tick.price, tick.size)
            state.buckets.append(current)
        # 结算之后才计入 K 线：这笔成交可能已属下一周期，先计入会让被结算的旧秒用上之后周期的 ATR。
        # 同一秒的后续成交也要计入，否则漏掉该秒的高低价。
        self._update_bar(state, tick.price, timestamp)
        return alerts

    def _settle_through(
        self,
        symbol: str,
        state: _SymbolState,
        second: datetime,
        price: float,
        timestamp: datetime,
        alerts: list[PriceAlert | StreakAlert],
    ) -> None:
        """结算新一秒之前已结束的秒：上一个有成交的秒，以及空档中需要补齐的秒。"""
        if not state.buckets:
            return
        completed = state.buckets[-1]
        if self._history_span is not None:
            # 进入结算时最后一个桶就是刚结束的真实秒（补齐桶在结算过程中才追加）。
            assert completed.last_price is not None
            completed_second = int(completed.timestamp.timestamp())
            self._history.setdefault(symbol, _SecondHistory()).append(
                completed_second,
                completed.last_price,
                completed.trades_through,
                completed_second - self._history_span,
            )
            if symbol in self._history_pending and self._live_start[symbol] <= completed_second - self._history_span:
                # 实时成交已攒满所需长度，不必再补。
                self._history_pending.discard(symbol)
        missing_seconds = int((second - completed.timestamp).total_seconds()) - 1
        # 等一秒结束才用整秒 VWAP。空档后才结算的秒 VWAP 已过期，要用当前成交价复核；
        # 紧邻结算只迟一秒，不复核，免得新一秒的单笔离群成交否掉本该发出的提醒。
        projections: dict[datetime, WilderAtr] = {}
        self._settle(
            symbol,
            state,
            completed,
            timestamp,
            alerts,
            price if missing_seconds > 0 else None,
            self._atr_before(state, completed.timestamp, projections),
        )
        # 稀疏合约在快速行情中常有空秒，用最后成交价补齐才能连续确认；空档超过最短窗口说明行情停滞，不再补齐。
        if 0 < missing_seconds <= self._fill_limit:
            last_price = completed.last_price
            assert last_price is not None
            for offset in range(1, missing_seconds + 1):
                filler = _SecondBucket.carried(
                    completed.timestamp + timedelta(seconds=offset),
                    last_price,
                    completed.trades_through,
                )
                state.buckets.append(filler)
                # 补齐桶本就不能触发提醒，无需复核。
                atr = self._atr_before(state, filler.timestamp, projections)
                self._settle(symbol, state, filler, timestamp, alerts, None, atr)

    def _settle(
        self,
        symbol: str,
        state: _SymbolState,
        bucket: _SecondBucket,
        timestamp: datetime,
        alerts: list[PriceAlert | StreakAlert],
        latest_price: float | None,
        atr: WilderAtr | None,
    ) -> None:
        # 保留最长窗口所需的全部秒，另留一个不晚于窗口起点的桶，空档中的基准价要靠它确定。
        oldest_required = bucket.timestamp - self._retention
        while len(state.buckets) > 2 and state.buckets[1].timestamp <= oldest_required:
            state.buckets.popleft()
        alerts.extend(self._evaluate(symbol, state, bucket, timestamp, latest_price, atr))

    def _atr_before(
        self,
        state: _SymbolState,
        second: datetime,
        projections: dict[datetime, WilderAtr],
    ) -> WilderAtr | None:
        """评估 second 时应使用的 ATR：只含 second 所在周期之前已收盘的 K 线；没有这样的 ATR 时返回 None。"""
        bar_timestamp = self._floor_time(second)
        live_bar = state.live_bar
        if state.atr_stale or live_bar is None or bar_timestamp <= live_bar.timestamp:
            last_timestamp = state.atr.last_timestamp
            if not state.atr_stale and last_timestamp is not None and last_timestamp >= bar_timestamp:
                # 回补重建了 ATR，却留着回补前未结算的秒：它们早于重建后的实时 K 线时，ATR 已含其所在及之后的周期，
                # 旧 ATR 无从恢复，只能放弃判定。正常逐笔推进时 ATR 总早于实时 K 线，走不到这里。
                return None
            return state.atr
        # 补齐秒跨入了实时 K 线之后的周期，实时 K 线及其后的空周期对它都已收盘。只在副本上推演，
        # 真正计入仍由触发结算的成交完成，所以每笔成交处理完后的状态与分几段评估无关；同一周期共用一份推演。
        projected = projections.get(bar_timestamp)
        if projected is None:
            projected = state.atr.copy()
            previous = live_bar.to_candle()
            projected.update(previous)
            self._fill_flat_bars(projected, previous, bar_timestamp)
            projections[bar_timestamp] = projected
        return projected

    def _update_bar(self, state: _SymbolState, price: float, timestamp: datetime) -> None:
        bar_timestamp = self._floor_time(timestamp)
        if state.live_bar is None:
            state.live_bar = _LiveBar(bar_timestamp, price, price, price, price)
            return
        if bar_timestamp == state.live_bar.timestamp:
            state.live_bar.update(price)
            return
        if bar_timestamp > state.live_bar.timestamp:
            if not state.atr_stale:
                # 断线期间 ATR 等待 REST 重建，此时喂入缺口两侧的 K 线只会引入错误数据。
                previous = state.live_bar.to_candle()
                state.atr.update(previous)
                self._fill_flat_bars(state.atr, previous, bar_timestamp)
            state.live_bar = _LiveBar(bar_timestamp, price, price, price, price)

    def _fill_flat_bars(self, atr: WilderAtr, previous: Candle, next_bar: datetime) -> None:
        # 连接正常时跨过的周期确实无人成交，Gate 对它返回开高低收都等于上一收盘价的平线；实时保持同样口径，
        # ATR 才与预热数据一致，冷门合约停摆后 ATR 也不会被误判过期。
        missing = int((next_bar - previous.timestamp) / self._candle_interval) - 1
        if missing <= 0:
            return
        # 连续平线让 Wilder ATR 按 (n-1)/n 几何衰减，4n 根后原值只剩约 2%，更早的平线无需逐根计算。
        count = min(missing, self.atr_period * 4)
        for offset in range(count, 0, -1):
            flat_time = next_bar - self._candle_interval * offset
            atr.update(Candle(flat_time, previous.close, previous.close, previous.close, previous.close))

    def _floor_time(self, timestamp: datetime) -> datetime:
        epoch = int(timestamp.timestamp())
        floored = epoch - (epoch % self.candle_interval_seconds)
        return datetime.fromtimestamp(floored, tz=UTC)

    def _evaluate(
        self,
        symbol: str,
        state: _SymbolState,
        current: _SecondBucket,
        timestamp: datetime,
        latest_price: float | None,
        atr_state: WilderAtr | None,
    ) -> list[PriceAlert | StreakAlert]:
        """在所有窗口上评估一个已结束的秒；各窗口的冷却与确认进度互不影响，同一秒可以各报一条。"""
        evaluations = [
            self._evaluate_window(symbol, state, window, candidate, current, latest_price, atr_state)
            for window, candidate in zip(self.windows, state.candidates, strict=True)
        ]

        alerts: list[PriceAlert | StreakAlert] = []
        for index, evaluation in enumerate(evaluations):
            if evaluation.outcome is not Outcome.ALERT:
                continue
            window = self.windows[index]
            key = (symbol, window.name)
            previous_alert = self._last_alert.get(key)
            if previous_alert is not None and timestamp - previous_alert < self._cooldowns[index]:
                evaluations[index] = replace(evaluation, outcome=Outcome.COOLDOWN)
            else:
                alert = self._build_alert(symbol, state, window, evaluation, timestamp)
                alerts.append(alert)
                self._last_alert[key] = timestamp
                if self._streaks is not None and window.name == "short":
                    # 紧跟在触发它的那条短窗口提醒之后，读者先看到这一波，再看到“已经连续几波”。
                    streak_alert = self._streaks.observe(alert)
                    if streak_alert is not None:
                        alerts.append(streak_alert)
            state.candidates[index].reset()

        if self._observer is not None:
            for evaluation in evaluations:
                self._observer(evaluation)
        return alerts

    def _evaluate_window(
        self,
        symbol: str,
        state: _SymbolState,
        window: SpikeWindow,
        candidate: _Candidate,
        current: _SecondBucket,
        latest_price: float | None,
        atr_state: WilderAtr | None,
    ) -> WindowEvaluation:
        """评估一个窗口；latest_price 非空时还需该价格仍满足门槛才允许提醒。

        atr_state 由调用方按这一秒所在周期选定，不能直接用 state.atr：结算可能晚于下一周期开始；
        为 None 表示这一秒没有对应周期的 ATR 可用。
        返回 ALERT 只表示窗口本身的条件全部满足，冷却由调用方处理。
        """

        def result(outcome: Outcome, atr: float | None = None) -> WindowEvaluation:
            return WindowEvaluation(
                symbol, window.name, current.timestamp, outcome, current.price, not current.has_trades, atr
            )

        atr = atr_state.value if atr_state is not None else None
        atr_timestamp = atr_state.last_timestamp if atr_state is not None else None
        if state.atr_stale or atr is None or atr <= 0 or atr_timestamp is None:
            candidate.reset()
            return result(Outcome.ATR_UNAVAILABLE)
        # K 线时间戳是开盘时间，按收盘时间计算年龄，配置值才等于“ATR 最近一次更新距今多久”。
        if current.timestamp - (atr_timestamp + self._candle_interval) > self.max_atr_age:
            candidate.reset()
            return result(Outcome.ATR_EXPIRED, atr)

        baseline = self._find_baseline(state.buckets, current, window)
        if baseline is None:
            candidate.reset()
            return result(Outcome.NO_BASELINE, atr)
        baseline_bucket, baseline_price = baseline
        price_move = current.price - baseline_price
        trade_count = current.trades_through - baseline_bucket.trades_through

        def measured(outcome: Outcome, confirmed_seconds: int = 0) -> WindowEvaluation:
            return replace(
                result(outcome, atr),
                baseline_price=baseline_price,
                change_percent=price_move / baseline_price * 100.0,
                move_atr=abs(price_move) / atr,
                trade_count=trade_count,
                confirmed_seconds=confirmed_seconds,
            )

        if trade_count < window.min_window_trades:
            candidate.reset()
            return measured(Outcome.FEW_TRADES)
        if not window.exceeds_thresholds(price_move, baseline_price, atr):
            candidate.reset()
            return measured(Outcome.BELOW_THRESHOLD)

        direction: Literal["surge", "drop"] = "surge" if price_move > 0 else "drop"
        confirmed = candidate.confirm(direction, current.timestamp)
        if confirmed < window.confirmation_seconds:
            return measured(Outcome.CONFIRMING, confirmed)
        if not current.has_trades:
            # 补齐的空秒只延续确认进度，不能单独触发：否则一笔离群成交后恰好无人成交也会被当成持续异动。
            return measured(Outcome.CARRIED, confirmed)
        if latest_price is not None and not self._still_moving(window, latest_price, baseline_price, atr, direction):
            # 空档后才结算的秒价格已过期，触发结算的成交是唯一的当前价：它不满足门槛说明异动已在空档中结束。
            # 与补齐桶一样只拦提醒、不重置确认进度。
            return measured(Outcome.MOVE_FADED, confirmed)
        if window.min_prior_move_percent is None and window.min_trade_burst_ratio is None:
            return measured(Outcome.ALERT, confirmed)
        # 两个附加条件只在可以提醒的那一秒判定，不满足同样只拦这一秒、不清零确认，与研究的判定一致。
        history = self._history.get(symbol, _EMPTY_HISTORY)
        prior = self._prior_change(
            history, current.timestamp, baseline_price, direction, window.lookback_seconds + window.prior_seconds
        )
        burst = self._trade_burst(history, current)
        gated = replace(
            measured(Outcome.ALERT, confirmed),
            # 内部按「顺急动方向为正」判定，记录里换回带符号的实际涨跌幅：急拉之前是负的（下跌）。
            prior_change_percent=None if prior is None else (prior if direction == "surge" else -prior),
            trade_burst_ratio=burst,
        )
        if window.min_prior_move_percent is not None and (prior is None or prior > -window.min_prior_move_percent):
            return replace(gated, outcome=Outcome.PRIOR_TREND)
        if window.min_trade_burst_ratio is not None and (burst is None or burst < window.min_trade_burst_ratio):
            return replace(gated, outcome=Outcome.NO_TRADE_BURST)
        return gated

    @staticmethod
    def _prior_change(
        history: _SecondHistory,
        second: datetime,
        baseline_price: float,
        direction: Literal["surge", "drop"],
        offset_seconds: int,
    ) -> float | None:
        """second 之前 offset_seconds 秒那一刻（取该时刻最后一个有成交的秒的最后成交价）到基准价的涨跌幅，

        顺急动方向为正；没有那么早的成交时返回 None。
        """
        index = history.index_at(int(second.timestamp()) - offset_seconds)
        if index < 0:
            return None
        change = (baseline_price / history.prices[index] - 1.0) * 100.0
        return change if direction == "surge" else -change

    @staticmethod
    def _trades_at(history: _SecondHistory, moment: int) -> int | None:
        index = history.index_at(moment)
        return history.trades[index] if index >= 0 else None

    @classmethod
    def _trade_burst(cls, history: _SecondHistory, current: _SecondBucket) -> float | None:
        """最近 1 分钟成交笔数 ÷ 最近 1 小时每分钟平均笔数；不足 1 小时的成交历史时返回 None。"""
        second = int(current.timestamp.timestamp())
        hour_start = cls._trades_at(history, second - BURST_BASE_SECONDS)
        if hour_start is None:
            return None
        hour = current.trades_through - hour_start
        if hour <= 0:
            return None
        # 1 小时前有记录，1 分钟前必然也有。
        minute_start = cls._trades_at(history, second - BURST_SECONDS)
        assert minute_start is not None
        return (current.trades_through - minute_start) / hour * (BURST_BASE_SECONDS / BURST_SECONDS)

    @staticmethod
    def _find_baseline(
        buckets: deque[_SecondBucket],
        current: _SecondBucket,
        window: SpikeWindow,
    ) -> tuple[_SecondBucket, float] | None:
        """找窗口起点的基准桶与基准价；空档过长、基准价已无意义时返回 None。"""
        target = current.timestamp - timedelta(seconds=window.lookback_seconds)
        # 桶按时间严格递增，二分查找让长窗口的定位成本与窗口长度基本无关。
        index = bisect_right(buckets, target, key=_bucket_time) - 1
        if index < 0:
            return None
        baseline = buckets[index]
        tolerance = timedelta(seconds=window.lookback_seconds + BASELINE_TOLERANCE_SECONDS)
        if current.timestamp - baseline.timestamp <= tolerance:
            return baseline, baseline.price
        # 窗口起点落在空档里：空档内价格停在最后成交价，空档不长于本窗口就沿用它。
        # 短窗口的这类空档都已补齐、走不到这里；长窗口据此跨过比补齐上限更长的空档，不至于整段失明。
        following = buckets[index + 1]
        missing = following.timestamp - baseline.timestamp - timedelta(seconds=1)
        if missing <= timedelta(seconds=window.lookback_seconds):
            assert baseline.last_price is not None
            return baseline, baseline.last_price
        return None

    @staticmethod
    def _still_moving(
        window: SpikeWindow,
        latest_price: float,
        baseline_price: float,
        atr: float,
        direction: Literal["surge", "drop"],
    ) -> bool:
        price_move = latest_price - baseline_price
        # 反向必须单独判断：仅看幅度的话，急涨过后直接砸穿基准价也能满足门槛而发出急涨提醒。
        if (price_move > 0) != (direction == "surge"):
            return False
        return window.exceeds_thresholds(price_move, baseline_price, atr)

    def _build_alert(
        self,
        symbol: str,
        state: _SymbolState,
        window: SpikeWindow,
        evaluation: WindowEvaluation,
        timestamp: datetime,
    ) -> PriceAlert:
        # 走到 ALERT 的判定一定带齐了全部指标，这里只做类型收窄。
        assert evaluation.baseline_price is not None and evaluation.change_percent is not None
        assert evaluation.move_atr is not None and evaluation.atr is not None
        assert evaluation.trade_count is not None
        return PriceAlert(
            symbol=symbol,
            direction="surge" if evaluation.change_percent > 0 else "drop",
            price=evaluation.price,
            reference_price=evaluation.baseline_price,
            change_percent=evaluation.change_percent,
            move_atr=evaluation.move_atr,
            atr=evaluation.atr,
            atr_period=self.atr_period,
            lookback_seconds=window.lookback_seconds,
            trade_count=evaluation.trade_count,
            volume_24h_quote=state.volume_24h_quote,
            timestamp=timestamp,
            window=window.name,
            kind=self.kind,
            price_decimals=state.price_decimals,
            # 只带窗口实际判了的条件，没开的那一项即使顺手算出了取值也不写进提醒。
            prior_change_percent=evaluation.prior_change_percent if window.min_prior_move_percent is not None else None,
            prior_seconds=window.prior_seconds if window.min_prior_move_percent is not None else None,
            trade_burst_ratio=evaluation.trade_burst_ratio if window.min_trade_burst_ratio is not None else None,
        )
