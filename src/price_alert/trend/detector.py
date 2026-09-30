"""K 线形态趋势检测器：由实时成交聚合 1 分钟 K 线，再合成各周期 K 线按形态判定，并用状态机输出提醒。

对外接口与 ATR 异动检测器保持一致（add_symbol / add_tick / mark_stream_gap / resync_symbol 等），
服务主循环因此不必知道它的存在；回放和调研则直接喂已收盘的 1 分钟 K 线（add_candle）。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import islice

from price_alert.models import Candle, PriceTick
from price_alert.trend.alerts import EndReason, TrendAlert, TrendStage
from price_alert.trend.pattern import (
    Direction,
    PatternMetrics,
    PeriodName,
    TrendRule,
    adverse_shadow_ratio,
    aggregate_candles,
    evaluate_pattern,
)

MINUTE = timedelta(minutes=1)
# Gate 单次最多返回 2000 根 K 线，其中一根是未收盘的当前 K 线。
MAX_WARMUP_CANDLES = 2000


@dataclass(frozen=True, slots=True)
class EpisodeSettings:
    """一段趋势从开始到结束的状态机参数，所有周期共用：同一段行情不论哪个周期先发现都只算一段。"""

    cooldown_minutes: int
    escalation_step_percent: float
    end_rebound_ratio: float
    stall_minutes: int
    notify_end: bool
    exhaustion_enabled: bool
    exhaustion_shadow_ratio: float
    exhaustion_candles: int
    # 检查长影线的周期。调研显示 1 分钟 K 线的长影线对趋势放缓几乎没有预示作用，默认只看较长周期。
    exhaustion_periods: frozenset[PeriodName]


class TrendOutcome(StrEnum):
    INSUFFICIENT_DATA = "insufficient_data"
    NO_DIRECTION = "no_direction"
    FAILED = "failed"
    EPISODE_ACTIVE = "episode_active"
    COOLDOWN = "cooldown"
    SUPERSEDED = "superseded"
    ALERT = "alert"


OUTCOME_LABELS: dict[TrendOutcome, str] = {
    TrendOutcome.INSUFFICIENT_DATA: "K 线不足",
    TrendOutcome.NO_DIRECTION: "首尾价格相同",
    TrendOutcome.FAILED: "未满足形态条件",
    TrendOutcome.EPISODE_ACTIVE: "形态成立，同方向趋势已在跟踪",
    TrendOutcome.COOLDOWN: "形态成立，冷却中",
    TrendOutcome.SUPERSEDED: "形态成立，同一分钟已由更短周期提醒",
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
    volume: float

    @classmethod
    def from_candle(cls, candle: Candle) -> _LiveMinute:
        return cls(candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume)

    def update(self, price: float, size: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size

    def to_candle(self) -> Candle:
        return Candle(self.timestamp, self.open, self.high, self.low, self.close, volume=self.volume)


@dataclass(slots=True)
class _Episode:
    direction: Direction
    period: PeriodName
    # 趋势起点（发现它的窗口第一根 K 线开盘时间）与被发现的时刻；持续时长从起点算才符合直觉。
    started_at: datetime
    detected_at: datetime
    anchor: float
    extreme: float
    extreme_at: datetime
    # 下一档升级的累计幅度（百分比，取绝对值）。
    next_level: float
    # 衰竭提示每档只发一次，升级后重新允许：趋势创出新一档后再出现长影线才有新的含义。
    exhaustion_armed: bool = True

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "surge" else -1.0


@dataclass(slots=True)
class _SymbolState:
    minutes: deque[Candle]
    volume_24h_quote: float
    price_decimals: int | None = None
    live: _LiveMinute | None = None
    last_tick_time: datetime | None = None
    # 断线期间的成交永久缺失，本地 K 线不完整，必须等 REST 回补后才能继续判定。
    stale: bool = False
    episode: _Episode | None = None


def _floor_minute(moment: datetime) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def _flat(timestamp: datetime, price: float) -> Candle:
    # 与 Gate 的口径一致：无成交的周期是开高低收都等于上一收盘价、成交量为 0 的平线 K 线。
    return Candle(timestamp, price, price, price, price)


@dataclass(slots=True)
class _Pass:
    rule: TrendRule
    metrics: PatternMetrics
    window_start: datetime
    evaluation_index: int


@dataclass(slots=True)
class _MinuteResult:
    alerts: list[TrendAlert] = field(default_factory=list)
    evaluations: list[TrendEvaluation] = field(default_factory=list)


class TrendDetector:
    # 各周期都由 1 分钟 K 线合成，预热只需要这一种 K 线。
    warmup_interval = "1m"

    def __init__(
        self,
        rules: Sequence[TrendRule],
        settings: EpisodeSettings,
        observer: Callable[[TrendEvaluation], None] | None = None,
    ) -> None:
        if not rules:
            raise ValueError("至少需要一个趋势周期")
        if len({rule.period for rule in rules}) != len(rules):
            raise ValueError("趋势周期不能重复")
        # 短周期在前：同一分钟多个周期同时成立时只发最先发现趋势的短周期。
        self.rules = tuple(sorted(rules, key=lambda rule: rule.minutes))
        self.settings = settings
        self._observer = observer
        self._retention = max(rule.history_minutes for rule in self.rules)
        self._cooldown = timedelta(minutes=settings.cooldown_minutes)
        self._stall = timedelta(minutes=settings.stall_minutes)
        self._states: dict[str, _SymbolState] = {}
        # 与 ATR 检测器一样，冷却记录不随合约移出而清除，避免重新入池后重复提醒同一段行情。
        self._last_start: dict[tuple[str, Direction], datetime] = {}

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

    def resync_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        live_candle: Candle | None,
        fetched_at: datetime,
    ) -> bool:
        """用 REST 1 分钟 K 线重建历史；合约已移除或已不需要回补时返回 False。

        趋势跟踪状态（进行中的趋势与冷却）保留：断线只让中间几分钟缺数据，行情本身并未重来。
        """
        state = self._states.get(symbol.upper())
        if state is None or not state.stale:
            return False
        seed = sorted(candles, key=lambda item: item.timestamp)
        live = state.live
        if live_candle is not None:
            if live is None or live.timestamp < live_candle.timestamp:
                live = _LiveMinute.from_candle(live_candle)
            elif live.timestamp == live_candle.timestamp:
                # 本地含请求返回后才到的成交，交易所含断线期间的成交：高低点取并集；
                # 两边成交有重叠无法精确合并，成交量取较大者。
                is_local_newer = state.last_tick_time is not None and state.last_tick_time > fetched_at
                live = _LiveMinute(
                    live.timestamp,
                    live_candle.open,
                    max(live.high, live_candle.high),
                    min(live.low, live_candle.low),
                    live.close if is_local_newer else live_candle.close,
                    max(live.volume, live_candle.volume),
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
            live.update(tick.price, tick.size)
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
        state.live = _LiveMinute(minute, tick.price, tick.price, tick.price, tick.price, tick.size)
        return alerts

    def add_candle(self, symbol: str, candle: Candle) -> list[TrendAlert]:
        """直接喂入一根已收盘的 1 分钟 K 线，供回放、模拟与调研使用；不能与 add_tick 混用。"""
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
            result = self._on_minute(symbol, state, minute)
            alerts.extend(result.alerts)
            if self._observer is not None:
                for evaluation in result.evaluations:
                    self._observer(evaluation)
        return alerts

    def _on_minute(self, symbol: str, state: _SymbolState, minute: Candle) -> _MinuteResult:
        result = _MinuteResult()
        closed_at = minute.timestamp + MINUTE
        self._track_episode(symbol, state, minute, closed_at, result)

        passes: list[_Pass] = []
        for rule in self.rules:
            if int(closed_at.timestamp()) % (rule.minutes * 60):
                continue
            candles = self._period_candles(state, rule)
            if len(candles) < rule.candles:
                insufficient = TrendEvaluation(symbol, rule.period, closed_at, TrendOutcome.INSUFFICIENT_DATA)
                result.evaluations.append(insufficient)
                continue
            window = candles[-rule.candles :]
            self._check_exhaustion(symbol, state, rule, window, closed_at, result)
            metrics = evaluate_pattern(window, candles[: -rule.candles], rule)
            if metrics is None:
                result.evaluations.append(TrendEvaluation(symbol, rule.period, closed_at, TrendOutcome.NO_DIRECTION))
                continue
            outcome = TrendOutcome.ALERT if metrics.passed else TrendOutcome.FAILED
            result.evaluations.append(TrendEvaluation(symbol, rule.period, closed_at, outcome, metrics))
            if metrics.passed:
                passes.append(_Pass(rule, metrics, window[0].timestamp, len(result.evaluations) - 1))

        self._arbitrate(symbol, state, passes, closed_at, result)
        return result

    def _period_candles(self, state: _SymbolState, rule: TrendRule) -> list[Candle]:
        # 1 分钟 K 线连续且以周期边界结尾，取周期整数倍的尾部即按边界对齐。
        available = min(len(state.minutes), rule.history_minutes)
        count = available - available % rule.minutes
        tail = list(islice(state.minutes, len(state.minutes) - count, None))
        return aggregate_candles(tail, rule.minutes)

    def _arbitrate(
        self,
        symbol: str,
        state: _SymbolState,
        passes: list[_Pass],
        closed_at: datetime,
        result: _MinuteResult,
    ) -> None:
        started = False
        for item in passes:
            direction = item.metrics.direction
            episode = state.episode
            outcome = TrendOutcome.ALERT
            # 先判断是否被本分钟更短的周期取代：它刚开始的趋势也会让后面的周期看起来“已在跟踪”。
            if started:
                outcome = TrendOutcome.SUPERSEDED
            elif episode is not None and episode.direction == direction:
                outcome = TrendOutcome.EPISODE_ACTIVE
            else:
                last = self._last_start.get((symbol, direction))
                if last is not None and closed_at - last < self._cooldown:
                    outcome = TrendOutcome.COOLDOWN
            if outcome is not TrendOutcome.ALERT:
                index = item.evaluation_index
                result.evaluations[index] = replace(result.evaluations[index], outcome=outcome)
                continue
            if episode is not None:
                # 反向趋势成立说明原趋势已经结束，先收尾再开始新的一段。
                self._end_episode(symbol, state, closed_at, item.metrics.end_price, "reversed", result)
            self._start_episode(symbol, state, item, closed_at, result)
            started = True

    def _start_episode(
        self,
        symbol: str,
        state: _SymbolState,
        item: _Pass,
        closed_at: datetime,
        result: _MinuteResult,
    ) -> None:
        metrics = item.metrics
        change = abs(metrics.change_percent)
        step = self.settings.escalation_step_percent
        state.episode = _Episode(
            direction=metrics.direction,
            period=item.rule.period,
            started_at=item.window_start,
            detected_at=closed_at,
            anchor=metrics.start_price,
            extreme=metrics.extreme_price,
            extreme_at=closed_at,
            next_level=(math.floor(change / step) + 1) * step,
        )
        self._last_start[(symbol, metrics.direction)] = closed_at
        result.alerts.append(
            TrendAlert(
                symbol=symbol,
                stage="start",
                direction=metrics.direction,
                period=item.rule.period,
                timestamp=closed_at,
                price=metrics.end_price,
                anchor_price=metrics.start_price,
                change_percent=metrics.change_percent,
                started_at=item.window_start,
                extreme_price=metrics.extreme_price,
                volume_24h_quote=state.volume_24h_quote,
                price_decimals=state.price_decimals,
                metrics=metrics,
                candles=item.rule.candles,
            )
        )

    def _track_episode(
        self,
        symbol: str,
        state: _SymbolState,
        minute: Candle,
        closed_at: datetime,
        result: _MinuteResult,
    ) -> None:
        """每根 1 分钟 K 线收盘时推进进行中的趋势：更新极值，判断结束或升级。"""
        episode = state.episode
        if episode is None:
            return
        close = minute.close
        sign = episode.sign
        if sign * (close - episode.extreme) > 0:
            episode.extreme = close
            episode.extreme_at = closed_at
        move = abs(episode.extreme - episode.anchor)
        pullback = sign * (episode.extreme - close)
        if pullback > 0 and pullback >= self.settings.end_rebound_ratio * move:
            self._end_episode(symbol, state, closed_at, close, "rebound", result)
            return
        if closed_at - episode.extreme_at >= self._stall:
            self._end_episode(symbol, state, closed_at, close, "stall", result)
            return
        change = (close - episode.anchor) / episode.anchor * 100.0
        if sign * change < episode.next_level:
            return
        step = self.settings.escalation_step_percent
        # 一根 K 线跨过多档时只报一次，下一档从当前幅度往上数。
        episode.next_level = (math.floor(sign * change / step) + 1) * step
        episode.exhaustion_armed = True
        result.alerts.append(self._episode_alert(symbol, state, episode, "extend", closed_at, close))

    def _check_exhaustion(
        self,
        symbol: str,
        state: _SymbolState,
        rule: TrendRule,
        window: Sequence[Candle],
        closed_at: datetime,
        result: _MinuteResult,
    ) -> None:
        """检查最近几根 K 线是否出现逆向长影线，作为趋势可能放缓的提示；只在配置的周期上检查。"""
        settings = self.settings
        episode = state.episode
        if not settings.exhaustion_enabled or episode is None or rule.period not in settings.exhaustion_periods:
            return
        if not episode.exhaustion_armed:
            return
        period = timedelta(minutes=rule.minutes)
        for candle in window[-settings.exhaustion_candles :]:
            # 只看趋势被发现之后收盘的 K 线：发现时就带着的影线已经被形态条件接受过了。
            if candle.timestamp + period <= episode.detected_at:
                continue
            ratio = adverse_shadow_ratio(candle, episode.direction)
            # 影线必须探到新的极值附近才算“探底回升/冲高回落”，趋势中途的小十字星不算。
            probe = candle.low if episode.direction == "drop" else candle.high
            reached_extreme = episode.sign * (probe - episode.extreme) >= 0
            if ratio >= settings.exhaustion_shadow_ratio and reached_extreme:
                episode.exhaustion_armed = False
                alert = self._episode_alert(symbol, state, episode, "exhaustion", closed_at, window[-1].close)
                # 提醒标注出现长影线的周期，而不是发现趋势的周期。
                result.alerts.append(replace(alert, period=rule.period, shadow_ratio=ratio))
                return

    def _end_episode(
        self,
        symbol: str,
        state: _SymbolState,
        closed_at: datetime,
        price: float,
        reason: EndReason,
        result: _MinuteResult,
    ) -> None:
        episode = state.episode
        assert episode is not None
        state.episode = None
        if self.settings.notify_end:
            alert = self._episode_alert(symbol, state, episode, "end", closed_at, price)
            result.alerts.append(replace(alert, end_reason=reason))

    @staticmethod
    def _episode_alert(
        symbol: str,
        state: _SymbolState,
        episode: _Episode,
        stage: TrendStage,
        closed_at: datetime,
        price: float,
    ) -> TrendAlert:
        return TrendAlert(
            symbol=symbol,
            stage=stage,
            direction=episode.direction,
            period=episode.period,
            timestamp=closed_at,
            price=price,
            anchor_price=episode.anchor,
            change_percent=(price - episode.anchor) / episode.anchor * 100.0,
            started_at=episode.started_at,
            extreme_price=episode.extreme,
            volume_24h_quote=state.volume_24h_quote,
            price_decimals=state.price_decimals,
        )
