"""短窗口连续同向提醒：一次急涨急跌之后，冷却一过又立刻再次达标，说明行情还在单边延续。

短窗口只比较 30 秒首尾，冷却又让同一合约 30 秒内只报一次；一段持续拉升因此表现为一串间隔约
「冷却 + 几秒确认」的同向短窗口提醒，单条提醒看不出它与一次性插针的区别，连成串才看得出来。
这里只读已经发出的短窗口提醒，不参与秒级判定，也不影响各窗口的冷却与确认。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from price_alert.formatting import beijing_time, describe_window, format_price
from price_alert.models import PriceAlert

Direction = Literal["surge", "drop"]

LABELS: dict[Direction, str] = {"surge": "连续急涨", "drop": "连续急跌"}


@dataclass(frozen=True, slots=True)
class StreakRule:
    # 第几条同向短窗口提醒时发出连续提醒；之后同一串里的提醒不再重复发。
    min_alerts: int
    # 相邻两条同向短窗口提醒的最大间隔（配置里的短窗口冷却 + 容差）：超过就说明中间行情停过，重新计数。
    max_gap_seconds: int

    def __post_init__(self) -> None:
        if self.min_alerts < 2:
            raise ValueError("min_alerts 至少为 2")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds 必须大于 0")


@dataclass(frozen=True, slots=True)
class StreakAlert:
    symbol: str
    direction: Direction
    # 这一串里已发出的同向短窗口提醒条数（含触发本提醒的那条）。
    alert_count: int
    # 第一条提醒的窗口起点与基准价：累计涨跌幅从这里算，才覆盖整段行情而不只是最后 30 秒。
    started_at: datetime
    start_price: float
    price: float
    change_percent: float
    lookback_seconds: int
    volume_24h_quote: float
    timestamp: datetime
    price_decimals: int | None = None

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        return LABELS[self.direction]

    @property
    def duration_seconds(self) -> int:
        return round((self.timestamp - self.started_at).total_seconds())

    def to_dict(self) -> dict[str, object]:
        return {
            # 与其他提醒写入同一个 JSONL/Webhook，下游靠 kind 区分。
            "kind": "streak",
            "symbol": self.symbol,
            "direction": self.direction,
            "alert_count": self.alert_count,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "start_price": self.start_price,
            "price": self.price,
            "change_percent": self.change_percent,
            "lookback_seconds": self.lookback_seconds,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


@dataclass(slots=True)
class _Streak:
    first: PriceAlert
    last: PriceAlert
    count: int = 1
    reported: bool = False


class StreakTracker:
    """按合约串联同向短窗口提醒；反向提醒或间隔过长都会从头计数。"""

    def __init__(self, rule: StreakRule) -> None:
        self.rule = rule
        self._max_gap = timedelta(seconds=rule.max_gap_seconds)
        # 与冷却记录一样不随合约移出清除：间隔判断本身就会让过期的串失效。
        self._streaks: dict[str, _Streak] = {}

    def observe(self, alert: PriceAlert) -> StreakAlert | None:
        streak = self._streaks.get(alert.symbol)
        if (
            streak is None
            or streak.last.direction != alert.direction
            or alert.timestamp - streak.last.timestamp > self._max_gap
        ):
            self._streaks[alert.symbol] = _Streak(alert, alert)
            return None
        streak.last = alert
        streak.count += 1
        # 一串只报一次：之后的短窗口提醒照常发出，已经足以说明行情仍在延续，再报只会刷屏。
        if streak.reported or streak.count < self.rule.min_alerts:
            return None
        first = streak.first
        change_percent = (alert.price / first.reference_price - 1.0) * 100.0
        if (change_percent > 0) != (alert.direction == "surge") or change_percent == 0:
            # 每条短窗口提醒只看自己的 30 秒，提醒之间的冷却期里价格可能急跌又拉回而没触发反向提醒；
            # 这样几次下来整段可能并没有朝提醒方向走，不能报「连续急涨」。先不报，计数照常累加，
            # 等后续提醒让整段方向一致时再报。
            return None
        streak.reported = True
        return StreakAlert(
            symbol=alert.symbol,
            direction=alert.direction,
            alert_count=streak.count,
            started_at=first.timestamp - timedelta(seconds=first.lookback_seconds),
            start_price=first.reference_price,
            price=alert.price,
            change_percent=change_percent,
            lookback_seconds=alert.lookback_seconds,
            volume_24h_quote=alert.volume_24h_quote,
            timestamp=alert.timestamp,
            price_decimals=alert.price_decimals,
        )


def format_change(alert: StreakAlert) -> str:
    # 着色靠在文本中查找这段字样，format_streak_alert 与 notifier 必须出自同一个函数。
    return f"{abs(alert.change_percent):.2f}%"


def describe_duration(seconds: int) -> str:
    # 一串行情通常持续一两分钟，「近147秒」不如「近2分27秒」好读；整分钟沿用窗口长度的说法。
    minutes, rest = divmod(seconds, 60)
    if minutes == 0 or rest == 0:
        return describe_window(seconds)
    return f"{minutes}分{rest}秒"


def format_streak_alert(alert: StreakAlert) -> str:
    move = "上涨" if alert.direction == "surge" else "下跌"
    kind = "急涨" if alert.direction == "surge" else "急跌"
    start = format_price(alert.start_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{describe_window(alert.lookback_seconds)}窗口连续 {alert.alert_count} 次{kind}，"
        f"近{describe_duration(alert.duration_seconds)}累计{move} {format_change(alert)} | {start} → {price}"
    )


def describe_streak_rule(rule: StreakRule) -> str:
    return (
        f"连续提醒：短窗口同方向提醒连续 {rule.min_alerts} 次、相邻间隔 ≤ {rule.max_gap_seconds} 秒时追加一条，"
        "每串只报一次"
    )
