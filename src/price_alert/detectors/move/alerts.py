"""秒级 ATR 异动提醒的数据模型与文本格式。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.common.formatting import beijing_time, describe_window, format_price
from price_alert.detectors.move.windows import WindowName


@dataclass(frozen=True, slots=True)
class PriceAlert:
    symbol: str
    direction: Literal["surge", "drop"]
    price: float
    reference_price: float
    change_percent: float
    move_atr: float
    atr: float
    atr_period: int
    lookback_seconds: int
    trade_count: int
    volume_24h_quote: float
    timestamp: datetime
    # 触发提醒的观察窗口；lookback_seconds 已给出长度，名字便于下游按窗口类型过滤。
    window: WindowName = "short"
    price_decimals: int | None = None

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.astimezone(UTC).isoformat()
        # 结构化通知也携带展示语义，Webhook 接收方无需重复判断方向。
        payload["color"] = self.color
        return payload


def format_change(alert: PriceAlert) -> str:
    # format_move_alert 与 notifier 着色共用同一格式，保证着色时能在文本中准确找到百分比。
    return f"{abs(alert.change_percent):.2f}%"


def format_window(alert: PriceAlert) -> str:
    # 与 format_change 同理：着色靠在文本中查找这段字样，两处必须出自同一个函数。
    return describe_window(alert.lookback_seconds)


def format_move_alert(alert: PriceAlert) -> str:
    label = "急涨" if alert.direction == "surge" else "急跌"
    move_label = "上涨" if alert.direction == "surge" else "下跌"
    reference = format_price(alert.reference_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    return (
        f"[{label}提醒] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{format_window(alert)}内价格{move_label} "
        f"{format_change(alert)} | {reference} → {price}"
        f" | 异动强度 {alert.move_atr:.2f} ATR"
    )
