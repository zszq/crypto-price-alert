"""价格异动监控的领域模型。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.windows import WindowName


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp 必须包含时区")
        prices = (self.open, self.high, self.low, self.close)
        # NaN 与任何数比较都为假，必须显式检查有限性，否则会绕过正数校验污染 ATR。
        if not all(math.isfinite(price) for price in prices) or min(prices) <= 0:
            raise ValueError("K 线价格必须是大于 0 的有限数")
        if self.high < self.low:
            raise ValueError("K 线最高价不能低于最低价")
        if not math.isfinite(self.quote_volume):
            raise ValueError("K 线成交额必须是有限数")


@dataclass(frozen=True, slots=True)
class PriceTick:
    symbol: str
    price: float
    size: float
    timestamp: datetime
    trade_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price 必须是大于 0 的有限数")
        if not math.isfinite(self.size) or self.size < 0:
            raise ValueError("size 必须是不小于 0 的有限数")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp 必须包含时区")


@dataclass(frozen=True, slots=True)
class ContractTicker:
    symbol: str
    last_price: float
    volume_24h_quote: float
    # 交易所报价精度（小数位数），只用于展示；缺失时由展示层按有效数字自行决定。
    price_decimals: int | None = None
    # 一张合约对应的币数：实时成交的 size 是张数，换算计价成交额要乘它和价格。无法识别时为 None。
    quanto_multiplier: float | None = None


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
