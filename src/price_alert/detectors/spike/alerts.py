"""瞬时异动、短时异动提醒的数据模型与文本格式（两者共用同一套判定，只是参数不同）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.common.formatting import beijing_time, describe_window, format_price
from price_alert.detectors.spike.windows import WindowName

# 同一个 AtrSpikeDetector 按两组参数跑成两种提醒：瞬时异动（instant，即研究第四轮之前的短时异动，
# 急拉急砸都报、量大）与短时异动（spike，研究第四轮方案 A，只报逆势、成交涌入的急动）。kind 写进 JSONL，下游据此区分。
SpikeKind = Literal["instant", "spike"]
NAMES: dict[SpikeKind, str] = {"instant": "瞬时异动", "spike": "短时异动"}
# 提醒标签的前缀，接「急涨」「急跌」。
PREFIXES: dict[SpikeKind, str] = {"instant": "瞬时", "spike": "短时"}


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
    kind: SpikeKind = "spike"
    price_decimals: int | None = None
    # 窗口开了前置走势、成交暴增两个附加条件时，提醒带上它们的取值（前置走势为带符号的实际涨跌幅），否则为 None。
    prior_change_percent: float | None = None
    prior_seconds: int | None = None
    trade_burst_ratio: float | None = None

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        return f"{PREFIXES[self.kind]}{'急涨' if self.direction == 'surge' else '急跌'}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.astimezone(UTC).isoformat()
        # 结构化通知也携带展示语义，Webhook 接收方无需重复判断方向。
        payload["color"] = self.color
        return payload


def format_change(alert: PriceAlert) -> str:
    # format_spike_alert 与 notifier 着色共用同一格式，保证着色时能在文本中准确找到百分比。
    return f"{abs(alert.change_percent):.2f}%"


def format_window(alert: PriceAlert) -> str:
    # 与 format_change 同理：着色靠在文本中查找这段字样，两处必须出自同一个函数。
    return describe_window(alert.lookback_seconds)


def format_spike_alert(alert: PriceAlert) -> str:
    move_label = "上涨" if alert.direction == "surge" else "下跌"
    reference = format_price(alert.reference_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    # 附加条件放在最后：着色按首次出现查找涨跌幅与窗口长度，不能让它们先出现。
    extra = ""
    if alert.prior_change_percent is not None and alert.prior_seconds is not None:
        extra += f" | 此前{describe_window(alert.prior_seconds)} {alert.prior_change_percent:+.2f}%"
    if alert.trade_burst_ratio is not None:
        extra += f" | 成交笔数 {alert.trade_burst_ratio:.1f} 倍"
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{format_window(alert)}内价格{move_label} "
        f"{format_change(alert)} | {reference} → {price}"
        f" | 异动强度 {alert.move_atr:.2f} ATR{extra}"
    )
