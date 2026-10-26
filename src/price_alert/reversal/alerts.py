"""反弹回落提醒的数据模型与文本格式。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.formatting import beijing_time, describe_window, format_price
from price_alert.reversal.rule import ReversalRule

# 按急动方向：急拉是对此前下跌的反弹，急砸是对此前上涨的回落。
LABELS: dict[Literal["surge", "drop"], str] = {"surge": "下跌反弹", "drop": "冲高回落"}
NAME = "反弹回落"


@dataclass(frozen=True, slots=True)
class ReversalAlert:
    symbol: str
    # 急动方向（surge 急拉 / drop 急砸），也是提醒之后预期延续的方向。
    direction: Literal["surge", "drop"]
    # 触发提醒的那笔成交的时间（与秒级异动相同：下一秒第一笔成交结算上一秒）。
    timestamp: datetime
    # 急动那一秒的 VWAP 与窗口起点的基准价。
    price: float
    reference_price: float
    change_percent: float
    lookback_seconds: int
    move_atr: float
    trade_count: int
    # 窗口开始前 prior_seconds 秒的价格到基准价的涨跌幅（带符号，与急动方向相反）。
    prior_change_percent: float
    prior_seconds: int
    # 最近 1 分钟成交笔数 ÷ 最近 1 小时每分钟平均笔数。
    trade_burst_ratio: float
    volume_24h_quote: float
    price_decimals: int | None = None

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        return LABELS[self.direction]

    def to_dict(self) -> dict[str, object]:
        return {
            # 与其他提醒写入同一个 JSONL/Webhook，下游靠 kind 区分。
            "kind": "reversal",
            "symbol": self.symbol,
            "direction": self.direction,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "price": self.price,
            "reference_price": self.reference_price,
            "change_percent": self.change_percent,
            "lookback_seconds": self.lookback_seconds,
            "move_atr": self.move_atr,
            "trade_count": self.trade_count,
            "prior_change_percent": self.prior_change_percent,
            "prior_seconds": self.prior_seconds,
            "trade_burst_ratio": self.trade_burst_ratio,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


def describe_reversal_rule(rule: ReversalRule) -> str:
    return (
        f"{NAME}：此前 {describe_window(rule.prior_seconds)}已反向走了 ≥ {rule.min_prior_move_percent:g}%，"
        f"{describe_window(rule.lookback_seconds)}内急动 ≥ {rule.min_change_percent:g}% 且 ≥ "
        f"{rule.trigger_atr_multiple:g} ATR，窗口成交 ≥ {rule.min_window_trades} 笔、"
        f"连续 {rule.confirmation_seconds} 秒，"
        f"最近 1 分钟成交笔数 ≥ 1 小时均值 × {rule.min_trade_burst_ratio:g}；冷却 {rule.cooldown_seconds} 秒"
    )


def format_change(alert: ReversalAlert) -> str:
    # 文本与着色共用同一格式，着色时才能在文本中准确找到百分比。
    return f"{abs(alert.change_percent):.2f}%"


def format_reversal_alert(alert: ReversalAlert) -> str:
    prior_move = "下跌" if alert.prior_change_percent < 0 else "上涨"
    move = "上涨" if alert.direction == "surge" else "下跌"
    reference = format_price(alert.reference_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    # 急动涨跌幅写在前置涨跌幅之前：着色按首次出现替换，前置的「11.85%」不能被当成急动的「1.85%」。
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{describe_window(alert.lookback_seconds)}内{move} {format_change(alert)}"
        f"（此前{describe_window(alert.prior_seconds)}{prior_move} {abs(alert.prior_change_percent):.2f}%） | "
        f"成交笔数为1小时均值的 {alert.trade_burst_ratio:.0f} 倍 | {reference} → {price}"
    )
