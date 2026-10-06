"""趋势提醒的数据模型与文本格式；何时提醒见 trend.detector。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.formatting import beijing_time, describe_window, format_price
from price_alert.trend.pattern import PERIOD_MINUTES, Direction, PatternMetrics, PeriodName, TrendRule

LABELS: dict[Direction, str] = {"surge": "趋势上涨", "drop": "趋势下跌"}


@dataclass(frozen=True, slots=True)
class TrendAlert:
    symbol: str
    direction: Direction
    period: PeriodName
    candles: int
    # 依据的最后一根 K 线的收盘时间。
    timestamp: datetime
    # 窗口第一根 K 线的开盘时间，即这段走势的起点。
    started_at: datetime
    metrics: PatternMetrics
    volume_24h_quote: float
    price_decimals: int | None = None
    # 同一段走势在上一根周期 K 线收盘时就已成立，这条是它的延续，不是新形成的走势。
    continuing: bool = False

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        label = LABELS[self.direction]
        return f"{label}·延续" if self.continuing else label

    def to_dict(self) -> dict[str, object]:
        metrics = self.metrics
        return {
            # 与异动提醒写入同一个 JSONL/Webhook，下游靠 kind 区分两类提醒。
            "kind": "trend",
            "symbol": self.symbol,
            "direction": self.direction,
            "period": self.period,
            "candles": self.candles,
            "continuing": self.continuing,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "start_price": metrics.start_price,
            "price": metrics.end_price,
            "change_percent": metrics.change_percent,
            "counter_candles": metrics.counter_candles,
            "rebound_ratio": metrics.rebound_ratio,
            "body_ratio": metrics.body_ratio,
            "single_candle_ratio": metrics.single_candle_ratio,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


def describe_period(period: PeriodName) -> str:
    return f"{describe_window(PERIOD_MINUTES[period] * 60)}K线"


def describe_trend_rule(rule: TrendRule) -> str:
    # 0 表示不限制单根实体，此时只有真正反向的 K 线算反向，不写「< 0% 也算」这种无意义的说明。
    small_body = f"（单根实体 < {rule.min_candle_body_ratio:.0%} 也算）" if rule.min_candle_body_ratio > 0 else ""
    return (
        f"趋势 {describe_period(rule.period)}：最近 {rule.candles} 根累计涨跌 ≥ {rule.min_change_percent:g}%，"
        f"反向 K 线 ≤ {rule.max_counter_candles} 根{small_body}，"
        f"最大反弹 ≤ {rule.max_rebound_ratio:.0%}，"
        f"总实体占比 ≥ {rule.min_body_ratio:.0%}，最大实体占比 ≤ {rule.max_single_candle_ratio:.0%}"
    )


def format_change(alert: TrendAlert) -> str:
    # 文本与着色共用同一格式，着色时才能在文本中准确找到百分比。
    return f"{abs(alert.metrics.change_percent):.2f}%"


def format_trend_alert(alert: TrendAlert) -> str:
    metrics = alert.metrics
    move = "上涨" if alert.direction == "surge" else "下跌"
    start = format_price(metrics.start_price, alert.price_decimals)
    price = format_price(metrics.end_price, alert.price_decimals)
    details = (
        # 小实体 K 线即使收在顺势一侧也计入反向，按阴阳线称呼会出现「五根全阴却显示 4/5」。
        f"顺势 {alert.candles - metrics.counter_candles}/{alert.candles}"
        f" · 最大反弹 {metrics.rebound_ratio:.0%} · 总实体占比 {metrics.body_ratio:.0%}"
        f" · 最大实体占比 {metrics.single_candle_ratio:.0%}"
    )
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{describe_period(alert.period)} {alert.candles} 根持续{move}，累计{move} {format_change(alert)}"
        f"（自 {beijing_time(alert.started_at, '%H:%M')} 起） | {start} → {price} | {details}"
    )
