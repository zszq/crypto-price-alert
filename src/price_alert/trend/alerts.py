"""趋势提醒的数据模型与文本格式。

一段趋势按状态机输出：开始 → 每累计一档升级 →（可选）衰竭提示 → 结束，
整段行情只报几条，而不是在趋势途中每隔几分钟重复刷屏。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.formatting import beijing_time, describe_window, format_price
from price_alert.trend.pattern import (
    PERIOD_MINUTES,
    Direction,
    PatternMetrics,
    PeriodName,
    TrendRule,
    format_multiple,
)

TrendStage = Literal["start", "extend", "exhaustion", "end"]
EndReason = Literal["rebound", "stall", "reversed"]

STAGE_LABELS: dict[TrendStage, dict[Direction, str]] = {
    "start": {"surge": "趋势上涨", "drop": "趋势下跌"},
    "extend": {"surge": "上涨延续", "drop": "下跌延续"},
    "exhaustion": {"surge": "上涨衰竭提示", "drop": "下跌衰竭提示"},
    "end": {"surge": "上涨趋势结束", "drop": "下跌趋势结束"},
}
END_REASON_LABELS: dict[EndReason, str] = {
    "rebound": "反向回撤过大",
    "stall": "长时间未创新极值",
    "reversed": "出现反向趋势",
}


@dataclass(frozen=True, slots=True)
class TrendAlert:
    symbol: str
    stage: TrendStage
    direction: Direction
    # 发现这段趋势的周期；升级与结束沿用它，便于把同一段行情的提醒串起来；衰竭提示为出现长影线的周期。
    period: PeriodName
    # 依据的 K 线收盘时间。
    timestamp: datetime
    price: float
    # 趋势起点（发现趋势的窗口第一根 K 线开盘价），累计涨跌幅都相对它计算。
    anchor_price: float
    change_percent: float
    started_at: datetime
    extreme_price: float
    volume_24h_quote: float
    price_decimals: int | None = None
    # 以下字段只在对应阶段有值。
    metrics: PatternMetrics | None = None
    candles: int | None = None
    shadow_ratio: float | None = None
    end_reason: EndReason | None = None

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        return STAGE_LABELS[self.stage][self.direction]

    @property
    def duration_minutes(self) -> int:
        return int((self.timestamp - self.started_at).total_seconds() // 60)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            # 与异动提醒写入同一个 JSONL/Webhook，下游靠 kind 区分两类提醒。
            "kind": "trend",
            "symbol": self.symbol,
            "stage": self.stage,
            "direction": self.direction,
            "period": self.period,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "price": self.price,
            "anchor_price": self.anchor_price,
            "change_percent": self.change_percent,
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "extreme_price": self.extreme_price,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "candles": self.candles,
            "shadow_ratio": self.shadow_ratio,
            "end_reason": self.end_reason,
            "color": self.color,
        }
        if self.metrics is not None:
            metrics = self.metrics
            payload["metrics"] = {
                "rebound_ratio": metrics.rebound_ratio,
                "trend_candle_ratio": metrics.trend_candle_ratio,
                "step_ratio": metrics.step_ratio,
                "body_ratio": metrics.body_ratio,
                "volume_ratio": metrics.volume_ratio,
                "change_atr": metrics.change_atr,
            }
        return payload


def describe_period(period: PeriodName) -> str:
    return f"{describe_window(PERIOD_MINUTES[period] * 60)}K线"


def describe_trend_rule(rule: TrendRule) -> str:
    parts = [
        f"最近 {rule.candles} 根累计涨跌 ≥ {rule.min_change_percent:g}%",
        f"顺势 K 线 ≥ {rule.min_trend_candle_ratio:.0%}",
        f"收盘递进 ≥ {rule.min_step_ratio:.0%}",
        f"最大反弹 ≤ {rule.max_rebound_ratio:.0%}",
        f"实体占比 ≥ {rule.min_body_ratio:.0%}",
    ]
    if rule.min_volume_ratio is not None:
        parts.append(f"量能 ≥ 前 {rule.baseline_candles} 根中位数的 {rule.min_volume_ratio:g} 倍")
    if rule.min_change_atr is not None:
        parts.append(f"位移 ≥ 前 {rule.baseline_candles} 根平均真实波幅的 {rule.min_change_atr:g} 倍")
    return f"趋势 {describe_period(rule.period)}：{'，'.join(parts)}"


def format_change(alert: TrendAlert) -> str:
    # 文本与着色共用同一格式，着色时才能在文本中准确找到百分比。
    return f"{abs(alert.change_percent):.2f}%"


def format_trend_alert(alert: TrendAlert) -> str:
    move = "上涨" if alert.direction == "surge" else "下跌"
    candle_word = "阳线" if alert.direction == "surge" else "阴线"
    anchor = format_price(alert.anchor_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    head = f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
    since = f"自 {beijing_time(alert.started_at, '%H:%M')} 起 {alert.duration_minutes} 分钟"
    if alert.stage == "start":
        assert alert.metrics is not None and alert.candles is not None
        metrics = alert.metrics
        details = [
            f"{candle_word} {metrics.trend_candle_ratio:.0%}",
            f"收盘递进 {metrics.step_ratio:.0%}",
            f"最大反弹 {metrics.rebound_ratio:.0%}",
            f"实体占比 {metrics.body_ratio:.0%}",
        ]
        if metrics.volume_ratio is not None:
            details.append(f"量能 {format_multiple(metrics.volume_ratio)} 倍")
        if metrics.change_atr is not None:
            details.append(f"{format_multiple(metrics.change_atr)} ATR")
        return (
            f"{head}{describe_period(alert.period)} {alert.candles} 根持续{move}，累计{move} {format_change(alert)}"
            f" | {anchor} → {price} | {' · '.join(details)}"
        )
    if alert.stage == "extend":
        return f"{head}累计{move} {format_change(alert)}（{since}） | {anchor} → {price}"
    if alert.stage == "exhaustion":
        assert alert.shadow_ratio is not None
        shadow = "下影线" if alert.direction == "drop" else "上影线"
        return (
            f"{head}{describe_period(alert.period)}出现长{shadow}（占振幅 {alert.shadow_ratio:.0%}），{move}可能放缓"
            f" | 累计{move} {format_change(alert)}（{since}） | {anchor} → {price}"
        )
    assert alert.end_reason is not None
    extreme = format_price(alert.extreme_price, alert.price_decimals)
    extreme_change = abs(alert.extreme_price - alert.anchor_price) / alert.anchor_price * 100.0
    extreme_word = "最高" if alert.direction == "surge" else "最低"
    return (
        f"{head}{END_REASON_LABELS[alert.end_reason]}，{since}"
        f" | {anchor} → {extreme_word} {extreme}（最大{move} {extreme_change:.2f}%），当前 {price}"
    )
