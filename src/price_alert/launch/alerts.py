"""放量启动提醒的数据模型与文本格式。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.formatting import beijing_time, describe_minutes, format_amount, format_price
from price_alert.launch.rule import Direction, LaunchMetrics, LaunchRule

LABELS: dict[Direction, str] = {"surge": "放量拉升", "drop": "放量下跌"}


@dataclass(frozen=True, slots=True)
class LaunchAlert:
    symbol: str
    # 依据的最后一根 1 分钟 K 线的收盘时间。
    timestamp: datetime
    # 观察窗口第一根 K 线的开盘时间。
    started_at: datetime
    window_minutes: int
    baseline_minutes: int
    metrics: LaunchMetrics
    volume_24h_quote: float
    price_decimals: int | None = None
    # 同方向第几波：1 是这段行情的启动，之后每次价格再推进一截、重新放量就加一。
    wave: int = 1

    @property
    def direction(self) -> Direction:
        return self.metrics.direction

    @property
    def color(self) -> Literal["green", "red"]:
        return "green" if self.direction == "surge" else "red"

    @property
    def label(self) -> str:
        label = LABELS[self.direction]
        return f"{label}·第{self.wave}波" if self.wave > 1 else label

    def to_dict(self) -> dict[str, object]:
        metrics = self.metrics
        return {
            # 与其他提醒写入同一个 JSONL/Webhook，下游靠 kind 区分。
            "kind": "launch",
            "symbol": self.symbol,
            "direction": self.direction,
            "wave": self.wave,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "window_minutes": self.window_minutes,
            "baseline_minutes": self.baseline_minutes,
            "start_price": metrics.start_price,
            "price": metrics.end_price,
            "change_percent": metrics.change_percent,
            "window_quote_volume": metrics.window_quote,
            "baseline_quote_volume": metrics.baseline_quote,
            "volume_ratio": metrics.volume_ratio,
            "breakout_price": metrics.breakout_price,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


def describe_launch_rule(rule: LaunchRule) -> str:
    baseline = describe_minutes(rule.baseline_minutes)
    breakout = f"，收盘突破近{baseline}高低点" if rule.require_breakout else ""
    directions = "涨跌都提醒" if rule.alert_drops else "只提醒上涨"
    return (
        f"放量启动：最近 {rule.window_minutes} 分钟成交额 ≥ {format_amount(rule.min_window_quote)} USDT "
        f"且 ≥ 近{baseline}同长度均量的 {rule.min_volume_ratio:g} 倍，涨跌幅 ≥ {rule.min_change_percent:g}%{breakout}；"
        f"同方向再次提醒需间隔 ≥ {rule.cooldown_minutes} 分钟且价格比上次提醒再推进 ≥ {rule.realert_step_percent:g}%"
        f"（超过{baseline}重新计为启动）；{directions}"
    )


def format_change(alert: LaunchAlert) -> str:
    # 文本与着色共用同一格式，着色时才能在文本中准确找到百分比。
    return f"{abs(alert.metrics.change_percent):.2f}%"


def format_ratio(metrics: LaunchMetrics, baseline_minutes: int) -> str:
    if metrics.volume_ratio is None:
        return f"近{describe_minutes(baseline_minutes)}无成交"
    return f"为近{describe_minutes(baseline_minutes)}均量的 {metrics.volume_ratio:.1f} 倍"


def format_launch_alert(alert: LaunchAlert) -> str:
    metrics = alert.metrics
    move = "上涨" if alert.direction == "surge" else "下跌"
    start = format_price(metrics.start_price, alert.price_decimals)
    price = format_price(metrics.end_price, alert.price_decimals)
    level = "高点" if alert.direction == "surge" else "低点"
    breakout = (
        f"，突破近{describe_minutes(alert.baseline_minutes)}{level} "
        f"{format_price(metrics.breakout_price, alert.price_decimals)}"
        if metrics.broke_out
        else ""
    )
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{alert.window_minutes}分钟{move} {format_change(alert)}，成交额 {format_amount(metrics.window_quote)} USDT，"
        f"{format_ratio(metrics, alert.baseline_minutes)}{breakout} | {start} → {price}"
    )
