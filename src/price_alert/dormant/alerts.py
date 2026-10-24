"""冷门启动提醒的数据模型与文本格式。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.dormant.rule import CALM_MINUTES, DormantMetrics, DormantRule, late_minutes
from price_alert.formatting import beijing_time, format_amount, format_price

LABEL = "冷门启动"


@dataclass(frozen=True, slots=True)
class DormantAlert:
    symbol: str
    # 依据的最后一根 1 分钟 K 线的收盘时间。
    timestamp: datetime
    # 观察窗口第一根 K 线的开盘时间。
    started_at: datetime
    window_minutes: int
    metrics: DormantMetrics
    volume_24h_quote: float
    price_decimals: int | None = None
    # 冷却期内价格继续上涨的补报从 2 起算。
    wave: int = 1

    # 只检测拉升；方向与颜色给按方向着色的通知代码用，与其他提醒一致。
    @property
    def direction(self) -> Literal["surge"]:
        return "surge"

    @property
    def color(self) -> Literal["green"]:
        return "green"

    @property
    def label(self) -> str:
        return f"{LABEL}·第{self.wave}波" if self.wave > 1 else LABEL

    def to_dict(self) -> dict[str, object]:
        metrics = self.metrics
        return {
            # 与其他提醒写入同一个 JSONL/Webhook，下游靠 kind 区分。
            "kind": "dormant",
            "symbol": self.symbol,
            "direction": self.direction,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "window_minutes": self.window_minutes,
            "wave": self.wave,
            "start_price": metrics.start_price,
            "price": metrics.end_price,
            "change_percent": metrics.change_percent,
            "window_quote_volume": metrics.window_quote,
            "active_minutes": metrics.active_minutes,
            "prior_quote_volume": metrics.prior_quote,
            # 倍数在涨前一字横盘时为无穷大，JSON 表示不了，下游可由涨幅与涨前波动自行计算。
            "pre_volatility_percent": metrics.pre_volatility_percent,
            "max_minute_change_percent": metrics.max_minute_change_percent,
            "body_ratio": metrics.body_ratio,
            "last_minute_quote_volume": metrics.last_minute_quote,
            "late_quote_share": metrics.late_quote_share,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


def describe_dormant_rule(rule: DormantRule) -> str:
    return (
        f"{LABEL}：之前 24 小时成交额 ≤ {format_amount(rule.max_prior_quote)} USDT 的币，"
        f"最近 {rule.window_minutes} 分钟涨幅 ≥ {rule.min_change_percent:g}%、"
        f"成交额 ≥ {format_amount(rule.min_window_quote)} USDT、至少 {rule.min_active_minutes} 分钟有成交，"
        f"{describe_calm(rule)}、"
        f"单分钟涨幅 ≤ {rule.max_minute_change_percent:g}%、总实体占比 ≥ {rule.min_body_ratio:.0%}"
        f"{describe_volume_shape(rule)}；"
        f"{describe_repeat(rule)}"
    )


def describe_volume_shape(rule: DormantRule) -> str:
    # 两项代码默认不限，未开启时不写，规则说明与开启前一致。
    parts = []
    if rule.min_last_minute_quote > 0:
        parts.append(f"最后一分钟成交额 ≥ {format_amount(rule.min_last_minute_quote)} USDT")
    if rule.min_late_quote_share > 0:
        parts.append(f"后 {late_minutes(rule.window_minutes)} 分钟成交额占比 ≥ {rule.min_late_quote_share:.0%}")
    return "".join(f"、{part}" for part in parts)


def describe_calm(rule: DormantRule) -> str:
    multiple = f"涨幅 ≥ 涨前 {CALM_MINUTES} 分钟平均每分钟波动的 {rule.min_calm_multiple:g} 倍"
    if rule.max_pre_volatility_percent is None:
        return multiple
    return f"涨前 {CALM_MINUTES} 分钟平均每分钟波动 ≤ {rule.max_pre_volatility_percent:g}%、{multiple}"


def describe_repeat(rule: DormantRule) -> str:
    if rule.realert_step_percent is None:
        return f"同一合约 {rule.cooldown_minutes} 分钟内只提醒一次"
    return (
        f"同一合约距上次提醒不足 {rule.cooldown_minutes} 分钟时，"
        f"价格比上次提醒再涨 ≥ {rule.realert_step_percent:g}% 才再报「第 N 波」"
    )


def format_change(alert: DormantAlert) -> str:
    # 文本与着色共用同一格式，着色时才能在文本中准确找到百分比。
    return f"{alert.metrics.change_percent:.2f}%"


def format_calm(metrics: DormantMetrics) -> str:
    """涨前波动与倍数；一字横盘时倍数无穷大，直接说横盘。"""
    if metrics.pre_volatility_percent <= 0:
        return f"涨前{CALM_MINUTES}分钟一字横盘"
    return (
        f"涨前{CALM_MINUTES}分钟平均每分钟波动 {metrics.pre_volatility_percent:.2f}%"
        f"（涨幅为其 {metrics.calm_multiple:.0f} 倍）"
    )


def format_dormant_alert(alert: DormantAlert) -> str:
    metrics = alert.metrics
    start = format_price(metrics.start_price, alert.price_decimals)
    price = format_price(metrics.end_price, alert.price_decimals)
    return (
        f"[{alert.label}] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{alert.window_minutes}分钟上涨 {format_change(alert)}，成交额 {format_amount(metrics.window_quote)} USDT"
        f"（{metrics.active_minutes} 分钟有成交），之前24小时成交额 {format_amount(metrics.prior_quote)} USDT，"
        f"{format_calm(metrics)} | {start} → {price}"
    )
