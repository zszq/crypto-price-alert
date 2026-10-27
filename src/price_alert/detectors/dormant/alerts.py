"""冷门启动提醒的数据模型与文本格式。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from price_alert.common.formatting import beijing_time, format_amount, format_price
from price_alert.detectors.dormant.rule import CALM_MINUTES, DormantGroup, DormantMetrics, DormantRule, late_minutes

LABELS: dict[DormantGroup, str] = {DormantGroup.MAIN: "冷门启动", DormantGroup.SLOW: "冷门慢启动"}


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
    # 慢启动一组准确率明显低于现行一组，标签分开，看到提醒就知道该信几分。
    group: DormantGroup = DormantGroup.MAIN

    # 只检测拉升；方向与颜色给按方向着色的通知代码用，与其他提醒一致。
    @property
    def direction(self) -> Literal["surge"]:
        return "surge"

    @property
    def color(self) -> Literal["green"]:
        return "green"

    @property
    def label(self) -> str:
        label = LABELS[self.group]
        return f"{label}·第{self.wave}波" if self.wave > 1 else label

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
            "efficiency": metrics.efficiency,
            # 之前 24 小时没有成交时量比为无穷大，JSON 表示不了，与放量启动一样写 null。
            "volume_ratio": metrics.volume_ratio if math.isfinite(metrics.volume_ratio) else None,
            "retrace": metrics.retrace,
            "group": self.group.value,
            "volume_24h_quote": self.volume_24h_quote,
            "price_decimals": self.price_decimals,
            "color": self.color,
        }


def describe_dormant_rule(rule: DormantRule) -> str:
    return (
        f"{LABELS[rule.group]}：之前 24 小时成交额 ≤ {format_amount(rule.max_prior_quote)} USDT 的币，"
        f"最近 {rule.window_minutes} 分钟涨幅 ≥ {rule.min_change_percent:g}%、"
        f"成交额 ≥ {format_amount(rule.min_window_quote)} USDT、至少 {rule.min_active_minutes} 分钟有成交，"
        f"{'、'.join(describe_shape(rule))}；"
        f"{describe_repeat(rule)}"
    )


def describe_shape(rule: DormantRule) -> list[str]:
    # 设为不限的项不写：两组参数开启的条件不同，各自只列真正在判定的门槛。
    parts = describe_calm(rule)
    parts.append(f"单分钟涨幅 ≤ {rule.max_minute_change_percent:g}%")
    if rule.min_body_ratio > 0:
        parts.append(f"总实体占比 ≥ {rule.min_body_ratio:.0%}")
    if rule.min_last_minute_quote > 0:
        parts.append(f"最后一分钟成交额 ≥ {format_amount(rule.min_last_minute_quote)} USDT")
    if rule.min_late_quote_share > 0:
        parts.append(f"后 {late_minutes(rule.window_minutes)} 分钟成交额占比 ≥ {rule.min_late_quote_share:.0%}")
    if rule.min_efficiency > 0:
        parts.append(f"效率（净涨幅 ÷ 逐分钟涨跌路程）≥ {rule.min_efficiency:g}")
    if rule.min_volume_ratio > 0:
        parts.append(f"量比（相对之前 24 小时同长度均值）≥ {rule.min_volume_ratio:g} 倍")
    if rule.max_retrace is not None:
        parts.append(f"从窗口最高价回落 ≤ 窗口最大涨幅的 {rule.max_retrace:.0%}")
    return parts


def describe_calm(rule: DormantRule) -> list[str]:
    parts = []
    if rule.max_pre_volatility_percent is not None:
        parts.append(f"涨前 {CALM_MINUTES} 分钟平均每分钟波动 ≤ {rule.max_pre_volatility_percent:g}%")
    if rule.min_calm_multiple > 0:
        parts.append(f"涨幅 ≥ 涨前 {CALM_MINUTES} 分钟平均每分钟波动的 {rule.min_calm_multiple:g} 倍")
    return parts


def describe_repeat(rule: DormantRule) -> str:
    if rule.realert_step_percent is None:
        text = f"同一合约 {rule.cooldown_minutes} 分钟内只提醒一次"
    else:
        text = (
            f"同一合约距上次提醒不足 {rule.cooldown_minutes} 分钟时，"
            f"价格比上次提醒再涨 ≥ {rule.realert_step_percent:g}% 才再报「第 N 波」"
        )
    if rule.follow_step_percent is None:
        return text
    return (
        f"{text}；提醒后跟踪价格，比上次提醒（含补报）再涨 ≥ {rule.follow_step_percent:g}% 就补报「第 N 波」"
        f"（不再检查其他条件），距上次提醒超过 {rule.follow_minutes} 分钟没再涨够就停止跟踪"
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
