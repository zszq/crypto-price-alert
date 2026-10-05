"""按配置组装检测器与 REST 客户端。

实时监控、simulate 与 replay 三个入口共用这里的组装逻辑：新增检测参数时只需改这一处，
各入口也因此不必为了拿到检测器而依赖实时编排模块 service。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from price_alert.config import INTERVAL_SECONDS, AppConfig, WindowConfig
from price_alert.detection import MonitoredDetector
from price_alert.detector import AtrMoveDetector
from price_alert.gate import GateRestClient, RateLimiter
from price_alert.trend.config import TrendConfig
from price_alert.trend.detector import TrendDetector, TrendEvaluation
from price_alert.trend.pattern import TrendRule
from price_alert.universe import VolumeRange
from price_alert.windows import MoveWindow, WindowEvaluation, WindowName


def build_windows(config: AppConfig, cooldown_seconds: int | None = None) -> list[MoveWindow]:
    """按配置生成观察窗口，短窗口在前：同一秒两个窗口都提醒时，更紧迫的短窗口提醒排在前面。

    cooldown_seconds 覆盖全部窗口的冷却，供 simulate 这类不需要冷却的场景使用。
    """
    indicator = config.indicator
    configured: list[tuple[WindowName, WindowConfig]] = [("short", indicator.short_window)]
    if indicator.long_window.enabled:
        configured.append(("long", indicator.long_window))
    return [
        MoveWindow(
            name=name,
            lookback_seconds=window.lookback_seconds,
            trigger_atr_multiple=window.trigger_atr_multiple,
            min_change_percent=window.min_change_percent,
            confirmation_seconds=window.confirmation_seconds,
            min_window_trades=window.min_window_trades,
            cooldown_seconds=window.cooldown_seconds if cooldown_seconds is None else cooldown_seconds,
        )
        for name, window in configured
    ]


def build_detector(
    config: AppConfig,
    cooldown_seconds: int | None = None,
    *,
    windows: Sequence[MoveWindow] | None = None,
    observer: Callable[[WindowEvaluation], None] | None = None,
) -> AtrMoveDetector:
    indicator = config.indicator
    return AtrMoveDetector(
        atr_period=indicator.atr_period,
        candle_interval_seconds=INTERVAL_SECONDS[indicator.candle_interval],
        max_atr_age_seconds=indicator.max_atr_age_seconds,
        windows=build_windows(config, cooldown_seconds) if windows is None else windows,
        observer=observer,
    )


def build_trend_detector(
    config: TrendConfig,
    *,
    rules: Sequence[TrendRule] | None = None,
    observer: Callable[[TrendEvaluation], None] | None = None,
) -> TrendDetector:
    # rules 覆盖配置中启用的周期，供 simulate 逐个周期单独验证。
    return TrendDetector(config.rules() if rules is None else rules, observer, alert_continuing=config.alert_continuing)


def build_detectors(config: AppConfig) -> list[MonitoredDetector]:
    """实时监控运行的全部检测器；趋势提醒关闭时不创建，也就不会多拉 K 线、多占内存。"""
    indicator = config.indicator
    detectors = [
        MonitoredDetector(
            "ATR 异动",
            build_detector(config),
            indicator.candle_interval,
            indicator.warmup_candles,
            VolumeRange(config.gate.min_volume_24h_quote),
        )
    ]
    if config.trend.enabled:
        trend = build_trend_detector(config.trend)
        volume = VolumeRange(config.trend.min_volume_24h_quote, config.trend.max_volume_24h_quote)
        detectors.append(MonitoredDetector("K 线趋势", trend, trend.warmup_interval, trend.warmup_candles, volume))
    return detectors


def build_rest_client(config: AppConfig) -> GateRestClient:
    # 所有入口共用同一份限速与重试口径，命令行查询也不会绕过令牌桶打满交易所限频。
    gate = config.gate
    return GateRestClient(
        gate.rest_url,
        gate.settle,
        gate.rest_timeout_seconds,
        gate.rest_retries,
        RateLimiter(gate.rest_rate_limit_per_second, gate.rest_rate_limit_burst),
    )
