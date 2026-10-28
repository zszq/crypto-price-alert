"""按配置组装检测器与 REST 客户端。

实时监控、simulate 与 replay 三个入口共用这里的组装逻辑：新增检测参数时只需改这一处，
各入口也因此不必为了拿到检测器而依赖实时编排模块 service。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from price_alert.common.models import INTERVAL_SECONDS
from price_alert.config import AppConfig
from price_alert.detection import MonitoredDetector
from price_alert.detectors.dormant.detector import DormantDetector, DormantEvaluation
from price_alert.detectors.dormant.rule import DormantRule
from price_alert.detectors.launch.alerts import LABELS as LAUNCH_LABELS
from price_alert.detectors.launch.detector import LaunchDetector, LaunchEvaluation
from price_alert.detectors.launch.rule import LaunchRule
from price_alert.detectors.reversal.alerts import NAME as REVERSAL_NAME
from price_alert.detectors.reversal.detector import ReversalDetector
from price_alert.detectors.reversal.rule import CANDLE_INTERVAL as REVERSAL_INTERVAL
from price_alert.detectors.reversal.rule import ReversalRule
from price_alert.detectors.spike.alerts import NAMES as SPIKE_NAMES
from price_alert.detectors.spike.config import AtrSpikeConfig, InstantConfig, WindowConfig
from price_alert.detectors.spike.detector import AtrSpikeDetector
from price_alert.detectors.spike.streak import StreakRule
from price_alert.detectors.spike.windows import SpikeWindow, WindowEvaluation, WindowName
from price_alert.detectors.trend.config import TrendConfig
from price_alert.detectors.trend.detector import TrendDetector, TrendEvaluation
from price_alert.detectors.trend.pattern import TrendRule
from price_alert.exchange.gate import GateRestClient, RateLimiter
from price_alert.exchange.universe import VolumeRange


def build_windows(spike: AtrSpikeConfig, cooldown_seconds: int | None = None) -> list[SpikeWindow]:
    """按配置生成观察窗口，短窗口在前：同一秒两个窗口都提醒时，更紧迫的短窗口提醒排在前面。

    cooldown_seconds 覆盖全部窗口的冷却，供 simulate 这类不需要冷却的场景使用。
    """
    configured: list[tuple[WindowName, WindowConfig]] = [("short", spike.short_window)]
    if spike.long_window.enabled:
        configured.append(("long", spike.long_window))
    return [
        SpikeWindow(
            name=name,
            lookback_seconds=window.lookback_seconds,
            trigger_atr_multiple=window.trigger_atr_multiple,
            min_change_percent=window.min_change_percent,
            confirmation_seconds=window.confirmation_seconds,
            min_window_trades=window.min_window_trades,
            cooldown_seconds=window.cooldown_seconds if cooldown_seconds is None else cooldown_seconds,
            prior_seconds=window.prior_seconds,
            min_prior_move_percent=window.min_prior_move_percent,
            min_trade_burst_ratio=window.min_trade_burst_ratio,
        )
        for name, window in configured
    ]


def build_detector(
    spike: AtrSpikeConfig,
    cooldown_seconds: int | None = None,
    *,
    windows: Sequence[SpikeWindow] | None = None,
    observer: Callable[[WindowEvaluation], None] | None = None,
    streak: bool = True,
) -> AtrSpikeDetector:
    """按瞬时异动（InstantConfig）或短时异动（SpikeConfig）的参数组装；只有瞬时异动带连续提醒。

    streak=False 不追加连续提醒，供 simulate 这类只验证单个窗口门槛的场景使用。
    """
    instant = isinstance(spike, InstantConfig)
    return AtrSpikeDetector(
        atr_period=spike.atr_period,
        candle_interval_seconds=INTERVAL_SECONDS[spike.candle_interval],
        max_atr_age_seconds=spike.max_atr_age_seconds,
        windows=build_windows(spike, cooldown_seconds) if windows is None else windows,
        observer=observer,
        streak=(
            StreakRule(
                spike.streak.min_alerts,
                spike.short_window.cooldown_seconds + spike.streak.gap_tolerance_seconds,
            )
            if isinstance(spike, InstantConfig) and streak and spike.streak.enabled
            else None
        ),
        kind="instant" if instant else "spike",
    )


def build_trend_detector(
    config: TrendConfig,
    *,
    rules: Sequence[TrendRule] | None = None,
    observer: Callable[[TrendEvaluation], None] | None = None,
) -> TrendDetector:
    # rules 覆盖配置中启用的周期，供 simulate 逐个周期单独验证。
    return TrendDetector(config.rules() if rules is None else rules, observer, alert_continuing=config.alert_continuing)


def build_launch_detector(
    rule: LaunchRule,
    *,
    observer: Callable[[LaunchEvaluation], None] | None = None,
) -> LaunchDetector:
    # 一个检测器只判定一个方向；规则由 LaunchSideConfig.to_rule 生成，simulate 可以先调整冷却再传进来。
    return LaunchDetector(rule, observer)


def build_dormant_detector(
    rules: Sequence[DormantRule],
    *,
    observer: Callable[[DormantEvaluation], None] | None = None,
) -> DormantDetector:
    # 现行一组与慢启动一组共用一个检测器（同一份 K 线），规则由 DormantConfig.rules 生成。
    return DormantDetector(rules, observer)


def build_reversal_detector(rule: ReversalRule) -> ReversalDetector:
    return ReversalDetector(rule)


def build_detectors(config: AppConfig) -> list[MonitoredDetector]:
    """实时监控运行的全部检测器，顺序固定为瞬时异动、短时异动、放量上涨、放量下跌、趋势、冷门启动、反弹回落；
    关闭的检测器不创建，不多订阅、不多拉 K 线。"""
    detectors: list[MonitoredDetector] = []
    # 瞬时异动与短时异动是同一套检测器按两组参数各跑一个：合约池、冷却、秒级数据互不影响。
    for spike in (config.instant, config.spike):
        if not spike.enabled:
            continue
        detector = build_detector(spike)
        detectors.append(
            MonitoredDetector(
                SPIKE_NAMES[detector.kind],
                detector,
                spike.candle_interval,
                spike.warmup_candles,
                VolumeRange(spike.min_volume_24h_quote),
                # 窗口开了前置走势或成交暴增时才要补逐秒成交历史。
                detector if detector.history_seconds else None,
            )
        )
    # 放量上涨与放量下跌是两个检测器：各自的门槛、K 线保留长度与合约池（服务按各自的成交额范围增删合约）。
    for direction, side in config.launch.sides():
        launch = build_launch_detector(side.to_rule(direction))
        volume = VolumeRange(side.min_volume_24h_quote, side.max_volume_24h_quote)
        detectors.append(
            MonitoredDetector(LAUNCH_LABELS[direction], launch, launch.warmup_interval, launch.warmup_candles, volume)
        )
    if config.trend.enabled:
        trend = build_trend_detector(config.trend)
        volume = VolumeRange(config.trend.min_volume_24h_quote, config.trend.max_volume_24h_quote)
        detectors.append(MonitoredDetector("K 线趋势", trend, trend.warmup_interval, trend.warmup_candles, volume))
    if config.dormant.enabled:
        dormant = build_dormant_detector(config.dormant.rules())
        volume = VolumeRange(config.dormant.min_volume_24h_quote, config.dormant.max_volume_24h_quote)
        detectors.append(
            MonitoredDetector("冷门启动", dormant, dormant.warmup_interval, dormant.warmup_candles, volume)
        )
    reversal = config.reversal
    if reversal.enabled:
        volume = VolumeRange(reversal.min_volume_24h_quote, reversal.max_volume_24h_quote)
        rule = reversal.to_rule()
        detector = build_reversal_detector(rule)
        detectors.append(
            MonitoredDetector(REVERSAL_NAME, detector, REVERSAL_INTERVAL, rule.warmup_candles, volume, detector)
        )
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
