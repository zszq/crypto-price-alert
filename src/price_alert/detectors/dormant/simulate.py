"""合成行情验证冷门启动配置：之前 24 小时少量成交、一字横盘，之后一段拉升，每个启用的参数组恰好产生一条提醒。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from price_alert.assembly import build_dormant_detector
from price_alert.common.models import Candle
from price_alert.detectors.dormant.alerts import DormantAlert
from price_alert.detectors.dormant.config import DormantConfig
from price_alert.detectors.dormant.rule import PRIOR_MINUTES, DormantRule, late_minutes

SYMBOL = "BTC_USDT"
_PRICE = 100.0


def simulate_dormant(config: DormantConfig) -> list[DormantAlert]:
    # 每组单独用一个只含该组的检测器跑：同一段合成行情可能同时满足两组，合在一起就不是每组恰好一条了。
    return [alert for rule in config.rules() for alert in _simulate_rule(rule)]


def _simulate_rule(rule: DormantRule) -> list[DormantAlert]:
    # 拉升过程中条件可能在最后一分钟之前就满足，之后继续上涨还可能报第 N 波；模拟只验证会报出第一条，报出即停。
    detector = build_dormant_detector([rule])
    now = int(datetime.now(UTC).timestamp())
    launch_start = datetime.fromtimestamp(now - now % 60, tz=UTC) - timedelta(minutes=rule.window_minutes)
    # 横盘段要覆盖判定所需的全部历史，拉升的第一分钟就能判定。
    flat_start = launch_start - timedelta(minutes=rule.history_minutes)
    # 每分钟的成交额按之前 24 小时合计为冷门门槛的一半分摊。
    prior_quote = rule.max_prior_quote / 2 / PRIOR_MINUTES
    flat = [
        Candle(flat_start + timedelta(minutes=index), _PRICE, _PRICE, _PRICE, _PRICE, prior_quote)
        for index in range(rule.history_minutes)
    ]
    detector.add_symbol(SYMBOL, flat, rule.max_prior_quote / 2, quanto_multiplier=1.0)

    # 窗口每分钟都有成交，成交额（同时满足量比）与涨幅取门槛的 1.5 倍，逐分钟等额上涨、无影线、收在最高价：
    # 效率为 1、回落为 0；横盘段一字不动，倍数无穷大。单分钟涨幅为总涨幅 ÷ 窗口，配置已保证不超上限。
    window_quote = max(rule.min_window_quote, rule.min_volume_ratio * prior_quote * rule.window_minutes, 1.0) * 1.5
    change = min(
        rule.min_change_percent * 1.5,
        (rule.min_change_percent + 100) / 2,
        rule.window_minutes * rule.max_minute_change_percent,
    )
    step = _PRICE * change / 100 / rule.window_minutes
    # 后半窗口每分钟放大 late_scale 倍，使后半成交额占比为门槛的 1.5 倍以上（门槛 < 1 由配置保证）；
    # 最后一分钟成交额也取门槛的 1.5 倍。只加不减，窗口成交额照样满足。
    late = late_minutes(rule.window_minutes)
    early = rule.window_minutes - late
    share = min(rule.min_late_quote_share * 1.5, (rule.min_late_quote_share + 1) / 2)
    late_scale = max(1.0, share * early / (late * (1 - share))) if early else 1.0
    base_quote = window_quote / rule.window_minutes
    late_quote = max(base_quote * late_scale, rule.min_last_minute_quote * 1.5)
    alerts: list[DormantAlert] = []
    for index in range(rule.window_minutes):
        open_price = _PRICE + step * index
        close_price = open_price + step
        candle = Candle(
            launch_start + timedelta(minutes=index),
            open_price,
            close_price,
            open_price,
            close_price,
            late_quote if index >= early else base_quote,
        )
        alerts.extend(detector.add_candle(SYMBOL, candle))
        if alerts:
            break
    return alerts
