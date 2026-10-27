"""全部提醒类型的汇总：通知通道、提示音与服务约定按它分派，新增一种提醒只在这里加一项。"""

from __future__ import annotations

from price_alert.detectors.dormant.alerts import DormantAlert
from price_alert.detectors.launch.alerts import LaunchAlert
from price_alert.detectors.reversal.alerts import ReversalAlert
from price_alert.detectors.spike.alerts import PriceAlert
from price_alert.detectors.spike.streak import StreakAlert
from price_alert.detectors.trend.alerts import TrendAlert

Alert = PriceAlert | StreakAlert | TrendAlert | LaunchAlert | DormantAlert | ReversalAlert
