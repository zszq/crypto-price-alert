"""面向用户的数值与时间展示格式，提醒与回放共用，保证两处看到的价格和窗口描述一致。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

BEIJING_TIME = timezone(timedelta(hours=8))
# 没有交易所报价精度时按有效数字显示；8 位足以让十万级价格保留到小数、极小价格不丢尾数。
FALLBACK_SIGNIFICANT_DIGITS = 8


def format_price(value: float, decimals: int | None = None) -> str:
    """按交易所报价精度显示价格；精度未知时按有效数字显示，且绝不截断整数部分或用科学计数法。"""
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    magnitude = math.floor(math.log10(abs(value)))
    text = f"{value:.{max(0, FALLBACK_SIGNIFICANT_DIGITS - 1 - magnitude)}f}"
    # VWAP 常带一长串无意义的尾数，而整数价格补出的尾随零同样没有信息量，统一去掉。
    return text.rstrip("0").rstrip(".") if "." in text else text


def describe_window(seconds: int) -> str:
    # 整分钟的长窗口按分钟说，“180秒内”不如“3分钟内”直观；其余保持秒数原样。
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}分钟"
    return f"{seconds}秒"


def beijing_time(moment: datetime, pattern: str = "%Y-%m-%d %H:%M:%S") -> str:
    return moment.astimezone(BEIJING_TIME).strftime(pattern)
