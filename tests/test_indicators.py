from datetime import UTC, datetime, timedelta

import pytest

from price_alert.indicators import WilderAtr
from price_alert.models import Candle


def candle(minute: int, high: float, low: float, close: float) -> Candle:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
    return Candle(timestamp, close, high, low, close)


def test_wilder_atr_warmup_and_recursive_update():
    atr = WilderAtr(period=3)
    atr.seed([candle(0, 101, 99, 100), candle(1, 101, 99, 100), candle(2, 101, 99, 100)])

    assert atr.ready
    assert atr.value == 2.0

    atr.update(candle(3, 104, 100, 103))
    assert atr.value == pytest.approx((2 * 2 + 4) / 3)


def test_wilder_atr_ignores_duplicate_or_old_candle():
    atr = WilderAtr(period=2)
    atr.seed([candle(0, 101, 99, 100), candle(1, 101, 99, 100)])

    assert atr.update(candle(1, 120, 80, 100)) == 2.0


def test_wilder_atr_copy_evolves_independently_even_during_warmup():
    # 预热未完成时状态在列表里，浅拷贝会让副本的更新写回原对象。
    original = WilderAtr(period=3)
    original.seed([candle(0, 101, 99, 100), candle(1, 101, 99, 100)])

    clone = original.copy()
    clone.update(candle(2, 104, 100, 103))

    assert clone.value == pytest.approx((2 + 2 + 4) / 3)
    assert clone.last_timestamp == candle(2, 104, 100, 103).timestamp
    assert original.value is None
    assert original.last_timestamp == candle(1, 101, 99, 100).timestamp
    # 原对象随后仍按自己的历史继续预热，不受副本影响。
    original.update(candle(2, 101, 99, 100))
    assert original.value == 2.0
