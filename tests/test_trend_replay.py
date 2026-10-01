from datetime import UTC, datetime, timedelta

import pytest

from price_alert.assembly import build_trend_detector
from price_alert.config import AppConfig
from price_alert.formatting import BEIJING_TIME
from price_alert.models import Candle
from price_alert.trend.replay import (
    fetch_minute_candles,
    plan_trend_replay,
    render_trend_replay,
    run_trend_replay,
)

START = datetime(2026, 9, 30, 22, 0, tzinfo=BEIJING_TIME)


def trend_config():
    # 只开 1 分钟周期、关闭量能，构造数据简单直接。
    periods = {"1m": {"candles": 5, "baseline_candles": 5, "min_volume_ratio": None}}
    periods |= {name: {"enabled": False} for name in ("3m", "5m", "15m")}
    return AppConfig.model_validate({"trend": {"periods": periods}}).trend


def minute_candles(start: datetime, closes: list[float]) -> list[Candle]:
    candles = []
    previous = closes[0]
    for index, close in enumerate(closes):
        high, low = max(previous, close) + 0.01, min(previous, close) - 0.01
        candles.append(Candle(start + timedelta(minutes=index), previous, high, low, close, volume=10))
        previous = close
    return candles


def test_plan_starts_warmup_from_longest_history():
    detector = build_trend_detector(AppConfig().trend)

    plan = plan_trend_replay("ARK_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), detector)

    assert plan.warmup_start == START - timedelta(minutes=detector.warmup_candles - 1)
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_trend_replay("ARK_USDT", START, START, detector)


class ChunkRest:
    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []

    def fetch_candles_between(self, symbol, interval, start, end):
        assert interval == "1m"
        self.calls.append((start, end))
        count = int((end - start) / timedelta(minutes=1)) + 1
        # 两端都包含，且故意多返回一根区间外的 K 线，验证调用方会去重与裁剪。
        return minute_candles(start, [100.0] * (count + 1))


def test_fetch_splits_long_ranges_without_overlap():
    rest = ChunkRest()
    start = datetime(2026, 9, 30, tzinfo=UTC)

    candles = fetch_minute_candles(rest, "ARK_USDT", start, start + timedelta(minutes=2500))

    assert len(rest.calls) == 3
    assert rest.calls[0] == (start, start + timedelta(minutes=999))
    assert rest.calls[1][0] == start + timedelta(minutes=1000)
    assert [candle.timestamp for candle in candles] == [start + timedelta(minutes=i) for i in range(2500)]


def test_run_and_render_explain_alerts_and_failures():
    config = trend_config()
    detector = build_trend_detector(config)
    end = START + timedelta(minutes=20)
    plan = plan_trend_replay("ARK_USDT", START, end, detector)
    # 预热段平稳，区间开始后逐分钟下跌 1%，再横盘。
    history = minute_candles(plan.warmup_start, [100.0] * 10)
    trend = minute_candles(START, [100.0 - i for i in range(1, 8)] + [93.0] * 13)

    result = run_trend_replay(config, plan, history + trend, price_decimals=2)
    lines = render_trend_replay(result)
    text = "\n".join(lines)

    assert result.warmup_candles == 10
    assert result.candles_fed == 20
    assert [alert.stage for alert in result.alerts] == ["start", "extend"]
    assert "提醒 2 条：" in text
    assert "[趋势下跌] 2026-09-30 22:05:00 | ARK_USDT | 1分钟K线 5 根持续下跌" in text
    assert "各周期判定统计：" in text and "未满足的条件：" in text
    assert "形态成立的时刻" in text
    assert "22:05 | 1m | 99.00 → 95.00 -4.04% | 顺势 80%" in text
    all_lines = render_trend_replay(result, show_all=True)
    assert len(all_lines) > len(lines)
    assert "逐根判定：" in all_lines


def test_render_without_alerts():
    config = trend_config()
    detector = build_trend_detector(config)
    plan = plan_trend_replay("ARK_USDT", START, START + timedelta(minutes=10), detector)

    result = run_trend_replay(config, plan, minute_candles(plan.warmup_start, [100.0, 100.2] * 10))
    text = "\n".join(render_trend_replay(result))

    assert "区间内没有趋势提醒。" in text
    assert "没有任何周期的形态成立" in text
