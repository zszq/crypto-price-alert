from datetime import UTC, datetime, timedelta

import pytest

from price_alert.assembly import build_trend_detector
from price_alert.config import AppConfig
from price_alert.formatting import BEIJING_TIME
from price_alert.gate import fetch_minute_candles
from price_alert.models import Candle
from price_alert.trend.replay import (
    plan_trend_replay,
    render_trend_replay,
    run_trend_replay,
)

START = datetime(2026, 9, 30, 22, 0, tzinfo=BEIJING_TIME)
# 固定“当前时间”：回放会检查区间是否超出 Gate 的保留范围，不能依赖真实时钟。
NOW = START + timedelta(days=1)


def trend_config():
    # 只开 3 分钟周期，构造数据简单直接。
    periods = {"3m": {"candles": 3, "min_change_percent": 4.0, "max_counter_candles": 0}}
    periods |= {name: {"enabled": False} for name in ("5m", "15m")}
    return AppConfig.model_validate({"trend": {"periods": periods}}).trend


def minute_candles(start: datetime, closes: list[float]) -> list[Candle]:
    candles = []
    previous = closes[0]
    for index, close in enumerate(closes):
        high, low = max(previous, close) + 0.01, min(previous, close) - 0.01
        candles.append(Candle(start + timedelta(minutes=index), previous, high, low, close))
        previous = close
    return candles


def test_plan_starts_warmup_from_longest_history():
    detector = build_trend_detector(AppConfig().trend)

    plan = plan_trend_replay("ARK_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), detector, NOW)

    assert plan.warmup_start == START - timedelta(minutes=detector.warmup_candles - 1)
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_trend_replay("ARK_USDT", START, START, detector, NOW)


def test_plan_rejects_range_beyond_gate_history_with_readable_message():
    detector = build_trend_detector(AppConfig().trend)
    now = START + timedelta(days=10)

    with pytest.raises(ValueError, match="Gate 只保留最近 10000 根 1 分钟 K 线"):
        plan_trend_replay("ARK_USDT", START, START + timedelta(hours=1), detector, now)
    # 预热也要落在保留范围内（10000 根含当前一根，另留一分钟余量）：恰好在边界上的开始时间可以回放。
    earliest = now - timedelta(minutes=9998) + timedelta(minutes=detector.warmup_candles - 1)
    plan = plan_trend_replay("ARK_USDT", earliest, earliest + timedelta(hours=1), detector, now)
    assert plan.warmup_start == now - timedelta(minutes=9998)
    with pytest.raises(ValueError):
        plan_trend_replay("ARK_USDT", earliest - timedelta(minutes=1), earliest, detector, now)


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
    plan = plan_trend_replay("ARK_USDT", START, end, detector, NOW)
    # 预热段平稳，区间开始后逐分钟下跌 1% 共 13 分钟，再横盘。
    history = minute_candles(plan.warmup_start, [100.0] * (detector.warmup_candles - 1))
    trend = minute_candles(START, [100.0 - i for i in range(1, 14)] + [87.0] * 7)

    result = run_trend_replay(config, plan, history + trend, price_decimals=2)
    lines = render_trend_replay(result)
    text = "\n".join(lines)

    assert result.warmup_candles == 9
    assert result.candles_fed == 20
    # 下跌形态在 22:09 形成，22:12、22:15 仍成立，各报一条延续提醒。
    assert [(alert.timestamp, alert.continuing) for alert in result.alerts] == [
        (START + timedelta(minutes=9), False),
        (START + timedelta(minutes=12), True),
        (START + timedelta(minutes=15), True),
    ]
    assert "提醒 3 条：" in text
    assert "[趋势下跌] 2026-09-30 22:09:00 | ARK_USDT | 3分钟K线 3 根持续下跌" in text
    assert "[趋势下跌·延续] 2026-09-30 22:12:00 | ARK_USDT | 3分钟K线 3 根持续下跌" in text
    assert "各周期判定统计：" in text and "未满足的条件：反向 K 线过多" in text
    assert "22:09 | 3m | 99.00 → 91.00 -8.08% | 反向 0 根 反弹 0% 总实体 99% 最大实体 38% | 提醒" in text
    assert (
        "22:12 | 3m | 97.00 → 88.00 -9.28% | 反向 0 根 反弹 0% 总实体 99% 最大实体 33% | 形态延续，延续提醒已发" in text
    )
    all_lines = render_trend_replay(result, show_all=True)
    assert len(all_lines) > len(lines)
    assert "逐根判定：" in all_lines

    # 关闭延续提醒后同一段只报一次，之后的判定记为延续。
    quiet = run_trend_replay(config.model_copy(update={"alert_continuing": False}), plan, history + trend, 2)
    assert [alert.timestamp for alert in quiet.alerts] == [START + timedelta(minutes=9)]
    assert "最大实体 33% | 形态延续，本段已提醒" in "\n".join(render_trend_replay(quiet))


def test_render_without_alerts():
    config = trend_config()
    detector = build_trend_detector(config)
    plan = plan_trend_replay("ARK_USDT", START, START + timedelta(minutes=10), detector, NOW)

    result = run_trend_replay(config, plan, minute_candles(plan.warmup_start, [100.0, 100.2] * 10))
    text = "\n".join(render_trend_replay(result))

    assert "区间内没有趋势提醒。" in text
    assert "没有任何周期的形态成立" in text
