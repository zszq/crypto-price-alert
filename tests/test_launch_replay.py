from datetime import datetime, timedelta

import pytest

from price_alert.config import AppConfig
from price_alert.formatting import BEIJING_TIME
from price_alert.launch.replay import plan_launch_replay, render_launch_replay, run_launch_replay
from price_alert.models import Candle

START = datetime(2026, 10, 5, 17, 50, tzinfo=BEIJING_TIME)
# 固定“当前时间”：回放会检查区间是否超出 Gate 的保留范围，不能依赖真实时钟。
NOW = START + timedelta(days=1)


def launch_config():
    # 基准期缩短到 30 分钟，构造数据简单直接。
    return AppConfig.model_validate(
        {"launch": {"baseline_minutes": 30, "window_minutes": 3, "min_window_quote": 3000, "cooldown_minutes": 5}}
    ).launch


def candles(start, closes, quote):
    result = []
    previous = closes[0]
    for index, close in enumerate(closes):
        stamp = start + timedelta(minutes=index)
        result.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return result


def test_plan_aligns_warmup_and_rejects_ranges_beyond_gate_history():
    rule = launch_config().to_rule()
    plan = plan_launch_replay("RLC_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), rule, NOW)
    assert plan.warmup_start == START - timedelta(minutes=33)

    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_launch_replay("RLC_USDT", START, START, rule, NOW)
    with pytest.raises(ValueError, match="10000 根"):
        plan_launch_replay("RLC_USDT", NOW - timedelta(days=7), NOW, rule, NOW)


def test_run_and_render_explain_alerts_and_failures():
    config = launch_config()
    plan = plan_launch_replay("RLC_USDT", START, START + timedelta(minutes=10), config.to_rule(), NOW)
    history = candles(plan.warmup_start, [1.0] * 33, 100.0)
    # 区间开始后放量逐分钟上涨 2%，再横盘。
    rally = candles(START, [1.0, 1.02, 1.04, 1.06, 1.08] + [1.08] * 5, 2000.0)

    result = run_launch_replay(config, plan, history + rally, price_decimals=4)
    lines = render_launch_replay(result)

    assert [alert.timestamp for alert in result.alerts] == [START + timedelta(minutes=3)]
    assert result.warmup_candles == 33 and result.candles_fed == 10
    text = "\n".join(lines)
    assert "提醒 1 条：" in text and "[放量拉升] 2026-10-05 17:53:00 | RLC_USDT | 3分钟上涨 4.00%" in text
    assert "满足条件，冷却中" in text
    assert "未满足的条件：" in text and "最接近触发：" in text

    all_lines = render_launch_replay(result, show_all=True)
    # 区间内每一分钟收盘都有一行判定：17:51 到 18:00。
    rows = all_lines[all_lines.index("逐分钟判定：") + 1 :]
    assert [row.split(" | ")[0].strip() for row in rows] == [
        f"{minute:%H:%M}" for minute in (START + timedelta(minutes=offset) for offset in range(1, 11))
    ]
