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
    rise = {"baseline_minutes": 30, "window_minutes": 3, "min_window_quote": 3000, "cooldown_minutes": 5}
    return AppConfig.model_validate({"launch": {"rise": rise}}).launch


def candles(start, closes, quote):
    result = []
    previous = closes[0]
    for index, close in enumerate(closes):
        stamp = start + timedelta(minutes=index)
        result.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return result


def test_plan_aligns_warmup_and_rejects_ranges_beyond_gate_history():
    rules = launch_config().rules()
    plan = plan_launch_replay("RLC_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), rules, NOW)
    assert plan.warmup_start == START - timedelta(minutes=33)

    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_launch_replay("RLC_USDT", START, START, rules, NOW)
    with pytest.raises(ValueError, match="10000 根"):
        plan_launch_replay("RLC_USDT", NOW - timedelta(days=7), NOW, rules, NOW)
    with pytest.raises(ValueError, match="都已关闭"):
        plan_launch_replay("RLC_USDT", START, NOW, [], NOW)


def test_run_and_render_explain_alerts_and_failures():
    config = launch_config()
    plan = plan_launch_replay("RLC_USDT", START, START + timedelta(minutes=10), config.rules(), NOW)
    history = candles(plan.warmup_start, [1.0] * 33, 100.0)
    # 区间开始后放量逐分钟上涨 2%，再横盘。
    rally = candles(START, [1.0, 1.02, 1.04, 1.06, 1.08] + [1.08] * 5, 2000.0)

    result = run_launch_replay(config, plan, history + rally, price_decimals=4)
    lines = render_launch_replay(result)

    assert [alert.timestamp for alert in result.alerts] == [START + timedelta(minutes=3)]
    assert result.warmup_candles == 33 and result.candles_fed == 10
    text = "\n".join(lines)
    assert "提醒 1 条：" in text and "[放量上涨] 2026-10-05 17:53:00 | RLC_USDT | 3分钟上涨 4.00%" in text
    assert "满足条件，冷却中" in text
    assert "未满足的条件：" in text and "最接近触发：" in text

    all_lines = render_launch_replay(result, show_all=True)
    # 区间内每一分钟收盘都有一行判定：17:51 到 18:00。
    rows = all_lines[all_lines.index("逐分钟判定：") + 1 :]
    assert [row.split(" | ")[0].strip() for row in rows] == [
        f"{minute:%H:%M}" for minute in (START + timedelta(minutes=offset) for offset in range(1, 11))
    ]


def test_overshoot_minute_is_listed_to_explain_later_cooldown():
    config = launch_config()
    plan = plan_launch_replay("RLC_USDT", START, START + timedelta(minutes=6), config.rules(), NOW)
    history = candles(plan.warmup_start, [1.0] * 33, 100.0)
    # 3 分钟涨 18% 超过上限，不提醒；窗口滑过后涨幅回到区间内，也只是冷却中，不在高位补报。
    spike = candles(START, [1.0, 1.02, 1.18, 1.20, 1.22, 1.24], 2000.0)

    result = run_launch_replay(config, plan, history + spike)
    text = "\n".join(render_launch_replay(result))

    assert result.alerts == []
    assert "17:53 | 1 → 1.18 +18.00% | 3分钟成交额 6000 量比 20.0 | 只因涨跌幅过大（插针）不提醒" in text
    assert "满足条件，冷却中" in text


def test_both_directions_replay_with_their_own_baselines():
    # 下跌基准期更长：按较长的一边预热，两个方向各自成段输出，提醒按时间合并。
    config = AppConfig.model_validate(
        {
            "launch": {
                "rise": {"baseline_minutes": 30, "window_minutes": 3, "min_window_quote": 3000, "cooldown_minutes": 5},
                "fall": {
                    "enabled": True,
                    "baseline_minutes": 40,
                    "window_minutes": 3,
                    "min_window_quote": 3000,
                    "cooldown_minutes": 5,
                },
            }
        }
    ).launch
    plan = plan_launch_replay("RLC_USDT", START, START + timedelta(minutes=20), config.rules(), NOW)
    assert plan.warmup_start == START - timedelta(minutes=43)
    history = candles(plan.warmup_start, [1.0] * 43, 100.0)
    # 先放量上涨 3 分钟，横盘，再放量下跌 3 分钟。
    moves = candles(START, [1.0, 1.02, 1.04, 1.06] + [1.06] * 6 + [1.04, 1.01, 0.98] + [0.98] * 7, 2000.0)

    result = run_launch_replay(config, plan, history + moves)
    text = "\n".join(render_launch_replay(result))

    assert [(alert.direction, alert.timestamp) for alert in result.alerts] == [
        ("surge", START + timedelta(minutes=3)),
        ("drop", START + timedelta(minutes=13)),
    ]
    assert "【放量上涨】放量上涨：" in text and "【放量下跌】放量下跌：" in text
