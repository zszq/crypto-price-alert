from datetime import datetime, timedelta

import pytest

from price_alert.dormant.replay import plan_dormant_replay, render_dormant_replay, run_dormant_replay
from price_alert.dormant.rule import PRIOR_MINUTES, DormantRule
from price_alert.formatting import BEIJING_TIME
from price_alert.models import Candle

START = datetime(2026, 10, 21, 14, 50, tzinfo=BEIJING_TIME)
# 固定“当前时间”：回放会检查区间是否超出 Gate 的保留范围，不能依赖真实时钟。
NOW = START + timedelta(days=1)
RULE = DormantRule(window_minutes=3, max_prior_quote=10_000, min_window_quote=1_000, min_active_minutes=2)


def candles(start, closes, quote):
    result = []
    previous = closes[0]
    for index, close in enumerate(closes):
        stamp = start + timedelta(minutes=index)
        result.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return result


def test_plan_aligns_warmup_and_rejects_ranges_beyond_gate_history():
    plan = plan_dormant_replay("IRYS_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), RULE, NOW)
    assert plan.warmup_start == START - timedelta(minutes=PRIOR_MINUTES + 3)

    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_dormant_replay("IRYS_USDT", START, START, RULE, NOW)
    with pytest.raises(ValueError, match="10000 根"):
        plan_dormant_replay("IRYS_USDT", NOW - timedelta(days=7), NOW, RULE, NOW)


def test_run_and_render_explain_alerts_and_failures():
    plan = plan_dormant_replay("IRYS_USDT", START, START + timedelta(minutes=8), RULE, NOW)
    history = candles(plan.warmup_start, [1.0] * (PRIOR_MINUTES + 3), 5.0)
    # 区间开始后逐分钟上涨 3%，再横盘。
    rally = candles(START, [1.03, 1.06, 1.09, 1.12] + [1.12] * 4, 500.0)

    result = run_dormant_replay(RULE, plan, history + rally, price_decimals=4)
    text = "\n".join(render_dormant_replay(result))

    assert [alert.timestamp for alert in result.alerts] == [START + timedelta(minutes=3)]
    assert result.warmup_candles == PRIOR_MINUTES + 3 and result.candles_fed == 8
    assert "提醒 1 条：" in text and "[冷门启动] 2026-10-21 14:53:00 | IRYS_USDT | 3分钟上涨 9.00%" in text
    assert "满足条件，冷却中" in text
    assert "未满足的条件：" in text and "最接近触发：" in text
    all_rows = "\n".join(render_dormant_replay(result, show_all=True))
    assert "逐分钟判定：" in all_rows and all_rows.count(" | ") > text.count(" | ")
