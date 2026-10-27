from datetime import datetime, timedelta

import pytest

from price_alert.common.formatting import BEIJING_TIME
from price_alert.common.models import Candle
from price_alert.detectors.dormant.replay import plan_dormant_replay, render_dormant_replay, run_dormant_replay
from price_alert.detectors.dormant.rule import PRIOR_MINUTES, DormantGroup, DormantRule

START = datetime(2026, 10, 21, 14, 50, tzinfo=BEIJING_TIME)
# 固定“当前时间”：回放会检查区间是否超出 Gate 的保留范围，不能依赖真实时钟。
NOW = START + timedelta(days=1)
RULES = (DormantRule(window_minutes=3, max_prior_quote=10_000, min_window_quote=1_000, min_active_minutes=2),)


def candles(start, closes, quote):
    result = []
    previous = closes[0]
    for index, close in enumerate(closes):
        stamp = start + timedelta(minutes=index)
        result.append(Candle(stamp, previous, max(previous, close), min(previous, close), close, quote))
        previous = close
    return result


def test_plan_aligns_warmup_and_rejects_ranges_beyond_gate_history():
    plan = plan_dormant_replay("IRYS_USDT", START + timedelta(seconds=20), START + timedelta(hours=1), RULES, NOW)
    assert plan.warmup_start == START - timedelta(minutes=PRIOR_MINUTES + 3)

    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_dormant_replay("IRYS_USDT", START, START, RULES, NOW)
    with pytest.raises(ValueError, match="10000 根"):
        plan_dormant_replay("IRYS_USDT", NOW - timedelta(days=7), NOW, RULES, NOW)


def test_run_and_render_explain_alerts_and_failures():
    plan = plan_dormant_replay("IRYS_USDT", START, START + timedelta(minutes=8), RULES, NOW)
    history = candles(plan.warmup_start, [1.0] * (PRIOR_MINUTES + 3), 5.0)
    # 区间开始后逐分钟上涨 3%，再横盘。
    rally = candles(START, [1.03, 1.06, 1.09, 1.12] + [1.12] * 4, 500.0)

    result = run_dormant_replay(RULES, plan, history + rally, price_decimals=4)
    text = "\n".join(render_dormant_replay(result))

    assert [alert.timestamp for alert in result.alerts] == [START + timedelta(minutes=3)]
    assert result.warmup_candles == PRIOR_MINUTES + 3 and result.candles_fed == 8
    assert "提醒 1 条：" in text and "[冷门启动] 2026-10-21 14:53:00 | IRYS_USDT | 3分钟上涨 9.00%" in text
    assert "满足条件，冷却中" in text
    assert "未满足的条件：" in text and "最接近触发：" in text
    all_rows = "\n".join(render_dormant_replay(result, show_all=True))
    assert "逐分钟判定：" in all_rows and all_rows.count(" | ") > text.count(" | ")


def test_two_groups_share_one_replay_and_render_separately():
    slow = DormantRule(
        window_minutes=6, max_prior_quote=10_000, min_window_quote=1_000, min_active_minutes=3,
        min_change_percent=12.0, min_efficiency=0.5, group=DormantGroup.SLOW,
    )
    rules = (*RULES, slow)
    plan = plan_dormant_replay("IRYS_USDT", START, START + timedelta(minutes=8), rules, NOW)
    # 预热按窗口最长的一组。
    assert plan.warmup_start == START - timedelta(minutes=PRIOR_MINUTES + 6)
    history = candles(plan.warmup_start, [1.0] * (PRIOR_MINUTES + 6), 5.0)
    rally = candles(START, [1.03, 1.06, 1.09, 1.12] + [1.12] * 4, 500.0)

    result = run_dormant_replay(rules, plan, history + rally, price_decimals=4)
    text = "\n".join(render_dormant_replay(result))

    # 现行一组 14:53 报，慢启动一组 14:54（6 分钟涨 12%）报。
    assert [(alert.label, alert.timestamp) for alert in result.alerts] == [
        ("冷门启动", START + timedelta(minutes=3)),
        ("冷门慢启动", START + timedelta(minutes=4)),
    ]
    assert "【冷门启动】" in text and "【冷门慢启动】" in text
    assert "冷门慢启动：之前 24 小时成交额" in text and "效率 1.00 量比" in text
