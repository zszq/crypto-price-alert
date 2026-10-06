import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from colorama import Style

from price_alert.formatting import format_amount
from price_alert.launch.alerts import LaunchAlert, describe_launch_rule, format_launch_alert
from price_alert.launch.rule import LaunchMetrics, LaunchRule
from price_alert.notifier import (
    LAUNCH_LABEL_COLOR,
    SURGE_CHANGE_COLOR,
    SURGE_COLOR,
    SYMBOL_COLOR,
    JsonlNotifier,
    colorize_alert,
    format_alert,
)

# 北京时间 2026-10-05 17:56。
DETECTED = datetime(2026, 10, 5, 9, 56, tzinfo=UTC)
METRICS = LaunchMetrics(
    direction="surge",
    start_price=0.3975,
    end_price=0.4155,
    change_percent=4.528,
    window_quote=36_900.0,
    baseline_quote=283.0,
    volume_ratio=130.4,
    breakout_price=0.3914,
    failures=(),
)
ALERT = LaunchAlert(
    symbol="RLC_USDT",
    timestamp=DETECTED,
    started_at=DETECTED - timedelta(minutes=5),
    window_minutes=5,
    baseline_minutes=360,
    metrics=METRICS,
    volume_24h_quote=71_000.0,
    price_decimals=4,
)


def test_alert_text_explains_volume_ratio_and_breakout():
    assert format_launch_alert(ALERT) == (
        "[放量拉升] 2026-10-05 17:56:00 | RLC_USDT | 5分钟上涨 4.53%，成交额 3.69万 USDT，"
        "为近6小时均量的 130.4 倍，突破近6小时高点 0.3914 | 0.3975 → 0.4155"
    )
    assert format_alert(ALERT) == format_launch_alert(ALERT)


def test_wave_drop_silent_baseline_and_no_breakout_texts():
    later = replace(ALERT, wave=3)
    assert format_launch_alert(later).startswith("[放量拉升·第3波]")

    drop_metrics = replace(METRICS, direction="drop", end_price=0.38, change_percent=-4.4, breakout_price=0.385)
    drop = format_launch_alert(replace(ALERT, metrics=drop_metrics))
    assert drop.startswith("[放量下跌]") and "5分钟下跌 4.40%" in drop and "突破近6小时低点 0.3850" in drop

    silent = format_launch_alert(replace(ALERT, metrics=replace(METRICS, volume_ratio=None, breakout_price=0.5)))
    assert "近6小时无成交" in silent and "突破" not in silent


def test_to_dict_is_json_ready_and_tagged_as_launch(tmp_path):
    payload = ALERT.to_dict()

    assert payload["kind"] == "launch"
    assert (payload["direction"], payload["wave"], payload["color"]) == ("surge", 1, "green")
    assert payload["timestamp"] == "2026-10-05T09:56:00+00:00"
    assert (payload["window_quote_volume"], payload["volume_ratio"]) == (36_900.0, 130.4)

    path = tmp_path / "alerts.jsonl"
    asyncio.run(JsonlNotifier(path).send(ALERT))
    assert json.loads(path.read_text(encoding="utf-8"))["symbol"] == "RLC_USDT"


def test_colorize_highlights_label_symbol_and_change():
    text = colorize_alert(ALERT, format_alert(ALERT))

    assert text.startswith(f"{SURGE_COLOR}{LAUNCH_LABEL_COLOR}[放量拉升]{Style.RESET_ALL}{SURGE_COLOR}")
    assert f"{SYMBOL_COLOR}RLC_USDT{Style.RESET_ALL}" in text
    assert f"{SURGE_CHANGE_COLOR}4.53%{Style.RESET_ALL}" in text


def test_amounts_use_wan_and_yi_without_trailing_zeros():
    assert [format_amount(value) for value in (850, 30_000, 36_900, 123_456_789)] == ["850", "3万", "3.69万", "1.23亿"]


def test_rule_description_lists_thresholds_and_repeat_policy():
    rule = LaunchRule(5, 360, 5.0, 30_000, 3.0)
    assert describe_launch_rule(rule) == (
        "放量启动：最近 5 分钟成交额 ≥ 3万 USDT 且 ≥ 近6小时同长度均量的 5 倍，涨跌幅 3%~15%，收盘突破近6小时高低点；"
        "同方向再次提醒需间隔 ≥ 15 分钟且价格比上次提醒再推进 ≥ 5%（超过6小时重新计为启动）；只提醒上涨"
    )
    loose = replace(
        rule,
        require_breakout=False,
        alert_drops=True,
        baseline_minutes=90,
        max_change_percent=None,
    )
    assert "突破" not in describe_launch_rule(loose)
    assert "涨跌幅 ≥ 3%" in describe_launch_rule(loose)
    assert "近90分钟" in describe_launch_rule(loose) and "涨跌都提醒" in describe_launch_rule(loose)
