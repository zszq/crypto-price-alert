import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from colorama import Style

from price_alert.notifier import (
    DROP_CHANGE_COLOR,
    DROP_COLOR,
    SYMBOL_COLOR,
    TREND_LABEL_COLOR,
    JsonlNotifier,
    colorize_alert,
    format_alert,
)
from price_alert.trend.alerts import TrendAlert, describe_trend_rule, format_trend_alert
from price_alert.trend.pattern import PatternMetrics, TrendRule

# 北京时间 2026-09-30 22:17。
DETECTED = datetime(2026, 9, 30, 14, 17, tzinfo=UTC)
METRICS = PatternMetrics(
    direction="drop",
    start_price=0.4129,
    end_price=0.3797,
    extreme_price=0.3797,
    change_percent=-8.04,
    rebound_ratio=0.08,
    trend_candle_ratio=0.875,
    step_ratio=0.875,
    body_ratio=0.58,
    volume_ratio=1.8,
    change_atr=None,
    failures=(),
)
START = TrendAlert(
    symbol="ARK_USDT",
    stage="start",
    direction="drop",
    period="1m",
    timestamp=DETECTED,
    price=0.3797,
    anchor_price=0.4129,
    change_percent=-8.04,
    started_at=DETECTED - timedelta(minutes=8),
    extreme_price=0.3797,
    volume_24h_quote=1e7,
    price_decimals=4,
    metrics=METRICS,
    candles=8,
)


def test_start_alert_text_lists_every_condition():
    text = format_trend_alert(START)

    assert text == (
        "[趋势下跌] 2026-09-30 22:17:00 | ARK_USDT | 1分钟K线 8 根持续下跌，累计下跌 8.04% | 0.4129 → 0.3797"
        " | 阴线 88% · 收盘递进 88% · 最大反弹 8% · 实体占比 58% · 量能 1.8 倍"
    )
    surge = replace(START, direction="surge", metrics=replace(METRICS, direction="surge", volume_ratio=None))
    assert "15分钟K线" in format_trend_alert(replace(surge, period="15m"))
    assert "[趋势上涨]" in format_trend_alert(surge) and "阳线 88%" in format_trend_alert(surge)
    assert "量能" not in format_trend_alert(surge)


def test_extend_exhaustion_and_end_texts():
    later = DETECTED + timedelta(minutes=18)
    extend = replace(START, stage="extend", timestamp=later, price=0.3706, change_percent=-10.24, metrics=None)
    exhaustion = replace(extend, stage="exhaustion", period="5m", shadow_ratio=0.68)
    end = replace(extend, stage="end", extreme_price=0.3305, end_reason="rebound")

    assert format_trend_alert(extend) == (
        "[下跌延续] 2026-09-30 22:35:00 | ARK_USDT | 累计下跌 10.24%（自 22:09 起 26 分钟） | 0.4129 → 0.3706"
    )
    assert "5分钟K线出现长下影线（占振幅 68%），下跌可能放缓" in format_trend_alert(exhaustion)
    assert format_trend_alert(end).endswith("0.4129 → 最低 0.3305（最大下跌 19.96%），当前 0.3706")
    assert "反向回撤过大，自 22:09 起 26 分钟" in format_trend_alert(end)


def test_to_dict_is_json_ready_and_tagged_as_trend():
    payload = START.to_dict()

    assert payload["kind"] == "trend"
    assert payload["stage"] == "start"
    assert payload["timestamp"] == "2026-09-30T14:17:00+00:00"
    assert payload["color"] == "red"
    assert payload["metrics"] == {
        "rebound_ratio": 0.08,
        "trend_candle_ratio": 0.875,
        "step_ratio": 0.875,
        "body_ratio": 0.58,
        "volume_ratio": 1.8,
        "change_atr": None,
    }
    json.dumps(payload)
    assert "metrics" not in replace(START, metrics=None).to_dict()


def test_notifier_dispatches_trend_format_and_colors_label():
    assert format_alert(START) == format_trend_alert(START)

    colored = colorize_alert(START, format_alert(START))

    assert colored.startswith(f"{DROP_COLOR}{TREND_LABEL_COLOR}[趋势下跌]{Style.RESET_ALL}{DROP_COLOR}")
    assert f"{SYMBOL_COLOR}ARK_USDT{Style.RESET_ALL}{DROP_COLOR}" in colored
    assert f"{DROP_CHANGE_COLOR}8.04%{Style.RESET_ALL}{DROP_COLOR}" in colored
    assert colorize_alert(START, "plain", enabled=False) == "plain"


def test_jsonl_writes_trend_alert(tmp_path):
    path = tmp_path / "alerts.jsonl"
    asyncio.run(JsonlNotifier(path).send(START))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["kind"], record["symbol"], record["period"]) == ("trend", "ARK_USDT", "1m")


def test_describe_rule_mentions_optional_conditions_only_when_enabled():
    rule = TrendRule("5m", 6, 6.0, None, 0.3, 0.7, 0.7, 0.5, None, 20)

    plain = describe_trend_rule(rule)
    full = describe_trend_rule(replace(rule, min_volume_ratio=1.5, min_change_atr=3.0))

    assert plain.startswith("趋势 5分钟K线：最近 6 根累计涨跌 ≥ 6%")
    assert "量能" not in plain and "平均真实波幅" not in plain
    assert "量能 ≥ 前 20 根中位数的 1.5 倍" in full
    assert "位移 ≥ 前 20 根平均真实波幅的 3 倍" in full
