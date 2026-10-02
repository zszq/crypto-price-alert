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
    change_percent=-8.04,
    counter_candles=1,
    rebound_ratio=0.08,
    body_ratio=0.58,
    single_candle_ratio=0.31,
    failures=(),
)
ALERT = TrendAlert(
    symbol="ARK_USDT",
    direction="drop",
    period="1m",
    candles=8,
    timestamp=DETECTED,
    started_at=DETECTED - timedelta(minutes=8),
    metrics=METRICS,
    volume_24h_quote=1e7,
    price_decimals=4,
)


def test_alert_text_lists_every_condition():
    assert format_trend_alert(ALERT) == (
        "[趋势下跌] 2026-09-30 22:17:00 | ARK_USDT | 1分钟K线 8 根持续下跌，累计下跌 8.04%（自 22:09 起）"
        " | 0.4129 → 0.3797 | 阴线 7/8 · 最大反弹 8% · 实体占比 58% · 单根占比 31%"
    )
    surge = replace(ALERT, direction="surge", period="15m", metrics=replace(METRICS, direction="surge"))
    text = format_trend_alert(surge)
    assert text.startswith("[趋势上涨]")
    assert "15分钟K线 8 根持续上涨" in text and "阳线 7/8" in text


def test_to_dict_is_json_ready_and_tagged_as_trend():
    payload = ALERT.to_dict()

    assert payload["kind"] == "trend"
    assert payload["timestamp"] == "2026-09-30T14:17:00+00:00"
    assert payload["started_at"] == "2026-09-30T14:09:00+00:00"
    assert (payload["start_price"], payload["price"], payload["change_percent"]) == (0.4129, 0.3797, -8.04)
    assert (payload["counter_candles"], payload["rebound_ratio"], payload["body_ratio"]) == (1, 0.08, 0.58)
    assert payload["single_candle_ratio"] == 0.31
    assert payload["continuing"] is False
    assert payload["color"] == "red"
    json.dumps(payload)


def test_notifier_dispatches_trend_format_and_colors_label():
    assert format_alert(ALERT) == format_trend_alert(ALERT)

    colored = colorize_alert(ALERT, format_alert(ALERT))

    assert colored.startswith(f"{DROP_COLOR}{TREND_LABEL_COLOR}[趋势下跌]{Style.RESET_ALL}{DROP_COLOR}")
    assert f"{SYMBOL_COLOR}ARK_USDT{Style.RESET_ALL}{DROP_COLOR}" in colored
    assert f"{DROP_CHANGE_COLOR}8.04%{Style.RESET_ALL}{DROP_COLOR}" in colored
    assert colorize_alert(ALERT, "plain", enabled=False) == "plain"


def test_jsonl_writes_trend_alert(tmp_path):
    path = tmp_path / "alerts.jsonl"
    asyncio.run(JsonlNotifier(path).send(ALERT))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["kind"], record["symbol"], record["period"]) == ("trend", "ARK_USDT", "1m")


def test_describe_rule():
    rule = TrendRule("5m", 4, 6.0, 1, 0.3, 0.5, 0.5)

    assert describe_trend_rule(rule) == (
        "趋势 5分钟K线：最近 4 根累计涨跌 ≥ 6%，反向 K 线 ≤ 1 根，最大反弹 ≤ 30%，实体占比 ≥ 50%，单根占比 ≤ 50%"
    )


def test_continuing_alert_is_labelled_in_text_color_and_payload():
    continuing = replace(ALERT, continuing=True)

    text = format_alert(continuing)
    colored = colorize_alert(continuing, text)

    assert text.startswith("[趋势下跌·延续] ")
    assert colored.startswith(f"{DROP_COLOR}{TREND_LABEL_COLOR}[趋势下跌·延续]{Style.RESET_ALL}{DROP_COLOR}")
    assert continuing.to_dict()["continuing"] is True
