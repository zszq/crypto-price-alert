import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig
from price_alert.detectors.dormant.alerts import DormantAlert, describe_dormant_rule, format_dormant_alert
from price_alert.detectors.dormant.config import DormantConfig
from price_alert.detectors.dormant.rule import DormantMetrics, DormantRule
from price_alert.exchange.universe import VolumeRange


def test_defaults_are_the_research_selection():
    # research/launch_trade/dormant_calm.md 开发期选出的一组。
    assert DormantConfig().to_rule() == DormantRule(
        window_minutes=10,
        max_prior_quote=1_000_000,
        min_window_quote=5_000,
        min_active_minutes=5,
        min_change_percent=8.0,
        max_minute_change_percent=5.0,
        min_body_ratio=0.6,
        min_calm_multiple=10.0,
        max_pre_volatility_percent=0.5,
        cooldown_minutes=60,
        realert_step_percent=3.0,
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"window_minutes": 3, "min_active_minutes": 4}, "min_active_minutes 不能大于 window_minutes"),
        ({"cooldown_minutes": 5}, "cooldown_minutes 不能小于 window_minutes"),
        (
            {"window_minutes": 2, "min_active_minutes": 2, "max_minute_change_percent": 3.0},
            "min_change_percent 不能大于 window_minutes × max_minute_change_percent",
        ),
        ({"min_body_ratio": 1.5}, "less than or equal to 1"),
        ({"min_late_quote_share": 1}, "less than 1"),
        ({"min_last_minute_quote": -1}, "greater than or equal to 0"),
        ({"realert_step_percent": 0}, "greater than 0"),
        ({"max_pre_volatility_percent": 0}, "greater than 0"),
        ({"min_volume_24h_quote": 5_000_000}, "max_volume_24h_quote 必须大于 min_volume_24h_quote"),
        ({"unknown": 1}, "Extra inputs"),
    ],
)
def test_rejects_inconsistent_values(values, message):
    with pytest.raises(ValidationError, match=message):
        DormantConfig.model_validate(values)


def test_joins_monitor_as_its_own_detector_and_pool():
    entries = {entry.name: entry for entry in build_detectors(AppConfig())}

    # 冷门启动之后只有反弹回落（顺序与 default.yaml 一致）。
    assert list(entries)[-2:] == ["冷门启动", "反弹回落"]
    entry = entries["冷门启动"]
    assert entry.volume == VolumeRange(0, 3_000_000)
    assert (entry.candle_interval, entry.warmup_candles) == ("1m", 1440 + 10 + 1)
    disabled = AppConfig.model_validate({"dormant": {"enabled": False}})
    assert "冷门启动" not in [entry.name for entry in build_detectors(disabled)]


def make_alert() -> DormantAlert:
    now = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
    metrics = DormantMetrics(0.01542, 0.01685, 9.27, 51_156.0, 8, 423_566.0, 0.3, 30.9, 3.9, 0.82, 6_120.0, 0.55, ())
    return DormantAlert("IRYS_USDT", now, now - timedelta(minutes=10), 10, metrics, 2_000_000, 5)


def test_alert_text_and_structured_fields():
    alert = make_alert()

    assert format_dormant_alert(alert) == (
        "[冷门启动] 2026-10-21 17:00:00 | IRYS_USDT | 10分钟上涨 9.27%，成交额 5.12万 USDT（8 分钟有成交），"
        "之前24小时成交额 42.36万 USDT，涨前240分钟平均每分钟波动 0.30%（涨幅为其 31 倍） | 0.01542 → 0.01685"
    )
    data = alert.to_dict()
    assert data["kind"] == "dormant" and data["direction"] == "surge" and data["color"] == "green"
    assert (data["active_minutes"], data["prior_quote_volume"], data["window_quote_volume"]) == (8, 423_566.0, 51_156.0)
    assert (data["pre_volatility_percent"], data["max_minute_change_percent"], data["body_ratio"]) == (0.3, 3.9, 0.82)
    assert (data["last_minute_quote_volume"], data["late_quote_share"]) == (6_120.0, 0.55)
    assert data["timestamp"] == "2026-10-21T09:00:00+00:00"


def test_alert_text_for_coin_that_was_flat_before():
    alert = make_alert()
    flat = replace(alert, metrics=replace(alert.metrics, pre_volatility_percent=0.0, calm_multiple=math.inf))

    assert "涨前240分钟一字横盘 | " in format_dormant_alert(flat)


def test_rule_description_lists_every_threshold():
    text = describe_dormant_rule(DormantConfig().to_rule())

    assert text == (
        "冷门启动：之前 24 小时成交额 ≤ 100万 USDT 的币，最近 10 分钟涨幅 ≥ 8%、成交额 ≥ 5000 USDT、"
        "至少 5 分钟有成交，涨前 240 分钟平均每分钟波动 ≤ 0.5%、涨幅 ≥ 涨前 240 分钟平均每分钟波动的 10 倍、"
        "单分钟涨幅 ≤ 5%、总实体占比 ≥ 60%；"
        "同一合约距上次提醒不足 60 分钟时，价格比上次提醒再涨 ≥ 3% 才再报「第 N 波」"
    )
