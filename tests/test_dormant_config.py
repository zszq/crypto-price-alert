import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig
from price_alert.detectors.dormant.alerts import DormantAlert, describe_dormant_rule, format_dormant_alert
from price_alert.detectors.dormant.config import DormantConfig
from price_alert.detectors.dormant.rule import DormantGroup, DormantMetrics, DormantRule
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
        # 效率、量比、回落只有慢启动一组有。
        ({"min_efficiency": 0.3}, "Extra inputs"),
        ({"min_volume_ratio": 3}, "Extra inputs"),
        ({"max_retrace": 0.3}, "Extra inputs"),
        ({"follow_step_percent": 5}, "Extra inputs"),
    ],
)
def test_rejects_inconsistent_values(values, message):
    with pytest.raises(ValidationError, match=message):
        DormantConfig.model_validate(values)


def test_joins_monitor_as_its_own_detector_and_pool():
    entries = {entry.name: entry for entry in build_detectors(AppConfig())}

    # 冷门启动之后只有反弹回落（顺序固定）。
    assert list(entries)[-2:] == ["冷门启动", "反弹回落"]
    entry = entries["冷门启动"]
    assert entry.volume == VolumeRange(0, 3_000_000)
    assert (entry.candle_interval, entry.warmup_candles) == ("1m", 1440 + 10 + 1)
    disabled = AppConfig.model_validate({"dormant": {"enabled": False}})
    assert "冷门启动" not in [entry.name for entry in build_detectors(disabled)]


def make_alert() -> DormantAlert:
    now = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
    metrics = DormantMetrics(
        0.01542, 0.01685, 9.27, 51_156.0, 8, 423_566.0, 0.3, 30.9, 3.9, 0.82, 6_120.0, 0.55, 0.6, 12.0, 0.1, ()
    )
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


def test_slow_group_is_off_by_default_and_uses_the_research_shape():
    config = DormantConfig()
    assert not config.slow.enabled
    assert config.rules() == (config.to_rule(),)

    slow = DormantConfig.model_validate({"slow": {"enabled": True}})
    # research/launch_trade/dormant_slow.md 第三节第 2 小节按 JCT 形态挑的第一组。
    assert slow.rules() == (
        config.to_rule(),
        DormantRule(
            window_minutes=60,
            max_prior_quote=1_000_000,
            min_window_quote=10_000,
            min_active_minutes=48,
            min_change_percent=6.0,
            max_minute_change_percent=3.0,
            min_body_ratio=0.0,
            min_calm_multiple=0.0,
            max_pre_volatility_percent=0.2,
            min_late_quote_share=0.5,
            min_efficiency=0.3,
            min_volume_ratio=3.0,
            max_retrace=0.3,
            cooldown_minutes=120,
            realert_step_percent=None,
            group=DormantGroup.SLOW,
        ),
    )
    # 整个冷门启动关闭时，慢启动一组也不报。
    assert DormantConfig.model_validate({"enabled": False, "slow": {"enabled": True}}).rules() == ()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"window_minutes": 121}, "less than or equal to 120"),
        ({"min_active_minutes": 61}, "min_active_minutes 不能大于 window_minutes"),
        ({"max_retrace": 1.5}, "less than or equal to 1"),
        ({"follow_step_percent": 0}, "greater than 0"),
        ({"follow_minutes": 0}, "greater than or equal to 1"),
        ({"min_volume_24h_quote": 0}, "Extra inputs"),
    ],
)
def test_slow_group_rejects_inconsistent_values(values, message):
    with pytest.raises(ValidationError, match=message):
        DormantConfig.model_validate({"slow": values})


def test_slow_group_shares_the_detector_and_pool_with_longer_warmup():
    config = AppConfig.model_validate({"dormant": {"slow": {"enabled": True}}})
    entries = [entry for entry in build_detectors(config) if entry.name.startswith("冷门")]

    assert [entry.name for entry in entries] == ["冷门启动"]
    assert entries[0].warmup_candles == 1440 + 60 + 1
    assert [rule.group for rule in entries[0].detector.rules] == [DormantGroup.MAIN, DormantGroup.SLOW]


def test_slow_alert_label_and_fields():
    alert = replace(make_alert(), group=DormantGroup.SLOW, wave=1)

    assert format_dormant_alert(alert).startswith("[冷门慢启动] ")
    data = alert.to_dict()
    assert (data["group"], data["efficiency"], data["volume_ratio"], data["retrace"]) == ("slow", 0.6, 12.0, 0.1)
    # 之前 24 小时没有成交时量比为无穷大，写成 null。
    flat = replace(alert, metrics=replace(alert.metrics, volume_ratio=math.inf))
    assert flat.to_dict()["volume_ratio"] is None
    assert make_alert().to_dict()["group"] == "main"


def test_slow_rule_description_lists_only_active_thresholds():
    text = describe_dormant_rule(DormantConfig.model_validate({"slow": {"enabled": True}}).slow.to_rule())

    assert text == (
        "冷门慢启动：之前 24 小时成交额 ≤ 100万 USDT 的币，最近 60 分钟涨幅 ≥ 6%、成交额 ≥ 1万 USDT、"
        "至少 48 分钟有成交，涨前 240 分钟平均每分钟波动 ≤ 0.2%、单分钟涨幅 ≤ 3%、后 30 分钟成交额占比 ≥ 50%、"
        "效率（净涨幅 ÷ 逐分钟涨跌路程）≥ 0.3、量比（相对之前 24 小时同长度均值）≥ 3 倍、"
        "从窗口最高价回落 ≤ 窗口最大涨幅的 30%；同一合约 120 分钟内只提醒一次"
    )


def test_slow_follow_is_off_by_default_and_described_when_enabled():
    assert DormantConfig().slow.to_rule().follow_step_percent is None
    rule = DormantConfig.model_validate({"slow": {"follow_step_percent": 5, "follow_minutes": 240}}).slow.to_rule()

    assert (rule.follow_step_percent, rule.follow_minutes) == (5.0, 240)
    assert describe_dormant_rule(rule).endswith(
        "同一合约 120 分钟内只提醒一次；提醒后跟踪价格，比上次提醒（含补报）再涨 ≥ 5% 就补报「第 N 波」"
        "（不再检查其他条件），距上次提醒超过 240 分钟没再涨够就停止跟踪"
    )
