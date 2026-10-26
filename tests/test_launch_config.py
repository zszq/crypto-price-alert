import pytest
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig
from price_alert.launch.config import LaunchConfig
from price_alert.universe import VolumeRange


def test_defaults_and_build_rule():
    config = LaunchConfig()

    # 默认只开放量上涨；下跌关闭，但两边门槛的默认值相同。
    assert (config.rise.enabled, config.fall.enabled, config.enabled) == (True, False, True)
    assert [direction for direction, _ in config.sides()] == ["surge"]
    # YAML 键是 rise/fall，方向值（JSONL 的 direction）仍是 surge/drop。
    assert config.configured() == [("surge", config.rise), ("drop", config.fall)]
    rule = config.rise.to_rule("surge")
    assert (config.rise.min_volume_24h_quote, config.rise.max_volume_24h_quote) == (1_000_000, None)
    assert (rule.direction, rule.window_minutes, rule.baseline_minutes, rule.history_minutes) == ("surge", 5, 360, 365)
    assert (rule.min_volume_ratio, rule.min_window_quote, rule.min_change_percent) == (3.0, 100_000, 3.0)
    assert rule.require_breakout is True
    assert (rule.cooldown_minutes, rule.realert_step_percent, rule.max_change_percent) == (15, 5.0, 15.0)
    assert config.fall.model_dump(exclude={"enabled"}) == config.rise.model_dump(exclude={"enabled"})


def test_sides_are_configured_independently():
    config = AppConfig.model_validate(
        {
            "launch": {
                "rise": {"min_change_percent": 4, "max_change_percent": None},
                "fall": {"enabled": True, "baseline_minutes": 180, "min_volume_ratio": 2, "min_window_quote": 50_000},
            }
        }
    ).launch

    rise, drop = config.rules()
    assert (rise.direction, rise.min_change_percent, rise.max_change_percent, rise.baseline_minutes) == (
        "surge",
        4.0,
        None,
        360,
    )
    assert (drop.direction, drop.baseline_minutes, drop.min_volume_ratio, drop.min_window_quote) == (
        "drop",
        180,
        2.0,
        50_000,
    )
    # 没写的项取代码默认值，不受另一个方向影响。
    assert (drop.min_change_percent, drop.max_change_percent) == (3.0, 15.0)


def test_each_side_joins_monitor_as_its_own_detector_and_pool():
    config = AppConfig.model_validate(
        {
            "launch": {
                "rise": {"min_volume_24h_quote": 50_000, "max_volume_24h_quote": 5_000_000},
                "fall": {"enabled": True, "baseline_minutes": 180},
            }
        }
    )

    entries = {entry.name: entry for entry in build_detectors(config)}
    assert list(entries) == ["ATR 异动", "放量上涨", "放量下跌", "K 线趋势", "冷门启动", "反弹回落"]
    assert entries["放量上涨"].volume == VolumeRange(50_000, 5_000_000)
    assert entries["放量下跌"].volume == VolumeRange(1_000_000)
    # 两个方向的基准期不同，各自只保留并预热自己需要的 K 线。
    assert (entries["放量上涨"].candle_interval, entries["放量上涨"].warmup_candles) == ("1m", 366)
    assert entries["放量下跌"].warmup_candles == 186
    disabled = AppConfig.model_validate({"launch": {"rise": {"enabled": False}}})
    assert not disabled.launch.enabled
    assert not {"放量上涨", "放量下跌"} & {entry.name for entry in build_detectors(disabled)}


@pytest.mark.parametrize("side", ["rise", "fall"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"window_minutes": 30, "baseline_minutes": 30}, "baseline_minutes 必须大于 window_minutes"),
        ({"window_minutes": 10, "cooldown_minutes": 5}, "cooldown_minutes 不能小于 window_minutes"),
        ({"min_volume_24h_quote": 100, "max_volume_24h_quote": 100}, "max_volume_24h_quote"),
        ({"min_volume_ratio": 1}, "min_volume_ratio"),
        ({"min_change_percent": 5, "max_change_percent": 5}, "max_change_percent 必须大于 min_change_percent"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_rejects_inconsistent_settings(side, values, message):
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate({"launch": {side: values}})


def test_old_flat_keys_are_rejected():
    # 旧版写在 launch 顶层的键已经移到 rise / fall 下，留在顶层要直接报错，不能悄悄被忽略；
    # 下跌段曾叫 drop，同样要报错，否则旧配置里的下跌参数会被当作没写、退回默认关闭。
    for values in ({"enabled": True}, {"alert_drops": True}, {"min_volume_ratio": 3}, {"drop": {"enabled": True}}):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"launch": values})
