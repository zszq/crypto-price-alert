from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from price_alert.assembly import build_detectors
from price_alert.config import AppConfig, load_config


def test_load_config_and_webhook_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
spike:
  min_volume_24h_quote: 10000001
  short_window:
    trigger_atr_multiple: 1.25
alerts:
  webhook_url: null
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRICE_ALERT_WEBHOOK_URL", "https://example.test/hook")

    config = load_config(path)

    assert config.spike.min_volume_24h_quote == 10_000_001
    assert config.spike.short_window.trigger_atr_multiple == 1.25
    assert config.alerts.webhook_url == "https://example.test/hook"


def test_rejects_atr_configuration_without_enough_warmup_data():
    with pytest.raises(ValidationError, match="warmup_candles"):
        AppConfig.model_validate({"spike": {"atr_period": 20, "warmup_candles": 20}})


def test_noise_protection_defaults_are_enabled():
    config = AppConfig()

    short = config.instant.short_window
    assert short.trigger_atr_multiple == 1.5
    assert short.min_change_percent == 1.0
    assert short.confirmation_seconds == 3
    assert short.cooldown_seconds == 30


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default"


def test_default_config_passes_validation_and_covers_every_section(monkeypatch):
    # 日常监控配置的取值允许偏离代码默认值（运营随时调参），但仍须能通过校验，
    # 否则未知键或越界值要等到启动服务时才暴露。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)

    load_config(DEFAULT_CONFIG)

    # 拆成多个文件后，漏掉一个文件不会报错、只会悄悄退回代码默认值，所以要求每一段都在目录里写明。
    keys = [key for file in DEFAULT_CONFIG.glob("*.yaml") for key in yaml.safe_load(file.read_text(encoding="utf-8"))]
    assert sorted(keys) == sorted(AppConfig.model_fields)


def test_directory_config_merges_top_level_keys_of_every_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    (tmp_path / "common.yaml").write_text("gate:\n  rest_retries: 5\nalerts:\n  beep: false\n", encoding="utf-8")
    (tmp_path / "spike.yaml").write_text("spike:\n  min_volume_24h_quote: 123\n", encoding="utf-8")
    (tmp_path / "empty.yaml").write_text("# 只有注释\n", encoding="utf-8")
    # 只读 .yaml：备份、草稿等其他扩展名不参与合并。
    (tmp_path / "trend.yaml.bak").write_text("trend:\n  unknown: 1\n", encoding="utf-8")

    config = load_config(tmp_path)

    assert (config.gate.rest_retries, config.alerts.beep, config.spike.min_volume_24h_quote) == (5, False, 123)


def test_directory_config_rejects_a_section_split_across_files(tmp_path):
    (tmp_path / "a.yaml").write_text("spike:\n  enabled: true\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("spike:\n  min_volume_24h_quote: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="顶层键 spike 同时出现在 a.yaml 与 b.yaml"):
        load_config(tmp_path)


def test_rejects_empty_directory_and_non_mapping_file(tmp_path):
    with pytest.raises(ValueError, match="没有 .yaml 文件"):
        load_config(tmp_path)

    (tmp_path / "list.yaml").write_text("- spike\n", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层必须是键值对"):
        load_config(tmp_path)


def test_launch_trade_yaml_runs_only_the_launch_detector(monkeypatch):
    # 交易用的配置与日常监控同时运行，只能有放量启动一个检测器，且不能与日常监控写同一个 JSONL。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    config_dir = Path(__file__).resolve().parents[1] / "config"

    config = load_config(config_dir / "launch-trade.yaml")

    assert [entry.name for entry in build_detectors(config)] == ["放量上涨"]
    assert config.alerts.jsonl_path != load_config(DEFAULT_CONFIG).alerts.jsonl_path
    (rule,) = config.launch.rules()
    assert (rule.direction, rule.min_window_quote, rule.require_breakout, rule.max_change_percent) == (
        "surge",
        300_000,
        False,
        None,
    )


def test_launch_f1_yaml_runs_both_launch_directions_on_its_own_file(monkeypatch):
    # 待验证的放量参数与日常监控同时运行对照，只能有放量两个检测器，且不能与其他实例写同一个 JSONL。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    config_dir = Path(__file__).resolve().parents[1] / "config"

    config = load_config(config_dir / "launch-f1.yaml")

    assert [entry.name for entry in build_detectors(config)] == ["放量上涨", "放量下跌"]
    others = {load_config(path).alerts.jsonl_path for path in (DEFAULT_CONFIG, config_dir / "launch-trade.yaml")}
    assert config.alerts.jsonl_path not in others


def test_spike_can_be_disabled_but_not_every_detector():
    names = ["瞬时异动", "短时异动", "放量上涨", "K 线趋势", "冷门启动", "反弹回落"]
    assert [entry.name for entry in build_detectors(AppConfig())] == names
    # 瞬时异动与短时异动各自开关，关掉一个不影响另一个。
    for section, name in (("instant", "瞬时异动"), ("spike", "短时异动")):
        config = AppConfig.model_validate({section: {"enabled": False}})
        assert [entry.name for entry in build_detectors(config)] == [item for item in names if item != name]
    with pytest.raises(ValidationError, match="至少要启用一个"):
        AppConfig.model_validate(
            {
                "instant": {"enabled": False},
                "spike": {"enabled": False},
                "trend": {"enabled": False},
                "launch": {"rise": {"enabled": False}},
                "dormant": {"enabled": False},
                "reversal": {"enabled": False},
            }
        )


def test_rejects_reconnect_max_below_initial_delay():
    with pytest.raises(ValidationError, match="reconnect_max_seconds"):
        AppConfig.model_validate({"gate": {"reconnect_initial_seconds": 10, "reconnect_max_seconds": 5}})


def test_rejects_confirmation_longer_than_lookback():
    # 用精确的报错文字而不只是字段名：未知键的报错里同样会出现字段名，只匹配字段名会被误判为通过。
    # 长窗口的取值范围（窗口 ≥ 60、确认 ≤ 10）使它不可能违反此规则，只需验证短窗口。
    with pytest.raises(ValidationError, match="confirmation_seconds 不能大于 lookback_seconds"):
        AppConfig.model_validate({"spike": {"short_window": {"lookback_seconds": 5, "confirmation_seconds": 6}}})


def test_atr_age_default_follows_candle_interval_but_explicit_value_is_validated():
    assert AppConfig.model_validate({"spike": {"candle_interval": "5m"}}).spike.max_atr_age_seconds == 900

    with pytest.raises(ValidationError, match="max_atr_age_seconds"):
        AppConfig.model_validate({"spike": {"candle_interval": "5m", "max_atr_age_seconds": 180}})


def test_default_config_long_window_cooldown_follows_lookback(monkeypatch, tmp_path):
    # 运营只改长窗口长度时不应被冷却校验卡住，所以日常监控配置刻意不写死冷却。
    monkeypatch.delenv("PRICE_ALERT_WEBHOOK_URL", raising=False)
    spike_yaml = (DEFAULT_CONFIG / "spike.yaml").read_text(encoding="utf-8")
    edited = tmp_path / "config.yaml"
    edited.write_text(spike_yaml.replace("lookback_seconds: 180", "lookback_seconds: 300"), encoding="utf-8")

    assert load_config(edited).spike.long_window.cooldown_seconds == 300


def test_long_window_defaults_and_cooldown_follows_lookback():
    long_window = AppConfig().spike.long_window
    assert long_window.enabled is True
    assert (long_window.lookback_seconds, long_window.cooldown_seconds) == (180, 180)

    # 只改观察窗口长度时冷却自动跟随，不会因为忘改冷却而重复提醒同一段行情。
    configured = AppConfig.model_validate({"spike": {"long_window": {"lookback_seconds": 300}}})
    assert configured.spike.long_window.cooldown_seconds == 300


def test_rejects_long_window_cooldown_shorter_than_its_lookback():
    # 匹配完整报错文字：只匹配字段名的话，键被改名后未知键的报错同样含该字段名，测试会误判通过。
    with pytest.raises(ValidationError, match="cooldown_seconds 不能小于 lookback_seconds"):
        AppConfig.model_validate({"spike": {"long_window": {"lookback_seconds": 300, "cooldown_seconds": 120}}})


def test_rejects_long_window_not_longer_than_short_window():
    with pytest.raises(ValidationError, match="必须大于 short_window.lookback_seconds"):
        AppConfig.model_validate({"spike": {"short_window": {"lookback_seconds": 180}}})

    # 关闭长窗口后不再要求它比短窗口长。
    AppConfig.model_validate(
        {"spike": {"short_window": {"lookback_seconds": 180}, "long_window": {"enabled": False}}}
    )


@pytest.mark.parametrize(
    "legacy",
    [
        {"spike": {"lookback_seconds": 30}},
        {"spike": {"cooldown_seconds": 30}},
        {"alerts": {"cooldown_seconds": 30}},
        # 短时异动段先由 indicator 改名为 move、再改为 spike，合约池门槛从 gate 移到了这一段。
        {"indicator": {"atr_period": 14}},
        {"move": {"enabled": False}},
        {"gate": {"min_volume_24h_quote": 8_000_000}},
    ],
)
def test_legacy_window_keys_are_rejected_instead_of_ignored(legacy):
    # 旧位置直接报错而不是静默忽略，避免以为改了参数其实没生效。
    # 按错误类型判断而不是匹配报错文字：pydantic 的英文提示可能随版本变化。
    with pytest.raises(ValidationError) as caught:
        AppConfig.model_validate(legacy)
    assert [error["type"] for error in caught.value.errors()] == ["extra_forbidden"]


def test_short_window_cooldown_is_configured_inside_the_window():
    config = AppConfig.model_validate({"spike": {"short_window": {"cooldown_seconds": 45}}})

    assert config.spike.short_window.cooldown_seconds == 45


def test_rejects_unknown_long_window_keys():
    with pytest.raises(ValidationError, match="long_window"):
        AppConfig.model_validate({"spike": {"long_window": {"lookback": 300}}})


def test_streak_defaults_and_validation():
    streak = AppConfig().instant.streak
    assert (streak.enabled, streak.min_alerts, streak.gap_tolerance_seconds) == (True, 3, 30)

    # 连续提醒不能反过来卡住原有参数：调大短窗口冷却仍然是合法配置。
    AppConfig.model_validate({"instant": {"short_window": {"cooldown_seconds": 600}}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"instant": {"streak": {"min_alerts": 1}}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"instant": {"streak": {"gap_tolerance_seconds": 0}}})
    # 连续提醒只属于瞬时异动，短时异动段没有这个键。
    with pytest.raises(ValidationError, match="streak"):
        AppConfig.model_validate({"spike": {"streak": {"enabled": False}}})


def test_spike_gates_default_off_and_validate_ranges():
    # 瞬时异动代码默认不判两个附加条件；短时异动代码默认就是研究第四轮方案 A，两个窗口都开。
    config = AppConfig()
    for window in (config.instant.short_window, config.instant.long_window):
        assert (window.min_prior_move_percent, window.min_trade_burst_ratio, window.prior_seconds) == (None, None, 300)
    for window in (config.spike.short_window, config.spike.long_window):
        assert (window.min_prior_move_percent, window.min_trade_burst_ratio, window.prior_seconds) == (2.0, 4.0, 300)
    # 填 null 关掉某一项。
    off = AppConfig.model_validate({"spike": {"short_window": {"min_trade_burst_ratio": None}}})
    assert off.spike.short_window.min_trade_burst_ratio is None
    # 倍数理论上限是 60（全部成交都落在最近 1 分钟），门槛不小于它就永远不会提醒。
    with pytest.raises(ValidationError, match="min_trade_burst_ratio"):
        AppConfig.model_validate({"spike": {"long_window": {"min_trade_burst_ratio": 60}}})
    with pytest.raises(ValidationError, match="min_prior_move_percent"):
        AppConfig.model_validate({"spike": {"short_window": {"min_prior_move_percent": -1}}})


def test_only_detectors_judging_trade_history_ask_for_backfill():
    # 代码默认瞬时异动不判附加条件、不需要补逐秒成交；短时异动（方案 A）与反弹回落都要 1 小时成交暴增。
    entries = {entry.name: entry for entry in build_detectors(AppConfig())}
    assert entries["瞬时异动"].history is None
    for name in ("短时异动", "反弹回落"):
        assert entries[name].history is entries[name].detector
        assert entries[name].history.history_seconds == 3600
    for name in ("K 线趋势", "放量上涨", "冷门启动"):
        assert entries[name].history is None

    # 瞬时异动若也开了附加条件，同样要补。
    gated = AppConfig.model_validate({"instant": {"short_window": {"min_trade_burst_ratio": 4}}})
    instant = next(entry for entry in build_detectors(gated) if entry.name == "瞬时异动")
    assert instant.history is instant.detector and instant.history.history_seconds == 3600

    # 日常配置的瞬时异动（研究第七轮方案 C）只开前置走势，补到最长的长窗口 180 + 300 秒即可，不需要 1 小时。
    daily = {entry.name: entry for entry in build_detectors(load_config(Path("config/default")))}
    assert daily["瞬时异动"].history is daily["瞬时异动"].detector
    assert daily["瞬时异动"].history.history_seconds == 480
    assert daily["短时异动"].history is not None
