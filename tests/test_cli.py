import asyncio

import pytest

from price_alert import cli
from price_alert.config import AppConfig


@pytest.mark.parametrize(
    ("indicator", "expected_alerts"),
    [
        ({}, 2),
        ({"short_window": {"lookback_seconds": 60}}, 2),
        ({"short_window": {"lookback_seconds": 5, "confirmation_seconds": 5, "min_window_trades": 100}}, 2),
        (
            {
                "candle_interval": "5m",
                "atr_period": 2,
                "warmup_candles": 15,
                "short_window": {"trigger_atr_multiple": 20},
            },
            2,
        ),
        (
            {
                "max_atr_age_seconds": 120,
                "short_window": {"lookback_seconds": 3600},
                "long_window": {"enabled": False},
            },
            1,
        ),
        ({"long_window": {"lookback_seconds": 300, "min_window_trades": 1000, "trigger_atr_multiple": 20}}, 2),
    ],
)
def test_simulate_produces_one_alert_per_enabled_window(indicator, expected_alerts, capsys):
    config = AppConfig.model_validate({"indicator": indicator, "alerts": {"console_colors": False}})

    assert asyncio.run(cli.simulate(config)) == expected_alerts
    output = capsys.readouterr().out
    assert "急涨提醒" in output
    if expected_alerts == 2:
        assert "分钟内价格上涨" in output


def test_missing_config_file_exits_with_readable_message(tmp_path):
    with pytest.raises(SystemExit, match="配置文件不存在"):
        cli._load_config_or_exit(str(tmp_path / "missing.yaml"))


def test_invalid_config_lists_field_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("indicator:\n  atr_period: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"indicator\.atr_period"):
        cli._load_config_or_exit(str(path))


def test_replay_arguments_require_symbol_and_start():
    args = cli.build_parser().parse_args(["replay", "qnt", "--start", "09:25", "--all"])

    assert (args.symbol, args.start, args.end, args.all) == ("qnt", "09:25", None, True)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["replay", "QNT_USDT"])


def test_replay_end_with_only_clock_time_uses_start_date(monkeypatch):
    captured = {}

    def fake_plan(symbol, start, end, config):
        captured.update(symbol=symbol, start=start, end=end)
        raise ValueError("stop")

    monkeypatch.setattr(cli, "plan_replay", fake_plan)
    args = cli.build_parser().parse_args(["replay", "QNT", "--start", "2026-09-28 09:25", "--end", "09:36"])

    with pytest.raises(SystemExit):
        cli.replay(AppConfig(), args)
    assert captured["symbol"] == "QNT_USDT"
    assert captured["end"].date().isoformat() == "2026-09-28"


def test_replay_rejects_end_before_start(tmp_path):
    args = cli.build_parser().parse_args(["replay", "QNT", "--start", "09:30", "--end", "09:20"])

    with pytest.raises(SystemExit, match="结束时间必须晚于开始时间"):
        cli.replay(AppConfig(), args)
