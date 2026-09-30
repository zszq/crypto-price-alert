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
    # 关闭趋势提醒，只验证观察窗口；趋势周期由下面的用例单独覆盖。
    config = AppConfig.model_validate(
        {"indicator": indicator, "trend": {"enabled": False}, "alerts": {"console_colors": False}}
    )

    assert asyncio.run(cli.simulate(config)) == expected_alerts == cli.expected_simulated_alerts(config)
    output = capsys.readouterr().out
    assert "急涨提醒" in output
    if expected_alerts == 2:
        assert "分钟内价格上涨" in output


@pytest.mark.parametrize(
    ("periods", "expected_trend_alerts"),
    [
        ({}, 4),
        ({"1m": {"enabled": False}, "3m": {"enabled": False}}, 2),
        # 启用可选的 ATR 倍数条件、放宽量能到很高的门槛，合成行情仍应满足。
        ({"5m": {"min_change_atr": 50, "min_volume_ratio": 5}, "15m": {"candles": 12, "baseline_candles": 40}}, 4),
    ],
)
def test_simulate_produces_one_start_alert_per_enabled_trend_period(periods, expected_trend_alerts, capsys):
    config = AppConfig.model_validate({"trend": {"periods": periods}, "alerts": {"console_colors": False}})
    windows = 2

    assert asyncio.run(cli.simulate(config)) == windows + expected_trend_alerts
    assert cli.expected_simulated_alerts(config) == windows + expected_trend_alerts
    assert capsys.readouterr().out.count("[趋势下跌]") == expected_trend_alerts


def test_trend_replay_arguments_default_to_three_hours(monkeypatch):
    captured = {}

    def fake_plan(symbol, start, end, detector):
        captured.update(symbol=symbol, start=start, end=end)
        raise ValueError("stop")

    monkeypatch.setattr(cli, "plan_trend_replay", fake_plan)
    args = cli.build_parser().parse_args(["trend-replay", "ark", "--start", "2026-09-30 21:30"])

    with pytest.raises(SystemExit, match="stop"):
        cli.trend_replay(AppConfig(), args)
    assert captured["symbol"] == "ARK_USDT"
    assert (captured["end"] - captured["start"]).total_seconds() == 3 * 3600


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


def test_trend_replay_survives_missing_contract_precision(monkeypatch, capsys):
    from datetime import UTC, datetime, timedelta

    from price_alert.models import Candle

    class FakeRest:
        def fetch_candles_between(self, symbol, interval, start, end):
            count = int((end - start) / timedelta(minutes=1)) + 1
            return [Candle(start + timedelta(minutes=i), 1.0, 1.0, 1.0, 1.0) for i in range(count)]

        def fetch_contract(self, symbol):
            raise ConnectionError("contract down")

    monkeypatch.setattr(cli, "build_rest_client", lambda config: FakeRest())
    start = datetime(2026, 9, 30, 13, 30, tzinfo=UTC).isoformat()
    args = cli.build_parser().parse_args(["trend-replay", "ARK", "--start", start, "--end", "2026-09-30 21:40"])

    cli.trend_replay(AppConfig(), args)

    assert "趋势回放 ARK_USDT" in capsys.readouterr().out


def test_trend_replay_without_enabled_periods_exits_readably():
    periods = {name: {"enabled": False} for name in ("1m", "3m", "5m", "15m")}
    config = AppConfig.model_validate({"trend": {"enabled": False, "periods": periods}})
    args = cli.build_parser().parse_args(["trend-replay", "ARK", "--start", "09:30"])

    with pytest.raises(SystemExit, match="没有启用任何周期"):
        cli.trend_replay(config, args)
