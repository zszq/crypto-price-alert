from datetime import date, datetime, timedelta, timezone

import pytest

from price_alert.config import AppConfig
from price_alert.formatting import BEIJING_TIME
from price_alert.models import Candle, PriceTick
from price_alert.replay import (
    SETTLE_MARGIN,
    ReplayData,
    fetch_replay_data,
    normalize_symbol,
    parse_time,
    plan_replay,
    render_replay,
    run_replay,
)
from price_alert.windows import Outcome

START = datetime(2026, 9, 29, 9, 27, tzinfo=BEIJING_TIME)
END = datetime(2026, 9, 29, 9, 30, tzinfo=BEIJING_TIME)


def test_parse_time_defaults_to_beijing_and_today():
    assert parse_time("2026-09-29 09:27") == START
    assert parse_time("09:27", today=date(2026, 9, 29)) == START
    # 显式带时区的时间保持原样，不被强行改成北京时间。
    utc = parse_time("2026-09-29T01:27:00+00:00")
    assert utc == START and utc.utcoffset() == timedelta(0)
    with pytest.raises(ValueError, match="无法识别"):
        parse_time("昨天早上")


def test_normalize_symbol_appends_usdt_suffix():
    assert normalize_symbol(" qnt ") == "QNT_USDT"
    assert normalize_symbol("btc_usdt") == "BTC_USDT"


def test_plan_reserves_longest_window_and_aligns_to_candle_boundary():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())

    # 最长窗口 180 秒加 2 秒容差，从 09:23:58 向前对齐到 K 线边界。
    assert plan.feed_start == datetime(2026, 9, 29, 9, 23, tzinfo=BEIJING_TIME)
    assert plan.warmup_start == plan.feed_start - timedelta(minutes=50)
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_replay("QNT_USDT", END, START, AppConfig())


def flat_candles(plan, count: int = 50) -> list[Candle]:
    return [
        Candle(plan.feed_start - timedelta(minutes=count - index), 100.0, 100.5, 99.5, 100.0)
        for index in range(count)
    ]


def trades_per_second(first: datetime, last: datetime, price_at) -> list[PriceTick]:
    ticks = []
    moment = first
    index = 0
    while moment <= last:
        ticks.append(PriceTick("QNT_USDT", price_at(moment), 1.0, moment, str(index)))
        moment += timedelta(seconds=1)
        index += 1
    return ticks


def jump_scenario() -> tuple:
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    jump = datetime(2026, 9, 29, 9, 28, tzinfo=BEIJING_TIME)
    trades = trades_per_second(plan.feed_start, END + SETTLE_MARGIN, lambda moment: 103.0 if moment >= jump else 100.0)
    # feed_start 之后开盘的 K 线不能进入预热：之后的行情必须全部由成交重建。
    candles = [*flat_candles(plan), Candle(plan.feed_start, 100.0, 200.0, 50.0, 100.0)]
    return plan, ReplayData(candles, trades, price_decimals=2)


def test_run_replay_reproduces_alert_and_explains_each_second():
    plan, data = jump_scenario()

    result = run_replay(AppConfig(), plan, data)

    assert result.warmup_candles == 50
    assert [alert.window for alert in result.alerts] == ["short"]
    assert result.alerts[0].price_decimals == 2
    assert all(START <= item.second <= END for item in result.evaluations)
    # 同一秒长窗口也满足，但只发一条，长窗口记为被短窗口抢先。
    alert_second = next(item.second for item in result.evaluations if item.outcome is Outcome.ALERT)
    assert any(
        item.window == "long" and item.second == alert_second and item.outcome is Outcome.SUPERSEDED
        for item in result.evaluations
    )


def test_render_lists_alerts_rules_and_threshold_rows():
    plan, data = jump_scenario()
    text = "\n".join(render_replay(run_replay(AppConfig(), plan, data)))

    assert "回放 QNT_USDT：2026-09-29 09:27:00 ～ 2026-09-29 09:30:00" in text
    assert "提醒 1 条" in text
    assert "短窗口：30秒内涨跌 ≥ 1% 且 ≥ 1.5 ATR" in text
    assert "长窗口：3分钟内涨跌 ≥ 2% 且 ≥ 2 ATR" in text
    assert "最接近触发" in text
    assert "达标，连续确认中" in text
    assert "100.00 → 103.00" in text


def test_render_explains_quiet_market_without_alerts():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    trades = trades_per_second(plan.feed_start, END + SETTLE_MARGIN, lambda _moment: 100.0)

    result = run_replay(AppConfig(), plan, ReplayData(flat_candles(plan), trades))
    lines = render_replay(result)

    assert result.alerts == []
    assert "区间内没有提醒。" in lines
    assert any("没有任何一秒同时达到涨跌幅与 ATR 门槛" in line for line in lines)
    assert len(render_replay(result, show_all=True)) > len(lines)


class FakeReplayRest:
    def __init__(self, contract_error: Exception | None = None) -> None:
        self.contract_error = contract_error
        self.calls: list[tuple] = []

    def fetch_contract(self, symbol):
        if self.contract_error is not None:
            raise self.contract_error
        return {"order_price_round": "0.01"}

    def fetch_candles_between(self, symbol, interval, start, end):
        self.calls.append(("candles", symbol, interval, start, end))
        return []

    def fetch_trades(self, symbol, start, end):
        self.calls.append(("trades", symbol, start, end))
        return []


def test_fetch_replay_data_requests_warmup_candles_and_settling_margin():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    rest = FakeReplayRest()

    data = fetch_replay_data(rest, plan, AppConfig())

    assert data.price_decimals == 2
    assert rest.calls == [
        ("candles", "QNT_USDT", "1m", plan.warmup_start, plan.feed_start - timedelta(minutes=1)),
        ("trades", "QNT_USDT", plan.feed_start, END + SETTLE_MARGIN),
    ]


def test_fetch_replay_data_tolerates_missing_contract_precision():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())

    data = fetch_replay_data(FakeReplayRest(ConnectionError("down")), plan, AppConfig())

    assert data.price_decimals is None


def test_parse_time_keeps_explicit_offsets():
    moment = parse_time("2026-09-29 09:27:00+09:00")

    assert moment.utcoffset() == timezone(timedelta(hours=9)).utcoffset(None)
