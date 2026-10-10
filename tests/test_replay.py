from datetime import date, datetime, timedelta, timezone

import pytest

from price_alert.config import AppConfig
from price_alert.formatting import BEIJING_TIME
from price_alert.models import Candle, PriceAlert, PriceTick
from price_alert.replay import (
    SETTLE_MARGIN,
    SETTLE_SEARCH_LIMIT,
    ReplayData,
    fetch_replay_data,
    normalize_symbol,
    parse_time,
    plan_replay,
    render_replay,
    run_replay,
)

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
    # 两个窗口互不影响，同一秒都满足时各报一条。
    assert [alert.window for alert in result.alerts] == ["short", "long"]
    assert result.alerts[0].timestamp == result.alerts[1].timestamp
    assert all(alert.price_decimals == 2 for alert in result.alerts)
    assert all(START <= item.second <= END for item in result.evaluations)


def test_render_lists_alerts_rules_and_threshold_rows():
    plan, data = jump_scenario()
    text = "\n".join(render_replay(run_replay(AppConfig(), plan, data)))

    assert "回放 QNT_USDT：2026-09-29 09:27:00 ～ 2026-09-29 09:30:00" in text
    assert "提醒 2 条" in text
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


def tail_jump_trades(plan) -> list[PriceTick]:
    # 区间最后三秒跳到 103，最后一秒正好完成短窗口的 3 秒确认，但它要等下一笔成交才会结算。
    return trades_per_second(
        plan.feed_start, END, lambda moment: 103.0 if moment >= END - timedelta(seconds=2) else 100.0
    )


def test_run_replay_flags_unsettled_last_second():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())

    result = run_replay(AppConfig(), plan, ReplayData(flat_candles(plan), tail_jump_trades(plan)))
    lines = render_replay(result)

    assert result.alerts == []
    assert result.unsettled_second == END
    assert max(item.second for item in result.evaluations) == END - timedelta(seconds=1)
    notice = next(index for index, line in enumerate(lines) if line.startswith("注意：结果不完整"))
    assert lines[notice - 1] == "区间内没有提醒。"
    assert "09:30:00" in lines[notice]


def test_run_replay_settles_last_second_with_late_trade():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    late = PriceTick("QNT_USDT", 103.0, 1.0, END + timedelta(seconds=12), "late")

    result = run_replay(AppConfig(), plan, ReplayData(flat_candles(plan), [*tail_jump_trades(plan), late]))

    assert [alert.window for alert in result.alerts] == ["short", "long"]
    assert result.unsettled_second is None
    assert not any(line.startswith("注意：结果不完整") for line in render_replay(result))


def test_last_second_outside_range_is_not_flagged():
    plan, data = jump_scenario()
    # 成交一直延续到区间之后，区间内的秒全部由后续成交结算，最后那一秒在区间外无关紧要。
    assert run_replay(AppConfig(), plan, data).unsettled_second is None


class FakeReplayRest:
    def __init__(
        self,
        contract_error: Exception | None = None,
        trades: list[PriceTick] | None = None,
    ) -> None:
        self.contract_error = contract_error
        self.trades = trades or []
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
        return [tick for tick in self.trades if start <= tick.timestamp < end]


def test_fetch_replay_data_requests_warmup_candles_and_settling_margin():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    rest = FakeReplayRest()

    data = fetch_replay_data(rest, plan, AppConfig())

    assert data.price_decimals == 2
    assert rest.calls == [
        ("candles", "QNT_USDT", "1m", plan.warmup_start, plan.feed_start - timedelta(minutes=1)),
        ("trades", "QNT_USDT", plan.feed_start, END + SETTLE_MARGIN),
    ]


def test_fetch_replay_data_searches_settling_trade_beyond_margin():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    late = PriceTick("QNT_USDT", 103.0, 1.0, END + timedelta(seconds=45), "late")
    later = PriceTick("QNT_USDT", 90.0, 1.0, END + timedelta(seconds=46), "later")
    rest = FakeReplayRest(trades=[*tail_jump_trades(plan), late, later])

    data = fetch_replay_data(rest, plan, AppConfig())

    # 只补第一笔：结算只看新一秒的第一笔成交。
    assert data.trades[-1] is late
    search = [call[2:] for call in rest.calls if call[0] == "trades"][1:]
    first = END + SETTLE_MARGIN
    assert search == [
        (first, first + timedelta(seconds=10)),
        (first + timedelta(seconds=10), first + timedelta(seconds=30)),
        (first + timedelta(seconds=30), first + timedelta(seconds=70)),
    ]
    assert run_replay(AppConfig(), plan, data).unsettled_second is None


def test_fetch_replay_data_skips_search_when_trades_continue_past_end():
    plan, data = jump_scenario()
    rest = FakeReplayRest(trades=data.trades)

    fetch_replay_data(rest, plan, AppConfig())

    assert [call[0] for call in rest.calls] == ["candles", "trades"]


def test_fetch_replay_data_gives_up_after_search_limit():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    trades = tail_jump_trades(plan)
    rest = FakeReplayRest(trades=trades)

    data = fetch_replay_data(rest, plan, AppConfig())

    assert data.trades == trades
    search = [call[2:] for call in rest.calls if call[0] == "trades"][1:]
    # 逐段翻倍、首尾相接，恰好覆盖到查找上限为止。
    assert search[0][0] == END + SETTLE_MARGIN
    assert all(prev[1] == nxt[0] for prev, nxt in zip(search, search[1:], strict=False))
    assert search[-1][1] == END + SETTLE_MARGIN + SETTLE_SEARCH_LIMIT
    assert len(search) < 10
    assert run_replay(AppConfig(), plan, data).unsettled_second == END


def test_fetch_replay_data_tolerates_missing_contract_precision():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())

    data = fetch_replay_data(FakeReplayRest(ConnectionError("down")), plan, AppConfig())

    assert data.price_decimals is None


def test_parse_time_keeps_explicit_offsets():
    moment = parse_time("2026-09-29 09:27:00+09:00")

    assert moment.utcoffset() == timezone(timedelta(hours=9)).utcoffset(None)


def test_replay_keeps_streak_alert_right_after_its_short_alert():
    plan = plan_replay("QNT_USDT", START, END, AppConfig())
    # 自 START 起每秒涨 0.1：短窗口冷却一过又达标，同向短窗口提醒一条接一条。
    trades = trades_per_second(
        plan.feed_start,
        END + SETTLE_MARGIN,
        lambda moment: 100.0 + max(0.0, (moment - START).total_seconds()) * 0.1,
    )

    result = run_replay(AppConfig(), plan, ReplayData(flat_candles(plan), trades, price_decimals=2))

    kinds = [alert.window if isinstance(alert, PriceAlert) else "streak" for alert in result.alerts]
    assert kinds.count("streak") == 1
    index = kinds.index("streak")
    assert [kind for kind in kinds[:index] if kind == "short"] == ["short"] * 3
    assert kinds[index - 1] == "short" and result.alerts[index].timestamp == result.alerts[index - 1].timestamp
    text = "\n".join(render_replay(result))
    assert "连续提醒：短窗口同方向提醒连续 3 次" in text
    assert "[连续急涨]" in text
