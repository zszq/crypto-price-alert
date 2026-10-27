from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from price_alert import cli
from price_alert.assembly import build_detectors
from price_alert.common.formatting import BEIJING_TIME, beijing_time
from price_alert.common.models import Candle, PriceTick
from price_alert.config import AppConfig
from price_alert.detectors.replay import ReplayData
from price_alert.detectors.reversal.config import ReversalConfig
from price_alert.detectors.reversal.detector import ReversalDetector
from price_alert.detectors.reversal.replay import plan_reversal_replay, render_reversal_replay, run_reversal_replay
from price_alert.detectors.reversal.rule import CANDLE_INTERVAL as REVERSAL_INTERVAL
from price_alert.detectors.reversal.simulate import simulate_reversal
from price_alert.exchange.universe import VolumeRange
from price_alert.notify import sounds
from price_alert.notify.notifier import REVERSAL_LABEL_COLOR, colorize_alert, format_alert

BASE = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "LAB_USDT"
# 时间线（秒）：平稳到 3700，3700~4590 走一段前置行情，4590 起停在低位（高位），4600 起成交涌入，4650 急动。
FLAT_END = 3700
TREND_END = 4590
BUSY_START = 4600
MOVE = 4650


def detector() -> ReversalDetector:
    return ReversalDetector(ReversalConfig().to_rule())


def warmup_candles() -> list[Candle]:
    return [Candle(BASE - timedelta(minutes=15 - index), 100.0, 100.05, 99.95, 100.0) for index in range(15)]


def trades(*, prior_move: float = -9.0, jump: float = 2.7, busy: int = 40, first: int = 0) -> list[PriceTick]:
    """prior_move 是前置行情的涨跌幅（%），jump 是急动幅度（%，方向与前置行情相反）。"""
    turn = 100.0 * (1 + prior_move / 100)
    moved = turn * (1 - jump / 100) if prior_move > 0 else turn * (1 + jump / 100)
    ticks = []
    for second in range(first, MOVE + 2):
        if second < FLAT_END:
            price = 100.0
        elif second < TREND_END:
            price = 100.0 + (turn - 100.0) * (second - FLAT_END) / (TREND_END - FLAT_END)
        elif second < MOVE:
            price = turn
        else:
            price = moved
        count = busy if second >= BUSY_START else 1
        for index in range(count):
            moment = BASE + timedelta(seconds=second + index / count)
            ticks.append(PriceTick(SYMBOL, price, 1.0, moment, f"{second}-{index}"))
    return ticks


def feed(instance: ReversalDetector, **kwargs):
    """喂入 trades(**kwargs) 生成的行情，返回全部提醒。"""
    instance.add_symbol(SYMBOL, warmup_candles(), 50_000_000, price_decimals=2)
    alerts = []
    for tick in trades(**kwargs):
        alerts.extend(instance.add_tick(tick))
    return alerts


def test_rebound_after_drop_with_rush_of_trades_alerts_once():
    alerts = feed(detector())

    assert len(alerts) == 1
    alert = alerts[0]
    assert (alert.label, alert.direction, alert.color) == ("下跌反弹", "surge", "green")
    # 窗口起点前 15 分钟（第 3740 秒）还在前置行情的开头，到低位 91 跌了约 8.6%。
    assert alert.prior_change_percent == pytest.approx((91 / (100 - 9 * 40 / 890) - 1) * 100)
    assert alert.trade_burst_ratio > 16
    assert alert.change_percent == pytest.approx(2.7)
    text = format_alert(alert)
    assert text.startswith("[下跌反弹]") and "10秒内上涨 2.70%（此前15分钟下跌 8.6" in text
    assert "91.00 → 93.46" in text


def test_pullback_after_rise_alerts_as_drop():
    (alert,) = feed(detector(), prior_move=9.0)

    assert (alert.label, alert.direction, alert.color) == ("冲高回落", "drop", "red")
    assert alert.prior_change_percent > 6
    assert "10秒内下跌 2.70%（此前15分钟上涨" in format_alert(alert)


def test_move_without_a_big_reverse_move_before_is_not_reported():
    # 此前只跌了 3%，不够 6%：顺着平淡走势的急拉多半回吐。
    assert feed(detector(), prior_move=-3.0) == []


def test_move_without_a_rush_of_trades_is_not_reported():
    assert feed(detector(), busy=1) == []


def test_needs_an_hour_of_trades_before_reporting():
    # 从第 1200 秒才有成交：急动时回看 1 小时没有记录，宁可不报也不拿不到 1 小时的平均充数。
    assert feed(detector(), first=1200) == []


def test_history_is_bounded_and_dropped_with_the_symbol():
    instance = detector()
    feed(instance, busy=1)

    # 只需留 1 小时（加窗口与前置回看中较长者）左右的逐秒记录。
    assert len(instance._history[SYMBOL]) <= 3602
    instance.remove_symbols({SYMBOL})
    assert SYMBOL not in instance._history


def test_structured_fields():
    (alert,) = feed(detector())
    payload = alert.to_dict()

    assert payload["kind"] == "reversal"
    assert payload["direction"] == "surge" and payload["color"] == "green"
    assert payload["prior_seconds"] == 900 and payload["lookback_seconds"] == 10
    assert payload["timestamp"] == (BASE + timedelta(seconds=MOVE + 1)).isoformat()
    assert set(payload) >= {"prior_change_percent", "trade_burst_ratio", "reference_price", "price", "move_atr"}


def test_console_label_color_and_sound():
    (alert,) = feed(detector())

    assert f"{REVERSAL_LABEL_COLOR}[下跌反弹]" in colorize_alert(alert, format_alert(alert))
    assert sounds.sound_for(alert) == "reversal"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"lookback_seconds": 5, "confirmation_seconds": 6}, "confirmation_seconds 不能大于 lookback_seconds"),
        # 最近 1 分钟最多占 1 小时的全部成交，倍数到不了 60。
        ({"min_trade_burst_ratio": 60}, "min_trade_burst_ratio"),
        ({"min_volume_24h_quote": 100, "max_volume_24h_quote": 100}, "max_volume_24h_quote"),
        ({"prior_seconds": 30}, "prior_seconds"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_config_rejects_invalid_values(values, message):
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate({"reversal": values})


def test_joins_monitor_as_its_own_detector_and_pool():
    entries = {entry.name: entry for entry in build_detectors(AppConfig())}

    entry = entries["反弹回落"]
    assert entry.volume == VolumeRange(8_000_000)
    assert (entry.candle_interval, entry.warmup_candles) == ("1m", 50)
    disabled = AppConfig.model_validate({"reversal": {"enabled": False}})
    assert "反弹回落" not in [entry.name for entry in build_detectors(disabled)]


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"lookback_seconds": 60, "confirmation_seconds": 5, "min_trade_burst_ratio": 40, "prior_seconds": 60},
        {"min_prior_move_percent": 50, "min_window_trades": 300, "trigger_atr_multiple": 5, "cooldown_seconds": 0},
    ],
)
def test_simulate_produces_exactly_one_alert(values):
    alerts = simulate_reversal(ReversalConfig(**values))

    assert len(alerts) == 1 and alerts[0].label == "下跌反弹"


def replay_plan():
    # 区间只覆盖急动前后，回放要自己往前取 1 小时加前置回看的成交。
    start, end = BASE + timedelta(seconds=4640), BASE + timedelta(seconds=4660)
    return plan_reversal_replay(SYMBOL, start, end, ReversalConfig().to_rule())


def test_replay_plan_feeds_an_hour_and_prior_lookback_before_start():
    plan = replay_plan()

    # 开始时刻减 1 小时（长于 10 秒窗口 + 15 分钟前置）再减 2 秒容差，向前对齐到分钟。
    assert plan.feed_start == datetime.fromtimestamp((int(plan.start.timestamp()) - 3602) // 60 * 60, tz=BEIJING_TIME)
    assert plan.warmup_start == plan.feed_start - timedelta(minutes=50)
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        plan_reversal_replay(SYMBOL, plan.end, plan.start, ReversalConfig().to_rule())


def test_replay_reports_the_alert_and_why_it_fired():
    plan = replay_plan()
    data = ReplayData(warmup_candles(), trades(), price_decimals=2)

    result = run_reversal_replay(ReversalConfig().to_rule(), plan, data)
    text = "\n".join(render_reversal_replay(result))

    assert len(result.alerts) == 1 and result.alerts[0].label == "下跌反弹"
    assert "提醒 1 条：" in text and "[下跌反弹]" in text
    assert "急动达标、可以提醒 1 秒" in text
    move_second = beijing_time(BASE + timedelta(seconds=MOVE), "%H:%M:%S")
    row = next(line for line in text.splitlines() if line.strip().startswith(move_second))
    assert "此前15分钟 -8.6" in row and "成交 ×" in row and row.endswith("提醒")


def test_replay_explains_a_move_blocked_by_missing_trade_rush():
    plan = replay_plan()
    data = ReplayData(warmup_candles(), trades(busy=1), price_decimals=2)

    result = run_reversal_replay(ReversalConfig().to_rule(), plan, data)
    text = "\n".join(render_reversal_replay(result))

    assert result.alerts == []
    assert "区间内没有提醒。" in text
    assert "成交笔数未暴增（或不足 1 小时历史） 1 秒" in text
    assert "成交 ×1.0" in text


def test_replay_command_fetches_one_minute_candles(monkeypatch, capsys):
    args = cli.build_parser().parse_args(["reversal-replay", "LAB", "--start", "2026-01-01 09:30", "--all"])
    assert (args.symbol, args.all) == ("LAB", True)

    rest = FakeRest()
    monkeypatch.setattr(cli, "build_rest_client", lambda config: rest)
    # 秒级异动改用 5 分钟 K 线时，反弹回落的回放仍按自己固定的 1 分钟预热。
    cli.reversal_replay(AppConfig.model_validate({"move": {"candle_interval": "5m"}}), args)
    assert rest.intervals == [REVERSAL_INTERVAL] == ["1m"]


class FakeRest:
    def __init__(self) -> None:
        self.intervals: list[str] = []

    def fetch_contract(self, symbol):
        return {"order_price_round": "0.01"}

    def fetch_candles_between(self, symbol, interval, start, end):
        self.intervals.append(interval)
        return []

    def fetch_trades(self, symbol, start, end):
        return [PriceTick(SYMBOL, 100.0, 1.0, end + timedelta(seconds=1), "after")]


def test_stream_gap_clears_history_so_an_hour_is_needed_again():
    # 断线后累计笔数从 0 重新计数、断线期间成交缺失，留着旧记录会算出负的或偏大的倍数，所以清空重攒。
    instance = detector()
    instance.add_symbol(SYMBOL, warmup_candles(), 50_000_000)
    for tick in trades()[:4000]:
        instance.add_tick(tick)
    assert instance._history[SYMBOL]

    instance.mark_stream_gap()
    assert SYMBOL not in instance._history
    for tick in trades()[4000:4001]:
        instance.add_tick(tick)
    instance.mark_symbol_gap(SYMBOL.lower())
    assert SYMBOL not in instance._history


def day_candles(low: float = 99.95) -> list[Candle]:
    """BASE 之前整 24 小时（另加 2 根）的平稳 1 分钟 K 线，10 小时前那根的最低价为 low。"""
    candles = [Candle(BASE - timedelta(minutes=1442 - index), 100.0, 100.05, 99.95, 100.0) for index in range(1442)]
    candles[-600] = Candle(candles[-600].timestamp, 100.0, 100.05, low, 100.0)
    return candles


def extreme_detector(limit: float = 10.0) -> ReversalDetector:
    return ReversalDetector(ReversalConfig(max_extreme_distance_percent=limit).to_rule())


def feed_day(instance: ReversalDetector, candles: list[Candle], **kwargs):
    instance.add_symbol(SYMBOL, candles, 50_000_000, price_decimals=2)
    alerts = []
    for tick in trades(**kwargs):
        alerts.extend(instance.add_tick(tick))
    return alerts


def test_extreme_distance_measured_from_the_low_of_closed_candles():
    # 低位 91 落在急拉之前已收盘的 K 线里：急拉到 93.46 离 24 小时最低价 2.7%，在 10% 以内。
    (alert,) = feed_day(extreme_detector(), day_candles())

    assert alert.extreme_distance_percent == pytest.approx(93.457 / 91 * 100 - 100, abs=1e-3)
    assert "距24小时低点 2.70%" in format_alert(alert)
    assert alert.to_dict()["extreme_distance_percent"] == alert.extreme_distance_percent


def test_rebound_far_above_the_day_low_is_not_reported():
    # 10 小时前探到过 80：急拉后的价格离 24 小时最低价 16.8%，反弹已不是「刚从低点拉起」。
    assert feed_day(extreme_detector(), day_candles(low=80.0)) == []
    assert feed_day(extreme_detector(limit=2.0), day_candles()) == []


def test_pullback_measured_from_the_day_high():
    (alert,) = feed_day(extreme_detector(), day_candles(), prior_move=9.0)

    assert alert.extreme_distance_percent == pytest.approx(2.7, abs=1e-3)
    assert "距24小时高点 2.70%" in format_alert(alert)


def test_needs_a_full_day_of_candles_when_extreme_distance_is_set():
    # 只有 15 根预热 K 线：不满 24 小时不知道当天极值在哪，宁可不报。
    assert feed_day(extreme_detector(), warmup_candles()) == []
    # 不配置这一条件时不记 K 线，行为与之前相同。
    instance = detector()
    assert len(feed_day(instance, warmup_candles())) == 1
    assert instance._bars == {}


def test_closed_candles_are_bounded_and_rebuilt_after_a_gap():
    instance = extreme_detector()
    feed_day(instance, day_candles(), busy=1)
    bars = instance._bars[SYMBOL]
    # 只留最近 24 小时再加一根；跨入新分钟时追加刚收盘的那根。
    assert len(bars) == 1441
    assert bars[-1][0] == BASE + timedelta(minutes=MOVE // 60 - 1)

    instance.mark_stream_gap()
    assert SYMBOL not in instance._bars
    live = Candle(BASE + timedelta(minutes=MOVE // 60), 91.0, 91.0, 91.0, 91.0)
    assert instance.resync_symbol(SYMBOL, day_candles(), live, BASE + timedelta(seconds=MOVE))
    assert len(instance._bars[SYMBOL]) == 1441
    instance.mark_symbol_gap(SYMBOL)
    assert SYMBOL not in instance._bars


def test_extreme_distance_needs_a_day_of_warmup_and_must_exceed_the_move():
    config = AppConfig.model_validate({"reversal": {"max_extreme_distance_percent": 10}})
    entry = next(entry for entry in build_detectors(config) if entry.name == "反弹回落")

    assert entry.warmup_candles == 1442
    # 距离不超过急动门槛时，基准价落在已收盘 K 线里的急动都报不出来。
    with pytest.raises(ValidationError, match="max_extreme_distance_percent 必须大于 min_change_percent"):
        AppConfig.model_validate({"reversal": {"max_extreme_distance_percent": 1.5}})


@pytest.mark.parametrize(
    "values",
    [
        {"max_extreme_distance_percent": 10, "confirmation_seconds": 2},
        {"max_extreme_distance_percent": 1.6},
        {"max_extreme_distance_percent": 30, "lookback_seconds": 60, "prior_seconds": 60},
    ],
)
def test_simulate_with_extreme_distance_produces_exactly_one_alert(values):
    alerts = simulate_reversal(ReversalConfig(**values))

    assert len(alerts) == 1 and alerts[0].extreme_distance_percent <= values["max_extreme_distance_percent"]


def test_replay_shows_extreme_distance_and_warms_up_a_day():
    rule = ReversalConfig(max_extreme_distance_percent=2.0).to_rule()
    start, end = BASE + timedelta(seconds=4640), BASE + timedelta(seconds=4660)
    plan = plan_reversal_replay(SYMBOL, start, end, rule)
    assert plan.warmup_start == plan.feed_start - timedelta(minutes=1442)

    feed_start = plan.feed_start.astimezone(UTC)
    data = ReplayData([c for c in day_candles() if c.timestamp < feed_start], trades(), price_decimals=2)
    result = run_reversal_replay(rule, plan, data)
    text = "\n".join(render_reversal_replay(result))

    assert result.alerts == []
    assert "离 24 小时反向极值太远（或 K 线不足 24 小时） 1 秒" in text
    assert "距24小时低点 +2.70%" in text and "离 24 小时反向极值 ≤ 2%" in text
