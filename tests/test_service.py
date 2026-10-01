import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from price_alert import service
from price_alert.assembly import build_detector, build_trend_detector
from price_alert.config import AppConfig
from price_alert.detection import MonitoredDetector
from price_alert.gate import GateTradeFeed
from price_alert.models import Candle, PriceTick

BASE = datetime(2026, 1, 1, tzinfo=UTC)


class StopLoop(BaseException):
    """跳出无限循环专用；继承 BaseException 以免被服务里的 except Exception 吞掉。"""


def record_sleeps(monkeypatch, limit: int) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= limit:
            raise StopLoop

    monkeypatch.setattr(service.asyncio, "sleep", fake_sleep)
    return delays


def semaphore() -> asyncio.Semaphore:
    """生产代码里两个循环共用一个信号量，测试各用例独立构造即可。"""
    return asyncio.Semaphore(8)


def config() -> AppConfig:
    return AppConfig.model_validate(
        {"gate": {"reconnect_initial_seconds": 1, "reconnect_max_seconds": 4, "universe_refresh_seconds": 600}}
    )


def monitor(detector, rest=None, feed=None, app_config=None, sem=None) -> service._Monitor:
    """各用例只触达自己关心的依赖，其余传占位对象，被误用时会直接报错而不是悄悄通过。

    detector 传单个检测器时按 ATR 检测器的预热口径包装；传列表时原样使用。
    """
    app_config = config() if app_config is None else app_config
    if isinstance(detector, list):
        detectors = tuple(detector)
    else:
        indicator = app_config.indicator
        detectors = (MonitoredDetector("ATR 异动", detector, indicator.candle_interval, indicator.warmup_candles),)
    return service._Monitor(
        detectors=detectors,
        rest=object() if rest is None else rest,
        feed=object() if feed is None else feed,
        config=app_config,
        semaphore=semaphore() if sem is None else sem,
    )


class FakeDetector:
    def __init__(self) -> None:
        self.gaps = 0

    @property
    def symbols(self) -> list[str]:
        return ["BTC_USDT"]

    @property
    def stale_symbols(self) -> list[str]:
        return ["BTC_USDT"] if self.gaps else []

    def add_tick(self, tick: PriceTick) -> list[str]:
        return [f"alert-{tick.trade_id}"]

    def mark_stream_gap(self) -> None:
        self.gaps += 1


class ScriptedFeed:
    """每次 stream() 依次执行一个脚本：先产出若干成交，再抛出指定异常。

    成交时间戳取当前时间，避免被 stream_loop 的滞后熔断当成积压数据断开；
    需要构造滞后成交的用例传 lag_seconds。
    """

    def __init__(self, scripts, lag_seconds: float = 0.0) -> None:
        self.scripts = list(scripts)
        self.lag_seconds = lag_seconds
        self.rejected_symbols: set[str] = set()
        self.symbols = ["BTC_USDT"]

    async def stream(self):
        ticks, error = self.scripts.pop(0)
        timestamp = datetime.now(UTC) - timedelta(seconds=self.lag_seconds)
        for trade_id in ticks:
            yield PriceTick("BTC_USDT", 100.0, 1.0, timestamp, trade_id)
        # 让出一次事件循环，使连接期间启动的后台回补任务真正开始运行。
        wakeup = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(wakeup.set_result, None)
        await wakeup
        raise error


class FakeDispatcher:
    def __init__(self) -> None:
        self.published: list[str] = []

    def publish(self, alert) -> None:
        self.published.append(alert)


def test_network_timeouts_use_backoff_mark_gap_and_resync_only_while_connected(monkeypatch, caplog):
    delays = record_sleeps(monkeypatch, limit=3)
    resync_started: list[int] = []
    resync_cancelled: list[int] = []

    async def fake_resync(_monitor):
        resync_started.append(len(resync_started))
        try:
            await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError:
            resync_cancelled.append(len(resync_cancelled))
            raise

    monkeypatch.setattr(service._Monitor, "resync_stale_symbols", fake_resync)
    detector, dispatcher = FakeDetector(), FakeDispatcher()
    feed = ScriptedFeed(
        [
            # 握手或 TCP 连接超时抛出的就是 TimeoutError，必须按故障退避而不是当作刷新周期。
            ((), TimeoutError("timed out during opening handshake")),
            (("1", "2"), ConnectionError("closed")),
            ((), TimeoutError("connect timeout")),
        ]
    )

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed).stream_loop(dispatcher))

    # 第二次连接收到成交后退避重置为初始值。
    assert delays == [1, 1, 2]
    # 三次断线各标记一次，第二次连接恢复时再标记一次，覆盖断线期间新入池的合约。
    assert detector.gaps == 4
    # 只有第二次连接真正收到成交时才开始回补，且断线时回补被取消。
    assert resync_started == [0]
    assert resync_cancelled == [0]
    assert dispatcher.published == ["alert-1", "alert-2"]
    assert "timed out during opening handshake" in caplog.text
    assert "刷新周期" not in caplog.text


def test_first_stale_tick_breaks_before_publishing(monkeypatch, caplog):
    """连接后的第一笔成交就滞后时必须立即熔断：不能留出放行积压成交的时间窗口。

    刻意不替换单调时钟——按时间间隔抽查的实现会让首笔落进盲区，用跳变的假时钟
    反而会掩盖它。
    """
    delays = record_sleeps(monkeypatch, limit=1)
    detector, dispatcher = FakeDetector(), FakeDispatcher()
    feed = ScriptedFeed([(("1", "2"), ConnectionError("unused"))], lag_seconds=30)

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed).stream_loop(dispatcher))

    assert "行情数据滞后" in caplog.text
    assert "30.0 秒" in caplog.text
    # 校验在判定之前，且首笔即熔断，超限的成交一笔都不会产生已经过期的提醒。
    assert dispatcher.published == []
    # 熔断走正常的断线路径：清空秒级窗口并按初始退避重连。
    assert detector.gaps == 1
    assert delays == [1]


def test_backlog_burst_within_one_second_is_not_let_through(monkeypatch):
    """同一秒内涌入的大批积压成交必须全部拦下，不能只抽查其中一笔。"""
    record_sleeps(monkeypatch, limit=1)
    detector, dispatcher = FakeDetector(), FakeDispatcher()
    # 一次性给出 50 笔滞后成交，模拟积压排空时的突发。
    feed = ScriptedFeed([(tuple(str(i) for i in range(50)), ConnectionError("unused"))], lag_seconds=30)

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed).stream_loop(dispatcher))

    assert dispatcher.published == []


def test_persistent_lag_keeps_backing_off(monkeypatch, caplog):
    """每次重连的首笔都滞后时（拥塞未缓解或本地时钟偏移）退避必须继续增长。

    滞后的成交不算连接可用的证据，否则退避被反复重置，会变成每秒一次重连、
    每次都重新订阅全部合约，把已经拥塞的链路压得更死。
    """
    # “连接成功”是 INFO，caplog 默认只收 WARNING 及以上，不降级这条断言就永远为真。
    caplog.set_level(logging.INFO, logger="price_alert.service")
    delays = record_sleeps(monkeypatch, limit=3)
    detector, dispatcher = FakeDetector(), FakeDispatcher()
    feed = ScriptedFeed([(("1",), ConnectionError("unused"))] * 3, lag_seconds=30)

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed).stream_loop(dispatcher))

    # 一直没有可用行情，不应报出“连接成功”。
    assert "连接成功" not in caplog.text
    assert delays == [1, 2, 4]


def test_fresh_market_data_does_not_trigger_lag_breaker(monkeypatch):
    """数据新鲜时不得误断连接。"""
    record_sleeps(monkeypatch, limit=1)
    detector, dispatcher = FakeDetector(), FakeDispatcher()
    feed = ScriptedFeed([(("1", "2", "3"), ConnectionError("down"))], lag_seconds=0)

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed).stream_loop(dispatcher))

    assert dispatcher.published == ["alert-1", "alert-2", "alert-3"]


class FakeRest:
    def __init__(self, tickers, contracts, fail_times: int = 0) -> None:
        self.tickers = tickers
        self.contracts = contracts
        self.fail_times = fail_times
        self.candle_requests: list[str] = []

    def fetch_tickers(self):
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("rest down")
        return self.tickers

    def fetch_contracts(self):
        return self.contracts

    def fetch_candles(self, symbol, interval, limit):
        self.candle_requests.append(symbol)
        return []


def ticker(symbol: str, volume: float) -> dict:
    return {"contract": symbol, "last": "100", "volume_24h_quote": str(volume)}


def contract(symbol: str) -> dict:
    return {"name": symbol, "contract_type": "", "status": "trading"}


def test_refresh_warms_new_symbols_keeps_hysteresis_and_updates_feed_without_reconnect():
    async def scenario():
        app_config = config()
        detector = build_detector(app_config)
        feed = GateTradeFeed("wss://example.test")
        first = FakeRest(
            [ticker("BTC_USDT", 20e6), ticker("ETH_USDT", 20e6)],
            [contract("BTC_USDT"), contract("ETH_USDT")],
        )
        await monitor(detector, first, feed, app_config).refresh_universe()

        # ETH 成交额回落到门槛与退出门槛之间应保留，BTC 跌破退出门槛被移除，SOL 新入池需要预热。
        second = FakeRest(
            [ticker("BTC_USDT", 5e6), ticker("ETH_USDT", 9e6), ticker("SOL_USDT", 20e6)],
            [contract("BTC_USDT"), contract("ETH_USDT"), contract("SOL_USDT")],
        )
        await monitor(detector, second, feed, app_config).refresh_universe()
        return detector, feed, first, second

    detector, feed, first, second = asyncio.run(scenario())

    # 新合约经 to_thread 并发预热，请求先后取决于线程调度，只校验请求集合。
    assert sorted(first.candle_requests) == ["BTC_USDT", "ETH_USDT"]
    assert second.candle_requests == ["SOL_USDT"]
    assert detector.symbols == ["ETH_USDT", "SOL_USDT"]
    assert feed.symbols == ["ETH_USDT", "SOL_USDT"]


def test_warmup_logs_progress_and_failures(monkeypatch, caplog):
    """预热耗时较长时要持续输出进度，否则启动阶段看起来像假死。"""
    caplog.set_level(logging.INFO, logger="price_alert.service")
    # 节流间隔置零，让每完成一个合约都满足报告条件，不依赖真实耗时。
    monkeypatch.setattr(service, "_WARMUP_PROGRESS_SECONDS", 0.0)
    symbols = ("BTC_USDT", "ETH_USDT", "SOL_USDT")

    class FailingEthRest(FakeRest):
        def fetch_candles(self, symbol, interval, limit):
            if symbol == "ETH_USDT":
                raise ConnectionError("candles down")
            return super().fetch_candles(symbol, interval, limit)

    rest = FailingEthRest([ticker(symbol, 20e6) for symbol in symbols], [contract(symbol) for symbol in symbols])
    app_config = config()
    detector = build_detector(app_config)
    asyncio.run(monitor(detector, rest, GateTradeFeed("wss://example.test"), app_config).initial_universe())

    assert "正在从 Gate.io 获取合约列表" in caplog.text
    assert "开始为 3 个新合约拉取 K 线预热（ATR 异动，并发 8）" in caplog.text
    # 最后一个完成时由“预热完成”收尾，不再重复报告 3/3。
    assert "K 线预热进度：2/3" in caplog.text
    assert "K 线预热进度：3/3" not in caplog.text
    assert "K 线预热完成：3 个合约（1 个失败" in caplog.text
    assert detector.symbols == list(symbols)


def test_split_candles_separates_unclosed_candle():
    now = BASE + timedelta(minutes=5, seconds=20)
    candles = [Candle(BASE + timedelta(minutes=minute), 100, 101, 99, 100) for minute in range(3, 6)]

    closed, live = service._split_candles(candles, 60, now)

    assert [candle.timestamp for candle in closed] == [BASE + timedelta(minutes=3), BASE + timedelta(minutes=4)]
    assert live is not None and live.timestamp == BASE + timedelta(minutes=5)


def test_universe_loop_keeps_running_after_refresh_failure(monkeypatch):
    delays = record_sleeps(monkeypatch, limit=3)
    app_config = config()
    detector = build_detector(app_config)
    rest = FakeRest([ticker("BTC_USDT", 20e6)], [contract("BTC_USDT")], fail_times=1)
    feed = GateTradeFeed("wss://example.test")

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, rest, feed, app_config).universe_loop())

    assert delays == [600, 60, 600]
    assert detector.symbols == ["BTC_USDT"]


def test_initial_universe_retries_with_backoff(monkeypatch):
    delays = record_sleeps(monkeypatch, limit=10)
    app_config = config()
    detector = build_detector(app_config)
    rest = FakeRest([ticker("BTC_USDT", 20e6)], [contract("BTC_USDT")], fail_times=4)
    feed = GateTradeFeed("wss://example.test")

    asyncio.run(monitor(detector, rest, feed, app_config).initial_universe())

    assert delays == [1, 2, 4, 4]
    assert feed.symbols == ["BTC_USDT"]


class CandleRest:
    """按合约返回 K 线，可让指定合约前几次请求失败。"""

    def __init__(self, candles, failures: dict[str, int] | None = None) -> None:
        self.candles = candles
        self.failures = dict(failures or {})
        self.requests: list[str] = []

    def fetch_candles(self, symbol, interval, limit):
        self.requests.append(symbol)
        if self.failures.get(symbol):
            self.failures[symbol] -= 1
            raise ConnectionError("candles down")
        return self.candles


def test_resync_rebuilds_stale_symbols_and_retries_failures(monkeypatch):
    delays = record_sleeps(monkeypatch, limit=10)
    app_config = config()
    detector = build_detector(app_config)
    now = datetime.now(UTC)
    candles = [
        Candle(now - timedelta(minutes=offset), 100, 101, 99, 100)
        for offset in range(app_config.indicator.warmup_candles, -1, -1)
    ]
    for symbol in ("BTC_USDT", "ETH_USDT"):
        detector.add_symbol(symbol, [], 1e9)
    detector.mark_stream_gap()
    rest = CandleRest(candles, failures={"ETH_USDT": 1})

    asyncio.run(monitor(detector, rest, app_config=app_config).resync_stale_symbols())

    assert detector.stale_symbols == []
    # 第一轮 ETH 失败，只按退避重试仍然失效的合约。
    assert sorted(rest.requests) == ["BTC_USDT", "ETH_USDT", "ETH_USDT"]
    assert delays == [1]


def test_symbols_added_during_disconnect_are_resynced_after_reconnect(monkeypatch):
    record_sleeps(monkeypatch, limit=2)
    app_config = config()
    detector = build_detector(app_config)
    detector.add_symbol("BTC_USDT", [], 1e9)
    resynced_batches: list[list[str]] = []

    async def fake_resync(_monitor):
        resynced_batches.append(detector.stale_symbols)

    class AddsSymbolWhileDisconnected(ScriptedFeed):
        async def stream(self):
            if len(self.scripts) == 1:
                # 模拟断线退避期间合约池刷新加入了新合约（预热早于重连）。
                detector.add_symbol("NEW_USDT", [], 1e9)
            async for tick in super().stream():
                yield tick

    monkeypatch.setattr(service._Monitor, "resync_stale_symbols", fake_resync)
    feed = AddsSymbolWhileDisconnected([((), ConnectionError("down")), (("1",), ConnectionError("down again"))])

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detector, feed=feed, app_config=app_config).stream_loop(FakeDispatcher()))

    assert resynced_batches == [["BTC_USDT", "NEW_USDT"]]


class TrackingSemaphore:
    """记录并发峰值的信号量替身，用于确认回补走的是外部传入的那一个。"""

    def __init__(self, limit: int) -> None:
        self._inner = asyncio.Semaphore(limit)
        self.active = 0
        self.peak = 0

    async def __aenter__(self) -> None:
        await self._inner.acquire()
        self.active += 1
        self.peak = max(self.peak, self.active)

    async def __aexit__(self, *exc_info) -> None:
        self.active -= 1
        self._inner.release()


def test_resync_limits_concurrency_with_the_shared_semaphore():
    """回补若自建信号量，与合约池刷新并行时峰值并发会翻倍，限速也就失去了唯一的收口。"""
    app_config = config()
    detector = build_detector(app_config)
    now = datetime.now(UTC)
    candles = [
        Candle(now - timedelta(minutes=offset), 100, 101, 99, 100)
        for offset in range(app_config.indicator.warmup_candles, -1, -1)
    ]
    for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"):
        detector.add_symbol(symbol, [], 1e9)
    detector.mark_stream_gap()
    rest = CandleRest(candles)
    tracker = TrackingSemaphore(2)

    asyncio.run(monitor(detector, rest, app_config=app_config, sem=tracker).resync_stale_symbols())

    assert detector.stale_symbols == []
    # peak 为 0 说明传入的信号量根本没被用到（即代码又在内部自建了一个）。
    assert 0 < tracker.peak <= 2


def test_sync_universe_uses_the_shared_semaphore():
    async def scenario():
        app_config = config()
        detector = build_detector(app_config)
        feed = GateTradeFeed("wss://example.test")
        rest = FakeRest(
            [ticker(symbol, 20e6) for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT")],
            [contract(symbol) for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT")],
        )
        tracker = TrackingSemaphore(2)
        await monitor(detector, rest, feed, app_config, tracker).refresh_universe()
        return tracker

    tracker = asyncio.run(scenario())

    assert 0 < tracker.peak <= 2


class RecordingDetector:
    """记录收到的预热数据与调用，用于验证服务对多检测器的编排。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._symbols: set[str] = set()
        self.stale: set[str] = set()
        self.warmups: dict[str, tuple[list[Candle], Candle | None]] = {}
        self.resyncs: list[str] = []
        self.gaps = 0

    @property
    def symbols(self) -> list[str]:
        return sorted(self._symbols)

    @property
    def stale_symbols(self) -> list[str]:
        return sorted(self.stale)

    def add_symbol(self, symbol, candles, volume_24h_quote, live_candle=None, price_decimals=None) -> None:
        if symbol not in self._symbols:
            self.warmups[symbol] = (candles, live_candle)
        self._symbols.add(symbol)

    def remove_symbols(self, symbols) -> None:
        self._symbols -= symbols

    def mark_stream_gap(self) -> None:
        self.gaps += 1
        self.stale = set(self._symbols)

    def resync_symbol(self, symbol, candles, live_candle, fetched_at) -> bool:
        if symbol not in self.stale:
            return False
        self.stale.discard(symbol)
        self.resyncs.append(symbol)
        return True

    def add_tick(self, tick: PriceTick) -> list[str]:
        return [f"{self.name}-{tick.trade_id}"]


class IntervalRest(FakeRest):
    """按请求的周期与根数生成截至当前的 K 线，最后一根是未收盘的当前 K 线。"""

    def __init__(self, tickers, contracts) -> None:
        super().__init__(tickers, contracts)
        self.requests: list[tuple[str, str, int]] = []

    def fetch_candles(self, symbol, interval, limit):
        self.requests.append((symbol, interval, limit))
        seconds = {"1m": 60, "5m": 300}[interval]
        current = int(datetime.now(UTC).timestamp()) // seconds * seconds
        return [
            Candle(datetime.fromtimestamp(current - seconds * offset, UTC), 100, 101, 99, 100)
            for offset in range(limit - 1, -1, -1)
        ]


def test_detectors_sharing_an_interval_share_one_request_and_keep_their_own_depth():
    atr, trend, other = RecordingDetector("atr"), RecordingDetector("trend"), RecordingDetector("other")
    detectors = [
        MonitoredDetector("ATR 异动", atr, "1m", 50),
        MonitoredDetector("K 线趋势", trend, "1m", 331),
        MonitoredDetector("其他", other, "5m", 20),
    ]
    rest = IntervalRest([ticker("BTC_USDT", 20e6)], [contract("BTC_USDT")])
    feed = GateTradeFeed("wss://example.test")

    asyncio.run(monitor(detectors, rest, feed).refresh_universe())

    # 同周期取最多的根数只请求一次，不同周期各请求一次。
    assert sorted(rest.requests) == [("BTC_USDT", "1m", 331), ("BTC_USDT", "5m", 20)]
    # 各自截取到要求的根数（含一根未收盘），与单独请求时拿到的数据相同。
    for detector, closed_count in ((atr, 49), (trend, 330), (other, 19)):
        closed, live = detector.warmups["BTC_USDT"]
        assert len(closed) == closed_count
        assert live is not None
    assert feed.symbols == ["BTC_USDT"]


def test_warmup_failure_still_adds_symbol_to_every_detector():
    atr, trend = RecordingDetector("atr"), RecordingDetector("trend")
    detectors = [MonitoredDetector("ATR 异动", atr, "1m", 50), MonitoredDetector("K 线趋势", trend, "1m", 331)]

    class DownRest(FakeRest):
        def fetch_candles(self, symbol, interval, limit):
            raise ConnectionError("candles down")

    rest = DownRest([ticker("BTC_USDT", 20e6)], [contract("BTC_USDT")])
    asyncio.run(monitor(detectors, rest, GateTradeFeed("wss://example.test")).refresh_universe())

    assert atr.warmups["BTC_USDT"] == ([], None)
    assert trend.warmups["BTC_USDT"] == ([], None)


def test_resync_covers_symbols_stale_in_any_detector():
    atr, trend = RecordingDetector("atr"), RecordingDetector("trend")
    for detector in (atr, trend):
        detector.add_symbol("BTC_USDT", [], 1e9)
        detector.add_symbol("ETH_USDT", [], 1e9)
    # 只有趋势检测器的 ETH 失效：同样要回补，且只对失效的检测器生效。
    trend.stale = {"ETH_USDT"}
    detectors = [MonitoredDetector("ATR 异动", atr, "1m", 50), MonitoredDetector("K 线趋势", trend, "1m", 331)]
    rest = IntervalRest([], [])

    asyncio.run(monitor(detectors, rest).resync_stale_symbols())

    assert rest.requests == [("ETH_USDT", "1m", 331)]
    assert (atr.resyncs, trend.resyncs) == ([], ["ETH_USDT"])


def test_stream_loop_feeds_every_detector_and_marks_gaps_on_all(monkeypatch):
    record_sleeps(monkeypatch, limit=1)
    atr, trend = RecordingDetector("atr"), RecordingDetector("trend")
    detectors = [MonitoredDetector("ATR 异动", atr, "1m", 50), MonitoredDetector("K 线趋势", trend, "1m", 331)]
    dispatcher = FakeDispatcher()
    feed = ScriptedFeed([(("1", "2"), ConnectionError("down"))])

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detectors, feed=feed).stream_loop(dispatcher))

    assert dispatcher.published == ["atr-1", "trend-1", "atr-2", "trend-2"]
    assert (atr.gaps, trend.gaps) == (1, 1)


def test_real_trend_detector_warms_up_through_monitor():
    app_config = config()
    trend = build_trend_detector(app_config.trend)
    detectors = [MonitoredDetector("K 线趋势", trend, trend.warmup_interval, trend.warmup_candles)]
    rest = IntervalRest([ticker("BTC_USDT", 20e6)], [contract("BTC_USDT")])

    asyncio.run(monitor(detectors, rest, GateTradeFeed("wss://example.test"), app_config).refresh_universe())

    assert trend.symbols == ["BTC_USDT"]
    assert len(trend._states["BTC_USDT"].minutes) == trend.warmup_candles - 1


def test_failing_detector_is_skipped_without_breaking_the_stream(monkeypatch, caplog):
    """一个检测器出错只跳过这笔成交并限流记日志，其他检测器照常提醒，也不能因此断线重连。"""
    delays = record_sleeps(monkeypatch, limit=1)

    class BrokenDetector(RecordingDetector):
        def add_tick(self, tick: PriceTick) -> list[str]:
            raise ValueError("boom")

    atr, broken = RecordingDetector("atr"), BrokenDetector("trend")
    detectors = [MonitoredDetector("ATR 异动", atr, "1m", 50), MonitoredDetector("K 线趋势", broken, "1m", 46)]
    dispatcher = FakeDispatcher()
    feed = ScriptedFeed([(("1", "2", "3"), ConnectionError("down"))])

    with pytest.raises(StopLoop):
        asyncio.run(monitor(detectors, feed=feed).stream_loop(dispatcher))

    assert dispatcher.published == ["atr-1", "atr-2", "atr-3"]
    # 三笔都出错，但只记一次带堆栈的日志；随后的断线来自脚本里的 ConnectionError，而不是检测器异常。
    assert caplog.text.count("K 线趋势检测器处理 BTC_USDT 成交出错") == 1
    assert "boom" in caplog.text
    assert "down" in caplog.text and delays == [1]
