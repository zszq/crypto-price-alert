import asyncio
import email.parser
import email.utils
import http.client
import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from price_alert import gate
from price_alert.gate import (
    GateRestClient,
    GateTradeFeed,
    build_subscription,
    futures_trade_url,
    parse_candle,
    parse_trade_message,
    parse_trade_payload,
    price_decimals,
)


def test_builds_gate_trade_subscription():
    request = json.loads(build_subscription(["BTC_USDT", "ETH_USDT"]))

    assert request["channel"] == "futures.trades"
    assert request["event"] == "subscribe"
    assert request["payload"] == ["BTC_USDT", "ETH_USDT"]


def test_parses_candle_and_filters_internal_trades():
    candle = parse_candle({"t": 1_700_000_000, "o": "100", "h": "102", "l": "99", "c": "101", "sum": "5000"})
    assert candle.high == 102
    assert candle.quote_volume == 5000

    message = json.dumps(
        {
            "channel": "futures.trades",
            "event": "update",
            "result": [
                {"id": 1, "contract": "BTC_USDT", "price": "70000", "size": "-2", "create_time_ms": 1_700_000_000_123},
                {
                    "id": 2,
                    "contract": "BTC_USDT",
                    "price": "1",
                    "size": "1",
                    "create_time_ms": 1_700_000_000_124,
                    "is_internal": True,
                },
            ],
        }
    )

    ticks = parse_trade_message(message)

    assert len(ticks) == 1
    assert ticks[0].price == 70000
    assert ticks[0].size == 2


def trade(contract: str = "BTC_USDT", price: object = "100", **extra):
    return {"id": 1, "contract": contract, "price": price, "size": "1", "create_time_ms": 1_700_000_000_000} | extra


def update(*trades):
    return {"channel": "futures.trades", "event": "update", "result": list(trades)}


def test_skips_malformed_trades_without_dropping_valid_ones():
    payload = update(
        "not-a-dict",
        {"contract": "BTC_USDT", "size": "1", "create_time_ms": 1},
        trade(price="nan"),
        trade(price="-1"),
        trade(price="abc"),
        trade(create_time_ms=10**30),
        trade("ETH_USDT", "2000"),
    )

    ticks = parse_trade_payload(payload)

    assert [(tick.symbol, tick.price) for tick in ticks] == [("ETH_USDT", 2000.0)]


def test_ignores_update_without_trade_list():
    assert parse_trade_payload({"channel": "futures.trades", "event": "update", "result": {"status": "x"}}) == []


def test_subscription_request_carries_event_and_id():
    request = json.loads(build_subscription(["BTC_USDT"], "unsubscribe", 7))

    assert request["event"] == "unsubscribe"
    assert request["id"] == 7


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None


class ResetResponse(FakeResponse):
    def read(self, *args):
        raise ConnectionResetError("reset while reading body")


def rest_with_responses(monkeypatch, responses, sleeps=None, rate_limiter=None):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        result = responses[len(calls) - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    def fake_sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gate.time, "sleep", fake_sleep)
    return GateRestClient("https://example.test", retries=3, rate_limiter=rate_limiter), calls


def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", code, "error", headers or {}, None)


def test_rest_does_not_retry_client_errors(monkeypatch):
    client, calls = rest_with_responses(monkeypatch, [http_error(400)])

    with pytest.raises(urllib.error.HTTPError):
        client.fetch_tickers()

    assert len(calls) == 1


def test_rest_retries_rate_limit_and_body_read_failures(monkeypatch):
    client, calls = rest_with_responses(
        monkeypatch,
        [http_error(429), ResetResponse(), FakeResponse(b"[]")],
    )

    assert client.fetch_tickers() == []
    assert len(calls) == 3


def test_rest_raises_after_exhausting_retries(monkeypatch):
    client, calls = rest_with_responses(monkeypatch, [http_error(503)] * 3)

    with pytest.raises(urllib.error.HTTPError):
        client.fetch_contracts()

    assert len(calls) == 3


class FakeWebSocket:
    def __init__(self, respond=None):
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.respond = respond or (lambda request: [])

    async def send(self, message: str) -> None:
        request = json.loads(message)
        self.sent.append(request)
        for reply in self.respond(request):
            self.incoming.put_nowait(json.dumps(reply))

    async def recv(self) -> str:
        return await self.incoming.get()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


def ack(request, error=None):
    return {"channel": "futures.trades", "event": request["event"], "id": request["id"], "error": error}


def feed_for(websocket: FakeWebSocket, **kwargs) -> GateTradeFeed:
    return GateTradeFeed("wss://example.test", connect=lambda url, **options: websocket, **kwargs)


def test_feed_subscribes_in_chunks_and_skips_undecodable_messages():
    async def scenario():
        def respond(request):
            if request["payload"] == ["SOL_USDT"]:
                return [ack(request), update(trade("SOL_USDT"))]
            return [ack(request)]

        websocket = FakeWebSocket(respond)
        websocket.incoming.put_nowait("{not json")
        feed = feed_for(websocket, chunk_size=2)
        await feed.set_symbols(["sol_usdt", "BTC_USDT", "ETH_USDT"])
        stream = feed.stream()
        tick = await anext(stream)
        await stream.aclose()
        return websocket, tick

    websocket, tick = asyncio.run(scenario())

    assert [request["payload"] for request in websocket.sent] == [["BTC_USDT", "ETH_USDT"], ["SOL_USDT"]]
    assert tick.symbol == "SOL_USDT"


def test_failed_batch_subscription_is_retried_per_symbol_and_bad_symbol_is_reported():
    async def scenario():
        def respond(request):
            symbols = request["payload"]
            if len(symbols) > 1 or symbols == ["BAD_USDT"]:
                return [ack(request, {"code": 2, "message": "unknown contract"})]
            return [ack(request), update(trade(symbols[0]))]

        websocket = FakeWebSocket(respond)
        feed = feed_for(websocket)
        await feed.set_symbols(["BAD_USDT", "GOOD_USDT"])
        stream = feed.stream()
        tick = await anext(stream)
        await stream.aclose()
        return websocket, feed, tick

    websocket, feed, tick = asyncio.run(scenario())

    assert [request["payload"] for request in websocket.sent] == [
        ["BAD_USDT", "GOOD_USDT"],
        ["BAD_USDT"],
        ["GOOD_USDT"],
    ]
    assert feed.rejected_symbols == {"BAD_USDT"}
    assert tick.symbol == "GOOD_USDT"


def test_symbol_changes_are_applied_incrementally_while_connected():
    async def scenario():
        def respond(request):
            return [ack(request), update(trade("BTC_USDT"))] if request["id"] == 1 else [ack(request)]

        websocket = FakeWebSocket(respond)
        feed = feed_for(websocket)
        await feed.set_symbols(["BTC_USDT", "ETH_USDT"])
        stream = feed.stream()
        await anext(stream)
        await feed.set_symbols(["ETH_USDT", "SOL_USDT"])
        await stream.aclose()
        # 断开后的变更只记录目标集合，不应再向旧连接发送。
        await feed.set_symbols(["XRP_USDT"])
        return websocket, feed

    websocket, feed = asyncio.run(scenario())

    assert [(request["event"], request["payload"]) for request in websocket.sent] == [
        ("subscribe", ["BTC_USDT", "ETH_USDT"]),
        ("unsubscribe", ["BTC_USDT"]),
        ("subscribe", ["SOL_USDT"]),
    ]
    assert feed.symbols == ["XRP_USDT"]


def test_receive_timeout_is_reported_as_connection_error():
    async def scenario():
        feed = feed_for(FakeWebSocket(), receive_timeout_seconds=0.01)
        await feed.set_symbols(["BTC_USDT"])
        stream = feed.stream()
        try:
            await anext(stream)
        finally:
            await stream.aclose()

    with pytest.raises(ConnectionError, match="未收到 Gate 行情"):
        asyncio.run(scenario())


def test_rejects_non_finite_candle_values():
    base = {"t": 1_700_000_000, "o": "100", "h": "102", "l": "99", "c": "101", "sum": "5000"}
    for key in ("o", "h", "l", "c", "sum", "v"):
        for value in ("nan", "inf"):
            with pytest.raises(ValueError):
                parse_candle(base | {key: value})


class FrozenClock:
    """时钟不随 sleep 前进，用于模拟多个线程在同一时刻并发取令牌。"""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return 0.0

    def time(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class SteppingClock(FrozenClock):
    """sleep 按时长推进时钟，用于模拟单线程顺序取令牌。"""

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_passes_burst_then_queues_waiters(monkeypatch):
    clock = FrozenClock()
    monkeypatch.setattr(gate, "time", clock)
    limiter = gate.RateLimiter(10, burst=2)

    for _ in range(5):
        limiter.acquire()

    # 前两次用掉突发额度不等待；其余三次必须排成递增的等待，而不是睡同样长后一起醒来抢令牌。
    assert clock.sleeps == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]


def test_rate_limiter_refills_tokens_over_time(monkeypatch):
    clock = SteppingClock()
    monkeypatch.setattr(gate, "time", clock)
    limiter = gate.RateLimiter(10, burst=2)

    for _ in range(4):
        limiter.acquire()
    clock.now += 1.0  # 静置 1 秒足够把桶重新充满
    limiter.acquire()
    limiter.acquire()

    # 突发额度恢复后又能连放两次，长期速率仍被锁定在 10 次/秒。
    assert clock.sleeps == [pytest.approx(0.1), pytest.approx(0.1)]


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        gate.RateLimiter(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", 7.0),
        ("0", 0.0),
        ("-5", 0.0),  # 负数按"立即可重试"处理
        ("999999", gate.MAX_RETRY_AFTER_SECONDS),  # 异常大的值必须封顶
        ("nan", None),
        ("下周二", None),
        (None, None),
    ],
)
def test_retry_after_parses_seconds_and_rejects_garbage(raw, expected):
    assert gate._retry_after_seconds({"Retry-After": raw} if raw is not None else {}) == expected


def test_retry_after_parses_http_date():
    future = email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=12))

    assert gate._retry_after_seconds({"Retry-After": future}) == pytest.approx(12, abs=2)

    # -0000 表示时区未知，解析结果是 naive datetime，少了补 UTC 这一步就会因为无法与 aware
    # 的当前时间相减而抛 TypeError。用"未来 20 秒"而不是写死的远期日期，测试才不会到期失效。
    unknown_zone = (datetime.now(UTC) + timedelta(seconds=20)).strftime("%a, %d %b %Y %H:%M:%S -0000")
    assert gate._retry_after_seconds({"Retry-After": unknown_zone}) == pytest.approx(20, abs=2)


def test_rest_waits_exactly_as_long_as_retry_after(monkeypatch):
    sleeps: list[float] = []
    client, calls = rest_with_responses(
        monkeypatch,
        [http_error(429, {"Retry-After": "7"}), FakeResponse(b"[]")],
        sleeps=sleeps,
    )

    assert client.fetch_tickers() == []
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_rest_backs_off_longer_for_rate_limit_without_retry_after(monkeypatch):
    sleeps: list[float] = []
    client, _ = rest_with_responses(monkeypatch, [http_error(429)] * 3, sleeps=sleeps)

    with pytest.raises(urllib.error.HTTPError):
        client.fetch_tickers()

    # 限频恢复以秒计，退避基数必须明显长于普通临时故障，否则重试只是继续踩在限频上。
    assert sleeps == [2.0, 4.0]


def test_rest_keeps_short_backoff_for_transient_failures(monkeypatch):
    sleeps: list[float] = []
    client, _ = rest_with_responses(monkeypatch, [http_error(503)] * 3, sleeps=sleeps)

    with pytest.raises(urllib.error.HTTPError):
        client.fetch_contracts()

    assert sleeps == [0.5, 1.0]


def test_rest_takes_a_token_for_every_attempt_including_retries(monkeypatch):
    class CountingLimiter:
        def __init__(self) -> None:
            self.acquired = 0

        def acquire(self) -> None:
            self.acquired += 1

    limiter = CountingLimiter()
    client, calls = rest_with_responses(
        monkeypatch,
        [http_error(429), http_error(503), FakeResponse(b"[]")],
        rate_limiter=limiter,
    )

    assert client.fetch_tickers() == []
    # 重试若绕过限速，就会在最该收敛的时刻额外加码请求。
    assert limiter.acquired == len(calls) == 3


def test_rate_limit_reset_header_is_converted_to_remaining_seconds():
    reset_at = datetime.now(UTC).timestamp() + 9

    seconds = gate._rate_limit_reset_seconds({gate.RATE_LIMIT_RESET_HEADER: str(int(reset_at))})

    assert seconds == pytest.approx(9, abs=2)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),  # 时间戳早已过去：限频窗口已重置，可以立即重试
        ("99999999999", gate.MAX_RETRY_AFTER_SECONDS),  # 异常大的时间戳必须封顶
        ("nan", None),
        ("很快", None),
        (None, None),
    ],
)
def test_rate_limit_reset_header_rejects_garbage(raw, expected):
    headers = {gate.RATE_LIMIT_RESET_HEADER: raw} if raw is not None else {}

    assert gate._rate_limit_reset_seconds(headers) == expected


def test_rest_waits_until_gate_rate_limit_resets(monkeypatch):
    """Gate 不返回标准的 Retry-After，限频恢复时刻只在这个专有头里。"""
    sleeps: list[float] = []
    reset_at = int(datetime.now(UTC).timestamp() + 6)
    client, calls = rest_with_responses(
        monkeypatch,
        [http_error(429, {gate.RATE_LIMIT_RESET_HEADER: str(reset_at)}), FakeResponse(b"[]")],
        sleeps=sleeps,
    )

    assert client.fetch_tickers() == []
    assert len(calls) == 2
    assert sleeps == [pytest.approx(6, abs=2)]


def test_standard_retry_after_takes_precedence_over_gate_header(monkeypatch):
    sleeps: list[float] = []
    reset_at = int(datetime.now(UTC).timestamp() + 1)
    client, _ = rest_with_responses(
        monkeypatch,
        [
            http_error(429, {"Retry-After": "8", gate.RATE_LIMIT_RESET_HEADER: str(reset_at)}),
            FakeResponse(b"[]"),
        ],
        sleeps=sleeps,
    )

    assert client.fetch_tickers() == []
    # 标准头语义更权威：两者冲突时按它等待，而不是取更短的那个继续踩限频。
    assert sleeps == [8.0]


def test_rate_limit_headers_are_read_case_insensitively():
    """HTTP/2 的头名一律小写，而真实的 HTTPError.headers 是大小写不敏感的 HTTPMessage。

    其余用例为了简洁用普通 dict 传头（大小写敏感），这里用真实类型锁住生产路径的取值行为。
    """
    raw = "x-gate-ratelimit-reset-timestamp: 99999999999\r\nx-gate-ratelimit-limit: 200\r\n\r\n"
    headers = email.parser.Parser(_class=http.client.HTTPMessage).parsestr(raw)

    exc = urllib.error.HTTPError("https://example.test", 429, "too many", headers, None)

    assert GateRestClient("https://example.test")._retry_delay(0, exc) == gate.MAX_RETRY_AFTER_SECONDS


class FakeTradeHistory:
    """按实测的 Gate 语义返回历史成交：[from, to) 内从新到旧，limit 截断，offset 跳过。"""

    def __init__(self, trades: list[dict]) -> None:
        self.trades = trades
        self.requests: list[dict] = []

    def __call__(self, path: str, params: dict | None = None):
        assert path == "/futures/usdt/trades"
        assert params is not None
        self.requests.append(dict(params))
        matched = [item for item in self.trades if params["from"] <= item["create_time"] < params["to"]]
        matched.sort(key=lambda item: (item["create_time"], item["id"]), reverse=True)
        offset = params.get("offset", 0)
        return matched[offset : offset + params["limit"]]


def history_trade(trade_id: int, create_time: float, price: str = "100", **extra) -> dict:
    # REST 的 create_time_ms 实际也是秒，故意放一个会被误当成毫秒的值，确保解析只认 create_time。
    return {"id": trade_id, "contract": "QNT_USDT", "create_time": create_time, "create_time_ms": create_time,
            "size": -3, "price": price} | extra


def test_rest_trade_timestamp_is_read_as_seconds_and_internal_trades_are_dropped():
    tick = gate.parse_rest_trade(history_trade(7, 1_790_645_229.845))

    assert tick is not None
    assert tick.timestamp == datetime.fromtimestamp(1_790_645_229.845, tz=UTC)
    assert (tick.size, tick.trade_id) == (3.0, "7")
    assert gate.parse_rest_trade(history_trade(8, 1_790_645_229.845, is_internal=True)) is None


def test_fetch_trades_pages_back_by_shrinking_to_and_deduplicates(monkeypatch):
    monkeypatch.setattr(gate, "TRADES_PAGE_SIZE", 3)
    base = 1_790_645_220
    trades = [history_trade(index, base + index * 0.5) for index in range(1, 11)]
    fake = FakeTradeHistory(trades)
    client = GateRestClient("https://example.test")
    monkeypatch.setattr(client, "_get", fake)

    ticks = client.fetch_trades(
        "QNT_USDT",
        datetime.fromtimestamp(base, tz=UTC),
        datetime.fromtimestamp(base + 10, tz=UTC),
    )

    assert [tick.trade_id for tick in ticks] == [str(index) for index in range(1, 11)]
    # 翻页只靠收缩 to，不需要 offset。
    assert all("offset" not in request for request in fake.requests)


def test_fetch_trades_uses_offset_when_a_whole_page_is_one_second(monkeypatch):
    monkeypatch.setattr(gate, "TRADES_PAGE_SIZE", 3)
    base = 1_790_645_220
    burst = [history_trade(index, base + 5.1 + index / 1000) for index in range(1, 8)]
    trades = [history_trade(100, base + 1.0), *burst, history_trade(200, base + 8.0)]
    fake = FakeTradeHistory(trades)
    client = GateRestClient("https://example.test")
    monkeypatch.setattr(client, "_get", fake)

    ticks = client.fetch_trades(
        "QNT_USDT",
        datetime.fromtimestamp(base, tz=UTC),
        datetime.fromtimestamp(base + 10, tz=UTC),
    )

    # 第 5 秒挤了 7 笔，超过一页：必须在这一秒内用 offset 翻完，前后两侧的成交也不能丢。
    assert sorted(int(tick.trade_id) for tick in ticks) == sorted(item["id"] for item in trades)
    assert [tick.timestamp for tick in ticks] == sorted(tick.timestamp for tick in ticks)
    assert any(request.get("offset") for request in fake.requests)


def test_fetch_trades_refuses_ranges_with_too_many_trades(monkeypatch):
    monkeypatch.setattr(gate, "TRADES_PAGE_SIZE", 3)
    base = 1_790_645_220
    client = GateRestClient("https://example.test")
    monkeypatch.setattr(client, "_get", FakeTradeHistory([history_trade(i, base + i) for i in range(10)]))

    with pytest.raises(ValueError, match="缩短时间范围"):
        client.fetch_trades(
            "QNT_USDT",
            datetime.fromtimestamp(base, tz=UTC),
            datetime.fromtimestamp(base + 10, tz=UTC),
            max_trades=5,
        )


def test_fetch_candles_between_sends_inclusive_unix_range(monkeypatch):
    requests = []

    def fake_get(path, params=None):
        requests.append((path, params))
        return [{"t": 1_790_645_160, "o": "1", "h": "2", "l": "1", "c": "2"},
                {"t": 1_790_645_100, "o": "1", "h": "2", "l": "1", "c": "1"}]

    client = GateRestClient("https://example.test")
    monkeypatch.setattr(client, "_get", fake_get)
    start = datetime.fromtimestamp(1_790_645_100, tz=UTC)

    candles = client.fetch_candles_between("QNT_USDT", "1m", start, start + timedelta(minutes=1))

    assert requests == [
        (
            "/futures/usdt/candlesticks",
            {"contract": "QNT_USDT", "interval": "1m", "from": 1_790_645_100, "to": 1_790_645_160},
        )
    ]
    assert [candle.timestamp for candle in candles] == sorted(candle.timestamp for candle in candles)


def test_fetch_contract_quotes_symbol_and_requires_object(monkeypatch):
    paths = []
    client = GateRestClient("https://example.test")
    monkeypatch.setattr(client, "_get", lambda path, params=None: paths.append(path) or {"order_price_round": "0.1"})

    assert client.fetch_contract("A/B_USDT") == {"order_price_round": "0.1"}
    assert paths == ["/futures/usdt/contracts/A%2FB_USDT"]

    monkeypatch.setattr(client, "_get", lambda path, params=None: [])
    with pytest.raises(TypeError, match="合约响应不是对象"):
        client.fetch_contract("BTC_USDT")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.01", 2), ("0.010", 2), ("0.1", 1), ("1", 0), ("5", 0), ("1e-8", 8), (None, None), ("abc", None),
     ("0", None), ("-0.1", None), ("nan", None), ("inf", None)],
)
def test_price_decimals_parses_gate_price_step(raw, expected):
    assert price_decimals(raw) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("BTC_USDT", "https://www.gate.com/zh/futures/USDT/BTC_USDT"),
        # 中文原样保留便于辨认；ASCII 保留字符与非 ASCII 空白、不可见字符仍需编码。
        ("龙虾_USDT", "https://www.gate.com/zh/futures/USDT/龙虾_USDT"),
        ("A/B?_USDT", "https://www.gate.com/zh/futures/USDT/A%2FB%3F_USDT"),
        ("龙\u3000虾\u200b_USDT", "https://www.gate.com/zh/futures/USDT/龙%E3%80%80虾%E2%80%8B_USDT"),
    ],
)
def test_futures_trade_url_keeps_symbol_readable_but_path_safe(symbol, expected):
    assert futures_trade_url(symbol) == expected


def test_parse_candle_reads_contract_volume():
    candle = parse_candle({"t": 1_700_000_000, "o": "1", "h": "1", "l": "1", "c": "1", "v": 37284, "sum": "7972.9"})

    assert candle.volume == 37284
    assert parse_candle({"t": 1_700_000_000, "o": "1", "h": "1", "l": "1", "c": "1"}).volume == 0
