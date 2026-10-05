"""Gate.io USDT 永续公共 REST 与 WebSocket 行情适配器。"""

from __future__ import annotations

import asyncio
import email.utils
import http.client
import itertools
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import websockets

from price_alert.formatting import beijing_time
from price_alert.models import Candle, PriceTick

LOGGER = logging.getLogger(__name__)

# 只有限频和服务端临时故障值得重试；其余 4xx 多为参数或合约不存在，重试只会拖慢预热。
RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# 普通临时故障的退避基数；限频恢复通常以秒计，用同一个基数会过早重试，所以 429 单独用下面的基数。
RETRY_BACKOFF_SECONDS = 0.5
RATE_LIMITED_BACKOFF_SECONDS = 2.0
# 退避时长由交易所控制，异常大的值会让线程长期占着不放并拖住整轮预热，必须封顶。
MAX_RETRY_AFTER_SECONDS = 30.0
# Gate 不返回标准的 Retry-After，限频恢复时刻放在这个专有头里（Unix 秒级绝对时间戳）。
RATE_LIMIT_RESET_HEADER = "X-Gate-RateLimit-Reset-Timestamp"
# 历史成交接口单页上限；offset 翻页在十万级就会被拒绝（offset too big），所以翻页主要靠收缩 to。
TRADES_PAGE_SIZE = 1000
TRADES_PROGRESS_PAGES = 20
# Gate 只保留最近 10000 根 1 分钟 K 线（约 6.9 天），更早的请求会直接返回 400。
GATE_MINUTE_HISTORY = 10_000
# 分段取 1 分钟 K 线时单次请求的根数，低于 Gate 的 2000 上限并留出余量。
MINUTE_CANDLES_PER_REQUEST = 1000
# 提醒里附带的合约交易页，末尾拼接合约名。
TRADE_PAGE_URL = "https://www.gate.com/zh/futures/USDT/"


def price_decimals(order_price_round: Any) -> int | None:
    """把 Gate 的报价步长（如 "0.01"）换算成展示用的小数位数，无法识别时返回 None。"""
    try:
        step = Decimal(str(order_price_round))
    except (InvalidOperation, ValueError):
        return None
    if not step.is_finite() or step <= 0:
        return None
    # normalize 去掉尾随零（"0.010" 与 "0.01" 同为两位）；步长为整数时指数非负，按 0 位小数显示。
    return max(0, -step.normalize().as_tuple().exponent)


def quanto_multiplier(value: Any) -> float | None:
    """合约面值（一张合约对应的币数），无法识别或不是正的有限数时返回 None。"""
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        return None
    return multiplier if math.isfinite(multiplier) and multiplier > 0 else None


def futures_trade_url(symbol: str) -> str:
    """合约在 Gate 网页上的交易地址。"""
    # 中文等非 ASCII 字符原样保留，否则地址显示成一串 %XX 难以辨认；浏览器打开时会自行编码。
    # ASCII 保留字符（/、?、# 等）会破坏路径结构，非 ASCII 的空白和不可见字符会截断终端的链接识别，仍需编码。
    segment = "".join(
        char if not char.isascii() and char.isprintable() and not char.isspace() else urllib.parse.quote(char, safe="")
        for char in symbol
    )
    return f"{TRADE_PAGE_URL}{segment}"


class RateLimiter:
    """令牌桶，给所有 REST 出口统一限速。

    限频按单位时间的请求数计算，而合约池大小只决定单轮请求数，所以并发信号量挡不住重连风暴这类
    "轮数多"的场景——真正的上限必须加在时间维度上，由这里统一承担。

    REST 客户端是同步实现、经 asyncio.to_thread 在多个线程中并发调用，因此用 threading 原语而
    不是 asyncio 原语。
    """

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second 必须大于 0")
        self.rate_per_second = float(rate_per_second)
        self.burst = float(burst) if burst is not None else max(1.0, float(rate_per_second))
        self._tokens = self.burst
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate_per_second)
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self.rate_per_second
            self._tokens = 0.0
            # 把等待时长预支进 _updated：后续线程读到的是"欠账"状态，于是依次排到更晚的时刻，
            # 而不是全部睡同样长再一起醒来抢同一个令牌。
            self._updated = now + wait
        # sleep 必须在锁外，否则等待的线程会连带堵住其他线程取令牌。
        time.sleep(wait)


def _rate_limit_reset_seconds(headers: Any) -> float | None:
    """解析 Gate 专有的限频重置头，换算成还需等待的秒数。"""
    raw = headers.get(RATE_LIMIT_RESET_HEADER) if headers is not None else None
    if not raw:
        return None
    try:
        reset_at = float(raw)
    except (TypeError, ValueError):
        return None
    if reset_at != reset_at:  # NaN 能通过解析，但比较运算全为假，会绕过下面的封顶
        return None
    # 头里是绝对时间戳，必须与本地时钟相减。本地时钟快于交易所时差值偏小甚至为负，会过早重试；
    # 但限速器对重试同样收令牌，最短也隔着 1/rate 秒，不会退化成无间隔重试。
    return min(max(reset_at - datetime.now(UTC).timestamp(), 0.0), MAX_RETRY_AFTER_SECONDS)


def _retry_after_seconds(headers: Any) -> float | None:
    """解析 Retry-After（秒数或 HTTP-date 两种合法格式），无法识别时返回 None。"""
    raw = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            parsed = email.utils.parsedate_to_datetime(str(raw))
        except (TypeError, ValueError):
            return None
        if parsed is None:
            return None
        # -0000 表示时区未知，解析结果是 naive datetime，直接与 aware 的当前时间相减会抛
        # TypeError 并中断整个请求。按规范这种情况等同 GMT，补上 UTC 即可。
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        seconds = (parsed - datetime.now(UTC)).total_seconds()
    if seconds != seconds:  # NaN：float("nan") 能通过上面的解析，但比较运算全为假，会绕过封顶
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


class _ThrottledLogger:
    """交易所数据格式变化时每笔成交都会出错，限流可避免日志刷屏淹没真正的故障信息。"""

    def __init__(self, interval_seconds: float = 60.0) -> None:
        self.interval_seconds = interval_seconds
        self._last_logged = float("-inf")
        self._suppressed = 0

    def warning(self, message: str, *args: object) -> None:
        now = time.monotonic()
        if now - self._last_logged < self.interval_seconds:
            self._suppressed += 1
            return
        if self._suppressed:
            message = f"{message}（此前 {self._suppressed} 条同类日志已省略）"
        LOGGER.warning(message, *args)
        self._last_logged = now
        self._suppressed = 0


_INVALID_DATA_LOG = _ThrottledLogger()


class GateRestClient:
    def __init__(
        self,
        base_url: str,
        settle: str = "usdt",
        timeout_seconds: float = 15.0,
        retries: int = 3,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.settle = settle
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.rate_limiter = rate_limiter

    def fetch_tickers(self) -> list[dict[str, Any]]:
        payload = self._get(f"/futures/{self.settle}/tickers")
        if not isinstance(payload, list):
            raise TypeError("Gate tickers 响应不是列表")
        return payload

    def fetch_contracts(self) -> list[dict[str, Any]]:
        payload = self._get(f"/futures/{self.settle}/contracts")
        if not isinstance(payload, list):
            raise TypeError("Gate contracts 响应不是列表")
        return payload

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        payload = self._get(
            f"/futures/{self.settle}/candlesticks",
            {"contract": symbol, "interval": interval, "limit": limit},
        )
        if not isinstance(payload, list):
            raise TypeError(f"Gate {symbol} K 线响应不是列表")
        candles = [parse_candle(item) for item in payload if isinstance(item, Mapping)]
        return sorted(candles, key=lambda item: item.timestamp)

    def fetch_contract(self, symbol: str) -> dict[str, Any]:
        payload = self._get(f"/futures/{self.settle}/contracts/{urllib.parse.quote(symbol, safe='')}")
        if not isinstance(payload, dict):
            raise TypeError(f"Gate {symbol} 合约响应不是对象")
        return payload

    def fetch_candles_between(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        """按开盘时间取 [start, end] 闭区间内的 K 线（Gate 的 from/to 两端都包含）。"""
        payload = self._get(
            f"/futures/{self.settle}/candlesticks",
            {"contract": symbol, "interval": interval, "from": int(start.timestamp()), "to": int(end.timestamp())},
        )
        if not isinstance(payload, list):
            raise TypeError(f"Gate {symbol} K 线响应不是列表")
        candles = [parse_candle(item) for item in payload if isinstance(item, Mapping)]
        return sorted(candles, key=lambda item: item.timestamp)

    def fetch_trades(self, symbol: str, start: datetime, end: datetime, max_trades: int = 300_000) -> list[PriceTick]:
        """取 [start, end) 内的全部历史成交，按时间升序返回。

        Gate 在时间范围内按从新到旧分页，每页最多 1000 笔。翻页靠把 to 收缩到本页最早那一秒：
        to 是开区间，所以取“最早秒 + 1”，同一秒里更早的成交留到下一页，重复的按 id 去重。
        整页都挤在同一秒时 to 收缩不动，只能在这一秒内用 offset 继续翻。
        """
        start_second = math.floor(start.timestamp())
        to = math.ceil(end.timestamp())
        trades: dict[str, PriceTick] = {}
        pages = 0
        while to > start_second:
            page = self._trades_page(symbol, start_second, to)
            self._collect_trades(page, trades, max_trades)
            pages += 1
            if pages % TRADES_PROGRESS_PAGES == 0:
                # 长区间要翻上百页、耗时以分钟计，定期报告进度，免得被误以为卡死。
                reached = beijing_time(datetime.fromtimestamp(to, tz=UTC))
                LOGGER.info("%s 已获取 %s 笔历史成交，回溯到北京时间 %s", symbol, f"{len(trades):,}", reached)
            if len(page) < TRADES_PAGE_SIZE:
                break
            oldest_second = _oldest_trade_second(page)
            if oldest_second + 1 < to:
                to = oldest_second + 1
                continue
            offset = TRADES_PAGE_SIZE
            while len(page) == TRADES_PAGE_SIZE:
                page = self._trades_page(symbol, oldest_second, oldest_second + 1, offset)
                self._collect_trades(page, trades, max_trades)
                offset += TRADES_PAGE_SIZE
            to = oldest_second
        return sorted(trades.values(), key=lambda tick: (tick.timestamp, _trade_order(tick)))

    def _trades_page(self, symbol: str, start: int, end: int, offset: int = 0) -> list[Mapping[str, Any]]:
        params: dict[str, object] = {"contract": symbol, "from": start, "to": end, "limit": TRADES_PAGE_SIZE}
        if offset:
            params["offset"] = offset
        payload = self._get(f"/futures/{self.settle}/trades", params)
        if not isinstance(payload, list):
            raise TypeError(f"Gate {symbol} 成交响应不是列表")
        return [item for item in payload if isinstance(item, Mapping)]

    @staticmethod
    def _collect_trades(page: list[Mapping[str, Any]], trades: dict[str, PriceTick], max_trades: int) -> None:
        for item in page:
            try:
                tick = parse_rest_trade(item)
            except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                _INVALID_DATA_LOG.warning("忽略无法解析的 Gate 历史成交（%s）：%s", exc, item)
                continue
            if tick is None:
                continue
            # 缺 id 的成交无法去重，用时间与价格拼一个键，宁可极少数重复也不丢数据。
            key = tick.trade_id or f"{tick.timestamp.isoformat()}|{tick.price}|{tick.size}"
            trades[key] = tick
        if len(trades) > max_trades:
            raise ValueError(f"成交超过 {max_trades:,} 笔，请缩短时间范围")

    def _get(self, path: str, params: Mapping[str, object] | None = None) -> Any:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        request = urllib.request.Request(  # noqa: S310 - 基础地址来自本地配置
            f"{self.base_url}{path}{query}",
            headers={"Accept": "application/json", "User-Agent": "price_alert/1.0"},
        )
        for attempt in range(self.retries):
            is_last_attempt = attempt == self.retries - 1
            if self.rate_limiter is not None:
                # 重试同样要取令牌：否则被限频后的重试会绕过限速，正好在最该收敛的时刻加码请求。
                self.rate_limiter.acquire()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # HTTPError 是 OSError 子类，必须先于下方分支处理，否则不可恢复的 4xx 也会被重试。
                if exc.code not in RETRYABLE_HTTP_STATUS or is_last_attempt:
                    raise
                delay = self._retry_delay(attempt, exc)
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                # 读响应体时的连接重置、截断和维护页等临时故障不会被包装成 URLError，需要单独纳入重试。
                if is_last_attempt:
                    raise
                delay = RETRY_BACKOFF_SECONDS * (2**attempt)
            time.sleep(delay)
        raise RuntimeError("Gate REST 重试状态异常")

    def _retry_delay(self, attempt: int, exc: urllib.error.HTTPError) -> float:
        if exc.code != 429:
            return RETRY_BACKOFF_SECONDS * (2**attempt)
        # 交易所明确告知还要等多久时以它为准，自行退避只会提前重试、继续踩在限频上。
        # 先认 HTTP 标准头：Gate 目前只发下面的专有头，但标准头若出现则语义更权威。
        for parse in (_retry_after_seconds, _rate_limit_reset_seconds):
            delay = parse(exc.headers)
            if delay is not None:
                return delay
        # 两个头都没有才盲目退避：等多久全靠猜，限频窗口未重置就会连续撞满重试次数。
        return RATE_LIMITED_BACKOFF_SECONDS * (2**attempt)


def fetch_minute_candles(rest: GateRestClient, symbol: str, start: datetime, end: datetime) -> list[Candle]:
    """取开盘时间在 [start, end) 内的 1 分钟 K 线，按单次上限分段请求；趋势与放量回放共用。"""
    minute = timedelta(minutes=1)
    candles: dict[datetime, Candle] = {}
    cursor = start
    while cursor < end:
        # Gate 的 from/to 两端都包含，分段的 to 取下一段起点前一分钟，避免重复。
        stop = min(cursor + minute * MINUTE_CANDLES_PER_REQUEST, end)
        for candle in rest.fetch_candles_between(symbol, "1m", cursor, stop - minute):
            candles[candle.timestamp] = candle
        cursor = stop
    return [candles[key] for key in sorted(candles) if start <= key < end]


def parse_candle(payload: Mapping[str, Any]) -> Candle:
    return Candle(
        timestamp=datetime.fromtimestamp(float(payload["t"]), tz=UTC),
        open=float(payload["o"]),
        high=float(payload["h"]),
        low=float(payload["l"]),
        close=float(payload["c"]),
        quote_volume=float(payload.get("sum", 0.0)),
    )


def build_subscription(symbols: Sequence[str], event: str = "subscribe", request_id: int | None = None) -> str:
    request: dict[str, object] = {
        "time": int(time.time()),
        "channel": "futures.trades",
        "event": event,
        "payload": list(symbols),
    }
    if request_id is not None:
        # Gate 会在响应中原样回传 id，据此才能知道是哪一批订阅失败。
        request["id"] = request_id
    return json.dumps(request, separators=(",", ":"))


def _tick_from_trade(trade: Mapping[str, Any], timestamp_seconds: float) -> PriceTick:
    return PriceTick(
        symbol=str(trade["contract"]).upper(),
        price=float(trade["price"]),
        size=abs(float(trade["size"])),
        timestamp=datetime.fromtimestamp(timestamp_seconds, tz=UTC),
        trade_id=str(trade.get("id")) if trade.get("id") is not None else None,
    )


def _parse_trade(trade: Any) -> PriceTick | None:
    if not isinstance(trade, Mapping):
        raise TypeError("成交记录不是对象")
    # 官方说明 internal trade 可能显著偏离盘口且不会进入 K 线，不能用于异动判定。
    if trade.get("is_internal"):
        return None
    timestamp_value = trade.get("create_time_ms")
    if timestamp_value is None:
        timestamp_value = float(trade["create_time"]) * 1000
    return _tick_from_trade(trade, float(timestamp_value) / 1000)


def parse_rest_trade(trade: Any) -> PriceTick | None:
    """解析 REST 历史成交；口径与实时推送一致，内部成交同样忽略。"""
    if not isinstance(trade, Mapping):
        raise TypeError("成交记录不是对象")
    if trade.get("is_internal"):
        return None
    # REST 响应里的 create_time_ms 实测单位是秒（带毫秒小数），与 WebSocket 的毫秒不同，只能用 create_time。
    return _tick_from_trade(trade, float(trade["create_time"]))


def _oldest_trade_second(page: Sequence[Mapping[str, Any]]) -> int:
    times: list[float] = []
    for item in page:
        try:
            times.append(float(item["create_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    finite = [value for value in times if math.isfinite(value)]
    if not finite:
        # 整页都没有可用时间就无法确定下一页的边界，继续请求只会原地打转。
        raise ValueError("Gate 历史成交整页缺少有效时间，无法继续翻页")
    return math.floor(min(finite))


def _trade_order(tick: PriceTick) -> int:
    # 同一毫秒内的成交按 id 排序才能还原撮合顺序；id 缺失或非数字时退化为不排序。
    try:
        return int(tick.trade_id) if tick.trade_id is not None else 0
    except ValueError:
        return 0


def parse_trade_payload(payload: Mapping[str, Any]) -> list[PriceTick]:
    if payload.get("event") != "update" or payload.get("channel") != "futures.trades":
        return []

    result = payload.get("result")
    if not isinstance(result, list):
        _INVALID_DATA_LOG.warning("忽略 result 不是列表的 Gate 成交推送：%s", payload)
        return []

    ticks: list[PriceTick] = []
    for trade in result:
        # 单笔坏数据只丢弃这一笔；向上抛出会断开整条连接，让全部合约一起丢行情。
        try:
            tick = _parse_trade(trade)
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
            _INVALID_DATA_LOG.warning("忽略无法解析的 Gate 成交（%s）：%s", exc, trade)
            continue
        if tick is not None:
            ticks.append(tick)
    return ticks


def parse_trade_message(message: str | bytes) -> list[PriceTick]:
    return parse_trade_payload(json.loads(message))


class GateTradeFeed:
    """长连接成交订阅；合约池变化通过增量订阅/退订生效，不必断线重连。"""

    def __init__(
        self,
        url: str,
        chunk_size: int = 100,
        receive_timeout_seconds: float = 30.0,
        connect: Callable[..., Any] = websockets.connect,
    ) -> None:
        self.url = url
        self.chunk_size = chunk_size
        self.receive_timeout_seconds = receive_timeout_seconds
        self.rejected_symbols: set[str] = set()
        self._connect = connect
        self._symbols: set[str] = set()
        self._websocket: Any = None
        # 订阅请求与集合变更必须串行，否则初始分批订阅与增量退订交错时，已移除的合约可能又被订阅回来。
        self._subscription_lock = asyncio.Lock()
        self._request_ids = itertools.count(1)
        self._pending: dict[int, tuple[str, list[str]]] = {}

    @property
    def symbols(self) -> list[str]:
        return sorted(self._symbols)

    async def set_symbols(self, symbols: Iterable[str]) -> None:
        target = {symbol.upper() for symbol in symbols}
        async with self._subscription_lock:
            removed = self._symbols - target
            added = target - self._symbols
            self._symbols = target
            self.rejected_symbols &= target
            websocket = self._websocket
            if websocket is None:
                # 未连接时只记录目标集合，建立连接后会整体订阅。
                return
            try:
                await self._send_chunks(websocket, "unsubscribe", sorted(removed))
                await self._send_chunks(websocket, "subscribe", sorted(added))
            except Exception as exc:
                # 发送失败说明连接已断，stream 会重连并按最新集合整体订阅，这里无需重试。
                LOGGER.warning("增量更新订阅失败，将在重连后整体订阅：%s", exc)

    async def stream(self) -> AsyncIterator[PriceTick]:
        async with self._connect(
            self.url,
            ping_interval=20,
            # pong 与成交推送共用一条 TCP 流，链路拥塞时会被积压数据队头阻塞，RTT 可从
            # 200 毫秒涨到二十几秒。超时放宽到 30 秒，把断线判定让给 service 的 stream_loop 里按
            # 成交时间戳做的滞后熔断（阈值更低、语义明确），keepalive 只兜底真正的死连接。
            ping_timeout=30,
            close_timeout=5,
            max_queue=20_000,
        ) as websocket:
            async with self._subscription_lock:
                self._pending.clear()
                self._websocket = websocket
                await self._send_chunks(websocket, "subscribe", sorted(self._symbols))
            try:
                while True:
                    try:
                        async with asyncio.timeout(self.receive_timeout_seconds):
                            message = await websocket.recv()
                    except TimeoutError as exc:
                        raise ConnectionError(f"连续 {self.receive_timeout_seconds:g} 秒未收到 Gate 行情") from exc

                    try:
                        payload = json.loads(message)
                    except ValueError:
                        _INVALID_DATA_LOG.warning("忽略无法解析的 Gate WebSocket 消息：%.200r", message)
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    if payload.get("event") in ("subscribe", "unsubscribe") or payload.get("error"):
                        await self._handle_response(websocket, payload)
                        continue
                    for tick in parse_trade_payload(payload):
                        yield tick
            finally:
                self._websocket = None

    async def _send_chunks(self, websocket: Any, event: str, symbols: list[str]) -> None:
        for start in range(0, len(symbols), self.chunk_size):
            chunk = symbols[start : start + self.chunk_size]
            request_id = next(self._request_ids)
            self._pending[request_id] = (event, chunk)
            await websocket.send(build_subscription(chunk, event, request_id))

    async def _handle_response(self, websocket: Any, payload: Mapping[str, Any]) -> None:
        request = self._pending.pop(payload["id"], None) if isinstance(payload.get("id"), int) else None
        error = payload.get("error")
        if not error:
            return
        if request is None:
            LOGGER.warning("Gate WebSocket 返回错误：%s", payload)
            return

        event, chunk = request
        if event != "subscribe":
            LOGGER.warning("Gate 退订 %s 失败：%s", ",".join(chunk), error)
            return
        async with self._subscription_lock:
            # 合约可能已在刷新中被移除，只处理仍需监控的部分。
            still_wanted = [symbol for symbol in chunk if symbol in self._symbols]
            if len(still_wanted) > 1:
                # 一个无效合约就可能让整批订阅失败，逐个重订阅以隔离问题合约，保住同批其他合约。
                LOGGER.warning("Gate 批量订阅 %d 个合约失败（%s），改为逐个订阅", len(still_wanted), error)
                for symbol in still_wanted:
                    await self._send_chunks(websocket, "subscribe", [symbol])
            elif still_wanted:
                self.rejected_symbols.update(still_wanted)
                LOGGER.error("Gate 拒绝订阅 %s，该合约不会被监控：%s", still_wanted[0], error)
