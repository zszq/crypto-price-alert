"""动态交易对池、ATR 预热、实时检测与重连编排。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from price_alert.config import INTERVAL_SECONDS, AppConfig, WindowConfig
from price_alert.detector import AtrMoveDetector
from price_alert.gate import GateRestClient, GateTradeFeed, RateLimiter
from price_alert.models import Candle, ContractTicker
from price_alert.notifier import AlertDispatcher, build_notifiers
from price_alert.universe import select_liquid_contracts
from price_alert.windows import MoveWindow, WindowEvaluation, WindowName

LOGGER = logging.getLogger(__name__)


def build_windows(config: AppConfig, cooldown_seconds: int | None = None) -> list[MoveWindow]:
    """按配置生成观察窗口，短窗口在前：同一秒两个窗口都满足时优先发更紧迫的短窗口提醒。

    cooldown_seconds 覆盖全部窗口的冷却，供 simulate 这类不需要冷却的场景使用。
    """
    indicator = config.indicator
    configured: list[tuple[WindowName, WindowConfig]] = [("short", indicator.short_window)]
    if indicator.long_window.enabled:
        configured.append(("long", indicator.long_window))
    return [
        MoveWindow(
            name=name,
            lookback_seconds=window.lookback_seconds,
            trigger_atr_multiple=window.trigger_atr_multiple,
            min_change_percent=window.min_change_percent,
            confirmation_seconds=window.confirmation_seconds,
            min_window_trades=window.min_window_trades,
            cooldown_seconds=window.cooldown_seconds if cooldown_seconds is None else cooldown_seconds,
        )
        for name, window in configured
    ]


def build_detector(
    config: AppConfig,
    cooldown_seconds: int | None = None,
    *,
    windows: Sequence[MoveWindow] | None = None,
    observer: Callable[[WindowEvaluation], None] | None = None,
) -> AtrMoveDetector:
    # 实时监控、simulate 与 replay 共用同一份组装逻辑，新增检测参数时只需改这里和 build_windows。
    indicator = config.indicator
    return AtrMoveDetector(
        atr_period=indicator.atr_period,
        candle_interval_seconds=INTERVAL_SECONDS[indicator.candle_interval],
        max_atr_age_seconds=indicator.max_atr_age_seconds,
        windows=build_windows(config, cooldown_seconds) if windows is None else windows,
        observer=observer,
    )


def build_rest_client(config: AppConfig) -> GateRestClient:
    # 所有入口共用同一份限速与重试口径，命令行查询也不会绕过令牌桶打满交易所限频。
    gate = config.gate
    return GateRestClient(
        gate.rest_url,
        gate.settle,
        gate.rest_timeout_seconds,
        gate.rest_retries,
        RateLimiter(gate.rest_rate_limit_per_second, gate.rest_rate_limit_burst),
    )


def _split_candles(
    candles: list[Candle],
    interval_seconds: int,
    now: datetime,
) -> tuple[list[Candle], Candle | None]:
    current_interval_start = int(now.timestamp()) // interval_seconds * interval_seconds
    closed = [candle for candle in candles if candle.timestamp.timestamp() < current_interval_start]
    current = next((candle for candle in candles if candle.timestamp.timestamp() == current_interval_start), None)
    return closed, current


async def _fetch_candles(
    rest: GateRestClient,
    symbol: str,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[list[Candle], Candle | None, datetime]:
    async with semaphore:
        fetched = await asyncio.to_thread(
            rest.fetch_candles,
            symbol,
            config.indicator.candle_interval,
            config.indicator.warmup_candles,
        )
    fetched_at = datetime.now(UTC)
    closed, live = _split_candles(fetched, INTERVAL_SECONDS[config.indicator.candle_interval], fetched_at)
    return closed, live, fetched_at


async def _resync_stale_symbols(
    detector: AtrMoveDetector,
    rest: GateRestClient,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    """重连后为断线期间失效的合约回补 K 线并重建 ATR，失败的合约按退避重试直到全部完成。"""
    delay = config.gate.reconnect_initial_seconds
    resynced = 0

    async def resync(symbol: str) -> bool:
        try:
            closed, live, fetched_at = await _fetch_candles(rest, symbol, config, semaphore)
        except Exception as exc:
            LOGGER.warning("%s 断线后 K 线回补失败：%s", symbol, exc)
            return False
        detector.resync_symbol(symbol, closed, live, fetched_at)
        return True

    while True:
        # 每轮重新读取：回补期间合约池刷新可能已移除部分合约。
        symbols = detector.stale_symbols
        if not symbols:
            return
        results = await asyncio.gather(*(resync(symbol) for symbol in symbols))
        failed = results.count(False)
        resynced += len(symbols) - failed
        if not failed:
            LOGGER.info("已为 %d 个合约回补 K 线并重建 ATR", resynced)
            return
        LOGGER.warning("%d 个合约 K 线回补失败，暂停其异动判定，%.1f 秒后重试", failed, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, config.gate.reconnect_max_seconds)


async def _sync_universe(
    detector: AtrMoveDetector,
    rest: GateRestClient,
    selected: list[ContractTicker],
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    selected_by_symbol = {ticker.symbol: ticker for ticker in selected}
    current = set(detector.symbols)
    target = set(selected_by_symbol)
    detector.remove_symbols(current - target)

    for symbol in current & target:
        ticker = selected_by_symbol[symbol]
        detector.add_symbol(symbol, [], ticker.volume_24h_quote, price_decimals=ticker.price_decimals)

    async def add_new_symbol(symbol: str) -> None:
        closed: list[Candle] = []
        live: Candle | None = None
        try:
            closed, live, _ = await _fetch_candles(rest, symbol, config, semaphore)
        except Exception as exc:
            # 单个新品种预热失败时仍加入监控，它会在实时 K 线积累后自行就绪。
            LOGGER.warning("%s ATR 预热失败：%s", symbol, exc)
        ticker = selected_by_symbol[symbol]
        detector.add_symbol(symbol, closed, ticker.volume_24h_quote, live, ticker.price_decimals)

    await asyncio.gather(*(add_new_symbol(symbol) for symbol in sorted(target - current)))


async def _refresh_universe(
    detector: AtrMoveDetector,
    rest: GateRestClient,
    feed: GateTradeFeed,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> list[ContractTicker]:
    raw_tickers, raw_contracts = await asyncio.gather(
        asyncio.to_thread(rest.fetch_tickers),
        asyncio.to_thread(rest.fetch_contracts),
    )
    selected = select_liquid_contracts(
        raw_tickers,
        raw_contracts,
        config.gate.min_volume_24h_quote,
        retained_symbols=detector.symbols,
        exit_volume_ratio=config.gate.universe_exit_volume_ratio,
    )
    if not selected:
        raise RuntimeError("Gate.io 没有满足成交额条件的可用虚拟币合约")
    await _sync_universe(detector, rest, selected, config, semaphore)
    # 先完成预热再订阅，新合约的第一笔实时成交到达时 ATR 已经就绪。
    await feed.set_symbols(detector.symbols)
    LOGGER.info(
        "交易对池已刷新：%d 个虚拟币合约，24h 计价成交额门槛 %.1fM USDT",
        len(selected),
        config.gate.min_volume_24h_quote / 1_000_000,
    )
    return selected


async def _initial_universe(
    detector: AtrMoveDetector,
    rest: GateRestClient,
    feed: GateTradeFeed,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    delay = config.gate.reconnect_initial_seconds
    while True:
        try:
            await _refresh_universe(detector, rest, feed, config, semaphore)
            return
        except Exception as exc:
            LOGGER.error("交易对池初始化失败：%s；%.1f 秒后重试", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, config.gate.reconnect_max_seconds)


async def _universe_loop(
    detector: AtrMoveDetector,
    rest: GateRestClient,
    feed: GateTradeFeed,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    refresh_seconds = config.gate.universe_refresh_seconds
    delay = refresh_seconds
    while True:
        await asyncio.sleep(delay)
        try:
            await _refresh_universe(detector, rest, feed, config, semaphore)
            delay = refresh_seconds
        except Exception as exc:
            # 刷新失败时沿用现有合约池继续监控，稍后重试即可，不影响实时连接。
            delay = min(60, refresh_seconds)
            LOGGER.error("交易对池刷新失败：%s；%d 秒后重试", exc, delay)


async def _stop_task(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # 只吞掉被取消任务自身的取消；当前协程也在被取消时必须继续向上传播。
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except Exception as exc:
        LOGGER.error("K 线回补任务异常退出：%s", exc)


async def _stream_loop(
    detector: AtrMoveDetector,
    feed: GateTradeFeed,
    rest: GateRestClient,
    dispatcher: AlertDispatcher,
    config: AppConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    reconnect_delay = config.gate.reconnect_initial_seconds
    last_status = time.monotonic()
    tick_count = 0
    had_stream_gap = False

    while True:
        connected = False
        resync_task: asyncio.Task[None] | None = None
        LOGGER.info("连接 Gate.io 实时成交：%d 个合约", len(feed.symbols))
        try:
            # aclosing 保证循环体抛错时也会立即关闭 WebSocket，而不是等垃圾回收。
            async with contextlib.aclosing(feed.stream()) as ticks:
                async for tick in ticks:
                    # 链路拥塞时成交会在网络中积压，推送的仍是几十秒前的行情：此时秒级判定
                    # 已失去意义，提醒发出时价格早已改变。滞后超限就主动断开——重连能清空
                    # 积压，也让日志直指真因，而不是等 keepalive 因 pong 被积压数据队头
                    # 阻塞而报出含义不明的 1011。
                    # 逐笔校验而不做节流：datetime.now 不到 1 微秒，相比 add_tick 可以忽略，
                    # 按时间间隔抽查反而会留出放行积压成交的窗口。
                    # 校验必须排在下面的“连接成功”之前：滞后的成交既不算判定依据，也不算
                    # 连接可用的证据——否则每次重连的首笔滞后成交都会重置退避，拥塞或本地
                    # 时钟偏移时会退化成每秒一次的重连风暴（每次都重新订阅全部合约）。
                    # 注意这里拿本地时钟与交易所时间戳直接相减（检测器内部只做时间戳之间的
                    # 相对比较，不依赖本地时钟）：本地时钟若快于交易所超过阈值，会被误判为
                    # 滞后而一直重连，此时应先校准系统时间而不是调高阈值。
                    lag = (datetime.now(UTC) - tick.timestamp).total_seconds()
                    if lag > config.gate.max_data_lag_seconds:
                        raise ConnectionError(
                            f"行情数据滞后 {lag:.1f} 秒（上限 {config.gate.max_data_lag_seconds:g} 秒），"
                            "网络链路拥塞，主动重连以清空积压"
                        )

                    if not connected:
                        connected = True
                        reconnect_delay = config.gate.reconnect_initial_seconds
                        LOGGER.info("Gate.io 实时行情连接成功，已收到 %s 成交", tick.symbol)
                        if had_stream_gap:
                            # 断线退避期间合约池刷新新增的合约，其预热数据同样早于本次重连，需在此重新标记。
                            detector.mark_stream_gap()
                            # 必须在实时成交恢复之后再拉 K 线：请求之前的缺口由 REST 覆盖，之后的成交由实时流覆盖。
                            resync_task = asyncio.create_task(
                                _resync_stale_symbols(detector, rest, config, semaphore),
                                name="resync-stale-atr",
                            )

                    tick_count += 1
                    current_time = time.monotonic()
                    for alert in detector.add_tick(tick):
                        dispatcher.publish(alert)

                    if current_time - last_status >= config.gate.status_interval_seconds:
                        rejected = len(feed.rejected_symbols)
                        LOGGER.info(
                            "监控正常：%d 个合约%s，累计 %s 条成交",
                            len(detector.symbols),
                            f"（{rejected} 个订阅被拒绝）" if rejected else "",
                            f"{tick_count:,}",
                        )
                        last_status = current_time
            raise ConnectionError("Gate.io 行情流意外结束")
        except Exception as exc:
            # 连接、握手、网络超时都会走到这里；合约池刷新已与连接解耦，任何异常都按故障退避重连。
            LOGGER.warning("Gate.io 行情连接异常：%s；%.1f 秒后重连", exc, reconnect_delay)
        finally:
            # 断线后本轮回补出的 K 线又会与新的缺口不一致，停止回补，等下次连接恢复后重新开始。
            await _stop_task(resync_task)
        detector.mark_stream_gap()
        had_stream_gap = True
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, config.gate.reconnect_max_seconds)


async def run_monitor(config: AppConfig) -> None:
    detector = build_detector(config)
    rest = build_rest_client(config)
    feed = GateTradeFeed(
        config.gate.websocket_url,
        config.gate.subscription_chunk_size,
        config.gate.receive_timeout_seconds,
    )
    # 合约池刷新与断线回补共用一个信号量：两个循环并行运行，各自持有一个的话峰值并发会翻倍。
    warmup_semaphore = asyncio.Semaphore(config.gate.warmup_concurrency)
    await _initial_universe(detector, rest, feed, config, warmup_semaphore)
    async with AlertDispatcher(build_notifiers(config.alerts), config.alerts.queue_size) as dispatcher:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(_universe_loop(detector, rest, feed, config, warmup_semaphore))
            tasks.create_task(_stream_loop(detector, feed, rest, dispatcher, config, warmup_semaphore))
