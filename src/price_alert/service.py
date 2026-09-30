"""动态交易对池、ATR 预热、实时检测与重连编排。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from price_alert.assembly import build_detector, build_rest_client
from price_alert.config import INTERVAL_SECONDS, AppConfig
from price_alert.detector import AtrMoveDetector
from price_alert.gate import GateRestClient, GateTradeFeed
from price_alert.models import Candle, ContractTicker
from price_alert.notifier import AlertDispatcher, build_notifiers
from price_alert.universe import select_liquid_contracts

LOGGER = logging.getLogger(__name__)

# 预热几百个合约受限速约束要数十秒，期间没有输出看起来像假死；逐个打印又会刷屏，按时间节流报告进度。
_WARMUP_PROGRESS_SECONDS = 5.0


class _Backoff:
    """指数退避：每次等待后翻倍直到上限。首次初始化、断线回补与重连共用，保证三处退避口径一致。"""

    def __init__(self, initial: float, maximum: float) -> None:
        self._initial = initial
        self._maximum = maximum
        self.delay = initial

    def reset(self) -> None:
        self.delay = self._initial

    async def wait(self) -> None:
        await asyncio.sleep(self.delay)
        self.delay = min(self.delay * 2, self._maximum)


def _split_candles(
    candles: list[Candle],
    interval_seconds: int,
    now: datetime,
) -> tuple[list[Candle], Candle | None]:
    current_interval_start = int(now.timestamp()) // interval_seconds * interval_seconds
    closed = [candle for candle in candles if candle.timestamp.timestamp() < current_interval_start]
    current = next((candle for candle in candles if candle.timestamp.timestamp() == current_interval_start), None)
    return closed, current


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


@dataclass(slots=True)
class _Monitor:
    """实时监控各循环共享的一组依赖：合约池刷新、断线回补与行情循环作用于同一个检测器和订阅。"""

    detector: AtrMoveDetector
    rest: GateRestClient
    feed: GateTradeFeed
    config: AppConfig
    # 合约池刷新与断线回补共用一个信号量：两个循环并行运行，各自持有一个的话峰值并发会翻倍。
    semaphore: asyncio.Semaphore

    def _backoff(self) -> _Backoff:
        gate = self.config.gate
        return _Backoff(gate.reconnect_initial_seconds, gate.reconnect_max_seconds)

    async def fetch_candles(self, symbol: str) -> tuple[list[Candle], Candle | None, datetime]:
        indicator = self.config.indicator
        async with self.semaphore:
            fetched = await asyncio.to_thread(
                self.rest.fetch_candles,
                symbol,
                indicator.candle_interval,
                indicator.warmup_candles,
            )
        fetched_at = datetime.now(UTC)
        closed, live = _split_candles(fetched, INTERVAL_SECONDS[indicator.candle_interval], fetched_at)
        return closed, live, fetched_at

    async def resync_stale_symbols(self) -> None:
        """重连后为断线期间失效的合约回补 K 线并重建 ATR，失败的合约按退避重试直到全部完成。"""
        backoff = self._backoff()
        resynced = 0

        async def resync(symbol: str) -> bool:
            try:
                closed, live, fetched_at = await self.fetch_candles(symbol)
            except Exception as exc:
                LOGGER.warning("%s 断线后 K 线回补失败：%s", symbol, exc)
                return False
            self.detector.resync_symbol(symbol, closed, live, fetched_at)
            return True

        while True:
            # 每轮重新读取：回补期间合约池刷新可能已移除部分合约。
            symbols = self.detector.stale_symbols
            if not symbols:
                return
            results = await asyncio.gather(*(resync(symbol) for symbol in symbols))
            failed = results.count(False)
            resynced += len(symbols) - failed
            if not failed:
                LOGGER.info("已为 %d 个合约回补 K 线并重建 ATR", resynced)
                return
            LOGGER.warning("%d 个合约 K 线回补失败，暂停其异动判定，%.1f 秒后重试", failed, backoff.delay)
            await backoff.wait()

    async def sync_universe(self, selected: list[ContractTicker]) -> None:
        detector = self.detector
        selected_by_symbol = {ticker.symbol: ticker for ticker in selected}
        current = set(detector.symbols)
        target = set(selected_by_symbol)
        detector.remove_symbols(current - target)

        for symbol in current & target:
            ticker = selected_by_symbol[symbol]
            detector.add_symbol(symbol, [], ticker.volume_24h_quote, price_decimals=ticker.price_decimals)

        new_symbols = sorted(target - current)
        if not new_symbols:
            return
        LOGGER.info(
            "开始为 %d 个新合约拉取 K 线预热 ATR（并发 %d）",
            len(new_symbols),
            self.config.gate.warmup_concurrency,
        )
        started = last_report = time.monotonic()
        done = failed = 0

        async def add_new_symbol(symbol: str) -> None:
            nonlocal done, failed, last_report
            closed: list[Candle] = []
            live: Candle | None = None
            try:
                closed, live, _ = await self.fetch_candles(symbol)
            except Exception as exc:
                # 单个新品种预热失败时仍加入监控，它会在实时 K 线积累后自行就绪。
                failed += 1
                LOGGER.warning("%s ATR 预热失败：%s", symbol, exc)
            ticker = selected_by_symbol[symbol]
            detector.add_symbol(symbol, closed, ticker.volume_24h_quote, live, ticker.price_decimals)
            done += 1
            now = time.monotonic()
            if now - last_report >= _WARMUP_PROGRESS_SECONDS and done < len(new_symbols):
                last_report = now
                LOGGER.info("ATR 预热进度：%d/%d", done, len(new_symbols))

        await asyncio.gather(*(add_new_symbol(symbol) for symbol in new_symbols))
        LOGGER.info(
            "ATR 预热完成：%d 个合约%s，耗时 %.1f 秒",
            len(new_symbols),
            f"（{failed} 个失败，等实时 K 线积累后就绪）" if failed else "",
            time.monotonic() - started,
        )

    async def refresh_universe(self) -> list[ContractTicker]:
        gate = self.config.gate
        raw_tickers, raw_contracts = await asyncio.gather(
            asyncio.to_thread(self.rest.fetch_tickers),
            asyncio.to_thread(self.rest.fetch_contracts),
        )
        selected = select_liquid_contracts(
            raw_tickers,
            raw_contracts,
            gate.min_volume_24h_quote,
            retained_symbols=self.detector.symbols,
            exit_volume_ratio=gate.universe_exit_volume_ratio,
        )
        if not selected:
            raise RuntimeError("Gate.io 没有满足成交额条件的可用虚拟币合约")
        await self.sync_universe(selected)
        # 先完成预热再订阅，新合约的第一笔实时成交到达时 ATR 已经就绪。
        await self.feed.set_symbols(self.detector.symbols)
        LOGGER.info(
            "交易对池已刷新：%d 个虚拟币合约，24h 计价成交额门槛 %.1fM USDT",
            len(selected),
            gate.min_volume_24h_quote / 1_000_000,
        )
        return selected

    async def initial_universe(self) -> None:
        backoff = self._backoff()
        # 只在首次初始化时提示：此前控制台还没有任何输出；定时刷新在后台进行、不阻塞监控，无需提示。
        LOGGER.info("正在从 Gate.io 获取合约列表并筛选交易对池…")
        while True:
            try:
                await self.refresh_universe()
                return
            except Exception as exc:
                LOGGER.error("交易对池初始化失败：%s；%.1f 秒后重试", exc, backoff.delay)
                await backoff.wait()

    async def universe_loop(self) -> None:
        refresh_seconds = self.config.gate.universe_refresh_seconds
        delay = refresh_seconds
        while True:
            await asyncio.sleep(delay)
            try:
                await self.refresh_universe()
                delay = refresh_seconds
            except Exception as exc:
                # 刷新失败时沿用现有合约池继续监控，稍后重试即可，不影响实时连接。
                delay = min(60, refresh_seconds)
                LOGGER.error("交易对池刷新失败：%s；%d 秒后重试", exc, delay)

    async def stream_loop(self, dispatcher: AlertDispatcher) -> None:
        gate = self.config.gate
        detector = self.detector
        backoff = self._backoff()
        last_status = time.monotonic()
        tick_count = 0
        had_stream_gap = False

        while True:
            connected = False
            resync_task: asyncio.Task[None] | None = None
            LOGGER.info("连接 Gate.io 实时成交：%d 个合约", len(self.feed.symbols))
            try:
                # aclosing 保证循环体抛错时也会立即关闭 WebSocket，而不是等垃圾回收。
                async with contextlib.aclosing(self.feed.stream()) as ticks:
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
                        if lag > gate.max_data_lag_seconds:
                            raise ConnectionError(
                                f"行情数据滞后 {lag:.1f} 秒（上限 {gate.max_data_lag_seconds:g} 秒），"
                                "网络链路拥塞，主动重连以清空积压"
                            )

                        if not connected:
                            connected = True
                            backoff.reset()
                            LOGGER.info("Gate.io 实时行情连接成功，已收到 %s 成交", tick.symbol)
                            if had_stream_gap:
                                # 断线退避期间合约池刷新新增的合约，其预热数据同样早于本次重连，需在此重新标记。
                                detector.mark_stream_gap()
                                # 必须在实时成交恢复之后再拉 K 线：请求之前的缺口由 REST 覆盖，之后的成交由实时流覆盖。
                                resync_task = asyncio.create_task(
                                    self.resync_stale_symbols(),
                                    name="resync-stale-atr",
                                )

                        tick_count += 1
                        current_time = time.monotonic()
                        for alert in detector.add_tick(tick):
                            dispatcher.publish(alert)

                        if current_time - last_status >= gate.status_interval_seconds:
                            rejected = len(self.feed.rejected_symbols)
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
                LOGGER.warning("Gate.io 行情连接异常：%s；%.1f 秒后重连", exc, backoff.delay)
            finally:
                # 断线后本轮回补出的 K 线又会与新的缺口不一致，停止回补，等下次连接恢复后重新开始。
                await _stop_task(resync_task)
            detector.mark_stream_gap()
            had_stream_gap = True
            await backoff.wait()


async def run_monitor(config: AppConfig) -> None:
    gate = config.gate
    monitor = _Monitor(
        detector=build_detector(config),
        rest=build_rest_client(config),
        feed=GateTradeFeed(gate.websocket_url, gate.subscription_chunk_size, gate.receive_timeout_seconds),
        config=config,
        semaphore=asyncio.Semaphore(gate.warmup_concurrency),
    )
    await monitor.initial_universe()
    async with AlertDispatcher(build_notifiers(config.alerts), config.alerts.queue_size) as dispatcher:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(monitor.universe_loop())
            tasks.create_task(monitor.stream_loop(dispatcher))
