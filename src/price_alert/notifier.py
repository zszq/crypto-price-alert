"""控制台、JSONL 与通用 Webhook 提醒通道。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Protocol

from colorama import Fore, Style, just_fix_windows_console

from price_alert import streak as streak_alerts
from price_alert.config import AlertConfig
from price_alert.detection import Alert
from price_alert.formatting import beijing_time, describe_window, format_price
from price_alert.gate import futures_trade_url
from price_alert.launch import alerts as launch_alerts
from price_alert.launch.alerts import LaunchAlert
from price_alert.models import PriceAlert
from price_alert.streak import StreakAlert
from price_alert.trend import alerts as trend_alerts
from price_alert.trend.alerts import TrendAlert

LOGGER = logging.getLogger(__name__)
MACOS_SOUND_PLAYER = "/usr/bin/afplay"
MACOS_ALERT_SOUND = "/System/Library/Sounds/Glass.aiff"
# 控制台颜色只用标准 16 色，各终端都能显示。
# 正文：急涨绿色、急跌红色。
SURGE_COLOR = Fore.GREEN
DROP_COLOR = Fore.RED
# 交易对：亮黄色。
SYMBOL_COLOR = Fore.LIGHTYELLOW_EX
# 涨跌幅：同方向的亮绿/亮红。
SURGE_CHANGE_COLOR = Fore.LIGHTGREEN_EX
DROP_CHANGE_COLOR = Fore.LIGHTRED_EX
# 交易地址：灰色，辅助信息不抢眼。
TRADE_URL_COLOR = Fore.LIGHTBLACK_EX
# 长窗口的窗口长度（3分钟）：亮青色。
LONG_WINDOW_COLOR = Fore.LIGHTCYAN_EX
# 趋势提醒的标签（[趋势下跌] 等）：亮紫色，与秒级异动提醒一眼区分。
TREND_LABEL_COLOR = Fore.LIGHTMAGENTA_EX
# 放量启动提醒的标签（[放量拉升] 等）：亮蓝色，与秒级异动、趋势提醒区分。
LAUNCH_LABEL_COLOR = Fore.LIGHTBLUE_EX
# 连续提醒的标签（[连续急涨] 等）：亮白色，与它前面那条普通短窗口提醒区分开。
STREAK_LABEL_COLOR = Fore.LIGHTWHITE_EX


class Notifier(Protocol):
    async def send(self, alert: Alert) -> None: ...


def format_alert(alert: Alert) -> str:
    if isinstance(alert, TrendAlert):
        return trend_alerts.format_trend_alert(alert)
    if isinstance(alert, LaunchAlert):
        return launch_alerts.format_launch_alert(alert)
    if isinstance(alert, StreakAlert):
        return streak_alerts.format_streak_alert(alert)
    label = "急涨" if alert.direction == "surge" else "急跌"
    move_label = "上涨" if alert.direction == "surge" else "下跌"
    reference = format_price(alert.reference_price, alert.price_decimals)
    price = format_price(alert.price, alert.price_decimals)
    return (
        f"[{label}提醒] {beijing_time(alert.timestamp)} | {alert.symbol} | "
        f"{format_window(alert)}内价格{move_label} "
        f"{format_change(alert)} | {reference} → {price}"
        f" | 异动强度 {alert.move_atr:.2f} ATR"
    )


def format_change(alert: PriceAlert) -> str:
    # format_alert 与 colorize_alert 共用同一格式，保证着色时能在文本中准确找到百分比。
    return f"{abs(alert.change_percent):.2f}%"


def format_window(alert: PriceAlert) -> str:
    # 与 format_change 同理：着色靠在文本中查找这段字样，两处必须出自同一个函数。
    return describe_window(alert.lookback_seconds)


def colorize_alert(alert: Alert, text: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    color = SURGE_COLOR if alert.direction == "surge" else DROP_COLOR
    bright = SURGE_CHANGE_COLOR if alert.direction == "surge" else DROP_CHANGE_COLOR
    # 标记结束后重新套上方向色，保证后半段文本颜色不丢。
    symbol = f"{SYMBOL_COLOR}{alert.symbol}{Style.RESET_ALL}{color}"
    if isinstance(alert, TrendAlert):
        change = trend_alerts.format_change(alert)
    elif isinstance(alert, LaunchAlert):
        change = launch_alerts.format_change(alert)
    elif isinstance(alert, StreakAlert):
        change = streak_alerts.format_change(alert)
    else:
        change = format_change(alert)
    highlighted_change = f"{bright}{change}{Style.RESET_ALL}{color}"
    text = text.replace(alert.symbol, symbol, 1).replace(change, highlighted_change, 1)
    if isinstance(alert, TrendAlert | LaunchAlert | StreakAlert):
        # 标签在行首，按首次出现替换不会误伤正文。
        label = f"[{alert.label}]"
        if isinstance(alert, TrendAlert):
            label_color = TREND_LABEL_COLOR
        elif isinstance(alert, LaunchAlert):
            label_color = LAUNCH_LABEL_COLOR
        else:
            label_color = STREAK_LABEL_COLOR
        text = text.replace(label, f"{label_color}{label}{Style.RESET_ALL}{color}", 1)
    elif alert.window == "long":
        # 只高亮长窗口，短窗口保持正文色。窗口字样带“秒”或“分钟”，按首次出现替换不会误伤。
        window = format_window(alert)
        text = text.replace(window, f"{LONG_WINDOW_COLOR}{window}{Style.RESET_ALL}{color}", 1)
    return f"{color}{text}{Style.RESET_ALL}"


class ConsoleNotifier:
    def __init__(self, beep: bool = True, colors: bool = True) -> None:
        self.beep = beep
        self.colors = colors
        # macOS 终端通常会忽略或禁用 ASCII 响铃，直接播放系统音效才能稳定发声。
        self._system_sound = beep and sys.platform == "darwin"
        # 持有引用，避免后台播放任务被垃圾回收，也用于判断上一次是否仍在播放。
        self._sound_task: asyncio.Task[None] | None = None
        if colors:
            # Windows 控制台实现差异较大，初始化兼容层可避免直接显示转义字符。
            just_fix_windows_console()

    async def send(self, alert: Alert) -> None:
        text = colorize_alert(alert, format_alert(alert), self.colors)
        # 完整网址独占一行，便于终端自动识别链接，不支持点击时也能直接复制。
        # 颜色转义只包在整行首尾、不插入网址中间，终端按显示文本识别链接，不受影响。
        url_line = f"交易地址：{futures_trade_url(alert.symbol)}"
        if self.colors:
            url_line = f"{TRADE_URL_COLOR}{url_line}{Style.RESET_ALL}"
        terminal_bell = "\a" if self.beep and not self._system_sound else ""
        print(f"{terminal_bell}{text}\n{url_line}", flush=True)
        # 音效约 1.65 秒，等待播完会让集中异动时的文字提醒逐条排队延迟，所以放到后台；
        # 上一次仍在播放时直接跳过，一波异动只响一次。
        if self._system_sound and (self._sound_task is None or self._sound_task.done()):
            self._sound_task = asyncio.create_task(self._play_macos_sound(), name="console-alert-sound")

    @staticmethod
    async def _play_macos_sound() -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                MACOS_SOUND_PLAYER,
                MACOS_ALERT_SOUND,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            # 提示音只是附加提醒，文字已经输出，失败时只记日志。
            LOGGER.warning("macOS 提示音播放失败：%s", exc)
            return
        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            # 退出时事件循环会取消后台任务；结束并回收子进程，避免 afplay 脱离事件循环后成为孤儿进程。
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        if return_code != 0:
            LOGGER.warning("macOS 提示音播放失败，afplay 退出码：%s", return_code)


class JsonlNotifier:
    def __init__(self, path: Path, max_bytes: int = 0, backup_count: int = 5) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    async def send(self, alert: Alert) -> None:
        await asyncio.to_thread(self._append, alert)

    def _append(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")

    def _rotate_if_needed(self) -> None:
        # 按整文件轮转而不是截断，保证每个文件里的每一行仍是完整 JSON。
        if self.max_bytes <= 0 or not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))


class WebhookNotifier:
    def __init__(self, url: str, timeout_seconds: float = 5.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    async def send(self, alert: Alert) -> None:
        await asyncio.to_thread(self._post, alert)

    def _post(self, alert: Alert) -> None:
        payload = alert.to_dict() | {"text": format_alert(alert)}
        request = urllib.request.Request(  # noqa: S310 - URL 由用户配置
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        # urlopen 遇到 4xx/5xx 会直接抛出 HTTPError，由分发器统一记录。
        with urllib.request.urlopen(request, timeout=self.timeout_seconds):  # noqa: S310
            pass


class AlertDispatcher:
    """每个通道独立的队列与后台任务：慢 Webhook 既不阻塞行情处理，也不拖慢控制台提醒。"""

    def __init__(self, notifiers: Iterable[Notifier], queue_size: int = 1000, drain_timeout: float = 5.0) -> None:
        self.notifiers = list(notifiers)
        self.drain_timeout = drain_timeout
        self._queues: list[asyncio.Queue[Alert]] = [asyncio.Queue(queue_size) for _ in self.notifiers]
        self._workers: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> AlertDispatcher:
        self._workers = [
            asyncio.create_task(self._run(notifier, queue), name=f"notifier-{type(notifier).__name__}")
            for notifier, queue in zip(self.notifiers, self._queues, strict=True)
        ]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def publish(self, alert: Alert) -> None:
        for notifier, queue in zip(self.notifiers, self._queues, strict=True):
            try:
                queue.put_nowait(alert)
            except asyncio.QueueFull:
                # 宁可丢弃单个通道的提醒，也不能让积压反压到行情接收导致全市场断线。
                LOGGER.error("提醒通道 %s 积压已满，丢弃 %s 提醒", type(notifier).__name__, alert.symbol)

    async def aclose(self) -> None:
        if not self._workers:
            return
        # 停止前尽量把已触发的提醒发完；超时说明通道卡死，不能无限期阻塞退出。
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self.drain_timeout):
                await asyncio.gather(*(queue.join() for queue in self._queues))
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    @staticmethod
    async def _run(notifier: Notifier, queue: asyncio.Queue[Alert]) -> None:
        while True:
            alert = await queue.get()
            try:
                await notifier.send(alert)
            except Exception as exc:
                # 单个提醒出口失败不应中断全市场行情监控。
                LOGGER.error("提醒通道 %s 发送失败：%s", type(notifier).__name__, exc)
            finally:
                queue.task_done()


def build_notifiers(config: AlertConfig) -> list[Notifier]:
    notifiers: list[Notifier] = []
    if config.console:
        notifiers.append(ConsoleNotifier(config.beep, config.console_colors))
    if config.jsonl_path is not None:
        notifiers.append(JsonlNotifier(config.jsonl_path, config.jsonl_max_bytes, config.jsonl_backup_count))
    if config.webhook_url:
        notifiers.append(WebhookNotifier(config.webhook_url, config.webhook_timeout_seconds))
    return notifiers
