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

from price_alert.detectors.alerts import Alert
from price_alert.detectors.dormant import alerts as dormant_alerts
from price_alert.detectors.dormant.alerts import DormantAlert
from price_alert.detectors.launch import alerts as launch_alerts
from price_alert.detectors.launch.alerts import LaunchAlert
from price_alert.detectors.move import alerts as move_alerts
from price_alert.detectors.move import streak as streak_alerts
from price_alert.detectors.move.streak import StreakAlert
from price_alert.detectors.reversal import alerts as reversal_alerts
from price_alert.detectors.reversal.alerts import ReversalAlert
from price_alert.detectors.trend import alerts as trend_alerts
from price_alert.detectors.trend.alerts import TrendAlert
from price_alert.exchange.gate import futures_trade_url
from price_alert.notify import sounds
from price_alert.notify.config import AlertConfig

try:
    import winsound
except ImportError:  # 只有 Windows 有 winsound
    winsound = None

LOGGER = logging.getLogger(__name__)
MACOS_SOUND_PLAYER = "/usr/bin/afplay"
# winsound 按音效时长判断是否播完，但声音要晚一些才出来（设备唤醒、缓冲），不多等会被下一个 PlaySound 切掉尾音。
WINDOWS_PLAYBACK_MARGIN_SECONDS = 0.3
# 必须完整播放的音效开播后这么久以内到达的同类提醒不再补响：同一分钟结算的一批放量启动几乎同时到达，补响像重复。
MUST_PLAY_MERGE_SECONDS = 0.3
# 提示音文件重新生成失败的警告间隔：失败通常是持续的（目录不可写），每条提醒都记会刷屏。
SOUND_REGENERATE_LOG_INTERVAL_SECONDS = 60.0
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
# 放量启动提醒的标签（[放量上涨] 等）：金黄色（标准黄），偏橙，与淡黄的交易对分得开，不带涨跌含义。
LAUNCH_LABEL_COLOR = Fore.YELLOW
# 冷门启动提醒的标签：亮蓝色，与放量启动分开，准确率较低的这一类一眼就能认出。
DORMANT_LABEL_COLOR = Fore.LIGHTBLUE_EX
# 反弹回落提醒的标签（[下跌反弹]/[冲高回落]）：亮白色，不与方向色、交易对、其他标签撞色；
# 正文仍按急动方向着色，靠标签与秒级异动区分。
REVERSAL_LABEL_COLOR = Fore.LIGHTWHITE_EX


class Notifier(Protocol):
    async def send(self, alert: Alert) -> None: ...


def format_alert(alert: Alert) -> str:
    if isinstance(alert, TrendAlert):
        return trend_alerts.format_trend_alert(alert)
    if isinstance(alert, LaunchAlert):
        return launch_alerts.format_launch_alert(alert)
    if isinstance(alert, DormantAlert):
        return dormant_alerts.format_dormant_alert(alert)
    if isinstance(alert, ReversalAlert):
        return reversal_alerts.format_reversal_alert(alert)
    if isinstance(alert, StreakAlert):
        return streak_alerts.format_streak_alert(alert)
    return move_alerts.format_move_alert(alert)


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
    elif isinstance(alert, DormantAlert):
        change = dormant_alerts.format_change(alert)
    elif isinstance(alert, ReversalAlert):
        change = reversal_alerts.format_change(alert)
    elif isinstance(alert, StreakAlert):
        change = streak_alerts.format_change(alert)
    else:
        change = move_alerts.format_change(alert)
    highlighted_change = f"{bright}{change}{Style.RESET_ALL}{color}"
    text = text.replace(alert.symbol, symbol, 1).replace(change, highlighted_change, 1)
    if isinstance(alert, TrendAlert | LaunchAlert | DormantAlert | ReversalAlert):
        # 标签在行首，按首次出现替换不会误伤正文。
        label = f"[{alert.label}]"
        if isinstance(alert, TrendAlert):
            label_color = TREND_LABEL_COLOR
        elif isinstance(alert, LaunchAlert):
            label_color = LAUNCH_LABEL_COLOR
        elif isinstance(alert, ReversalAlert):
            label_color = REVERSAL_LABEL_COLOR
        else:
            label_color = DORMANT_LABEL_COLOR
        text = text.replace(label, f"{label_color}{label}{Style.RESET_ALL}{color}", 1)
    elif isinstance(alert, StreakAlert):
        # 连续提醒是秒级异动的延伸，标签沿用方向色，靠文字区分即可。
        pass
    elif alert.window == "long":
        # 只高亮长窗口，短窗口保持正文色。窗口字样带“秒”或“分钟”，按首次出现替换不会误伤。
        window = move_alerts.format_window(alert)
        text = text.replace(window, f"{LONG_WINDOW_COLOR}{window}{Style.RESET_ALL}{color}", 1)
    return f"{color}{text}{Style.RESET_ALL}"


class ConsoleNotifier:
    def __init__(self, beep: bool = True, colors: bool = True, sound_dir: Path | None = None) -> None:
        self.beep = beep
        self.colors = colors
        self._macos = sys.platform == "darwin"
        self._sound_dir = sound_dir or sounds.default_sound_dir()
        self._sound_files = self._prepare_sounds() if beep else None
        self._last_regenerate_failure: float | None = None
        # 持有引用，避免后台播放任务被垃圾回收，也用于判断上一次是否仍在播放。
        self._sound_task: asyncio.Task[None] | None = None
        self._playing_priority = 0
        self._playing_since = 0.0
        self._replay_pending = False
        if colors:
            # Windows 控制台实现差异较大，初始化兼容层可避免直接显示转义字符。
            just_fix_windows_console()

    def _prepare_sounds(self) -> dict[sounds.SoundName, Path] | None:
        # macOS 终端常禁用响铃，Windows 的响铃声取决于终端与系统设置，也分不出提醒类型，
        # 所以这两个平台播放同一份合成音效；其他平台没有可依赖的播放器，仍用终端响铃。
        if not self._macos and not (sys.platform == "win32" and winsound is not None):
            return None
        try:
            return sounds.write_sound_files(self._sound_dir)
        except OSError as exc:
            LOGGER.warning("提示音文件生成失败，改用终端响铃：%s", exc)
            return None

    async def send(self, alert: Alert) -> None:
        text = colorize_alert(alert, format_alert(alert), self.colors)
        # 网址独占一行，便于终端识别链接或直接复制；颜色转义只包在整行首尾，不插进网址中间。
        url_line = f"交易地址：{futures_trade_url(alert.symbol)}"
        if self.colors:
            url_line = f"{TRADE_URL_COLOR}{url_line}{Style.RESET_ALL}"
        terminal_bell = "\a" if self.beep and self._sound_files is None else ""
        print(f"{terminal_bell}{text}\n{url_line}", flush=True)
        if self._sound_files is not None:
            self._play(sounds.sound_for(alert))

    async def play_and_wait(self, name: sounds.SoundName) -> bool:
        """试听用：播放一种音效并等它播完；当前平台没有内置音效或音效文件不可用时返回 False。"""
        if self._sound_files is None:
            return False
        # 以这次是否真的开播为准：文件重新生成失败时没有新任务，只看 _sound_task 会误报成功。
        if not self._play(name):
            return False
        assert self._sound_task is not None
        await self._sound_task
        return True

    def _sound_path(self, name: sounds.SoundName, now: float) -> Path | None:
        """返回可播放的音效文件；文件在运行中被删掉时重新生成，生成失败返回 None。"""
        assert self._sound_files is not None
        path = self._sound_files[name]
        # 每次播放前都检查：文件被清理工具删掉后 winsound 静默不出声、日志也没有痕迹。一次 stat 开销可忽略。
        if path.exists():
            return path
        try:
            self._sound_files = sounds.write_sound_files(self._sound_dir)
        except OSError as exc:
            # 目录持续不可写时每条提醒都会走到这里，限流避免刷屏；文字提醒不受影响。
            last = self._last_regenerate_failure
            if last is None or now - last >= SOUND_REGENERATE_LOG_INTERVAL_SECONDS:
                self._last_regenerate_failure = now
                LOGGER.warning("提示音文件已被删除，重新生成失败，本次不发声：%s", exc)
            return None
        LOGGER.warning("提示音文件已被删除（可能是系统清理了文件），已重新生成：%s", self._sound_dir)
        return self._sound_files[name]

    def _play(self, name: sounds.SoundName) -> bool:
        """返回这次是否开始了新的播放；被跳过、并入补响或文件不可用时返回 False。"""
        # 放到后台播放：等待播完会让集中异动时的文字提醒排队延迟。正在播放时只让更重要的提醒打断，
        # 同级或更低的跳过：一波异动只响一次，又不会因秒级异动正在响而漏掉放量启动的声音。
        priority = sounds.SOUND_PRIORITY[name]
        now = asyncio.get_running_loop().time()
        if self._sound_task is not None and not self._sound_task.done():
            if priority < self._playing_priority:
                return False
            if priority == self._playing_priority:
                # 必须完整播放的音效不被同类打断，也不丢：播完后补响一次（多条合并成一次）。
                if name in sounds.MUST_PLAY and now - self._playing_since >= MUST_PLAY_MERGE_SECONDS:
                    self._replay_pending = True
                return False
        # 放在优先级判断之后：被跳过的提醒不需要文件，也不必为它重新生成。
        path = self._sound_path(name, now)
        if path is None:
            return False
        if self._sound_task is not None and not self._sound_task.done():
            self._sound_task.cancel()
        self._playing_priority = priority
        self._playing_since = now
        self._replay_pending = False
        self._sound_task = asyncio.create_task(self._play_sound(name, path), name="console-alert-sound")
        return True

    async def _play_sound(self, name: sounds.SoundName, path: Path) -> None:
        while True:
            if self._macos:
                await self._play_macos_sound(path)
            else:
                await self._play_windows_sound(path, sounds.sound_duration(name))
            if not self._replay_pending:
                return
            self._replay_pending = False
            self._playing_since = asyncio.get_running_loop().time()

    @staticmethod
    async def _play_macos_sound(path: Path) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                MACOS_SOUND_PLAYER,
                str(path),
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
            # 被打断或退出时取消任务：结束并回收子进程，让声音立即停下，也不留下孤儿进程。
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        if return_code != 0:
            LOGGER.warning("macOS 提示音播放失败，afplay 退出码：%s", return_code)

    @staticmethod
    async def _play_windows_sound(path: Path, duration: float) -> None:
        try:
            # SND_ASYNC 立即返回，且新的播放会自动停掉上一个，打断时不用另外停止。
            # 不放进 to_thread：多个线程的执行先后不确定，被打断的音效可能反而后播。
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as exc:
            LOGGER.warning("Windows 提示音播放失败：%s", exc)
            return
        # winsound 没有播放结束通知，按时长加出声延迟等待，任务存活期间视为正在播放，供优先级判断。
        await asyncio.sleep(duration + WINDOWS_PLAYBACK_MARGIN_SECONDS)


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
