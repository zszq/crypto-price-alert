import asyncio
import contextlib
import json
from dataclasses import replace
from datetime import UTC, datetime

from colorama import Fore, Style

from price_alert.models import PriceAlert
from price_alert.notifier import (
    DROP_CHANGE_COLOR,
    MACOS_ALERT_SOUND,
    MACOS_SOUND_PLAYER,
    SURGE_CHANGE_COLOR,
    SYMBOL_COLOR,
    TRADE_URL_COLOR,
    AlertDispatcher,
    ConsoleNotifier,
    JsonlNotifier,
    colorize_alert,
    format_alert,
)


def test_jsonl_notifier_writes_atr_alert(tmp_path):
    alert = PriceAlert(
        symbol="BTC_USDT",
        direction="surge",
        price=101.0,
        reference_price=100.0,
        change_percent=1.0,
        move_atr=1.25,
        atr=0.8,
        atr_period=14,
        lookback_seconds=30,
        trade_count=10,
        volume_24h_quote=1_000_000_000,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    target = tmp_path / "alerts.jsonl"

    asyncio.run(JsonlNotifier(target).send(alert))

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["move_atr"] == 1.25
    assert payload["color"] == "green"
    text = format_alert(alert)
    assert "2026-01-01 08:00:00" in text
    assert "价格上涨 1.00%" in text
    assert "异动强度 1.25 ATR" in text
    assert "ATR(14)=" not in text
    assert "24h成交额" not in text
    assert colorize_alert(alert, "surge") == f"{Fore.GREEN}surge{Style.RESET_ALL}"


def test_drop_alert_is_red():
    alert = PriceAlert(
        symbol="ETH_USDT",
        direction="drop",
        price=99.0,
        reference_price=100.0,
        change_percent=-1.0,
        move_atr=1.25,
        atr=0.8,
        atr_period=14,
        lookback_seconds=30,
        trade_count=10,
        volume_24h_quote=1_000_000_000,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert alert.to_dict()["color"] == "red"
    assert "价格下跌 1.00%" in format_alert(alert)
    assert colorize_alert(alert, "drop") == f"{Fore.RED}drop{Style.RESET_ALL}"


def test_colorize_highlights_symbol():
    alert = make_alert("ETH_USDT")

    text = colorize_alert(alert, "[急涨提醒] | ETH_USDT | 价格上涨")

    symbol = f"{SYMBOL_COLOR}ETH_USDT{Style.RESET_ALL}{Fore.GREEN}"
    assert text == f"{Fore.GREEN}[急涨提醒] | {symbol} | 价格上涨{Style.RESET_ALL}"
    assert colorize_alert(alert, "ETH_USDT", enabled=False) == "ETH_USDT"


def test_colorize_highlights_change_percent():
    surge = make_alert()
    text = colorize_alert(surge, "价格上涨 1.00% | 100 → 101")
    change = f"{SURGE_CHANGE_COLOR}1.00%{Style.RESET_ALL}{Fore.GREEN}"
    assert text == f"{Fore.GREEN}价格上涨 {change} | 100 → 101{Style.RESET_ALL}"

    # 急跌时 change_percent 为负，文本中显示绝对值，着色也要匹配到绝对值。
    drop = replace(make_alert(), direction="drop", price=99.0, change_percent=-1.0)
    text = colorize_alert(drop, "价格下跌 1.00% | 100 → 99")
    change = f"{DROP_CHANGE_COLOR}1.00%{Style.RESET_ALL}{Fore.RED}"
    assert text == f"{Fore.RED}价格下跌 {change} | 100 → 99{Style.RESET_ALL}"


def make_alert(symbol: str = "BTC_USDT") -> PriceAlert:
    return PriceAlert(
        symbol=symbol,
        direction="surge",
        price=101.0,
        reference_price=100.0,
        change_percent=1.0,
        move_atr=1.25,
        atr=0.8,
        atr_period=14,
        lookback_seconds=30,
        trade_count=10,
        volume_24h_quote=1_000_000_000,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeSoundProcess:
    """可控制播放何时结束的 afplay 替身。"""

    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.finished = asyncio.Event()
        self.killed = False

    async def wait(self) -> int:
        await self.finished.wait()
        return self.return_code

    def kill(self) -> None:
        self.killed = True
        self.finished.set()


def patch_macos_sound(monkeypatch, return_code: int = 0) -> list[tuple[tuple, FakeSoundProcess]]:
    calls: list[tuple[tuple, FakeSoundProcess]] = []

    async def create_subprocess_exec(*args, **kwargs):
        process = FakeSoundProcess(return_code)
        calls.append((args, process))
        return process

    monkeypatch.setattr("price_alert.notifier.sys.platform", "darwin")
    monkeypatch.setattr("price_alert.notifier.asyncio.create_subprocess_exec", create_subprocess_exec)
    return calls


def test_console_notifier_uses_afplay_on_macos(monkeypatch, capsys):
    calls = patch_macos_sound(monkeypatch)

    async def scenario():
        notifier = ConsoleNotifier(beep=True, colors=False)
        await notifier.send(make_alert())
        await asyncio.sleep(0)
        calls[0][1].finished.set()
        await notifier._sound_task

    asyncio.run(scenario())

    assert calls[0][0] == (MACOS_SOUND_PLAYER, MACOS_ALERT_SOUND)
    assert "\a" not in capsys.readouterr().out


def test_macos_sound_does_not_delay_burst_and_plays_once(monkeypatch, capsys):
    calls = patch_macos_sound(monkeypatch)

    async def scenario():
        notifier = ConsoleNotifier(beep=True, colors=False)
        # 声音未播完时连续提醒：文字必须立即输出，且不能叠加播放。
        for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT"):
            await notifier.send(make_alert(symbol))
            await asyncio.sleep(0)
        output = capsys.readouterr().out
        assert all(symbol in output for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT"))
        assert len(calls) == 1

        calls[0][1].finished.set()
        await notifier._sound_task
        await notifier.send(make_alert("XRP_USDT"))
        await asyncio.sleep(0)
        assert len(calls) == 2
        calls[1][1].finished.set()
        await notifier._sound_task

    asyncio.run(scenario())


def test_macos_sound_nonzero_exit_is_logged(monkeypatch, caplog):
    calls = patch_macos_sound(monkeypatch, return_code=1)

    async def scenario():
        notifier = ConsoleNotifier(beep=True, colors=False)
        await notifier.send(make_alert())
        await asyncio.sleep(0)
        calls[0][1].finished.set()
        await notifier._sound_task

    asyncio.run(scenario())

    assert "afplay 退出码：1" in caplog.text


def test_cancelled_macos_sound_kills_player(monkeypatch):
    calls = patch_macos_sound(monkeypatch)

    async def scenario():
        notifier = ConsoleNotifier(beep=True, colors=False)
        await notifier.send(make_alert())
        await asyncio.sleep(0)
        notifier._sound_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await notifier._sound_task

    asyncio.run(scenario())

    assert calls[0][1].killed


def test_console_notifier_keeps_terminal_bell_off_macos(monkeypatch, capsys):
    monkeypatch.setattr("price_alert.notifier.sys.platform", "linux")

    asyncio.run(ConsoleNotifier(beep=True, colors=False).send(make_alert()))

    assert capsys.readouterr().out.startswith("\a[急涨提醒]")


def test_macos_sound_failure_does_not_hide_alert(monkeypatch, capsys, caplog):
    async def create_subprocess_exec(*args, **kwargs):
        raise OSError("afplay 不可用")

    monkeypatch.setattr("price_alert.notifier.sys.platform", "darwin")
    monkeypatch.setattr("price_alert.notifier.asyncio.create_subprocess_exec", create_subprocess_exec)

    async def scenario():
        notifier = ConsoleNotifier(beep=True, colors=False)
        await notifier.send(make_alert())
        await notifier._sound_task

    asyncio.run(scenario())

    assert "[急涨提醒]" in capsys.readouterr().out
    assert "macOS 提示音播放失败" in caplog.text


class RecordingNotifier:
    def __init__(self) -> None:
        self.symbols: list[str] = []

    async def send(self, alert: PriceAlert) -> None:
        self.symbols.append(alert.symbol)


class BlockingNotifier:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.symbols: list[str] = []

    async def send(self, alert: PriceAlert) -> None:
        await self.release.wait()
        self.symbols.append(alert.symbol)


class FailingNotifier:
    async def send(self, alert: PriceAlert) -> None:
        raise RuntimeError("webhook down")


def test_slow_channel_does_not_block_publish_or_other_channels():
    async def scenario():
        fast, slow = RecordingNotifier(), BlockingNotifier()
        async with AlertDispatcher([slow, fast]) as dispatcher:
            dispatcher.publish(make_alert("A_USDT"))
            dispatcher.publish(make_alert("B_USDT"))
            for _ in range(5):
                await asyncio.sleep(0)
            fast_before_release = list(fast.symbols)
            slow_before_release = list(slow.symbols)
            slow.release.set()
        return fast_before_release, slow_before_release, slow.symbols

    fast_before, slow_before, slow_after = asyncio.run(scenario())

    assert fast_before == ["A_USDT", "B_USDT"]
    assert slow_before == []
    # 退出时会把已排队的提醒发完。
    assert slow_after == ["A_USDT", "B_USDT"]


def test_failed_channel_is_logged_and_keeps_processing(caplog):
    async def scenario():
        recorder = RecordingNotifier()
        async with AlertDispatcher([FailingNotifier(), recorder]) as dispatcher:
            dispatcher.publish(make_alert("A_USDT"))
            dispatcher.publish(make_alert("B_USDT"))
        return recorder.symbols

    assert asyncio.run(scenario()) == ["A_USDT", "B_USDT"]
    assert "FailingNotifier 发送失败" in caplog.text


def test_full_queue_drops_alert_instead_of_blocking(caplog):
    async def scenario():
        slow = BlockingNotifier()
        dispatcher = AlertDispatcher([slow], queue_size=1, drain_timeout=0.01)
        async with dispatcher:
            dispatcher.publish(make_alert("A_USDT"))
            # 让后台任务取走第一条并卡在发送中，队列才会只剩一个空位。
            await asyncio.sleep(0)
            dispatcher.publish(make_alert("B_USDT"))
            dispatcher.publish(make_alert("C_USDT"))

    asyncio.run(scenario())

    assert "积压已满，丢弃 C_USDT" in caplog.text


def test_jsonl_notifier_rotates_whole_files(tmp_path):
    target = tmp_path / "alerts.jsonl"
    notifier = JsonlNotifier(target, max_bytes=1, backup_count=2)

    for symbol in ("A_USDT", "B_USDT", "C_USDT", "D_USDT"):
        asyncio.run(notifier.send(make_alert(symbol)))

    def symbols(path):
        return [json.loads(line)["symbol"] for line in path.read_text(encoding="utf-8").splitlines()]

    assert symbols(target) == ["D_USDT"]
    assert symbols(tmp_path / "alerts.jsonl.1") == ["C_USDT"]
    assert symbols(tmp_path / "alerts.jsonl.2") == ["B_USDT"]
    assert not (tmp_path / "alerts.jsonl.3").exists()


def test_console_trade_url_keeps_chinese_symbol_readable(monkeypatch, capsys):
    monkeypatch.setattr("price_alert.notifier.sys.platform", "linux")

    asyncio.run(ConsoleNotifier(beep=False, colors=False).send(make_alert(symbol="龙虾_USDT")))

    assert "交易地址：https://www.gate.com/zh/futures/USDT/龙虾_USDT\n" in capsys.readouterr().out


def test_console_trade_url_is_gray_when_colors_enabled(monkeypatch, capsys):
    monkeypatch.setattr("price_alert.notifier.sys.platform", "linux")

    asyncio.run(ConsoleNotifier(beep=False, colors=True).send(make_alert(symbol="BTC_USDT")))

    url = "https://www.gate.com/zh/futures/USDT/BTC_USDT"
    assert f"\n{TRADE_URL_COLOR}交易地址：{url}{Style.RESET_ALL}\n" in capsys.readouterr().out


def test_console_trade_url_still_escapes_ascii_reserved_characters(monkeypatch, capsys):
    monkeypatch.setattr("price_alert.notifier.sys.platform", "linux")

    asyncio.run(ConsoleNotifier(beep=False, colors=False).send(make_alert(symbol="A/B?_USDT")))

    assert "/futures/USDT/A%2FB%3F_USDT\n" in capsys.readouterr().out


def test_console_trade_url_escapes_non_ascii_whitespace_and_invisible_characters(monkeypatch, capsys):
    monkeypatch.setattr("price_alert.notifier.sys.platform", "linux")

    asyncio.run(ConsoleNotifier(beep=False, colors=False).send(make_alert(symbol="龙\u3000虾\u200b_USDT")))

    assert "/futures/USDT/龙%E3%80%80虾%E2%80%8B_USDT\n" in capsys.readouterr().out
