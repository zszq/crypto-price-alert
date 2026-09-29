"""Gate.io 价格异动提醒命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from pydantic import ValidationError

from price_alert.config import INTERVAL_SECONDS, AppConfig, load_config
from price_alert.formatting import BEIJING_TIME
from price_alert.instance import AlreadyRunningError, ProcessLock
from price_alert.models import Candle, PriceTick
from price_alert.notifier import ConsoleNotifier
from price_alert.replay import (
    fetch_replay_data,
    normalize_symbol,
    parse_time,
    plan_replay,
    render_replay,
    run_replay,
)
from price_alert.service import build_detector, build_rest_client, build_windows, run_monitor
from price_alert.universe import select_liquid_contracts
from price_alert.windows import MoveWindow, describe_rule

DEMO_ATR = 2.0
# 只给开始时间时回放的时长：足够覆盖长窗口一个完整周期以及前后的行情。
DEFAULT_REPLAY_MINUTES = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate.io 合约价格异动检测与提醒")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "启动 Gate.io 全市场实时监控"),
        ("universe", "查看当前满足成交额条件的合约"),
        ("check-config", "校验配置文件"),
        ("simulate", "使用合成行情验证 ATR 异动提醒"),
        ("replay", "用历史成交回放某个合约，逐秒解释为什么提醒或没有提醒"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="config/default.yaml")
    _add_replay_arguments(commands.choices["replay"])
    return parser


def _add_replay_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("symbol", help="合约名，如 QNT_USDT；只写 QNT 会自动补 _USDT")
    command.add_argument("--start", required=True, help="开始时间（北京时间），如 '2026-09-29 09:25' 或 09:25（今天）")
    command.add_argument("--end", help=f"结束时间，格式同 --start；默认开始后 {DEFAULT_REPLAY_MINUTES} 分钟")
    command.add_argument("--all", action="store_true", help="输出每一秒每个窗口的判定，而不只是达标的秒")


def _build_demo_candles(end: datetime, count: int, interval_seconds: int) -> list[Candle]:
    # 每根 K 线高低差恰好为 DEMO_ATR 且收盘不跳空，种子 ATR 因此确定等于 DEMO_ATR。
    half_range = DEMO_ATR / 2
    return [
        Candle(
            end - timedelta(seconds=interval_seconds * (count - index)),
            100.0,
            100.0 + half_range,
            100.0 - half_range,
            100.0,
        )
        for index in range(count)
    ]


async def simulate(config: AppConfig) -> int:
    """为每个启用的观察窗口各跑一遍合成异动，返回提醒总数；正常时等于窗口数。"""
    notifier = ConsoleNotifier(beep=False, colors=config.alerts.console_colors)
    alert_count = 0
    for window in build_windows(config, cooldown_seconds=0):
        alert_count += await _simulate_window(config, window, notifier)
    return alert_count


async def _simulate_window(config: AppConfig, window: MoveWindow, notifier: ConsoleNotifier) -> int:
    indicator = config.indicator
    interval_seconds = INTERVAL_SECONDS[indicator.candle_interval]
    # 每个窗口用只含它自己的检测器：要验证的是这个窗口的门槛能否被满足，不能让另一个窗口先提醒掩盖问题。
    detector = build_detector(config, windows=[window])

    # 让异动恰好从 K 线边界开始：异动秒全部落在同一根未收盘 K 线里，不会被提前计入 ATR 而削弱倍数，
    # 同时之前的平稳 K 线按时收盘，ATR 始终新鲜，结果不随 lookback 等参数变化而失效。
    now = datetime.now(UTC)
    move_start = datetime.fromtimestamp(int(now.timestamp()) // interval_seconds * interval_seconds, tz=UTC)
    flat_start = move_start - timedelta(seconds=window.lookback_seconds + 1)
    history_end = datetime.fromtimestamp(
        int(flat_start.timestamp()) // interval_seconds * interval_seconds,
        tz=UTC,
    )
    detector.add_symbol(
        "BTC_USDT",
        _build_demo_candles(history_end, indicator.atr_period + 1, interval_seconds),
        1_000_000_000,
    )

    # 每秒成交笔数要足以让一个观察窗口满足 min_window_trades。
    trades_per_second = max(1, math.ceil(window.min_window_trades / window.lookback_seconds))

    def ticks_at(second: datetime, price: float, label: str) -> list[PriceTick]:
        return [
            PriceTick("BTC_USDT", price, 1.0, second + timedelta(milliseconds=index), f"{label}-{index}")
            for index in range(trades_per_second)
        ]

    for offset in range(window.lookback_seconds + 1):
        for tick in ticks_at(flat_start + timedelta(seconds=offset), 100.0, f"flat-{offset}"):
            detector.add_tick(tick)

    # 平稳段跨过的 K 线只会让 ATR 衰减，按 DEMO_ATR 计算的位移因此总能满足 ATR 倍数。
    moved_price = 100.0 + max(DEMO_ATR * window.trigger_atr_multiple, window.min_change_percent) * 1.1
    alert_count = 0
    # 多送一秒用于结算最后一个确认桶，与真实行情的完整秒检测保持一致。
    for offset in range(window.confirmation_seconds + 1):
        for tick in ticks_at(move_start + timedelta(seconds=offset), moved_price, f"demo-{offset}"):
            for alert in detector.add_tick(tick):
                alert_count += 1
                await notifier.send(alert)
    return alert_count


def replay(config: AppConfig, args: argparse.Namespace) -> None:
    try:
        start = parse_time(args.start)
        # 结束时间只写时分时取开始那天，否则回放昨天时 --end 会被当成今天，区间意外拉长到一整天。
        end_day = start.astimezone(BEIJING_TIME).date()
        end = parse_time(args.end, today=end_day) if args.end else start + timedelta(minutes=DEFAULT_REPLAY_MINUTES)
        plan = plan_replay(normalize_symbol(args.symbol), start, end, config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        data = fetch_replay_data(build_rest_client(config), plan, config)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
        # REST 重试用尽后原样抛出，这里转成一行可读的错误，而不是整段堆栈。
        raise SystemExit(f"获取 {plan.symbol} 历史数据失败：{exc}") from exc
    if not data.trades:
        raise SystemExit(f"{plan.symbol} 在该时间段没有取到成交：合约名可能有误，或 Gate 已不再提供这么早的成交")
    result = run_replay(config, plan, data)
    print("\n".join(render_replay(result, show_all=args.all)))


def print_universe(config: AppConfig) -> None:
    rest = build_rest_client(config)
    selected = select_liquid_contracts(
        rest.fetch_tickers(),
        rest.fetch_contracts(),
        config.gate.min_volume_24h_quote,
    )
    print(f"当前符合条件：{len(selected)} 个 Gate.io 虚拟币 USDT 永续合约")
    for ticker in selected:
        print(f"{ticker.symbol:<20} {ticker.volume_24h_quote / 1_000_000:>12.2f}M USDT")


def _load_config_or_exit(path: str) -> AppConfig:
    try:
        return load_config(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"配置文件不存在：{path}") from exc
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SystemExit(f"配置文件 {path} 读取失败：{exc}") from exc
    except ValidationError as exc:
        # 逐项列出字段路径，比 pydantic 默认的英文堆栈更容易定位到 YAML 中的具体键。
        details = "\n".join(
            f"  - {'.'.join(str(part) for part in error['loc']) or '(根节点)'}：{error['msg']}"
            for error in exc.errors()
        )
        raise SystemExit(f"配置文件 {path} 校验失败：\n{details}") from exc


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = _load_config_or_exit(args.config)

    if args.command == "check-config":
        print(f"配置有效：Gate.io USDT 永续，成交额门槛 {config.gate.min_volume_24h_quote / 1_000_000:.1f}M")
        for window in build_windows(config):
            print(f"  {describe_rule(window)}")
        return
    if args.command == "universe":
        print_universe(config)
        return
    if args.command == "simulate":
        expected = len(build_windows(config))
        produced = asyncio.run(simulate(config))
        if produced != expected:
            raise SystemExit(f"模拟应产生 {expected} 条提醒（每个窗口一条），实际 {produced} 条，请检查 indicator 配置")
        return
    if args.command == "replay":
        replay(config, args)
        return
    try:
        with ProcessLock(Path("data/price-alert.lock")):
            asyncio.run(run_monitor(config))
    except AlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    main()
