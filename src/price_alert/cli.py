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

from price_alert.assembly import (
    build_detector,
    build_detectors,
    build_rest_client,
    build_trend_detector,
    build_windows,
)
from price_alert.config import INTERVAL_SECONDS, AppConfig, load_config
from price_alert.formatting import BEIJING_TIME
from price_alert.gate import GateRestClient, fetch_minute_candles, price_decimals
from price_alert.instance import AlreadyRunningError, ProcessLock
from price_alert.launch.alerts import describe_launch_rule
from price_alert.launch.replay import plan_launch_replay, render_launch_replay, run_launch_replay
from price_alert.launch.simulate import simulate_launch
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
from price_alert.service import run_monitor
from price_alert.trend.alerts import describe_trend_rule
from price_alert.trend.replay import plan_trend_replay, render_trend_replay, run_trend_replay
from price_alert.trend.simulate import simulate_trend
from price_alert.universe import select_liquid_contracts
from price_alert.windows import MoveWindow, describe_rule

LOGGER = logging.getLogger(__name__)
DEMO_ATR = 2.0
# 只给开始时间时回放的时长：足够覆盖长窗口一个完整周期以及前后的行情。
DEFAULT_REPLAY_MINUTES = 15
# 趋势回放只给开始时间时的时长：趋势以十几分钟到数小时计，区间要能容下一段完整的开始到结束。
DEFAULT_TREND_REPLAY_HOURS = 3
# 放量回放只给开始时间时的时长：足够看完一段启动以及之后的几波拉升。
DEFAULT_LAUNCH_REPLAY_HOURS = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate.io 合约价格异动检测与提醒")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "启动 Gate.io 全市场实时监控"),
        ("universe", "查看当前满足成交额条件的合约"),
        ("check-config", "校验配置文件"),
        ("simulate", "使用合成行情验证 ATR 异动提醒"),
        ("replay", "用历史成交回放某个合约，逐秒解释为什么提醒或没有提醒"),
        ("trend-replay", "用历史 1 分钟 K 线回放某个合约的趋势提醒，逐根解释判定结果"),
        ("launch-replay", "用历史 1 分钟 K 线回放某个合约的放量启动提醒，逐分钟解释判定结果"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="config/default.yaml")
    _add_replay_arguments(
        commands.choices["replay"],
        f"默认开始后 {DEFAULT_REPLAY_MINUTES} 分钟",
        "输出每一秒每个窗口的判定，而不只是达标的秒",
    )
    _add_replay_arguments(
        commands.choices["trend-replay"],
        f"默认开始后 {DEFAULT_TREND_REPLAY_HOURS} 小时",
        "输出每个周期每根 K 线的判定，而不只是形态成立的时刻",
    )
    _add_replay_arguments(
        commands.choices["launch-replay"],
        f"默认开始后 {DEFAULT_LAUNCH_REPLAY_HOURS} 小时",
        "输出每一分钟的判定，而不只是满足条件的分钟",
    )
    return parser


def _add_replay_arguments(command: argparse.ArgumentParser, end_default: str, all_help: str) -> None:
    command.add_argument("symbol", help="合约名，如 QNT_USDT；只写 QNT 会自动补 _USDT")
    command.add_argument("--start", required=True, help="开始时间（北京时间），如 '2026-09-29 09:25' 或 09:25（今天）")
    command.add_argument("--end", help=f"结束时间，格式同 --start；{end_default}")
    command.add_argument("--all", action="store_true", help=all_help)


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


def expected_simulated_alerts(config: AppConfig) -> int:
    trend_periods = len(config.trend.rules()) if config.trend.enabled else 0
    return len(build_windows(config)) + trend_periods + int(config.launch.enabled)


async def simulate(config: AppConfig) -> int:
    """为每个启用的观察窗口、趋势周期和放量启动各跑一遍合成行情，返回提醒总数；正常时等于 expected_simulated_alerts。"""
    notifier = ConsoleNotifier(beep=False, colors=config.alerts.console_colors)
    alert_count = 0
    for window in build_windows(config, cooldown_seconds=0):
        alert_count += await _simulate_window(config, window, notifier)
    if config.trend.enabled:
        for alert in simulate_trend(config.trend):
            alert_count += 1
            await notifier.send(alert)
    if config.launch.enabled:
        for alert in simulate_launch(config.launch):
            alert_count += 1
            await notifier.send(alert)
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
        start, end = _replay_range(args, timedelta(minutes=DEFAULT_REPLAY_MINUTES))
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


def _replay_range(args: argparse.Namespace, default_length: timedelta) -> tuple[datetime, datetime]:
    start = parse_time(args.start)
    # 结束时间只写时分时取开始那天，否则回放昨天时 --end 会被当成今天，区间意外拉长到一整天。
    end_day = start.astimezone(BEIJING_TIME).date()
    end = parse_time(args.end, today=end_day) if args.end else start + default_length
    return start, end


def trend_replay(config: AppConfig, args: argparse.Namespace) -> None:
    if not config.trend.rules():
        raise SystemExit("trend.periods 没有启用任何周期，无法回放")
    detector = build_trend_detector(config.trend)
    try:
        start, end = _replay_range(args, timedelta(hours=DEFAULT_TREND_REPLAY_HOURS))
        plan = plan_trend_replay(normalize_symbol(args.symbol), start, end, detector)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rest = build_rest_client(config)
    try:
        candles = fetch_minute_candles(rest, plan.symbol, plan.warmup_start, plan.end)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
        raise SystemExit(f"获取 {plan.symbol} 历史 K 线失败：{exc}") from exc
    if not candles:
        raise SystemExit(f"{plan.symbol} 在该时间段没有取到 K 线：合约名可能有误")
    result = run_trend_replay(config.trend, plan, candles, _fetch_price_decimals(rest, plan.symbol))
    print("\n".join(render_trend_replay(result, show_all=args.all)))


def launch_replay(config: AppConfig, args: argparse.Namespace) -> None:
    rule = config.launch.to_rule()
    try:
        start, end = _replay_range(args, timedelta(hours=DEFAULT_LAUNCH_REPLAY_HOURS))
        plan = plan_launch_replay(normalize_symbol(args.symbol), start, end, rule)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rest = build_rest_client(config)
    try:
        candles = fetch_minute_candles(rest, plan.symbol, plan.warmup_start, plan.end)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
        raise SystemExit(f"获取 {plan.symbol} 历史 K 线失败：{exc}") from exc
    if not candles:
        raise SystemExit(f"{plan.symbol} 在该时间段没有取到 K 线：合约名可能有误")
    result = run_launch_replay(config.launch, plan, candles, _fetch_price_decimals(rest, plan.symbol))
    print("\n".join(render_launch_replay(result, show_all=args.all)))


def _fetch_price_decimals(rest: GateRestClient, symbol: str) -> int | None:
    try:
        return price_decimals(rest.fetch_contract(symbol).get("order_price_round"))
    except Exception as exc:
        # 与成交回放一致：精度只影响价格显示，取不到就按有效数字显示。
        LOGGER.warning("%s 合约信息获取失败，价格按有效数字显示：%s", symbol, exc)
        return None


def print_universe(config: AppConfig) -> None:
    rest = build_rest_client(config)
    tickers, contracts = rest.fetch_tickers(), rest.fetch_contracts()
    # 与实时监控同一来源取各检测器的成交额范围，展示的就是 run 启动时各自的合约池。
    for index, entry in enumerate(build_detectors(config)):
        volume = entry.volume
        selected = select_liquid_contracts(tickers, contracts, volume.minimum, volume.maximum)
        if index:
            print()
        print(f"{entry.name}（24h 计价成交额 {volume.describe()} USDT）：{len(selected)} 个虚拟币 USDT 永续合约")
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
        ranges = "，".join(f"{entry.name} {entry.volume.describe()}" for entry in build_detectors(config))
        print(f"配置有效：Gate.io USDT 永续，24h 计价成交额范围：{ranges}")
        for window in build_windows(config):
            print(f"  {describe_rule(window)}")
        if config.trend.enabled:
            for rule in config.trend.rules():
                print(f"  {describe_trend_rule(rule)}")
        else:
            print("  趋势提醒：已关闭")
        if config.launch.enabled:
            print(f"  {describe_launch_rule(config.launch.to_rule())}")
        else:
            print("  放量启动提醒：已关闭")
        return
    if args.command == "universe":
        print_universe(config)
        return
    if args.command == "simulate":
        expected = expected_simulated_alerts(config)
        produced = asyncio.run(simulate(config))
        if produced != expected:
            raise SystemExit(
                f"模拟应产生 {expected} 条提醒（每个窗口、每个趋势周期、放量启动各一条），实际 {produced} 条，"
                "请检查 indicator、trend 与 launch 配置"
            )
        return
    if args.command == "replay":
        replay(config, args)
        return
    if args.command == "trend-replay":
        trend_replay(config, args)
        return
    if args.command == "launch-replay":
        launch_replay(config, args)
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
