"""Gate.io 价格异动提醒命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from price_alert.assembly import (
    build_detector,
    build_detectors,
    build_rest_client,
    build_trend_detector,
    build_windows,
)
from price_alert.common.formatting import BEIJING_TIME, format_amount
from price_alert.common.models import Candle
from price_alert.config import AppConfig, load_config
from price_alert.detectors.dormant.alerts import describe_dormant_rule
from price_alert.detectors.dormant.replay import plan_dormant_replay, render_dormant_replay, run_dormant_replay
from price_alert.detectors.dormant.simulate import simulate_dormant
from price_alert.detectors.launch.alerts import LABELS as LAUNCH_LABELS
from price_alert.detectors.launch.alerts import describe_launch_rule
from price_alert.detectors.launch.replay import plan_launch_replay, render_launch_replay, run_launch_replay
from price_alert.detectors.launch.simulate import simulate_launch
from price_alert.detectors.replay import (
    MinuteReplayPlan,
    ReplayData,
    ReplayPlan,
    fetch_price_decimals,
    fetch_replay_data,
    normalize_symbol,
    parse_time,
)
from price_alert.detectors.reversal.alerts import describe_reversal_rule
from price_alert.detectors.reversal.replay import plan_reversal_replay, render_reversal_replay, run_reversal_replay
from price_alert.detectors.reversal.rule import CANDLE_INTERVAL as REVERSAL_INTERVAL
from price_alert.detectors.reversal.simulate import simulate_reversal
from price_alert.detectors.spike.alerts import NAMES as SPIKE_NAMES
from price_alert.detectors.spike.config import AtrSpikeConfig
from price_alert.detectors.spike.replay import plan_replay, render_replay, run_replay
from price_alert.detectors.spike.simulate import simulate_spike
from price_alert.detectors.spike.streak import describe_streak_rule
from price_alert.detectors.spike.windows import describe_rule
from price_alert.detectors.trend.alerts import describe_trend_rule
from price_alert.detectors.trend.replay import plan_trend_replay, render_trend_replay, run_trend_replay
from price_alert.detectors.trend.simulate import simulate_trend
from price_alert.exchange.gate import fetch_minute_candles
from price_alert.exchange.universe import select_liquid_contracts
from price_alert.instance import AlreadyRunningError, ProcessLock
from price_alert.notify.notifier import ConsoleNotifier
from price_alert.notify.sounds import SOUND_LABELS, SOUND_NAMES, SOUND_PRIORITY
from price_alert.service import run_monitor

LOGGER = logging.getLogger(__name__)
# 以下是只给开始时间时各回放的时长。瞬时、短时异动：覆盖长窗口一个完整周期及前后行情。
DEFAULT_REPLAY_MINUTES = 15
# 趋势以十几分钟到数小时计，要容下一段完整的开始到结束。
DEFAULT_TREND_REPLAY_HOURS = 3
# 放量、冷门启动：看完一段启动及之后的几波拉升。
DEFAULT_LAUNCH_REPLAY_HOURS = 6
DEFAULT_DORMANT_REPLAY_HOURS = 6
# 反弹回落是秒级判定，还要多取 1 小时成交作基准，区间不宜太长。
DEFAULT_REVERSAL_REPLAY_MINUTES = 30
SOUND_PREVIEW_PAUSE_SECONDS = 0.8
# REST 重试用尽后会原样抛出的错误：回放命令把它们转成一行提示退出。
FETCH_ERRORS = (OSError, ValueError, TypeError, http.client.HTTPException)

PlanT = TypeVar("PlanT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate.io 合约价格异动检测与提醒")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "启动 Gate.io 全市场实时监控"),
        ("universe", "查看当前满足成交额条件的合约"),
        ("check-config", "校验配置文件"),
        ("simulate", "使用合成行情验证各类提醒"),
        ("replay", "用历史成交回放某个合约的短时异动提醒，逐秒解释为什么提醒或没有提醒"),
        ("instant-replay", "用历史成交回放某个合约的瞬时异动提醒，逐秒解释为什么提醒或没有提醒"),
        ("trend-replay", "用历史 1 分钟 K 线回放某个合约的趋势提醒，逐根解释判定结果"),
        ("launch-replay", "用历史 1 分钟 K 线回放某个合约的放量启动提醒，逐分钟解释判定结果"),
        ("dormant-replay", "用历史 1 分钟 K 线回放某个合约的冷门启动提醒，逐分钟解释判定结果"),
        ("reversal-replay", "用历史成交回放某个合约的反弹回落提醒，逐秒解释判定结果"),
        ("sounds", "依次试听各类提醒的提示音"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="config/default", help="配置文件，或目录（合并其中全部 .yaml）")
    for name in ("replay", "instant-replay"):
        _add_replay_arguments(
            commands.choices[name],
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
    _add_replay_arguments(
        commands.choices["dormant-replay"],
        f"默认开始后 {DEFAULT_DORMANT_REPLAY_HOURS} 小时",
        "输出每一分钟的判定，而不只是满足条件的分钟",
    )
    _add_replay_arguments(
        commands.choices["reversal-replay"],
        f"默认开始后 {DEFAULT_REVERSAL_REPLAY_MINUTES} 分钟",
        "输出每一秒的判定，而不只是急动达标的秒",
    )
    return parser


def _add_replay_arguments(command: argparse.ArgumentParser, end_default: str, all_help: str) -> None:
    command.add_argument("symbol", help="合约名，如 QNT_USDT；只写 QNT 会自动补 _USDT")
    command.add_argument("--start", required=True, help="开始时间（北京时间），如 '2026-09-29 09:25' 或 09:25（今天）")
    command.add_argument("--end", help=f"结束时间，格式同 --start；{end_default}")
    command.add_argument("--all", action="store_true", help=all_help)


def expected_simulated_alerts(config: AppConfig) -> int:
    windows = sum(len(build_windows(spike)) for spike in (config.instant, config.spike) if spike.enabled)
    trend_periods = len(config.trend.rules()) if config.trend.enabled else 0
    singles = len(config.dormant.rules()) + int(config.reversal.enabled)
    return windows + trend_periods + len(config.launch.sides()) + singles


async def simulate(config: AppConfig) -> int:
    """为瞬时、短时异动每个启用的观察窗口和每个趋势周期、放量方向、冷门启动参数组、反弹回落各跑一遍合成行情，
    返回提醒总数；正常时等于 expected_simulated_alerts。"""
    notifier = ConsoleNotifier(beep=False, colors=config.alerts.console_colors)
    alert_count = 0
    for spike in (config.instant, config.spike):
        if not spike.enabled:
            continue
        for alert in simulate_spike(spike):
            alert_count += 1
            await notifier.send(alert)
    if config.trend.enabled:
        for alert in simulate_trend(config.trend):
            alert_count += 1
            await notifier.send(alert)
    if config.launch.enabled:
        for alert in simulate_launch(config.launch):
            alert_count += 1
            await notifier.send(alert)
    if config.dormant.enabled:
        for alert in simulate_dormant(config.dormant):
            alert_count += 1
            await notifier.send(alert)
    if config.reversal.enabled:
        for alert in simulate_reversal(config.reversal):
            alert_count += 1
            await notifier.send(alert)
    return alert_count


async def preview_sounds() -> None:
    # 试听不看 alerts.beep：关掉提醒声音时也要能先听一下再决定。
    notifier = ConsoleNotifier(beep=True, colors=False)
    for name in SOUND_NAMES:
        print(f"{SOUND_LABELS[name]}（打断优先级 {SOUND_PRIORITY[name]}）", flush=True)
        if not await notifier.play_and_wait(name):
            raise SystemExit("无法播放内置提示音：只支持 Windows 与 macOS，或音效文件生成失败（见上方日志）")
        # 每种之间留出停顿，前后两种不会听成一个声音。
        await asyncio.sleep(SOUND_PREVIEW_PAUSE_SECONDS)


def replay(config: AppConfig, args: argparse.Namespace, spike: AtrSpikeConfig) -> None:
    """replay 回放短时异动（spike 段），instant-replay 回放瞬时异动（instant 段）；两者判定代码相同、参数不同。"""
    plan = _plan_or_exit(
        args,
        timedelta(minutes=DEFAULT_REPLAY_MINUTES),
        lambda symbol, start, end: plan_replay(symbol, start, end, spike),
    )
    data = _fetch_trades_or_exit(config, plan, spike.candle_interval)
    print("\n".join(render_replay(run_replay(spike, plan, data), show_all=args.all)))


def reversal_replay(config: AppConfig, args: argparse.Namespace) -> None:
    rule = config.reversal.to_rule()
    plan = _plan_or_exit(
        args,
        timedelta(minutes=DEFAULT_REVERSAL_REPLAY_MINUTES),
        lambda symbol, start, end: plan_reversal_replay(symbol, start, end, rule),
    )
    data = _fetch_trades_or_exit(config, plan, REVERSAL_INTERVAL)
    print("\n".join(render_reversal_replay(run_reversal_replay(rule, plan, data), show_all=args.all)))


def trend_replay(config: AppConfig, args: argparse.Namespace) -> None:
    if not config.trend.rules():
        raise SystemExit("trend.periods 没有启用任何周期，无法回放")
    detector = build_trend_detector(config.trend)
    plan = _plan_or_exit(
        args,
        timedelta(hours=DEFAULT_TREND_REPLAY_HOURS),
        lambda symbol, start, end: plan_trend_replay(symbol, start, end, detector),
    )
    candles, decimals = _fetch_minutes_or_exit(config, plan)
    print("\n".join(render_trend_replay(run_trend_replay(config.trend, plan, candles, decimals), show_all=args.all)))


def launch_replay(config: AppConfig, args: argparse.Namespace) -> None:
    plan = _plan_or_exit(
        args,
        timedelta(hours=DEFAULT_LAUNCH_REPLAY_HOURS),
        lambda symbol, start, end: plan_launch_replay(symbol, start, end, config.launch.rules()),
    )
    candles, decimals = _fetch_minutes_or_exit(config, plan)
    result = run_launch_replay(config.launch, plan, candles, decimals)
    print("\n".join(render_launch_replay(result, show_all=args.all)))


def dormant_replay(config: AppConfig, args: argparse.Namespace) -> None:
    # 冷门启动整个关闭时仍按现行一组回放：回放常用来判断要不要打开它。
    rules = config.dormant.rules() or (config.dormant.main.to_rule(),)
    plan = _plan_or_exit(
        args,
        timedelta(hours=DEFAULT_DORMANT_REPLAY_HOURS),
        lambda symbol, start, end: plan_dormant_replay(symbol, start, end, rules),
    )
    candles, decimals = _fetch_minutes_or_exit(config, plan)
    print("\n".join(render_dormant_replay(run_dormant_replay(rules, plan, candles, decimals), show_all=args.all)))


def _plan_or_exit(
    args: argparse.Namespace,
    default_length: timedelta,
    plan: Callable[[str, datetime, datetime], PlanT],
) -> PlanT:
    """解析 --start/--end 并推算回放区间；时间写错、超出 Gate 保留范围等都转成一行错误退出。"""
    try:
        start = parse_time(args.start)
        # 结束时间只写时分时取开始那天，否则回放昨天时 --end 会被当成今天。
        end_day = start.astimezone(BEIJING_TIME).date()
        end = parse_time(args.end, today=end_day) if args.end else start + default_length
        return plan(normalize_symbol(args.symbol), start, end)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _fetch_trades_or_exit(config: AppConfig, plan: ReplayPlan, candle_interval: str) -> ReplayData:
    try:
        data = fetch_replay_data(build_rest_client(config), plan, candle_interval)
    except FETCH_ERRORS as exc:
        raise SystemExit(f"获取 {plan.symbol} 历史数据失败：{exc}") from exc
    if not data.trades:
        raise SystemExit(f"{plan.symbol} 在该时间段没有取到成交：合约名可能有误，或 Gate 已不再提供这么早的成交")
    return data


def _fetch_minutes_or_exit(config: AppConfig, plan: MinuteReplayPlan) -> tuple[list[Candle], int | None]:
    rest = build_rest_client(config)
    try:
        candles = fetch_minute_candles(rest, plan.symbol, plan.warmup_start, plan.end)
    except FETCH_ERRORS as exc:
        raise SystemExit(f"获取 {plan.symbol} 历史 K 线失败：{exc}") from exc
    if not candles:
        raise SystemExit(f"{plan.symbol} 在该时间段没有取到 K 线：合约名可能有误")
    return candles, fetch_price_decimals(rest, plan.symbol)


def print_universe(config: AppConfig) -> None:
    rest = build_rest_client(config)
    tickers, contracts = rest.fetch_tickers(), rest.fetch_contracts()
    # 与实时监控同一来源取成交额范围，展示的就是 run 启动时各自的合约池。
    for index, entry in enumerate(build_detectors(config)):
        volume = entry.volume
        selected = select_liquid_contracts(tickers, contracts, volume.minimum, volume.maximum)
        if index:
            print()
        print(f"{entry.name}（24h 计价成交额 {volume.describe()} USDT）：{len(selected)} 个虚拟币 USDT 永续合约")
        for ticker in selected:
            print(f"{ticker.symbol:<20} {ticker.volume_24h_quote / 1_000_000:>12.2f}M USDT")


def lock_path(config_path: str) -> Path:
    """同一份配置只允许一个实例（否则重复提醒、抢写同一个 JSONL），不同配置可同时运行。
    日常监控的 config/default 用不带文件名的 data/price-alert.lock，其他配置按文件名各用一把锁。"""
    stem = Path(config_path).stem
    return Path("data/price-alert.lock" if stem == "default" else f"data/price-alert-{stem}.lock")


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
    except ValueError as exc:
        # 顶层不是键值对、目录为空或顶层键重复；须放在 ValidationError（ValueError 的子类）之后。
        raise SystemExit(f"配置文件 {path} 读取失败：{exc}") from exc


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = _load_config_or_exit(args.config)

    if args.command == "check-config":
        ranges = "，".join(f"{entry.name} {entry.volume.describe()}" for entry in build_detectors(config))
        print(f"配置有效：Gate.io USDT 永续，24h 计价成交额范围：{ranges}")
        for spike in (config.instant, config.spike):
            detector = build_detector(spike)
            name = SPIKE_NAMES[detector.kind]
            if not spike.enabled:
                print(f"  {name}提醒：已关闭")
                continue
            for window in detector.windows:
                print(f"  {name}·{describe_rule(window)}")
            if detector.streak_rule is not None:
                print(f"  {name}·{describe_streak_rule(detector.streak_rule)}")
        if config.trend.enabled:
            for rule in config.trend.rules():
                print(f"  {describe_trend_rule(rule)}")
        else:
            print("  趋势提醒：已关闭")
        for direction, side in config.launch.configured():
            if side.enabled:
                volume = f"24 小时成交额 > {format_amount(side.min_volume_24h_quote)}"
                if side.max_volume_24h_quote is not None:
                    volume += f" 且 ≤ {format_amount(side.max_volume_24h_quote)}"
                print(f"  {describe_launch_rule(side.to_rule(direction))}；合约池：{volume} USDT")
            else:
                print(f"  {LAUNCH_LABELS[direction]}提醒：已关闭")
        dormant = config.dormant
        if dormant.enabled:
            volume = f"24 小时成交额 > {format_amount(dormant.min_volume_24h_quote)}"
            if dormant.max_volume_24h_quote is not None:
                volume += f" 且 ≤ {format_amount(dormant.max_volume_24h_quote)}"
            for rule in dormant.rules():
                print(f"  {describe_dormant_rule(rule)}；合约池：{volume} USDT")
            if not dormant.slow.enabled:
                print("  冷门慢启动提醒：已关闭")
        else:
            print("  冷门启动提醒：已关闭（含冷门慢启动）")
        reversal = config.reversal
        if reversal.enabled:
            volume = f"24 小时成交额 > {format_amount(reversal.min_volume_24h_quote)}"
            if reversal.max_volume_24h_quote is not None:
                volume += f" 且 ≤ {format_amount(reversal.max_volume_24h_quote)}"
            print(f"  {describe_reversal_rule(reversal.to_rule())}；合约池：{volume} USDT")
        else:
            print("  反弹回落提醒：已关闭")
        return
    if args.command == "universe":
        print_universe(config)
        return
    if args.command == "simulate":
        expected = expected_simulated_alerts(config)
        produced = asyncio.run(simulate(config))
        if produced != expected:
            raise SystemExit(
                f"模拟应产生 {expected} 条提醒（瞬时、短时异动每个窗口，每个趋势周期、每个放量方向、冷门启动、"
                f"反弹回落各一条），实际 {produced} 条，请检查 instant、spike、trend、launch、dormant 与 reversal 配置"
            )
        return
    if args.command == "replay":
        replay(config, args, config.spike)
        return
    if args.command == "instant-replay":
        replay(config, args, config.instant)
        return
    if args.command == "trend-replay":
        trend_replay(config, args)
        return
    if args.command == "launch-replay":
        launch_replay(config, args)
        return
    if args.command == "dormant-replay":
        dormant_replay(config, args)
        return
    if args.command == "reversal-replay":
        reversal_replay(config, args)
        return
    if args.command == "sounds":
        asyncio.run(preview_sounds())
        return
    try:
        with ProcessLock(lock_path(args.config)):
            asyncio.run(run_monitor(config))
    except AlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    main()
