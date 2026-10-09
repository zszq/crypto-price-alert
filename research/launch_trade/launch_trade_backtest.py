"""放量启动提醒的交易回测：提醒后做多（1 倍），可选确认入场条件，到时或触发止损/回撤/止盈平仓；默认只统计第 1 波。

判定与实时监控一致：
- 规则取 config/default.yaml 的 launch 段（实际运行的参数），逐分钟判定、冷却、波次、插针记入与 LaunchDetector 相同；
  判定用 numpy 向量化预筛，再按时间顺序走一遍与检测器相同的重复提醒状态机（--validate 可与真检测器逐条对照）。
- 合约池按 gate.universe_refresh_seconds 定时刷新：24 小时成交额严格大于 launch.min_volume_24h_quote 才入池，
  已在池中的放宽到 × universe_exit_volume_ratio；不在池里的分钟不判定（实时监控也收不到它的成交），
  提醒记录不随移出清除。
- 回放式判定不做「空档后结算复核」（与 launch-replay 相同），K 线一收盘就判定。

交易假设：
- 入场：默认在提醒所在分钟收盘后的下一分钟开盘价买入；设了 --entry-delay 时等这么多分钟再按那一分钟开盘价买，
  未满足 --confirm / --min-dip / --max-extension 的提醒放弃（skipped）。
- 出场：每分钟先看亏损的线（固定或支撑位止损、从最高价回撤），开盘就越过按开盘价成交，否则按线价成交；
  再看止盈（--take-fraction 小于 100 时只平一部分，剩余继续跑）；回撤线只用上一分钟为止的最高价；
  满 N 小时按最后一分钟收盘价平仓。
- 手续费按双边吃单计；不计资金费率与滑点。

用法（在仓库根目录，先运行 update_data.py）：
    data/backtest/.venv/Scripts/python.exe research/launch_trade/launch_trade_backtest.py
    data/backtest/.venv/Scripts/python.exe research/launch_trade/launch_trade_backtest.py --validate RLC_USDT
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import Series, load_contracts, load_series  # noqa: E402
from paths import REPO, RESULTS  # noqa: E402

from price_alert.config import load_config  # noqa: E402
from price_alert.launch.rule import LaunchRule  # noqa: E402

BEIJING = timezone(timedelta(hours=8))


@dataclass(frozen=True, slots=True)
class PoolRule:
    min_volume: float
    max_volume: float | None
    exit_ratio: float
    refresh_seconds: int
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TradeRule:
    hold_minutes: int
    stop_loss: float  # 例如 0.15 表示亏损 15% 止损
    fee: float  # 单边手续费率
    take_profit: float | None = None  # 例如 0.15 表示盈利 15% 止盈，None 不止盈
    # 回撤止损：从持仓期最高价回撤这么多就平仓，让拉升中的单子继续跑、回落时锁住利润；None 不用。
    trail: float | None = None
    # 确认入场：提醒后等 entry_delay 分钟，那一分钟开盘价比提醒价至少高 confirm 才买，否则放弃这次提醒。
    # 半年回测里大多数提醒之后很快回落，等几分钟确认仍在涨，能去掉大部分一触即落的假启动。
    entry_delay: int = 0
    confirm: float | None = None
    # 等待期内最低价须至少比提醒价低这么多（先回踩、再涨回确认价才买）：一路直拉没有换手的启动之后更容易回落。
    min_dip: float | None = None
    # 止损放在等待期最低价下方这么多（支撑位止损），且离入场价不超过 stop_max；None 用固定 stop_loss。
    stop_wait_low_gap: float | None = None
    stop_max: float = 0.09
    # 回撤幅度按波动放宽：取 trail 与「入场前 60 分钟平均振幅 × trail_atr」的较大者，波动大的币不被正常震荡洗出去。
    trail_atr: float | None = None
    # 止盈时只平这个比例，剩下的继续按回撤止损跑。
    take_fraction: float = 1.0
    # 不追高：入场价相对启动前（窗口开始前一根收盘）最多涨这么多，太高的放弃。
    max_extension: float | None = None


@dataclass(slots=True)
class Alert:
    symbol: str
    closed_at: int
    price: float
    wave: int
    change_percent: float
    window_quote: float
    volume_24h: float


@dataclass(slots=True)
class Trade:
    symbol: str
    alert_at: int
    wave: int
    entry_at: int
    entry: float
    exit_at: int | None
    exit: float | None
    reason: str  # stop / trail / take / time / incomplete / skipped（未通过确认入场）
    gross: float | None
    net: float | None
    max_gain: float | None
    volume_24h: float
    change_percent: float
    # 部分止盈：先平掉的那部分成交过，最终的 reason 是剩余仓位的平仓原因。
    partial_take: bool = False


def _pool_membership(series: Series, pool: PoolRule) -> np.ndarray:
    """每根 K 线收盘时是否在合约池里：取收盘前最近一次刷新的结果。"""
    n = len(series)
    if not pool.enabled:
        return np.ones(n, dtype=bool)
    cq = np.concatenate([[0.0], np.cumsum(series.quote)])
    t0 = series.t0
    end = t0 + n * 60
    first = (t0 // pool.refresh_seconds + 1) * pool.refresh_seconds
    refreshes = np.arange(first, end + 1, pool.refresh_seconds)
    # 刷新时刻的 24 小时成交额：开盘时间在 [R - 24h, R) 的分钟。
    hi = np.clip((refreshes - t0) // 60, 0, n)
    lo = np.clip((refreshes - 86400 - t0) // 60, 0, n)
    volume = cq[hi] - cq[lo]
    member = np.zeros(len(refreshes), dtype=bool)
    inside = False
    enter_min, exit_min = pool.min_volume, pool.min_volume * pool.exit_ratio
    enter_max = pool.max_volume
    exit_max = None if pool.max_volume is None else pool.max_volume / pool.exit_ratio
    for k, value in enumerate(volume):
        low, high = (exit_min, exit_max) if inside else (enter_min, enter_max)
        inside = value > low and (high is None or value <= high)
        member[k] = inside
    # K 线 i 在 C = t0 + (i+1)·60 收盘，用 C 之前最近一次刷新（R < C）的结果；首次刷新之前不在池里。
    closes = t0 + (np.arange(n) + 1) * 60
    slot = np.searchsorted(refreshes, closes, side="left") - 1
    result = np.zeros(n, dtype=bool)
    ok = slot >= 0
    result[ok] = member[slot[ok]]
    return result


def volume_24h_at(series: Series, cq: np.ndarray, i: int) -> float:
    return float(cq[i + 1] - cq[max(0, i + 1 - 1440)])


def detect(series: Series, rule: LaunchRule, pool: PoolRule) -> list[Alert]:
    n = len(series)
    window, baseline, history = rule.window_minutes, rule.baseline_minutes, rule.history_minutes
    if n < history:
        return []
    h, c, q = series.high, series.close, series.quote
    cq = np.concatenate([[0.0], np.cumsum(q)])
    idx = np.arange(history - 1, n)
    window_quote = cq[idx + 1] - cq[idx - window + 1]
    baseline_sum = cq[idx - window + 1] - cq[idx - history + 1]
    change = (c[idx] / c[idx - window] - 1) * 100
    # 整个判定窗口（基准期 + 观察窗口）都要有数据源覆盖；缺口里的平线是编出来的，不能拿来判定。
    gaps = np.concatenate([[0], np.cumsum(~series.covered)])
    covered = gaps[idx + 1] - gaps[idx - history + 1] == 0
    # 基准期完全无成交时量比视为满足；累加误差可能让 0 变成极小的正负数，留一点余量。
    ratio_ok = (baseline_sum <= 1e-6) | (window_quote * baseline >= rule.min_volume_ratio * baseline_sum * window)
    # 只提醒上涨（alert_drops 关闭）：下跌方向永远带着 DIRECTION 失败，也不会写提醒记录。
    if rule.alert_drops:
        raise SystemExit("回测只做多，不支持 alert_drops")
    candidate = covered & ratio_ok & (window_quote >= rule.min_window_quote) & (change >= rule.min_change_percent)
    in_pool = _pool_membership(series, pool)
    candidate &= in_pool[idx]
    alerts: list[Alert] = []
    last: tuple[int, float, int] | None = None  # (时间, 价格, 波次)
    for pos in np.flatnonzero(candidate):
        i = int(idx[pos])
        # 突破：收盘严格高于基准期最高价（只对候选分钟算，免得对每一分钟取 360 根的最大值）。
        if not c[i] > h[i - history + 1 : i - window + 1].max():
            continue
        closed_at = series.t0 + (i + 1) * 60
        overshoot = rule.overshoots(float(change[pos]))
        # 与 LaunchDetector._repeat_outcome 相同：冷却先于基准期重置；否则要比上次价格再推进一截。
        if last is not None and closed_at - last[0] < rule.cooldown_minutes * 60:
            continue
        if last is None or closed_at - last[0] >= rule.baseline_minutes * 60:
            wave = 1
        elif (c[i] / last[1] - 1) * 100 < rule.realert_step_percent:
            continue
        else:
            wave = last[2] + 1
        if overshoot:
            # 插针分钟：不提醒，但记入提醒记录，波次不加。
            last = (closed_at, float(c[i]), wave - 1)
            continue
        last = (closed_at, float(c[i]), wave)
        alerts.append(
            Alert(
                series.symbol,
                closed_at,
                float(c[i]),
                wave,
                float(change[pos]),
                float(window_quote[pos]),
                volume_24h_at(series, cq, i),
            )
        )
    return alerts


def simulate_trade(series: Series, alert: Alert, trade: TradeRule) -> Trade:
    i = (alert.closed_at - series.t0) // 60 - 1
    e = i + 1 + trade.entry_delay
    entry_at = series.t0 + e * 60
    result = Trade(
        alert.symbol,
        alert.closed_at,
        alert.wave,
        entry_at,
        float("nan"),
        None,
        None,
        "incomplete",
        None,
        None,
        None,
        alert.volume_24h,
        alert.change_percent,
    )
    # 等待期间数据中断同样无法判断。
    if e >= len(series) or not series.covered[i + 1 : e + 1].all():
        return result
    entry = float(series.open[e])
    result.entry = entry
    if trade.confirm is not None and entry < alert.price * (1 + trade.confirm):
        result.reason = "skipped"
        return result
    start_price = alert.price / (1 + alert.change_percent / 100)
    if trade.max_extension is not None and entry > start_price * (1 + trade.max_extension):
        result.reason = "skipped"
        return result
    # 等待期（提醒后到入场前已收盘的分钟）的最低价：回踩确认与支撑位止损都用它。
    wait_low = float(series.low[i + 1 : e].min()) if e > i + 1 else entry
    if trade.min_dip is not None and wait_low > alert.price * (1 - trade.min_dip):
        result.reason = "skipped"
        return result
    stop = entry * (1 - trade.stop_loss)
    if trade.stop_wait_low_gap is not None:
        support_stop = max(wait_low * (1 - trade.stop_wait_low_gap), entry * (1 - trade.stop_max))
        # 支撑位贴着入场价时止损太近，一点波动就出局，退回固定止损。
        if support_stop < entry * 0.995:
            stop = support_stop
    trail = trade.trail
    if trail is not None and trade.trail_atr is not None:
        span = slice(max(0, e - 60), e)
        ranges = (series.high[span] - series.low[span]) / series.close[span]
        if len(ranges):
            trail = max(trail, trade.trail_atr * float(ranges.mean()))
    take = None if trade.take_profit is None else entry * (1 + trade.take_profit)
    best = entry
    # 部分止盈后已平部分按成交价计入，最后与剩余部分的平仓价按比例合成一个平均平仓价。
    remaining = 1.0
    banked = 0.0
    for k in range(e, e + trade.hold_minutes):
        if k >= len(series) or not series.covered[k]:
            # 数据在持仓期内中断（到了数据末尾或缺口），之前也没平仓：无法知道结果，单独计数。
            return result
        closed_at = series.t0 + (k + 1) * 60
        opened = float(series.open[k])
        # 回撤止损线只用上一分钟为止的最高价：1 分钟 K 线看不出本分钟先到高点还是低点，不能先抬高止损线再触发。
        line, reason = stop, "stop"
        if trail is not None and best * (1 - trail) > stop:
            line, reason = best * (1 - trail), "trail"
        # 先看亏损的线（开盘跳空按开盘价），再看止盈：同一分钟两条线都碰到时按保守的一边算。
        if opened <= line:
            return _closed(result, closed_at, banked + remaining * opened, reason, best, trade)
        if series.low[k] <= line:
            return _closed(result, closed_at, banked + remaining * line, reason, best, trade)
        if take is not None and remaining == 1.0 and series.high[k] >= take:
            price = max(opened, take)
            if trade.take_fraction >= 1.0:
                return _closed(result, closed_at, price, "take", best, trade)
            banked += trade.take_fraction * price
            result.partial_take = True
            remaining -= trade.take_fraction
        best = max(best, float(series.high[k]))
    k = e + trade.hold_minutes - 1
    return _closed(result, series.t0 + (k + 1) * 60, banked + remaining * float(series.close[k]), "time", best, trade)


def _closed(result: Trade, at: int, price: float, reason: str, best: float, trade: TradeRule) -> Trade:
    result.exit_at = at
    result.exit = price
    result.reason = reason
    result.gross = price / result.entry - 1
    # 双边按名义价值收手续费：开仓按入场价、平仓按出场价。
    result.net = result.gross - trade.fee - trade.fee * price / result.entry
    result.max_gain = best / result.entry - 1
    return result


def run_symbol(args: tuple[str, float | None, LaunchRule, PoolRule, TradeRule]) -> tuple[list[Alert], list[Trade], int]:
    symbol, multiplier, rule, pool, trade = args
    series = load_series(symbol, multiplier)
    if series is None:
        return [], [], 0
    alerts = detect(series, rule, pool)
    trades = [simulate_trade(series, alert, trade) for alert in alerts]
    return alerts, trades, len(series)


def validate(symbol: str, multiplier: float | None, config_path: Path) -> None:
    """与真实 LaunchDetector 逐条对照（不按合约池筛选，只用第一段连续覆盖的数据）。"""
    from price_alert.assembly import build_launch_detector
    from price_alert.models import Candle

    config = load_config(config_path)
    rule = config.launch.to_rule()
    series = load_series(symbol, multiplier)
    assert series is not None, "没有数据"
    end = int(np.argmin(series.covered)) if not series.covered.all() else len(series)
    clipped = Series(
        symbol,
        series.t0,
        series.open[:end],
        series.high[:end],
        series.low[:end],
        series.close[:end],
        series.quote[:end],
        series.covered[:end],
    )
    mine = [(a.closed_at, a.wave) for a in detect(clipped, rule, PoolRule(0, None, 1, 600, enabled=False))]
    detector = build_launch_detector(config.launch)
    detector.add_symbol(symbol, [], 0.0, quanto_multiplier=1.0)
    real = []
    for k in range(end):
        candle = Candle(
            datetime.fromtimestamp(series.t0 + k * 60, tz=UTC),
            float(series.open[k]),
            float(series.high[k]),
            float(series.low[k]),
            float(series.close[k]),
            float(series.quote[k]),
        )
        for alert in detector.add_candle(symbol, candle):
            real.append((int(alert.timestamp.timestamp()), alert.wave))
    print(f"{symbol}：{end} 分钟，向量化 {len(mine)} 条，检测器 {len(real)} 条，一致：{mine == real}")
    for item in sorted(set(mine) ^ set(real)):
        print("  差异", beijing(item[0]), item[1], "向量化" if item in mine else "检测器")


def beijing(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=BEIJING).strftime("%Y-%m-%d %H:%M")


def summarize(title: str, trades: list[Trade], stake: float) -> list[str]:
    skipped = sum(1 for t in trades if t.reason == "skipped")
    trades = [t for t in trades if t.reason != "skipped"]
    done = [t for t in trades if t.net is not None]
    if not done:
        return [f"{title}：没有完成的交易"]
    nets = [t.net for t in done]
    gross = [t.gross for t in done]
    wins = sum(1 for value in nets if value > 0)
    stops = [t for t in done if t.reason == "stop"]
    takes = [t for t in done if t.reason == "take"]
    trails = [t for t in done if t.reason == "trail"]
    # 等额本金、按平仓时间累计的盈亏曲线，看最大回撤。
    curve = np.cumsum([t.net for t in sorted(done, key=lambda t: t.exit_at)]) * stake
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate([[0], curve]))[1:] - curve)) if len(curve) else 0.0
    events = sorted([(t.entry_at, 1) for t in done] + [(t.exit_at, -1) for t in done], key=lambda x: (x[0], x[1]))
    concurrent = peak = 0
    for _, step in events:
        concurrent += step
        peak = max(peak, concurrent)
    return [
        f"{title}：{len(done)} 笔（另有 {len(trades) - len(done)} 笔持仓期内数据不全未计"
        + (f"，{skipped} 次提醒未通过确认入场、没有买" if skipped else "")
        + "）",
        f"  胜率 {wins / len(done):.1%}，止损 {len(stops)} 笔（{len(stops) / len(done):.1%}），"
        f"回撤平仓 {len(trails)} 笔（{len(trails) / len(done):.1%}），"
        f"止盈 {len(takes)} 笔（{len(takes) / len(done):.1%}）"
        + (f"，部分止盈 {partial} 笔" if (partial := sum(1 for t in done if t.partial_take)) else ""),
        f"  每笔收益（扣费）：均值 {statistics.mean(nets):+.2%}，中位数 {statistics.median(nets):+.2%}，"
        f"最好 {max(nets):+.2%}，最差 {min(nets):+.2%}；未扣费均值 {statistics.mean(gross):+.2%}",
        f"  累计：每笔 {stake:.0f} USDT 等额开仓，合计 {sum(nets) * stake:+,.0f} USDT"
        f"（收益率之和 {sum(nets):+.1%}），未扣费 {sum(gross) * stake:+,.0f} USDT",
        f"  按平仓顺序累计盈亏的最大回撤 {drawdown:,.0f} USDT，最多同时持仓 {peak} 笔",
    ]


def describe_trade(args: argparse.Namespace, pool: PoolRule) -> str:
    """汇总第一行的交易规则说明，只列出用到的条件。"""
    parts = ["合约池不筛选" if args.no_pool else f"合约池 24h 成交额 > {pool.min_volume:,.0f}"]
    parts.append(f"持有 {args.hold_hours:g} 小时")
    parts.append(f"止损 -{args.stop_loss:g}%")
    parts.append("不止盈" if args.take_profit is None else f"止盈 +{args.take_profit:g}%")
    if args.take_profit is not None and args.take_fraction < 100:
        parts.append(f"止盈只平 {args.take_fraction:g}%")
    if args.trail is not None:
        parts.append(f"回撤 {args.trail:g}% 平仓")
    if args.trail_atr is not None:
        parts.append(f"回撤至少 {args.trail_atr:g} 倍平均振幅")
    if args.confirm is not None:
        parts.append(f"等 {args.entry_delay} 分钟、比提醒价高 {args.confirm:g}% 才买")
    if args.min_dip is not None:
        parts.append(f"等待期须回踩 {args.min_dip:g}%")
    if args.stop_wait_low is not None:
        parts.append(f"止损在等待期低点下 {args.stop_wait_low:g}%（最多 {args.stop_max:g}%）")
    if args.max_extension is not None:
        parts.append(f"入场价比启动前高超过 {args.max_extension:g}% 不追")
    parts.append(f"单边手续费 {args.fee:g}%")
    return "，".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "config" / "default.yaml"))
    parser.add_argument("--days", type=int, default=183, help="回测最近多少天的提醒（之前的数据只用于预热）")
    parser.add_argument("--hold-hours", type=float, default=5)
    parser.add_argument("--stop-loss", type=float, default=15, help="止损百分比")
    parser.add_argument("--take-profit", type=float, default=None, help="止盈百分比，默认不止盈")
    parser.add_argument(
        "--trail", type=float, default=None, help="回撤止损百分比：从持仓期最高价回撤这么多平仓，默认不用"
    )
    parser.add_argument("--entry-delay", type=int, default=0, help="提醒后等几分钟再入场，默认 0（下一分钟开盘）")
    parser.add_argument(
        "--confirm", type=float, default=None, help="确认入场：入场价须比提醒价至少高这么多百分比，否则不买"
    )
    parser.add_argument(
        "--min-dip", type=float, default=None, help="等待期最低价须比提醒价至少低这么多百分比（先回踩）才买"
    )
    parser.add_argument(
        "--stop-wait-low", type=float, default=None, help="止损放在等待期最低价下方这么多百分比（支撑位止损）"
    )
    parser.add_argument("--stop-max", type=float, default=9, help="支撑位止损离入场价最多这么多百分比")
    parser.add_argument("--trail-atr", type=float, default=None, help="回撤幅度至少为入场前 60 分钟平均振幅的这么多倍")
    parser.add_argument(
        "--max-extension", type=float, default=None, help="入场价相对启动前最多涨这么多百分比，太高不追"
    )
    parser.add_argument("--take-fraction", type=float, default=100, help="止盈时平掉的百分比，其余继续按回撤止损跑")
    parser.add_argument("--fee", type=float, default=0.05, help="单边手续费百分比，Gate 合约吃单默认 0.05")
    parser.add_argument("--stake", type=float, default=100, help="每笔开仓金额（USDT），只影响金额展示")
    parser.add_argument("--no-pool", action="store_true", help="不按合约池筛选（成交额下限视为 0）")
    parser.add_argument("--all-waves", action="store_true", help="月度与逐笔明细按所有波次统计（默认只看第 1 波）")
    parser.add_argument("--validate", metavar="SYMBOL", help="与真实检测器逐条对照一个合约")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    if args.validate:
        validate(args.validate, multiplier(args.validate), Path(args.config))
        return

    config = load_config(args.config)
    rule = config.launch.to_rule()
    pool = PoolRule(
        config.launch.min_volume_24h_quote,
        config.launch.max_volume_24h_quote,
        config.gate.universe_exit_volume_ratio,
        config.gate.universe_refresh_seconds,
        enabled=not args.no_pool,
    )
    trade_rule = TradeRule(
        int(args.hold_hours * 60),
        args.stop_loss / 100,
        args.fee / 100,
        None if args.take_profit is None else args.take_profit / 100,
        None if args.trail is None else args.trail / 100,
        args.entry_delay,
        None if args.confirm is None else args.confirm / 100,
        None if args.min_dip is None else args.min_dip / 100,
        None if args.stop_wait_low is None else args.stop_wait_low / 100,
        args.stop_max / 100,
        args.trail_atr,
        args.take_fraction / 100,
        None if args.max_extension is None else args.max_extension / 100,
    )
    jobs = [(symbol, multiplier(symbol), rule, pool, trade_rule) for symbol in sorted(contracts)]
    alerts: list[Alert] = []
    trades: list[Trade] = []
    minutes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for symbol_alerts, symbol_trades, length in executor.map(run_symbol, jobs, chunksize=4):
            alerts.extend(symbol_alerts)
            trades.extend(symbol_trades)
            minutes += length

    now = datetime.now(UTC)
    start = int((now - timedelta(days=args.days)).replace(hour=16, minute=0, second=0, microsecond=0).timestamp())
    alerts = [a for a in alerts if a.closed_at >= start]
    trades = [t for t in trades if t.alert_at >= start]
    first_wave = [t for t in trades if t.wave == 1]
    # 月度、逐笔明细与最好最差的统计对象：默认第 1 波，--all-waves 时是所有波次。
    focus = trades if args.all_waves else first_wave
    focus_label = "所有波次" if args.all_waves else "第 1 波"
    file_tag = "all_waves" if args.all_waves else "first_wave"
    seen: set[str] = set()
    first_per_symbol = []
    for t in sorted(first_wave, key=lambda t: t.alert_at):
        if t.symbol not in seen:
            seen.add(t.symbol)
            first_per_symbol.append(t)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"launch_{file_tag}_trades.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        fields = list(asdict(focus[0]).keys()) if focus else []
        writer.writerow(fields)
        for t in sorted(focus, key=lambda t: t.alert_at):
            row = asdict(t)
            for key in ("alert_at", "entry_at", "exit_at"):
                row[key] = beijing(row[key]) if row[key] else ""
            writer.writerow([row[key] for key in fields])

    days = (int(now.timestamp()) - start) / 86400
    lines = [
        f"回测区间：{beijing(start)} ～ {beijing(int(now.timestamp()))}（北京时间，约 {days:.0f} 天）",
        f"规则：{args.config} 的 launch 段；{describe_trade(args, pool)}",
        f"数据：{len(contracts)} 个合约，共 {minutes / 1e6:.1f} 百万分钟",
        f"提醒 {len(alerts)} 条（每天 {len(alerts) / days:.1f} 条），"
        f"其中第 1 波 {sum(a.wave == 1 for a in alerts)} 条，"
        f"涉及 {len({a.symbol for a in alerts if a.wave == 1})} 个合约",
        "",
        *summarize("每次第 1 波提醒都买入", first_wave, args.stake),
        "",
        *summarize("每个合约只买区间内第一次提醒", first_per_symbol, args.stake),
        "",
        *summarize("参照：所有波次都买入", trades, args.stake),
        "",
        f"按月（{focus_label}，扣费）：",
    ]
    by_month: dict[str, list[float]] = {}
    for t in focus:
        if t.net is not None:
            by_month.setdefault(beijing(t.alert_at)[:7], []).append(t.net)
    for month, nets in sorted(by_month.items()):
        wins = sum(1 for value in nets if value > 0)
        lines.append(
            f"  {month}：{len(nets):4d} 笔，胜率 {wins / len(nets):.0%}，均值 {statistics.mean(nets):+.2%}，"
            f"合计 {sum(nets) * args.stake:+,.0f} USDT"
        )
    if args.all_waves:
        lines.append("按波次（扣费）：")
        by_wave: dict[int, list[float]] = {}
        for t in focus:
            if t.net is not None:
                by_wave.setdefault(min(t.wave, 4), []).append(t.net)
        for wave, nets in sorted(by_wave.items()):
            wins = sum(1 for value in nets if value > 0)
            lines.append(
                f"  第 {wave}{'+' if wave == 4 else ''} 波：{len(nets):4d} 笔，胜率 {wins / len(nets):.0%}，"
                f"均值 {statistics.mean(nets):+.2%}，合计 {sum(nets) * args.stake:+,.0f} USDT"
            )
    done = sorted((t for t in focus if t.net is not None), key=lambda t: t.net)
    lines.append("最差 5 笔：" + "；".join(f"{t.symbol} {beijing(t.alert_at)} {t.net:+.1%}" for t in done[:5]))
    lines.append("最好 5 笔：" + "；".join(f"{t.symbol} {beijing(t.alert_at)} {t.net:+.1%}" for t in done[-5:][::-1]))
    lines.append(f"逐笔明细：{out}")
    text = "\n".join(lines)
    print(text)
    (RESULTS / f"launch_{file_tag}_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
