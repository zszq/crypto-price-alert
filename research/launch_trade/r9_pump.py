"""第九轮：合约池成交额下限调高、或者「最近大幅拉升过」的币，放量提醒（尤其放量下跌）之后是否更容易顺着走。

问题来自直觉：24 小时成交额大的币多是刚被炒过的，炒过的币之后更容易跌。拆成两件事验证：
- 合约池下限（launch.min_volume_24h_quote）：用真实的合约池模拟（刷新、进出池放宽），
  下限取 0 / 50 万 / 200 万 / 500 万 / 1000 万 / 3000 万；
- 提醒前的状态：每条提醒算窗口开始前（不含这次放量本身）的 24 小时成交额、相对 30 天日均的活跃度，
  以及 1 小时 ~ 30 天各尺度的涨跌、离区间低点 / 高点多远、区间振幅，按开发期分位数分五档看延续率。

延续率、开发期 / 样本外 / 半年段的口径与第八轮（r8_sides.py）相同。每一档另给同一合约池随机时刻的同档延续率：
炒过的币不管有没有提醒都可能更容易跌（整体漂移），只有提醒的延续率比同档随机时刻更高，才说明这类提醒更有用；
若只是同档随机也高，那是「拉升过的币之后容易跌」本身，提醒只是恰好落在这些币上。

提醒规则取运行时 config/default.yaml 的 launch 段（含 drop），两个方向都判定。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r9_pump.py run          # 约 40 分钟，写 results/r9/
    & $P research/launch_trade/r9_pump.py pool         # 合约池下限
    & $P research/launch_trade/r9_pump.py buckets      # 提醒前状态分五档（--feature 只看一项）
    & $P research/launch_trade/r9_pump.py long         # 1 / 3 / 7 天的更长周期结果，写 results/r9/*_long.npy
    & $P research/launch_trade/r9_pump.py long-report  # 拉升过 / 没拉升过的币在更长周期上的结果
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_trade_backtest import REPO, RESULTS, PoolRule  # noqa: E402
from load_data import load_contracts, load_series  # noqa: E402
from r5_export import detect_both  # noqa: E402
from r8_sides import DEV, FIELDS, LEVELS, OOS, SEGMENTS, drop_top_days, outcome, rate, select  # noqa: E402

OUT = RESULTS / "r9"
POOLS = {
    "pool_0": 0,
    "pool_50w": 500_000,
    "pool_200w": 2_000_000,
    "pool_500w": 5_000_000,
    "pool_1000w": 10_000_000,
    "pool_3000w": 30_000_000,
}
CURRENT = "pool_50w"
CONTROL_STEP = 720
RET_SPANS = {"1h": 60, "4h": 240, "24h": 1440, "3d": 4320, "7d": 10080, "30d": 43200}
RANGE_SPANS = {"24h": 1440, "3d": 4320, "7d": 10080, "30d": 43200}
FEATURES = [
    "vol24",  # 窗口开始前 24 小时成交额（不含这次放量）
    "activity",  # vol24 ÷ 之前 30 天日均成交额：今天比平时活跃多少倍
    *(f"ret_{name}" for name in RET_SPANS),  # 窗口开始前的对数涨跌
    *(f"up_{name}" for name in RANGE_SPANS),  # 离区间最低价多远（拉升了多少还没跌回去）
    *(f"off_{name}" for name in RANGE_SPANS),  # 离区间最高价多远（已经从高点回落多少）
    *(f"range_{name}" for name in RANGE_SPANS),  # 区间最高 ÷ 最低（这段时间炒得多猛，不管现在在哪）
]
COLUMNS = [*FIELDS, *FEATURES]


def features(series, cq: np.ndarray, p: int) -> list[float]:
    """p 是窗口开始前最后一根 K 线：之后的放量与涨跌都不算进特征。数据不够长的记 nan。"""
    c, h, lo = series.close, series.high, series.low
    price = c[p]
    vol24 = float(cq[p + 1] - cq[max(0, p + 1 - 1440)]) if p >= 1439 else np.nan
    month = cq[p + 1] - cq[p + 1 - 43200] if p >= 43199 else np.nan
    activity = vol24 / (month / 30) if month and month > 0 else np.nan
    row = [vol24, activity]
    row += [float(np.log(price / c[p - n])) if p - n >= 0 else np.nan for n in RET_SPANS.values()]
    ups, offs, ranges = [], [], []
    for n in RANGE_SPANS.values():
        if p - n + 1 < 0:
            for values in (ups, offs, ranges):
                values.append(np.nan)
            continue
        low, high = lo[p - n + 1 : p + 1].min(), h[p - n + 1 : p + 1].max()
        ups.append(float(np.log(price / low)))
        offs.append(float(np.log(high / price)))
        ranges.append(float(np.log(high / low)))
    return row + ups + offs + ranges


def run_symbol(args):
    symbol, multiplier, sym_index, rule, exit_ratio, refresh = args
    series = load_series(symbol, multiplier)
    if series is None:
        return {}
    cq = np.concatenate([[0.0], np.cumsum(series.quote)])
    window = rule.window_minutes
    memo: dict[tuple[int, int], list[float]] = {}

    def row(i: int, sign: int, wave: int, change: float) -> list[float]:
        key = (i, sign)
        if key not in memo:
            memo[key] = outcome(series, i, sign) + features(series, cq, i - window)
        return [series.t0 + (i + 1) * 60, sign, wave, change, sym_index, *memo[key]]

    result = {}
    for name, floor in POOLS.items():
        pool = PoolRule(floor, None, exit_ratio, refresh)
        rows = [row(i, sign, wave, change) for i, sign, wave, change, _q, _b in detect_both(series, rule, pool)]
        result[name] = np.array(rows, dtype=np.float64).reshape(-1, len(COLUMNS))
    # 随机对照：与当前合约池（50 万）同样在池内的分钟，每 CONTROL_STEP 分钟取一个，两个方向各算一次。
    from launch_trade_backtest import _pool_membership

    member = _pool_membership(series, PoolRule(POOLS[CURRENT], None, exit_ratio, refresh))
    picks = np.arange(1440 + sym_index % CONTROL_STEP, len(series) - 1, CONTROL_STEP)
    picks = picks[member[picks] & series.covered[picks]]
    result["control"] = np.array(
        [row(int(i), sign, 0, 0.0) for i in picks for sign in (1, -1)], dtype=np.float64
    ).reshape(-1, len(COLUMNS))
    return result


def run(args) -> None:
    from price_alert.config import load_config

    config = load_config(REPO / "config" / "default.yaml")
    rule = replace(config.launch.to_rule(), alert_drops=True)
    print(rule, flush=True)
    contracts = load_contracts()
    jobs = []
    symbols = sorted(contracts)
    for k, symbol in enumerate(symbols):
        try:
            multiplier = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            continue
        if multiplier <= 0:
            continue
        jobs.append(
            (symbol, multiplier, k, rule, config.gate.universe_exit_volume_ratio, config.gate.universe_refresh_seconds)
        )
    collected: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, part in enumerate(executor.map(run_symbol, jobs, chunksize=2), 1):
            for name, rows in part.items():
                collected.setdefault(name, []).append(rows)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "columns.txt").write_text("\n".join(COLUMNS), encoding="utf-8")
    for name, parts in collected.items():
        rows = np.concatenate(parts)
        np.save(OUT / f"{name}.npy", rows[np.argsort(rows[:, 0], kind="stable")])
    print("done", flush=True)


def load(name: str) -> dict[str, np.ndarray]:
    rows = np.load(OUT / f"{name}.npy")
    return {column: rows[:, k] for k, column in enumerate(COLUMNS)}


def _rates(data: dict, mask: np.ndarray) -> list[float]:
    return [rate(data[f"cont_{level}"][mask]) for level in LEVELS]


def pool(args) -> None:
    current = load(CURRENT)
    for sign, title in ((1, "放量上涨"), (-1, "放量下跌")):
        print(f"\n===== {title}：合约池下限（段胜：10 个半年段里 ±5% 高于 50 万的段数）=====")
        print(
            f"{'下限':<12}{'条/天 开/外':>14}{'开发 ±3/5/10':>20}{'样本外 ±3/5/10':>20}"
            f"{'去10天 开/外':>14}{'段胜':>6}{'4h收% 开/外':>16}"
        )
        for name in POOLS:
            data = load(name)
            cells = []
            for span in (DEV, OOS):
                mask = select(data, *span, sign)
                cells.append((mask, mask.sum() / ((span[1] - span[0]) / 86400)))
            wins = total = 0
            for span in SEGMENTS:
                a = rate(data["cont_5"][select(data, *span, sign)])
                b = rate(current["cont_5"][select(current, *span, sign)])
                if not (np.isnan(a) or np.isnan(b)):
                    total += 1
                    wins += a > b
            dev, oos = (_rates(data, mask) for mask, _ in cells)
            trims = [rate(data["cont_5"][drop_top_days(data, mask)]) for mask, _ in cells]
            ret = [np.nanmean(data["ret_240"][mask]) * 100 for mask, _ in cells]
            print(
                f"{name:<12}{cells[0][1]:>7.1f}/{cells[1][1]:<6.1f}"
                + "".join(f"{v:>6.1f}" for v in dev)
                + "  "
                + "".join(f"{v:>6.1f}" for v in oos)
                + f"{trims[0]:>8.1f}/{trims[1]:<5.1f}{wins:>3}/{total:<2}{ret[0]:>8.2f}/{ret[1]:.2f}"
            )


def buckets(args) -> None:
    data = load(CURRENT)
    control = load("control")
    names = [args.feature] if args.feature else FEATURES
    for sign, title in ((1, "放量上涨"), (-1, "放量下跌")):
        for feature in names:
            # 分档边界只用开发期的提醒定，样本外沿用，免得用到样本外的分布。
            dev_mask = select(data, *DEV, sign)
            values = data[feature][dev_mask]
            values = values[~np.isnan(values)]
            edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
            bins = np.concatenate([[-np.inf], edges, [np.inf]])
            print(f"\n{title} · {feature}（开发期分位边界 " + " / ".join(f"{e:.3g}" for e in edges) + "）")
            print(
                f"{'档':<4}{'提醒 ±5 开/外':>16}{'随机 ±5 开/外':>16}{'差 开/外':>14}"
                f"{'条/天 开':>9}{'段胜':>6}{'±10 开/外':>14}"
            )
            all_segments = [rate(data["cont_5"][select(data, *span, sign)]) for span in SEGMENTS]
            for k in range(5):
                inside = (data[feature] >= bins[k]) & (data[feature] < bins[k + 1])
                inside_c = (control[feature] >= bins[k]) & (control[feature] < bins[k + 1])
                a = [rate(data["cont_5"][select(data, *span, sign) & inside]) for span in (DEV, OOS)]
                c = [rate(control["cont_5"][select(control, *span, sign) & inside_c]) for span in (DEV, OOS)]
                a10 = [rate(data["cont_10"][select(data, *span, sign) & inside]) for span in (DEV, OOS)]
                per_day = (select(data, *DEV, sign) & inside).sum() / ((DEV[1] - DEV[0]) / 86400)
                wins = sum(
                    rate(data["cont_5"][select(data, *span, sign) & inside]) > base
                    for span, base in zip(SEGMENTS, all_segments, strict=True)
                )
                print(
                    f"Q{k + 1:<3}{a[0]:>8.1f}/{a[1]:<7.1f}{c[0]:>8.1f}/{c[1]:<7.1f}"
                    f"{a[0] - c[0]:>+7.1f}/{a[1] - c[1]:<+6.1f}"
                    f"{per_day:>9.1f}{wins:>4}/10{a10[0]:>8.1f}/{a10[1]:.1f}"
                )


LONG_SPANS = {"1d": 1440, "3d": 4320, "7d": 10080}


def long_symbol(args):
    """更长周期：提醒（或随机时刻）后 1 / 3 / 7 天收盘的顺向对数收益，与 7 天内 ±20% 先到哪边。
    「炒过的币之后会跌」也许要几天才兑现，24 小时、±5% 的口径看不到。"""
    symbol, multiplier, rows = args
    series = load_series(symbol, multiplier)
    out = []
    for i, sign in rows:
        stop = min(len(series), i + 1 + LONG_SPANS["7d"])
        covered = series.covered[i + 1 : stop]
        if not covered.all():
            stop = i + 1 + int(np.argmin(covered))
        price = series.close[i]
        close = np.log(series.close[i + 1 : stop] / price) * sign
        fav = np.log(series.high[i + 1 : stop] / price) if sign > 0 else -np.log(series.low[i + 1 : stop] / price)
        adv = -np.log(series.low[i + 1 : stop] / price) if sign > 0 else np.log(series.high[i + 1 : stop] / price)
        values = [float(close[n - 1]) if len(close) >= n else np.nan for n in LONG_SPANS.values()]
        x = np.log(1.2)
        hit_fav = int(np.argmax(fav >= x)) if (fav >= x).any() else -1
        hit_adv = int(np.argmax(adv >= x)) if (adv >= x).any() else -1
        if hit_fav < 0 and hit_adv < 0:
            values.append(np.nan)
        elif hit_adv < 0 or 0 <= hit_fav < hit_adv:
            values.append(1.0)
        elif hit_fav < 0 or hit_adv < hit_fav:
            values.append(0.0)
        else:
            values.append(0.5)
        out.append(values)
    return out


def long(args) -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    for name in (CURRENT, "control"):
        rows = np.load(OUT / f"{name}.npy")
        closed_at, sign, sym = rows[:, 0], rows[:, 1].astype(int), rows[:, 4].astype(int)
        jobs, order = [], []
        for k in np.unique(sym):
            picked = np.flatnonzero(sym == k)
            symbol = symbols[k]
            # 行里只存了收盘时间，换回分钟下标要合约序列的起点。
            t0 = load_series(symbol, float(contracts[symbol]["quanto_multiplier"])).t0
            index = ((closed_at[picked] - t0) // 60 - 1).astype(int)
            pairs = list(zip(index, sign[picked], strict=True))
            jobs.append((symbol, float(contracts[symbol]["quanto_multiplier"]), pairs))
            order.append(picked)
        result = np.full((len(rows), len(LONG_SPANS) + 1), np.nan)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for picked, values in zip(order, executor.map(long_symbol, jobs, chunksize=2), strict=True):
                result[picked] = values
        np.save(OUT / f"{name}_long.npy", result)
        print(name, "done", flush=True)


def long_report(args) -> None:
    data, control = load(CURRENT), load("control")
    names = [*(f"ret{key}" for key in LONG_SPANS), "first20"]
    for target, source in ((data, CURRENT), (control, "control")):
        extra = np.load(OUT / f"{source}_long.npy")
        for k, column in enumerate(names):
            target[column] = extra[:, k]
    groups = {
        "全部": lambda d: np.ones(len(d["vol24"]), bool),
        "up_7d ≥ ×2": lambda d: d["up_7d"] >= np.log(2),
        "up_3d ≥ ×2": lambda d: d["up_3d"] >= np.log(2),
        "up_30d ≥ ×3": lambda d: d["up_30d"] >= np.log(3),
        "range_7d ≥ ×2": lambda d: d["range_7d"] >= np.log(2),
        "range_7d ≥ ×2 且离高点 ≥ 30%": lambda d: (d["range_7d"] >= np.log(2)) & (d["off_7d"] >= np.log(1 / 0.7)),
        "up_7d < 5%": lambda d: d["up_7d"] < np.log(1.05),
        "vol24 ≥ 1000 万": lambda d: d["vol24"] >= 1e7,
    }
    for sign, title in ((-1, "放量下跌（顺向 = 下跌）"), (1, "放量上涨（顺向 = 上涨）")):
        print(
            f"\n===== {title}：之后 1 / 3 / 7 天收盘顺向对数收益均值 %，"
            "7 天内 ±20% 先顺向的比例 %（开发 / 样本外）====="
        )
        for label, rule in groups.items():
            cells = []
            for target in (data, control):
                mask = rule(target)
                part = []
                for span in (DEV, OOS):
                    m = select(target, *span, sign) & mask
                    part.append(
                        [np.nanmean(target[f"ret{key}"][m]) * 100 for key in LONG_SPANS] + [rate(target["first20"][m])]
                    )
                cells.append(part)
            (a_dev, a_oos), (c_dev, c_oos) = cells
            text = "  ".join(f"{x:+.1f}/{y:+.1f}" for x, y in zip(a_dev[:3], a_oos[:3], strict=True))
            ctext = "  ".join(f"{x:+.1f}/{y:+.1f}" for x, y in zip(c_dev[:3], c_oos[:3], strict=True))
            print(
                f"{label:<28} 提醒 {text}  ±20 {a_dev[3]:.1f}/{a_oos[3]:.1f} | "
                f"随机 {ctext}  ±20 {c_dev[3]:.1f}/{c_oos[3]:.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=14)
    sub.add_parser("pool")
    p_buckets = sub.add_parser("buckets")
    p_buckets.add_argument("--feature", choices=FEATURES)
    p_long = sub.add_parser("long")
    p_long.add_argument("--workers", type=int, default=14)
    sub.add_parser("long-report")
    args = parser.parse_args()
    {"run": run, "pool": pool, "buckets": buckets, "long": long, "long-report": long_report}[args.cmd](args)


if __name__ == "__main__":
    main()
