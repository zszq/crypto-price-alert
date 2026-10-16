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
    & $P research/launch_trade/r9_pump.py run          # 约 10 分钟，写 results/r9/
    & $P research/launch_trade/r9_pump.py pool         # 合约池下限
    & $P research/launch_trade/r9_pump.py buckets      # 提醒前状态分五档（--feature 只看一项）
    & $P research/launch_trade/r9_pump.py tails        # 拉升 / 振幅 / 成交额 / 活跃度的固定阈值尾部
    & $P research/launch_trade/r9_pump.py nearlow      # 窗口开始时贴近 N 天低点
    & $P research/launch_trade/r9_pump.py long         # 1 / 3 / 7 天的更长周期结果，约 2 分钟，写 results/r9/*_long.npy
    & $P research/launch_trade/r9_pump.py long-report  # 拉升过 / 没拉升过的币在更长周期上的顺向对数收益
    & $P research/launch_trade/r9_pump.py long-arith   # 7 天后按币价的算术均值、中位数、下跌占比、波动
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
# 表里「当前」那一档：须等于 default.yaml 的 launch.min_volume_24h_quote，run 时校验，YAML 改了要同步改这里。
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
    if config.launch.min_volume_24h_quote != POOLS[CURRENT]:
        raise SystemExit(
            f"default.yaml 的 launch.min_volume_24h_quote={config.launch.min_volume_24h_quote:g}，"
            f"与 CURRENT={CURRENT} 不一致，先改 POOLS / CURRENT"
        )
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
    # 行里的 symbol 列是这份名单的下标；long 读这份名单，不重新排序 contracts.json，免得中间更新过数据就对错合约。
    (OUT / "symbols.txt").write_text("\n".join(symbols), encoding="utf-8")
    for name, parts in collected.items():
        rows = np.concatenate(parts)
        np.save(OUT / f"{name}.npy", rows[np.argsort(rows[:, 0], kind="stable")])
    print("done", flush=True)


def load(name: str) -> dict[str, np.ndarray]:
    rows = np.load(OUT / f"{name}.npy")
    return {column: rows[:, k] for k, column in enumerate(COLUMNS)}


def _rates(data: dict, mask: np.ndarray) -> list[float]:
    return [rate(data[f"cont_{level}"][mask]) for level in LEVELS]


def _segment_wins(data: dict, mask: np.ndarray, sign: int) -> tuple[int, int]:
    """10 个半年段里，满足条件的提醒 ±5% 延续率高于同段全部提醒的段数，与有样本的段数；没样本的段不算输。"""
    wins = total = 0
    for span in SEGMENTS:
        period = select(data, *span, sign)
        a, b = rate(data["cont_5"][period & mask]), rate(data["cont_5"][period])
        if not (np.isnan(a) or np.isnan(b)):
            total += 1
            wins += a > b
    return wins, total


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
            for k in range(5):
                inside = (data[feature] >= bins[k]) & (data[feature] < bins[k + 1])
                inside_c = (control[feature] >= bins[k]) & (control[feature] < bins[k + 1])
                a = [rate(data["cont_5"][select(data, *span, sign) & inside]) for span in (DEV, OOS)]
                c = [rate(control["cont_5"][select(control, *span, sign) & inside_c]) for span in (DEV, OOS)]
                a10 = [rate(data["cont_10"][select(data, *span, sign) & inside]) for span in (DEV, OOS)]
                per_day = (select(data, *DEV, sign) & inside).sum() / ((DEV[1] - DEV[0]) / 86400)
                wins, total = _segment_wins(data, inside, sign)
                print(
                    f"Q{k + 1:<3}{a[0]:>8.1f}/{a[1]:<7.1f}{c[0]:>8.1f}/{c[1]:<7.1f}"
                    f"{a[0] - c[0]:>+7.1f}/{a[1] - c[1]:<+6.1f}"
                    f"{per_day:>9.1f}{wins:>4}/{total:<2}{a10[0]:>8.1f}/{a10[1]:.1f}"
                )


# 固定阈值的尾部：分位数分档把最极端的几百条和一般拉升混在一档里，这里单看翻倍、三倍这类真正炒过的币。
TAILS = [
    *(
        (f"{kind}_{span} ≥ ×{x:g}", f"{kind}_{span}", float(np.log(x)))
        for kind in ("range", "up")
        for span in RANGE_SPANS
        for x in (1.5, 2, 3)
    ),
    *((f"vol24 ≥ {x} 万", "vol24", x * 1e4) for x in (200, 1000, 3000, 10000)),
    *((f"activity ≥ {x}", "activity", float(x)) for x in (3, 5, 10)),
]


def tails(args) -> None:
    data, control = load(CURRENT), load("control")
    days = (DEV[1] - DEV[0]) / 86400
    for sign, title in ((-1, "放量下跌"), (1, "放量上涨")):
        print(f"\n===== {title}：提醒前状态的固定阈值尾部（±5% 延续率，开发 / 样本外）=====")
        rows = [("全部", None, None), *TAILS]
        for label, feature, threshold in rows:
            mask = np.ones(len(data["vol24"]), bool) if feature is None else data[feature] >= threshold
            mask_c = np.ones(len(control["vol24"]), bool) if feature is None else control[feature] >= threshold
            a = [rate(data["cont_5"][select(data, *span, sign) & mask]) for span in (DEV, OOS)]
            c = [rate(control["cont_5"][select(control, *span, sign) & mask_c]) for span in (DEV, OOS)]
            n = [int((select(data, *span, sign) & mask).sum()) for span in (DEV, OOS)]
            wins, total = _segment_wins(data, mask, sign)
            print(
                f"{label:<24} 提醒±5 {a[0]:5.1f}/{a[1]:5.1f}  随机 {c[0]:5.1f}/{c[1]:5.1f}  "
                f"条数 {n[0]}/{n[1]}（{n[0] / days:.1f}/天）  段胜 {wins}/{total}"
            )


def nearlow(args) -> None:
    """窗口开始时价格离 N 天最低价不到 X：贴着低点的放量下跌是不是更容易延续。
    去掉提醒最多的 30 天单列，因为全市场暴跌日所有币都贴着低点一起跌，会主导这一档。"""
    data, control = load(CURRENT), load("control")
    for sign, title in ((-1, "放量下跌"), (1, "放量上涨")):
        print(f"\n== {title}（离 N 天最低价 < X，即窗口开始时价格还贴着 N 天低点）==")
        everything = [select(data, *span, sign) for span in (DEV, OOS)]
        rows = [("全部", None, None)]
        # 阈值与特征同样写成 log(比值)：报价在最小变动单位上，比值恰好是 1.2 的提醒各尺度有 6~13 条，
        # log1p(0.2) 比 log(1.2) 大两个 ulp，会把它们分到另一侧。
        rows += [
            (f"up_{span} < {x}%", f"up_{span}", float(np.log(1 + x / 100)))
            for span in RANGE_SPANS
            for x in (2, 5, 10, 20)
        ]
        for label, feature, threshold in rows:
            mask = np.ones(len(data["vol24"]), bool) if feature is None else data[feature] < threshold
            mask_c = np.ones(len(control["vol24"]), bool) if feature is None else control[feature] < threshold
            picked = [period & mask for period in everything]
            r5 = [rate(data["cont_5"][m]) for m in picked]
            r10 = [rate(data["cont_10"][m]) for m in picked]
            trim = [rate(data["cont_5"][drop_top_days(data, m, 30)]) for m in picked]
            c = [rate(control["cont_5"][select(control, *span, sign) & mask_c]) for span in (DEV, OOS)]
            keep = [m.sum() / period.sum() * 100 for m, period in zip(picked, everything, strict=True)]
            wins, total = _segment_wins(data, mask, sign)
            print(
                f"{label:<16} ±5 {r5[0]:5.1f}/{r5[1]:5.1f}  ±10 {r10[0]:5.1f}/{r10[1]:5.1f}  "
                f"去30天±5 {trim[0]:5.1f}/{trim[1]:5.1f}  随机±5 {c[0]:5.1f}/{c[1]:5.1f}  "
                f"保留 {keep[0]:4.0f}%/{keep[1]:4.0f}%  段胜 {wins}/{total}"
            )


LONG_SPANS = {"1d": 1440, "3d": 4320, "7d": 10080}


def long_symbol(args):
    """更长周期：提醒（或随机时刻）后 1 / 3 / 7 天收盘的顺向对数收益，与 7 天内 ±20% 先到哪边。
    「炒过的币之后会跌」也许要几天才兑现，24 小时、±5% 的口径看不到。"""
    symbol, multiplier, closed_at, signs = args
    series = load_series(symbol, multiplier)
    # 行里只存了收盘时间，在这里按本合约序列的起点换回分钟下标，主进程不必为了 t0 再加载一遍序列。
    index = ((closed_at - series.t0) // 60 - 1).astype(int)
    out = []
    for i, sign in zip(index, signs, strict=True):
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
    symbols = (OUT / "symbols.txt").read_text(encoding="utf-8").splitlines()
    for name in (CURRENT, "control"):
        rows = np.load(OUT / f"{name}.npy")
        closed_at, sign, sym = rows[:, 0], rows[:, 1].astype(int), rows[:, 4].astype(int)
        jobs, order = [], []
        for k in np.unique(sym):
            picked = np.flatnonzero(sym == k)
            symbol = symbols[k]
            jobs.append((symbol, float(contracts[symbol]["quanto_multiplier"]), closed_at[picked], sign[picked]))
            order.append(picked)
        result = np.full((len(rows), len(LONG_SPANS) + 1), np.nan)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for picked, values in zip(order, executor.map(long_symbol, jobs, chunksize=2), strict=True):
                result[picked] = values
        np.save(OUT / f"{name}_long.npy", result)
        print(name, "done", flush=True)


LONG_GROUPS = {
    "全部": lambda d: np.ones(len(d["vol24"]), bool),
    "up_7d ≥ ×2": lambda d: d["up_7d"] >= np.log(2),
    "up_3d ≥ ×2": lambda d: d["up_3d"] >= np.log(2),
    "up_30d ≥ ×3": lambda d: d["up_30d"] >= np.log(3),
    "range_7d ≥ ×2": lambda d: d["range_7d"] >= np.log(2),
    "range_7d ≥ ×2 且离高点 ≥ 30%": lambda d: (d["range_7d"] >= np.log(2)) & (d["off_7d"] >= np.log(1 / 0.7)),
    "up_7d < 5%": lambda d: d["up_7d"] < np.log(1.05),
    "vol24 ≥ 1000 万": lambda d: d["vol24"] >= 1e7,
}


def _load_long() -> tuple[dict, dict]:
    data, control = load(CURRENT), load("control")
    names = [*(f"ret{key}" for key in LONG_SPANS), "first20"]
    for target, source in ((data, CURRENT), (control, "control")):
        extra = np.load(OUT / f"{source}_long.npy")
        for k, column in enumerate(names):
            target[column] = extra[:, k]
    return data, control


def long_report(args) -> None:
    data, control = _load_long()
    groups = LONG_GROUPS
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


def long_arith(args) -> None:
    """7 天后按币价（不按方向）的算术均值、中位数、下跌占比与对数收益标准差。
    long-report 的对数收益均值会被波动拖累：波动越大的币，没有方向时对数均值也越低，不能拿来说「更容易跌」。"""
    data, control = _load_long()

    def line(target: dict, mask: np.ndarray, span: tuple[int, int], sign: int) -> str:
        # 存的是顺向对数收益，乘回方向得到按币价的对数收益。
        r = target["ret7d"][select(target, *span, sign) & mask] * sign
        r = r[~np.isnan(r)]
        if not len(r):
            return f"{'—':>36}"
        change = np.expm1(r) * 100
        return (
            f"{change.mean():+6.1f} / {np.median(change):+5.1f} / {(r < 0).mean() * 100:4.1f} / "
            f"{r.std() * 100:4.0f}  n={len(r)}"
        )

    print("7 天后价格变化（按币价，不按方向）：算术均值 % / 中位数 % / 下跌占比 % / 对数收益标准差 —— 开发 | 样本外")
    # 随机时刻每个时刻存了两个方向，按币价的结果一样，只取一个方向。
    for title, target, sign in (("放量下跌之后", data, -1), ("放量上涨之后", data, 1), ("随机时刻", control, 1)):
        print(f"\n== {title} ==")
        for label, rule in LONG_GROUPS.items():
            mask = rule(target)
            print(f"{label:<28}{line(target, mask, DEV, sign)}  |  {line(target, mask, OOS, sign)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=14)
    sub.add_parser("pool")
    p_buckets = sub.add_parser("buckets")
    p_buckets.add_argument("--feature", choices=FEATURES)
    sub.add_parser("tails")
    sub.add_parser("nearlow")
    p_long = sub.add_parser("long")
    p_long.add_argument("--workers", type=int, default=14)
    sub.add_parser("long-report")
    sub.add_parser("long-arith")
    args = parser.parse_args()
    commands = {
        "run": run,
        "pool": pool,
        "buckets": buckets,
        "tails": tails,
        "nearlow": nearlow,
        "long": long,
        "long-report": long_report,
        "long-arith": long_arith,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
