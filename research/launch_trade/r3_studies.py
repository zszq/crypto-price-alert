"""第三轮的各项研究，复现 FINDINGS.md「第三轮：三年数据」中排名表以外的数字。

每个子命令对应一项研究，数据集是 r3_export.py 的导出目录（见 README「第三轮」）：
- r3：当前提醒规则、提醒后 48 小时（默认导出），持仓超过 18 小时的研究要用它；
- v_base / v_nobreak / v_nb_pool0_noover 等：提醒后 20 小时（--post 1200），提醒规则不同。

    & $P research/launch_trade/r3_studies.py waves         # 所有波次、1~2 波、不叠仓，12 种等待设置
    & $P research/launch_trade/r3_studies.py exits         # 出场网格（按 30 种入场平均）与回撤幅度的尖峰
    & $P research/launch_trade/r3_studies.py holds         # 持仓 10~47 小时
    & $P research/launch_trade/r3_studies.py shapes        # 部分止盈、延迟启用回撤、时间止损、保本、收紧回撤
    & $P research/launch_trade/r3_studies.py features      # 提醒特征分档、逐小时、距 30 天 / 7 天高点（6 段）
    & $P research/launch_trade/r3_studies.py walkforward   # 前 18 个月选出场、最差时段、距高点阈值，后 18 个月检验
    & $P research/launch_trade/r3_studies.py variants      # 提醒门槛变体（需先导出对应目录）
    & $P research/launch_trade/r3_studies.py breakout      # 不要求突破的提醒中，已突破 / 未突破两部分
    & $P research/launch_trade/r3_studies.py control       # 随机（合约, 小时）买入对照（需先运行 r3_market.py）
    & $P research/launch_trade/r3_studies.py confirm       # 确认涨幅扫描、按波动缩放的确认
    & $P research/launch_trade/r3_studies.py market        # 市场状态与策略近期盈亏分档（需先运行 r3_market.py）
    & $P research/launch_trade/r3_studies.py rolling       # 每月滚动重选参数
    & $P research/launch_trade/r3_studies.py short         # 做空失败的启动
    & $P research/launch_trade/r3_studies.py regime        # 亏损期、市场状态、过热过滤与扩展类开关（需 r3_market.py）
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from datetime import datetime

import numba as nb
import numpy as np
from paths import RESULTS
from r2_delays import DELAYS
from r2_lib import BEIJING, BEST, PRE, alert_features, fold_sums
from r3_lib import (
    BAD_HOURS,
    ENTRIES,
    HOT_BREADTH,
    TRAIL_SCALE,
    Book,
    Scheme,
    X,
    capped,
    hood,
    times,
)
from r3_market import MARKET, hourly_breadth

# 前推检验的分界：前 18 个月（2023-10 ~ 2025-03）选参数，后 18 个月检验。
HALF = int(datetime(2025, 4, 1, tzinfo=BEIJING).timestamp())
VARIANTS = [
    "v_base",
    "v_step3",
    "v_step10",
    "v_nobreak",
    "v_chg2",
    "v_noover",
    "v_wq50k",
    "v_wq200k",
    "v_ratio2",
    "v_ratio5",
    "v_pool0",
    "v_pool3m",
    "v_nb_pool0",
    "v_nb_noover",
    "v_nb_pool0_noover",
    "v_nb_step3",
    "v_nb_step10",
    "v_nb_wq200k",
]


def total(book: Book, rule, mask: np.ndarray) -> float:
    return float(np.nansum(book.run(rule)[0][mask])) * 100


def folds_of(book: Book) -> np.ndarray:
    """每条提醒属于 6 段中的哪一段。"""
    edges = [start for _, start in book.data.period.folds[1:]]
    return np.searchsorted(edges, book.data.closed_at, side="right")


def neighborhood_nets(book: Book, rule, scales=TRAIL_SCALE) -> np.ndarray:
    """邻域里每组参数的逐笔收益（NaN 记 0），形状 (组数, 提醒数)；分档统计时对每组求和再平均，等于邻域平均。"""
    rows = []
    for delay, confirm in ENTRIES:
        for scale in scales:
            net = book.run(replace(rule, delay=delay, confirm=confirm, trail=rule.trail * scale))[0]
            rows.append(np.nan_to_num(net))
    return np.array(rows)


def bucket_line(nets: np.ndarray, sel: np.ndarray, fold: np.ndarray) -> str:
    per = np.array([(nets[:, sel & (fold == k)].sum(1) * 100).mean() for k in range(6)])
    return f"合{per.sum():+6.0f} 正段{(per > 0).sum()}/6 | " + " ".join(f"{v:+5.0f}" for v in per)


def no_stack(book: Book, net, entry_at, exit_at, mask) -> np.ndarray:
    """同一个币已有持仓时不再开新仓（按入场时间顺序）。"""
    ok = np.flatnonzero(mask & ~np.isnan(net))
    t_in, t_out = (t[ok] for t in times(book.data, entry_at, exit_at))
    busy: dict[str, float] = {}
    keep = np.zeros_like(mask)
    for j in np.argsort(t_in, kind="stable"):
        symbol = book.data.symbol[ok[j]]
        if busy.get(symbol, -1) > t_in[j]:
            continue
        busy[symbol] = t_out[j]
        keep[ok[j]] = True
    return keep


def study_waves(args) -> None:
    book = Book(args.dataset)
    data = book.data
    allw = np.ones_like(book.first)
    net, entry_at, exit_at = book.run(BEST)
    for name, mask in [
        ("J 第1波", book.first),
        ("J 所有波次", allw),
        ("J 第1~2波", data.wave <= 2),
        ("J 第2波起", data.wave >= 2),
        ("J 所有波次不叠仓", no_stack(book, net, entry_at, exit_at, allw)),
    ]:
        print(f"{name:12s} {np.nansum(net[mask]) * 100:+6.0f}")
    print("第二轮的 12 种等待设置（等待分钟 + 确认涨幅）：第 1 波 / 所有波次")
    first = np.array([total(book, replace(BEST, delay=d, confirm=c), book.first) for d, c in DELAYS])
    every = np.array([total(book, replace(BEST, delay=d, confirm=c), allw) for d, c in DELAYS])
    for (d, c), a, b in zip(DELAYS, first, every, strict=True):
        print(f"  {d:2d}分+{c:.1%}: {a:+6.0f} / {b:+6.0f}")
    print(f"  平均 {first.mean():+.0f} / {every.mean():+.0f}，所有波次胜出 {(every > first).sum()}/12")


def study_exits(args) -> None:
    book = Book(args.dataset)
    allw = np.ones_like(book.first)

    def average(rule, mask):
        values = np.array([total(book, replace(rule, delay=d, confirm=c), mask) for d, c in ENTRIES])
        return values.mean(), values.min()

    print("出场网格（按 30 种入场的平均 / 最差）：")
    grid = itertools.product([0.02, 0.03], [0.09, 0.12, 0.15], [0.12, 0.15, 0.2], [np.nan, 3, 5], [420, 600, 700])
    rows = []
    for gap, smax, trail, atr, hold in grid:
        rule = replace(BEST, stop_support_gap=gap, stop_max=smax, trail=trail, trail_atr=atr, hold=hold)
        rows.append((average(rule, book.first), average(rule, allw), (gap, smax, trail, atr, hold)))
    for label, k in (("第1波", 0), ("所有波次", 1)):
        rows.sort(key=lambda r: -r[k][0])
        print(f"  {label}（前 5）")
        for row in rows[:5]:
            gap, smax, trail, atr, hold = row[2]
            print(f"    余量{gap} 上限{smax} 回撤{trail} 振幅{atr} 持{hold}: {row[k][0]:+.0f} / {row[k][1]:+.0f}")
        j = average(BEST, book.first if k == 0 else allw)
        print(f"    J 的出场: {j[0]:+.0f} / {j[1]:+.0f}")
    print("放宽余量与持仓（第 1 波 / 所有波次，按 30 种入场平均）：")
    for gap, smax, hold in itertools.product([0.03, 0.05, 0.07], [0.15, 0.2], [700, 1080]):
        rule = replace(X, stop_support_gap=gap, stop_max=smax, hold=hold)
        print(f"  余量{gap} 上限{smax} 持{hold}: {average(rule, book.first)[0]:+.0f} / {average(rule, allw)[0]:+.0f}")
    print("回撤幅度的尖峰（止损余量 5%、持 18h、不按振幅放宽、所有波次，等 10 分钟 +2%）：")
    base = replace(X, trail_atr=np.nan)
    for trail in (0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20):
        net = book.run(replace(base, trail=trail))[0]
        x = np.sort(net[allw & ~np.isnan(net)])
        print(f"  回撤{trail}: 合计 {x.sum() * 100:+.0f} 去掉最好20笔 {x[:-20].sum() * 100:+.0f}")


def study_holds(args) -> None:
    book = Book(args.dataset)
    for hold in (600, 900, 1080, 1440, 1800, 2400, 2800):
        rule = replace(BEST, hold=hold)
        print(f"J 的出场持 {hold / 60:.0f}h：第 1 波 {total(book, rule, book.first):+.0f}")


def study_shapes(args) -> None:
    book = Book(args.dataset)
    variants = [("X", X)]
    variants += [
        (f"涨{t:.0%}平{f:.0%}", replace(X, take=t, take_fraction=f)) for t, f in [(0.3, 0.5), (0.5, 0.5), (1, 0.5)]
    ]
    variants += [(f"涨{a:.0%}后才回撤", replace(X, trail_after=a)) for a in (0.03, 0.05, 0.1)]
    variants += [(f"{s}分未涨{g:.0%}离场", replace(X, time_stop=s, time_gain=g)) for s, g in [(60, 0.03), (120, 0.05)]]
    variants += [(f"涨{b:.0%}后保本", replace(X, breakeven_at=b)) for b in (0.2, 0.5)]
    variants += [
        (f"涨{a:.0%}后回撤改{t:.0%}", replace(X, trail_tighten_at=a, trail_tight=t))
        for a, t in [(0.5, 0.2), (0.3, 0.12)]
    ]
    for name, rule in variants:
        first = hood(book, Scheme(name, args.dataset, rule))[0]
        every = hood(book, Scheme(name, args.dataset, rule, all_waves=True))[0]
        print(f"{name:16s} 邻域平均：第 1 波 {first:+6.0f}  所有波次 {every:+6.0f}")


def study_features(args) -> None:
    book = Book(args.dataset)
    data, bars = book.data, book.data.bars
    nets = neighborhood_nets(book, X)
    fold = folds_of(book)
    features = {k: v for k, v in alert_features(data).items() if not k.startswith(("等待期", "入场"))}
    price = bars[:, 3, PRE - 1]
    features["距7d高"] = price / data.meta["high_7d"] - 1
    features["距30d高"] = price / data.meta["high_30d"] - 1
    features["24h振幅"] = data.meta["high_24h"] / data.meta["low_24h"] - 1
    for label, mask in (("第1波", book.first), ("所有波次", np.ones_like(book.first))):
        print(f"===== {label}：出场 X 的邻域平均，按 6 段")
        for name, values in features.items():
            ok = mask & ~np.isnan(values)
            edges = np.nanpercentile(values[ok], [20, 40, 60, 80])
            bucket = np.searchsorted(edges, values)
            print(f"  {name} 分位点 {np.round(edges, 3)}")
            for q in range(5):
                print(f"    档{q + 1} " + bucket_line(nets, ok & (bucket == q), fold))
        print("  逐小时（北京时，按提醒 K 线收盘时刻）")
        for hour in range(24):
            print(f"    {hour:2d}点 " + bucket_line(nets, mask & (book.hour == hour), fold))


def study_walkforward(args) -> None:
    book = Book(args.dataset)
    data = book.data
    early = data.closed_at < HALF
    distance = data.bars[:, 3, PRE - 1] / data.meta["high_30d"] - 1
    for label, waves in (("第1波", book.first), ("所有波次", np.ones_like(book.first))):
        print(f"===== {label}")

        def split_hood(rule, waves=waves):
            nets = neighborhood_nets(book, rule)
            return (nets[:, waves & early].sum(1) * 100).mean(), (nets[:, waves & ~early].sum(1) * 100).mean()

        j = split_hood(BEST)
        print(f"J：前18月 {j[0]:+.0f} 后18月 {j[1]:+.0f}")
        grid = []
        for gap, hold, atr in itertools.product([0.02, 0.03, 0.05, 0.07], [600, 700, 1080, 1440], [np.nan, 3, 5]):
            rule = replace(BEST, stop_support_gap=gap, stop_max=0.2, trail_atr=atr, hold=hold)
            grid.append((*split_hood(rule), rule))
        grid.sort(key=lambda g: -g[0])
        for a, b, rule in grid[:3]:
            print(
                f"前18月最好：余量{rule.stop_support_gap} 持{rule.hold} 振幅{rule.trail_atr}：前 {a:+.0f} → 后 {b:+.0f}"
            )
        later = np.array([g[1] for g in grid])
        print(f"{len(grid)} 组出场的后18月：中位 {np.median(later):+.0f}，胜过 J 的 {np.mean(later > j[1]):.0%}")
        nets = neighborhood_nets(book, grid[0][2])
        by_hour = [
            [(nets[:, waves & part & (book.hour == h)].sum(1) * 100).mean() for h in range(24)]
            for part in (early, ~early)
        ]
        windows = sorted((sum(by_hour[0][(s + k) % 24] for k in range(3)), s) for s in range(24))
        for value, s in windows[:3]:
            hours = [(s + k) % 24 for k in range(3)]
            print(f"前18月最差 3 小时 {hours}：前 {value:+.0f} → 后 {sum(by_hour[1][h] for h in hours):+.0f}")
        for threshold in (-0.5, -0.4, -0.3, -0.2, -0.1):
            cut = waves & (distance < threshold)
            a, b = ((nets[:, cut & part].sum(1) * 100).mean() for part in (early, ~early))
            print(f"距30d高<{threshold}（被剔除部分）：前 {a:+.0f} 后 {b:+.0f}")


def study_variants(args) -> None:
    rules = (("J", BEST), ("X", X), ("X确认0%", replace(X, confirm=0.0)))
    for dataset in VARIANTS:
        if not (RESULTS / dataset / "symbol.npy").exists():
            print(f"{dataset}：未导出，跳过")
            continue
        book = Book(dataset)
        cells = []
        for name, rule in rules:
            net, entry_at, exit_at = book.run(rule)
            for label, mask in (("1波", book.first), ("所有", np.ones_like(book.first))):
                ok = mask & ~np.isnan(net)
                limited = net[capped(book.data, entry_at, exit_at, ok, 10)].sum() * 100
                cells.append(f"{name}{label} {net[ok].sum() * 100:+6.0f}（限10 {limited:+5.0f}）")
        days = len(book.data.wave) / 1096
        print(f"{dataset:18s} 每天{days:4.1f}条 | " + " | ".join(cells), flush=True)


def study_breakout(args) -> None:
    book = Book(args.dataset)
    data = book.data
    broke = data.bars[:, 3, PRE - 1] > data.meta["breakout"]
    print(f"已突破基准期高点的占比：全部 {broke.mean():.0%}，第 1 波 {broke[book.first].mean():.0%}")
    for name, rule in (("J", BEST), ("X", X)):
        for label, mask in (("第1波", book.first), ("未突破", book.first & ~broke), ("已突破", book.first & broke)):
            print(f"{name} {label}: {total(book, rule, mask):+.0f}")


def _market():
    market = np.load(RESULTS / "r3_market.npz")
    return market["hours"], market["close"], market["quote"]


def study_control(args) -> None:
    hours, close, quote = _market()
    start = int(datetime(2023, 10, 10, tzinfo=BEIJING).timestamp())
    cumulative = np.cumsum(quote, 0)
    volume_24h = cumulative - np.roll(cumulative, 24, 0)
    volume_24h[:24] = 0
    # 与提醒同样只看合约池里的币（24 小时成交额 > 100 万）。
    pool = volume_24h > 1e6
    pool[: (start - int(hours[0])) // 3600] = False
    forward = np.roll(close, -18, 0) / close - 1
    forward[-18:] = np.nan
    sample = forward[pool & ~np.isnan(forward)]
    print(
        f"随机（合约, 小时）买入持 18h：{len(sample)} 个样本，"
        f"平均 {sample.mean():+.3%}，扣双边费 {sample.mean() - 0.001:+.3%}"
    )
    for dataset in ("v_base", "v_nobreak"):
        book = Book(dataset)
        bars = book.data.bars
        entry = PRE + 10
        ret = bars[:, 3, entry + 1079] / bars[:, 0, entry] - 1
        x = ret[book.first & ~np.isnan(ret)]
        print(f"{dataset} 第 1 波提醒后 10 分钟买入持 18h：{len(x)} 笔，平均 {x.mean():+.3%}")


def study_confirm(args) -> None:
    for dataset in ("v_nobreak", "v_base"):
        book = Book(dataset)
        bars = book.data.bars
        alert_price = bars[:, 3, PRE - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            span = slice(PRE - 366, PRE - 6)
            volatility = np.nanmean((bars[:, 1, span] - bars[:, 2, span]) / bars[:, 3, span], 1)
        print(f"===== {dataset}（出场 X，等 3/5/7/10/12/15/20 分钟的平均）")
        for label, mask in (("第1波", book.first), ("所有波次 避开7~10点", ~np.isin(book.hour, BAD_HOURS))):
            gains = {}
            for delay in (3, 5, 7, 10, 12, 15, 20):
                # 不设确认先全部模拟，再按入场时的涨幅筛：同一笔交易的结果与设了确认时相同。
                net = book.run(replace(X, delay=delay, confirm=np.nan))[0]
                gain = bars[:, 0, PRE + delay] / alert_price - 1
                for c in (None, -0.03, -0.01, 0.0, 0.01, 0.02):
                    keep = mask if c is None else mask & (gain >= c)
                    gains.setdefault(f"确认{'不要求' if c is None else f'{c:+.0%}'}", []).append(np.nansum(net[keep]))
                for k in (1, 2, 4):
                    gains.setdefault(f"确认≥{k}倍振幅", []).append(np.nansum(net[mask & (gain >= k * volatility)]))
            for name, values in gains.items():
                print(f"  {label} {name}: 平均 {np.mean(values) * 100:+.0f}")


def study_market(args) -> None:
    hours, close, quote = _market()
    start = int(hours[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        r24 = close / np.roll(close, 24, 0) - 1
        r24[:24] = np.nan
        r168 = close / np.roll(close, 168, 0) - 1
        r168[:168] = np.nan
        valid = ~np.isnan(r24)
        breadth = np.where(valid.sum(1) > 20, (r24 > 0).sum(1) / np.maximum(valid.sum(1), 1), np.nan)
        total_quote = np.concatenate([[0], np.cumsum(quote.sum(1))])
        index = np.arange(len(hours))
        activity = (total_quote[index + 1] - total_quote[np.maximum(index - 23, 0)]) / (
            (total_quote[index + 1] - total_quote[np.maximum(index - 719, 0)]) / 30
        )
        states = {
            "广度24h": breadth,
            "市场中位24h": np.nanmedian(r24, 1),
            "市场中位7d": np.nanmedian(r168, 1),
            "离散度": np.nanstd(np.clip(r24, -0.5, 0.5), 1),
            "活跃度": activity,
        }
    for dataset in ("v_nobreak", "v_base"):
        book = Book(dataset)
        data = book.data
        # 只用提醒之前已经收盘的那个小时。
        slot = (data.closed_at.astype(np.int64) - start) // 3600 - 1
        features = {name: values[slot] for name, values in states.items()}
        ordered = np.sort(data.closed_at)
        recent = np.searchsorted(ordered, data.closed_at) - np.searchsorted(ordered, data.closed_at - 86400)
        month = np.searchsorted(ordered, data.closed_at) - np.searchsorted(ordered, data.closed_at - 30 * 86400)
        features["提醒密度"] = recent / np.maximum(month / 30, 1)
        net, entry_at, exit_at = book.run(X)
        nets = neighborhood_nets(book, X, scales=(0.9, 1.0, 1.1))
        fold = folds_of(book)
        for label, mask in (("第1波", book.first), ("所有波次 避开7~10点", ~np.isin(book.hour, BAD_HOURS))):
            ok = mask & ~np.isnan(net)
            # 策略自身反馈：提醒之前 days 天内已经平仓的交易的合计。
            t_out = times(data, entry_at, exit_at)[1][ok]
            order = np.argsort(t_out)
            cum = np.concatenate([[0], np.cumsum(net[ok][order])])
            for days in (14, 30):
                hi = np.searchsorted(t_out[order], data.closed_at)
                lo = np.searchsorted(t_out[order], data.closed_at - days * 86400)
                features[f"近{days}天盈亏"] = np.where(hi - lo >= 10, (cum[hi] - cum[lo]) * 100, np.nan)
            print(f"===== {dataset} {label}")
            for name, values in features.items():
                sel = mask & ~np.isnan(values)
                edges = np.nanpercentile(values[sel], [20, 40, 60, 80])
                bucket = np.searchsorted(edges, values)
                print(f"  {name} 分位点 {np.round(edges, 3)}")
                for q in range(5):
                    print(f"    档{q + 1} " + bucket_line(nets, sel & (bucket == q), fold))


def study_rolling(args) -> None:
    exits = [
        dict(stop_support_gap=g, hold=h, trail_atr=a)
        for g, h, a in itertools.product([0.02, 0.05], [600, 1080], [np.nan, 3])
    ]
    entries = list(itertools.product([5, 10, 15], [0.0, 0.01, 0.02]))
    configs = [replace(BEST, stop_max=0.2, delay=d, confirm=c, **e) for e in exits for d, c in entries]
    for dataset in ("v_nobreak", "v_base"):
        book = Book(dataset)
        month = np.array([int(datetime.fromtimestamp(t, BEIJING).strftime("%Y%m")) for t in book.data.closed_at])
        months = sorted(set(month))
        nets = np.array([np.nan_to_num(book.run(c)[0]) for c in configs])
        j_net = np.nan_to_num(book.run(BEST)[0])
        for label, mask in (("第1波", book.first), ("所有波次 避开7~10点", ~np.isin(book.hour, BAD_HOURS))):
            table = np.array([[nets[k, mask & (month == m)].sum() * 100 for m in months] for k in range(len(configs))])
            test = [i for i, m in enumerate(months) if m >= 202404]
            j_sum = sum(j_net[mask & (month == months[i])].sum() * 100 for i in test)
            fixed = table[:, test].sum(1)
            print(f"===== {dataset} {label}：{months[test[0]]}~{months[test[-1]]}，固定 J {j_sum:+.0f}")
            print(
                f"  {len(configs)} 组固定参数：中位 {np.median(fixed):+.0f}，"
                f"最好 {fixed.max():+.0f}，最差 {fixed.min():+.0f}"
            )
            for window in (3, 6, 12, 99):
                best1 = sum(table[int(table[:, max(0, i - window) : i].sum(1).argmax()), i] for i in test)
                best5 = sum(table[np.argsort(-table[:, max(0, i - window) : i].sum(1))[:5], i].mean() for i in test)
                name = "全部历史" if window == 99 else f"过去{window}个月"
                print(f"  {name}：选最好 1 组 {best1:+.0f}，最好 5 组平均 {best5:+.0f}")


@nb.njit(cache=True)
def _short(bars, alert_price, delay, low, high, stop, take, hold, fee):
    """做空：等 delay 分钟，开盘价相对提醒价在 [low, high] 之间才开空；固定止损、可选止盈、到时平仓。

    与做多的模拟器同样保守：开盘越过线按开盘价，盘中先看止损再看止盈。
    """
    n, width = bars.shape[0], bars.shape[2]
    net = np.full(n, np.nan)
    for a in range(n):
        o, h, lo, c = bars[a, 0], bars[a, 1], bars[a, 2], bars[a, 3]
        e = PRE + delay
        if e + hold > width or np.isnan(o[e]) or np.isnan(c[e - 1]):
            continue
        price = o[e]
        gain = price / alert_price[a] - 1
        if gain < low or gain > high:
            continue
        stop_line = price * (1 + stop)
        target = price * (1 - take) if take > 0 else -1.0
        result = np.nan
        for k in range(e, e + hold):
            if np.isnan(c[k]):
                break
            exit_price = np.nan
            if target > 0 and o[k] <= target:
                exit_price = o[k]
            elif o[k] >= stop_line:
                exit_price = o[k]
            elif h[k] >= stop_line:
                exit_price = stop_line
            elif target > 0 and lo[k] <= target:
                exit_price = target
            elif k == e + hold - 1:
                exit_price = c[k]
            if not np.isnan(exit_price):
                # 与做多相同：开仓手续费按名义本金，平仓手续费按平仓价。
                result = 1 - exit_price / price - fee * exit_price / price - fee
                break
        if not np.isnan(result):
            net[a] = result
    return net


def study_short(args) -> None:
    book = Book(args.dataset)
    bars = book.data.bars
    alert_price = np.ascontiguousarray(bars[:, 3, PRE - 1])
    print("做空第 1 波提醒（6 段合计）")
    for delay, (low, high), (stop, take, hold) in itertools.product(
        (0, 5, 10),
        ((-1.0, 1.0), (-1.0, 0.0), (-1.0, -0.01), (0.02, 1.0)),
        ((0.05, 0.0, 60), (0.1, 0.0, 240), (0.1, 0.1, 600)),
    ):
        net = _short(bars, alert_price, delay, low, high, stop, take, hold, 0.0005)
        ok = book.first & ~np.isnan(net)
        folds = fold_sums(net, book.first, book.data)
        print(
            f"  等{delay:2d}分 开空涨幅[{low:+.2f},{high:+.2f}] 止损{stop} 止盈{take} 持{hold}: "
            f"{ok.sum():5d}笔 合{np.nansum(net[ok]) * 100:+6.0f} 段 " + " ".join(f"{v:+5.0f}" for v in folds)
        )


# 亏损期分析的「最近一个月」：2026-10-10 分析时取的区间起点，固定下来以便复现。
RECENT = int(datetime(2026, 9, 10, tzinfo=BEIJING).timestamp())


def _closed_before(data, net, entry_at, exit_at, mask, days):
    """每条提醒之前 days 天内已平仓的交易：(笔数, 合计, 赚 ≥30% 的笔数)，只用提醒时已知的结果。"""
    ok = mask & ~np.isnan(net)
    t_out = times(data, entry_at, exit_at)[1][ok]
    order = np.argsort(t_out)
    ts = t_out[order]
    cum = np.concatenate([[0], np.cumsum(net[ok][order])])
    big = np.concatenate([[0], np.cumsum(net[ok][order] >= 0.3)])
    hi = np.searchsorted(ts, data.closed_at)
    lo = np.searchsorted(ts, data.closed_at - days * 86400)
    return hi - lo, cum[hi] - cum[lo], big[hi] - big[lo]


def _regime_monthly(cases) -> None:
    """逐月：买了就没涨（最大浮盈 < 3%）与回吐（浮盈到过 ≥ 20% 却亏或 < 3%）的占比、赚 ≥30% 的大单数。"""
    for name, book, rule, mask in cases:
        bars = book.data.bars
        net, entry_at, exit_at = book.run(rule)
        ok = mask & ~np.isnan(net)
        peak = np.full(len(net), np.nan)
        for k in np.flatnonzero(ok):
            peak[k] = np.nanmax(bars[k, 1, entry_at[k] : exit_at[k] + 1]) / bars[k, 0, entry_at[k]] - 1
        month = np.array([datetime.fromtimestamp(t, BEIJING).strftime("%Y-%m") for t in book.data.closed_at])
        print(f"===== {name} 逐月：合计（每笔 100）| 笔数 | 胜率 | 买了就没涨 | 回吐 | 赚≥30%的笔数")
        for m in sorted(set(month[ok])):
            sel = ok & (month == m)
            x = net[sel]
            giveback = ((peak[sel] >= 0.2) & (x < 0.03)).mean()
            print(
                f"  {m} {x.sum() * 100:+7.0f} {sel.sum():5d} {(x > 0).mean():5.0%} "
                f"{(peak[sel] < 0.03).mean():6.0%} {giveback:6.0%} {(x >= 0.3).sum():4d}"
            )


def _regime_states(cases) -> None:
    """长周期市场状态与策略近期大单占比，分五档看 6 段。"""
    hours, above = hourly_breadth()
    market = np.load(MARKET)
    close, symbols = market["close"], list(market["symbols"])
    with np.errstate(invalid="ignore", divide="ignore"):
        r720 = close / np.roll(close, 720, 0) - 1
        r720[:720] = np.nan
        r168 = close / np.roll(close, 168, 0) - 1
        r168[:168] = np.nan
        states = {
            "全市场30天中位涨幅": np.nanmedian(r720, 1),
            "站上30天均线占比": above,
            "BTC30天涨幅": r720[:, symbols.index("BTC_USDT")],
            "全市场7天中位涨幅": np.nanmedian(r168, 1),
        }
    for name, book, rule, mask in cases:
        data = book.data
        net, entry_at, exit_at = book.run(rule)
        ok = mask & ~np.isnan(net)
        slot = (data.closed_at.astype(np.int64) - int(hours[0])) // 3600 - 1
        # r3_market.npz 比提醒导出旧时，超出范围的提醒记为缺失，不越界。
        inside = (slot >= 0) & (slot < len(hours))
        slot = np.clip(slot, 0, len(hours) - 1)
        features = {k: np.where(inside, v[slot], np.nan) for k, v in states.items()}
        count, _, big = _closed_before(data, net, entry_at, exit_at, mask, 14)
        features["近14天大单占比"] = np.where(count >= 20, big / np.maximum(count, 1), np.nan)
        fold = folds_of(book)
        print(f"===== {name} 按市场状态分档（每笔 100 合计，6 段）")
        for k, v in features.items():
            sel = ok & ~np.isnan(v)
            edges = np.nanpercentile(v[sel], [20, 40, 60, 80])
            bucket = np.searchsorted(edges, v)
            print(f"  {k} 分位点 {np.round(edges, 3)}")
            for q in range(5):
                part = sel & (bucket == q)
                per = np.array([net[part & (fold == f)].sum() * 100 for f in range(6)])
                print(
                    f"    档{q + 1} 每笔{np.nanmean(net[part]) * 100:+5.2f}% 合{per.sum():+6.0f} "
                    f"正段{(per > 0).sum()}/6 | " + " ".join(f"{x:+5.0f}" for x in per)
                )


def _regime_hot(book: Book, rule) -> None:
    """P3 的 J 类与扩展类在不同过热阈值下（6 段 + 最近一个月），只用前 18 个月选阈值，以及最近一个月按提醒类型。"""
    data = book.data
    net = book.run(rule)[0]
    ok = ~np.isnan(net)
    core = book.core(rule)
    breadth = np.nan_to_num(book.breadth, nan=0.0)
    fold = folds_of(book)
    recent = data.closed_at >= RECENT
    early = data.closed_at < HALF
    print("===== P3：J 类（已突破、24h成交额>100万、入场涨≥2%）与扩展类，按过热阈值")
    for threshold in (0.7, 0.75, 0.8, 0.85):
        for hot_name, hot in (("过热", breadth > threshold), ("不过热", breadth <= threshold)):
            for kind, sel in (("J类", core), ("扩展类", ~core)):
                part = ok & hot & sel
                per = [net[part & (fold == f)].sum() * 100 for f in range(6)]
                r = part & recent
                print(
                    f"  阈值{threshold:.0%} {hot_name:3s} {kind:3s} {part.sum():5d}笔 "
                    f"每笔{net[part].mean() * 100:+5.2f}% 正段{sum(p > 0 for p in per)}/6 | "
                    + " ".join(f"{p:+6.0f}" for p in per)
                    + f" | 近1月 {r.sum():3d}笔 {net[r].sum() * 100:+5.0f}"
                )
        hot_core = ok & core & (breadth > threshold)
        print(
            f"  阈值{threshold:.0%} 过热J类：前18月 {net[hot_core & early].sum() * 100:+5.0f}，"
            f"后18月 {net[hot_core & ~early].sum() * 100:+5.0f}"
        )
    print("===== P3 按提醒类型：三年 笔数/每笔 | 最近一个月 笔数/合计/每笔")
    alert_price = data.bars[:, 3, PRE - 1]
    gain = data.bars[:, 0, PRE + int(rule.delay)] / alert_price - 1
    kinds = {
        "已突破": alert_price > data.meta["breakout"],
        "未突破": ~(alert_price > data.meta["breakout"]),
        "24h成交额>100万": data.meta["volume_24h"] > 1e6,
        "冷门币（≤100万）": data.meta["volume_24h"] <= 1e6,
        "入场涨0~2%": (gain >= 0) & (gain < 0.02),
        "入场涨≥2%": gain >= 0.02,
    }
    for k, m in kinds.items():
        a = ok & m
        r = a & recent
        print(
            f"  {k:14s} {a.sum():6d} {net[a].mean() * 100:+5.2f}% | {r.sum():4d} {net[r].sum() * 100:+6.0f} "
            f"{net[r].mean() * 100 if r.any() else 0:+5.2f}%"
        )


def _regime_switch(book: Book, rule) -> None:
    """扩展类按近期盈亏开关：只开一个仓（每笔 1000），换等待分钟与窗口长度，看好处是否稳定。"""
    data = book.data
    print("===== 一个仓、每笔 1000：三年合计 / 回撤，等 5/7/10/12/15 分钟（确认 0%）；最后一列是每个提醒都买的平均")
    p5 = Scheme("P5", "v_nb_pool0_noover", rule, all_waves=True, hot_breadth=HOT_BREADTH, hot_core_only=True)
    variants = (
        ("P3", None),
        ("P5", 0),
        ("P5+扩展类近14天开关", 14),
        ("P5+扩展类近30天开关", 30),
        ("P5+扩展类近60天开关", 60),
    )
    for label, days in variants:
        cells = []
        for delay in (5, 7, 10, 12, 15):
            moved = replace(rule, delay=delay)
            net, entry_at, exit_at = book.run(moved)
            mask = np.ones_like(book.first) if days is None else book.mask(p5, moved)
            if days:
                ext = ~book.core(moved)
                count, recent_sum, _ = _closed_before(data, net, entry_at, exit_at, ext, days)
                # 扩展类只在它们近 days 天已平仓交易合计为正时才买；还没有记录时照买。
                mask = mask & (~ext | (count == 0) | (recent_sum > 0))
            ok = mask & ~np.isnan(net)
            one = capped(data, entry_at, exit_at, ok, 1)
            t_out = times(data, entry_at, exit_at)[1]
            index = np.flatnonzero(one)
            curve = np.cumsum(net[index[np.argsort(t_out[index])]] * 1000)
            drop = np.max(np.maximum.accumulate(np.concatenate([[0], curve]))[1:] - curve)
            cells.append((curve[-1], drop, net[ok].sum() * 1000))
        mean = np.mean([c[0] for c in cells])
        many = np.mean([c[2] for c in cells])
        print(
            f"  {label:20s}"
            + "".join(f"{a:+8.0f}/{b:5.0f}" for a, b, _ in cells)
            + f" | 平均 {mean:+7.0f} | 多个仓平均 {many:+8.0f}"
        )


def study_regime(args) -> None:
    """J 与 P3 的亏损期：信号失效还是出场回吐、和市场状态的关系、过热过滤与扩展类开关。"""
    jb = Book("v_base")
    pb = Book("v_nb_pool0_noover")
    p3 = replace(X, confirm=0.0)
    cases = (("J", jb, BEST, jb.first), ("P3", pb, p3, np.ones_like(pb.first)))
    _regime_monthly(cases)
    _regime_states(cases)
    _regime_hot(pb, p3)
    _regime_switch(pb, p3)


STUDIES = {
    "waves": (study_waves, "r3"),
    "exits": (study_exits, "r3"),
    "holds": (study_holds, "r3"),
    "shapes": (study_shapes, "r3"),
    "features": (study_features, "r3"),
    "walkforward": (study_walkforward, "r3"),
    "variants": (study_variants, None),
    "breakout": (study_breakout, "v_nobreak"),
    "control": (study_control, None),
    "confirm": (study_confirm, None),
    "market": (study_market, None),
    "rolling": (study_rolling, None),
    "short": (study_short, "r3"),
    "regime": (study_regime, None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--dataset", help="数据集目录，默认按研究而定")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    func, default = STUDIES[args.study]
    args.dataset = args.dataset or default
    func(args)


if __name__ == "__main__":
    main()
