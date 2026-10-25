"""短、长窗口参数网格：各窗口单独重算提醒（两窗口判定与冷却互不影响），再两两组合汇总。

输出 results/grid_<variant>.pkl（每个窗口参数的逐条提醒与大行情首报延迟）供 combine.py 组合。
用法：data/backtest/.venv/bin/python research/move_params/grid.py [--variant s30_l180] [--grid wide]
（wide 是第三轮的候选族，输出 results/grid_wide_<variant>.pkl）
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle
from multiprocessing import Pool

import evaluate
import numpy as np
import sim

SHORT_GRID = [
    sim.Win(c, a, n, k, cd)
    for c, a, n, k, cd in itertools.product([1.2, 1.5, 1.8, 2.0], [1.5, 2.0, 2.5], [15, 30], [2, 3], [30, 60, 120, 300])
]
LONG_GRID = [
    sim.Win(c, a, n, k, cd)
    for c, a, n, k, cd in itertools.product(
        [1.6, 2.0, 2.5, 3.0], [1.8, 2.2, 2.6, 3.0], [30, 60], [3, 5], [180, 300, 600, 900]
    )
]
# 第三轮（move_params.md 第九节）按质量选参：第二轮推荐落在短窗口涨跌幅、ATR 的上沿，两个窗口都向两侧扩；
# 长窗口冷却 300 以上在第一轮已明显丢大行情，只留 180、300。每项依次为（短窗口取值, 长窗口取值）
WIDE = {
    "change": ([1.2, 1.5, 1.8, 2.0, 2.5, 3.0], [1.2, 1.4, 1.6, 2.0, 2.5]),
    "atr": ([1.5, 2.0, 2.5, 3.0, 3.5], [1.5, 1.8, 2.2, 2.6]),
    "trades": ([15, 30, 60], [30, 60, 120]),
    "confirm": ([2, 3, 5], [2, 3, 5]),
    "cooldown": ([30, 60, 120, 300], [180, 300]),
}
WIDE_SHORT = [sim.Win(*v) for v in itertools.product(*(side[0] for side in WIDE.values()))]
WIDE_LONG = [sim.Win(*v) for v in itertools.product(*(side[1] for side in WIDE.values()))]
FIELDS = ("good", "fake", "cont", "rev", "held")


def events(variant: str):
    rows = []
    for sid, symbol in enumerate(evaluate.symbols(variant)):
        for level in (3.0, 5.0):
            for t0, t1, d in evaluate.big_moves(symbol, level, variant):
                rows.append((sid, t0, t1, d, level))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def run(job):
    name, win, variant, ev = job
    symbols = evaluate.symbols(variant)
    cols = {k: [] for k in ("sym", "t", "d", "chg", "r5", *FIELDS)}
    delay = np.full(len(ev), np.inf)
    for sid, symbol in enumerate(symbols):
        data = sim.load(symbol, variant)
        rec = data[name]
        alerts = []
        for i in sim.simulate_window(rec, win):
            r = rec[i]
            if sim.in_pool(symbol, r[sim.NOW], variant=variant):
                d = 1 if r[sim.CHG] > 0 else -1
                alerts.append(
                    (r[sim.NOW], name, d, r[sim.PRICE], r[sim.BASE], r[sim.CHG], r[sim.MATR], int(r[sim.TRADES]))
                )
        for o in outcomes_cached(symbol, variant, alerts):
            cols["sym"].append(sid)
            cols["t"].append(o["t"])
            cols["d"].append(o["d"])
            cols["chg"].append(o["chg"])
            cols["r5"].append(o["r5"])
            for k in FIELDS:
                cols[k].append(o[k])
        times = np.array([a[0] for a in alerts])
        dirs = np.array([a[2] for a in alerts])
        for e in np.flatnonzero(ev[:, 0] == sid):
            _, t0, t1, d, _ = ev[e]
            hit = np.flatnonzero((times >= t0) & (times <= t1 + 60) & (dirs == d))
            if len(hit):
                delay[e] = times[hit[0]] - t0
    return name, win, {k: np.array(v) for k, v in cols.items()}, delay


_cache: dict = {}


def outcomes_cached(symbol, variant, alerts):
    out = []
    missing = [a for a in alerts if (symbol, a[0], a[2]) not in _cache]
    for o in sim.outcomes(symbol, missing, variant):
        _cache[(symbol, o["t"], o["d"])] = o
    for a in alerts:
        o = dict(_cache[(symbol, a[0], a[2])])
        o["win"] = a[1]
        out.append(o)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s30_l180")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--grid", choices=("base", "wide"), default="base")
    ns = parser.parse_args()
    ev = events(ns.variant)
    shorts, longs = (SHORT_GRID, LONG_GRID) if ns.grid == "base" else (WIDE_SHORT, WIDE_LONG)
    jobs = [("short", w, ns.variant, ev) for w in shorts] + [("long", w, ns.variant, ev) for w in longs]
    results = {}
    with Pool(ns.workers) as pool:
        for i, (name, win, cols, delay) in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            results[(name, win)] = (cols, delay)
            if i % 50 == 0:
                print(i, "/", len(jobs), flush=True)
    out = sim.DATA / "results" / f"grid_{'' if ns.grid == 'base' else 'wide_'}{ns.variant}.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"events": ev, "symbols": evaluate.symbols(ns.variant), "results": results}, f)
    print("saved", out, math.fsum(len(c["t"]) for c, _ in results.values()))


if __name__ == "__main__":
    main()
