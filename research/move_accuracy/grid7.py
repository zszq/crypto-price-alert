"""第七轮写定的网格（PLAN.md 第七轮）：保住数量的瞬时异动，按检测器口径重算每组参数与条件下的提醒，按月汇总标签。

与第四轮 spike_grid.py 同一套重算（短窗口、长窗口各自冷却，分开算再按组合相加），只换候选族：
前置走势加上 +1% / −1% 这类温和档、成交暴增加上 1.5 / 2 / 3 倍，不算反向极值（2023-10 ~ 2024-03 没有这一列）。
输出 grid7_<名称>.npz：agg_<部分>[参数, 条件, 月份, 指标]，指标与第四轮相同。用法：grid7.py sel|oos|oos4
"""

from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
from common import (
    ALL_MONTHS,
    DATA,
    EXIT_OOS_MONTHS,
    FRESH_MONTHS,
    MONTHS,
    ROUND3_MONTHS,
    ROUND4_MONTHS,
    SPIKE_OOS_MONTHS,
)
from sim import Params, Simulator, load, scan
from spike_grid import KEEP, LONG, NMET, VARIANTS, files_of, labels

PERIODS = {
    "sel": ROUND4_MONTHS + ROUND3_MONTHS + FRESH_MONTHS + MONTHS,
    "oos": EXIT_OOS_MONTHS,
    # 第四轮的样本外，已不干净，只作参考
    "oos4": SPIKE_OOS_MONTHS,
}
A = (1.0, 1.3)
B = (1.0, 1.5)
C = (1, 2)
TRADES, COOLDOWN, POOL = 15, 30.0, 1
PRE = (None, (300, 1.0), (300, 0.0), (300, -1.0), (300, -2.0), (900, 1.0), (900, 0.0), (900, -1.0), (900, -2.0))
BURST = (None, 1.5, 2.0, 3.0, 4.0)
GATES = list(itertools.product(PRE, BURST))
SHORT_PARAMS = [Params(a, b, c, TRADES, COOLDOWN, POOL) for a, b, c in itertools.product(A, B, C)]


def gate_ok(rows, idx: np.ndarray, gate: tuple) -> np.ndarray:
    pre, r = gate
    ok = np.ones(len(idx), bool)
    if pre is not None:
        span, x = pre
        # NaN（窗口开始前没有成交记录）比较结果为 False，与检测器缺历史时不提醒一致
        ok &= rows["pre300" if span == 300 else "pre"][idx] <= x
    if r is not None:
        ok &= rows["burst"][idx] >= r
    return ok


def hits(rows, p: Params, idx: np.ndarray, gate: tuple) -> np.ndarray:
    ok = ((rows["pool"][idx] >> p.pool) & 1 == 1) & gate_ok(rows, idx, gate)
    args = (rows.group[idx], rows["sec"][idx], rows["t"][idx], rows["real"][idx], rows["lchg"][idx], rows["latrm"][idx])
    return idx[scan(*args, ok, p.a, p.b, p.c, p.cooldown)]


def chunk_job(args) -> tuple[str, np.ndarray]:
    part, variant, months, files, params = args
    rows = load(variant, months, KEEP, files)
    lab = labels(rows)
    sim = Simulator(rows)
    agg = np.zeros((len(params), len(GATES), len(ALL_MONTHS), NMET), np.int64)
    for i, p in enumerate(params):
        idx = sim.eligible(p)
        for g, gate in enumerate(GATES):
            h = hits(rows, p, idx, gate)
            np.add.at(agg[i, g], rows.month[h], lab[h])
    return part, agg


def run(name: str) -> None:
    months = PERIODS[name]
    jobs = []
    for length, (short, long) in VARIANTS.items():
        for files in files_of(short, months):
            jobs.append((f"short{length}", short, months, files, SHORT_PARAMS))
        for files in files_of(long, months):
            jobs.append((f"long{length}", long, months, files, [LONG]))
    out: dict[str, np.ndarray] = {}
    with Pool(int(os.environ.get("WORKERS", "6"))) as pool:
        for n, (part, agg) in enumerate(pool.imap_unordered(chunk_job, jobs), 1):
            out[part] = out.get(part, 0) + agg
            print(n, "/", len(jobs), part, flush=True)
    np.savez(DATA / f"grid7_{name}.npz", **{f"agg_{k}": v for k, v in out.items()})


def alerts_for(variant: str, months: list[str], p: Params, gate: tuple) -> dict[str, np.ndarray]:
    """单组参数的提醒明细（时间、方向、月份、合约、标签列），供自助法与分解。"""
    parts = []
    for files in files_of(variant, months):
        rows = load(variant, months, KEEP, files)
        sim = Simulator(rows)
        h = hits(rows, p, sim.eligible(p), gate)
        parts.append(
            {"t": rows["t"][h], "dir": rows["dir"][h], "month": rows.month[h],
             "symbol": np.array([rows.symbols[g] for g in rows.group[h]]), "lab": labels(rows)[h], "ret": rows["ret"][h]}
        )
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


if __name__ == "__main__":
    run(sys.argv[1])
