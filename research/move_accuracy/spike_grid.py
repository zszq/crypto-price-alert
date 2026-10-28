"""第四轮写定的网格（PLAN.md 第四轮）：按检测器口径重算短窗口、长窗口在每组参数与新条件下的提醒，按月汇总标签。

短窗口与长窗口各自冷却、互不影响，所以分开算：短窗口每组（窗口、涨跌幅、ATR、确认）× 108 种新条件，
长窗口固定现行参数 × 108 种新条件（其中「不限」那一档就是不加条件）。选参时按组合把两部分的计数相加。
输出 spike4_grid_<名称>.npz：agg_<部分>[参数, 条件, 月份, 指标]，指标为 条数、有效、hold5、side5、hold3、延迟 5 秒 hold5。
alerts_for() 供验证时取单组提醒明细。用法：spike_grid.py sel|oos
"""

from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA, FRESH_MONTHS, MONTHS, ROUND3_MONTHS, ROUND4_MONTHS, SPIKE_OOS_MONTHS
from engine import rows_dir
from engine4 import ROWS4, ROWS4_LONG10
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, load, row_count, scan
from spike_feat import OPP4

PERIODS = {"sel": ROUND4_MONTHS + ROUND3_MONTHS + FRESH_MONTHS + MONTHS, "oos": SPIKE_OOS_MONTHS}
# 短窗口长度 -> （短窗口变体，长窗口变体）；长窗口的补齐上限跟随短窗口
VARIANTS = {10: ("L10F10", "L180F10"), 30: ("L30", "L180F30")}
FOLDERS = {"L10F10": rows_dir("L10F10"), "L30": ROWS4, "L180F30": ROWS4, "L180F10": ROWS4_LONG10}
A = (1.0, 1.3, 2.0)
B = (1.0, 1.5)
C = (1, 2)
TRADES, COOLDOWN, POOL = 15, 30.0, 1
LONG = Params(1.6, 1.8, 3, 30, 180.0, POOL)
PRE = (None, (300, 0.0), (300, -1.0), (300, -2.0), (300, -3.0), (900, 0.0), (900, -2.0), (900, -4.0), (900, -6.0))
OPP = (None, 15.0, 10.0, 5.0)
BURST = (None, 4.0, 8.0)
GATES = list(itertools.product(PRE, OPP, BURST))
SHORT_PARAMS = [Params(a, b, c, TRADES, COOLDOWN, POOL) for a, b, c in itertools.product(A, B, C)]
KEEP = COLUMNS + EXTRA_COLUMNS
NMET = 6


def files_of(variant: str, months: list[str], limit: int = 3_000_000) -> list[list]:
    chunks, cur, size = [], [], 0
    for path in sorted(FOLDERS[variant].glob("*.npz")):
        if ".tmp" in path.name or path.stem.rsplit("-", 1)[1] not in months:
            continue
        n = row_count(path, variant)
        if not n:
            continue
        if cur and size + n > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(path)
        size += n
    return chunks + ([cur] if cur else [])


def load_chunk(variant: str, months: list[str], files: list):
    rows = load(variant, months, KEEP, files)
    opp = np.concatenate([np.load(OPP4 / variant / f"{p.stem}.npy") for p in files])
    assert len(opp) == len(rows), (variant, len(opp), len(rows))
    return rows, opp


def gate_ok(rows, opp: np.ndarray, idx: np.ndarray, gate: tuple) -> np.ndarray:
    pre, d, r = gate
    ok = np.ones(len(idx), bool)
    if pre is not None:
        span, x = pre
        ok &= rows["pre300" if span == 300 else "pre"][idx] <= x
    if d is not None:
        ok &= opp[idx] <= d
    if r is not None:
        ok &= rows["burst"][idx] >= r
    return ok


def labels(rows) -> np.ndarray:
    ret = rows["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    return np.column_stack(
        [np.ones(len(ret), bool), valid, pos[:, :5].all(1) & valid, pos[:, 4] & valid, pos[:, :3].all(1) & valid,
         rows["hold5d"] & valid]
    ).astype(np.int32)


def hits(rows, opp, sim: Simulator, p: Params, idx: np.ndarray, gate: tuple) -> np.ndarray:
    ok = ((rows["pool"][idx] >> p.pool) & 1 == 1) & gate_ok(rows, opp, idx, gate)
    args = (rows.group[idx], rows["sec"][idx], rows["t"][idx], rows["real"][idx], rows["lchg"][idx], rows["latrm"][idx])
    return idx[scan(*args, ok, p.a, p.b, p.c, p.cooldown)]


def chunk_job(args) -> tuple[str, np.ndarray]:
    part, variant, months, files, params = args
    rows, opp = load_chunk(variant, months, files)
    lab = labels(rows)
    sim = Simulator(rows)
    agg = np.zeros((len(params), len(GATES), len(ALL_MONTHS), NMET), np.int64)
    for i, p in enumerate(params):
        idx = sim.eligible(p)
        for g, gate in enumerate(GATES):
            h = hits(rows, opp, sim, p, idx, gate)
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
    np.savez(DATA / f"spike4_grid_{name}.npz", **{f"agg_{k}": v for k, v in out.items()})


def alerts_for(variant: str, months: list[str], p: Params, gate: tuple) -> dict[str, np.ndarray]:
    """单组参数的提醒明细（时间、方向、月份、合约、标签列），供自助法与分解。"""
    parts = []
    for files in files_of(variant, months):
        rows, opp = load_chunk(variant, months, files)
        h = hits(rows, opp, Simulator(rows), p, Simulator(rows).eligible(p), gate)
        lab = labels(rows)[h]
        parts.append(
            {"t": rows["t"][h], "dir": rows["dir"][h], "month": rows.month[h],
             "symbol": np.array([rows.symbols[g] for g in rows.group[h]]), "lab": lab, "ret": rows["ret"][h]}
        )
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


if __name__ == "__main__":
    run(sys.argv[1])
