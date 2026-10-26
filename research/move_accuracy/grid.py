"""第一、二步网格的候选族与公共函数：现有参数 × （不加或加一个新条件），只用开发期。

扫描由 fastgrid.py 执行（输出 grid_<变体>.npz：每个组合的参数下标与开发期逐月指标）。
"""

from __future__ import annotations

import itertools

import numpy as np
from common import DEV_MONTHS, MONTHS
from sim import files_for, row_count

A = (0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0)
B = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
C = (1, 2, 3, 5)
T = (5, 20, 60)
CD = (30, 120, 300, 900)
POOLS = (0, 1, 2)
GATES: tuple[tuple[str, float] | None, ...] = (
    None,
    ("imb", 0.0),
    ("imb", 0.2),
    ("imb", 0.4),
    ("vr", 2.0),
    ("vr", 4.0),
    ("vr", 8.0),
    ("retr", 0.1),
    ("retr", 0.25),
    ("retr", 0.5),
    ("pre", 0.0),
    ("pre_neg", 0.0),
    ("brk", 0.0),
    ("cap", 3.0),
    ("cap", 5.0),
    ("cap", 8.0),
    ("eff", 0.3),
    ("eff", 0.5),
)
DEV_IDX = [MONTHS.index(m) for m in DEV_MONTHS]


GRID_COLUMNS = (
    "sec",
    "t",
    "dir",
    "real",
    "chg",
    "atrm",
    "trades",
    "lchg",
    "latrm",
    "imb",
    "vr",
    "retr",
    "pre",
    "brk",
    "eff",
    "ret",
    "hold5d",
    "pool",
)
CHUNK_ROWS = 8_000_000


def combo_list() -> list[tuple[int, ...]]:
    out = []
    for ci in range(len(C)):
        for ai, bi, ti in itertools.product(range(len(A)), range(len(B)), range(len(T))):
            for pi in range(len(POOLS)):
                for gi in range(len(GATES)):
                    for di in range(len(CD)):
                        out.append((ai, bi, ci, ti, di, pi, gi))
    return out


def chunks(variant: str) -> list[tuple[str, list]]:
    out, cur, size = [], [], 0
    for path in files_for(DEV_MONTHS):
        n = row_count(path, variant)
        if not n:
            continue
        if cur and size + n > CHUNK_ROWS:
            out.append((variant, cur))
            cur, size = [], 0
        cur.append(path)
        size += n
    if cur:
        out.append((variant, cur))
    return out


def stats_from_alerts(ar: dict) -> np.ndarray:
    """与 fastgrid 内核相同的逐月汇总（条数、有标签条数与各标签之和），输入为 sim.alert_rows 收集的提醒行。"""
    ret = ar["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    r5 = np.where(valid, np.nan_to_num(ret[:, 4]), 0.0)
    m = ar["month"].astype(np.int64)
    k = len(MONTHS)
    weights = (
        valid,
        pos[:, :3].all(1) & valid,
        pos[:, :5].all(1) & valid,
        pos.all(1) & valid,
        r5,
        (r5 > 0.1) & valid,
        ar["hold5d"] & valid,
    )
    return np.concatenate([np.bincount(m, minlength=k)] + [np.bincount(m, weights=w, minlength=k) for w in weights])
