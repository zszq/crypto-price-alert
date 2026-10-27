"""第三轮写定的网格（PLAN.md 第三轮，3072 组）：在 rows_L10F10 上按检测器口径重算每组的提醒，写 grid3_<名称>.npz。

可以提醒的行都在 cand3 候选里（候选族的门槛都比候选严），所以确认连续性在全部行上算、扫描只走候选行，结果与全量扫描相同。
输出每组参数与全部提醒（组号、合约-月份、时间、方向、月份、标签），选参与验证都从这里取。用法：grid3.py dev|r3
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from cand3 import PERIODS
from common import DATA
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, chunked_files, load, scan

A = (1.0, 1.5, 2.0)
C = (1, 2)
POOLS = (0, 1)
PRE = (-4.0, -6.0, -8.0, -10.0)
BURST = (8.0, 12.0, 16.0, 24.0)
OPP = ((None, None),) + tuple((n, x) for n in (720, 1440, 2880) for x in (-2.0, 0.0, 2.0, 5.0, 10.0))
CD = (30.0, 300.0)
CURRENT = (1.5, 1, 1, -6.0, 16.0, None, None, 30.0)


def combos() -> list[tuple]:
    return [
        (a, c, pool, pre, burst, n, x, cd)
        for a, c, pool, pre, burst, (n, x), cd in itertools.product(A, C, POOLS, PRE, BURST, OPP, CD)
    ]


def run(name: str) -> None:
    months = PERIODS[name]
    cand = np.load(DATA / f"cand3_{name}.npz", allow_pickle=True)
    opp = np.load(DATA / f"cand3_{name}_opp.npz")
    lookup = {}
    for key in np.unique(cand["symbol"]):
        i = np.flatnonzero(cand["symbol"] == key)
        lookup[key] = (cand["sec"][i], {f"opp_{n}": opp[f"opp_{n}"][i] for n in (720, 1440, 2880)})
    table = combos()
    out: list[np.ndarray] = []
    symbols: list[str] = []
    for files in chunked_files("L10F10", months):
        rows = load("L10F10", months, COLUMNS + EXTRA_COLUMNS, files)
        n = len(rows)
        incand = np.zeros(n, bool)
        extra = {k: np.full(n, np.inf) for k in ("opp_720", "opp_1440", "opp_2880")}
        for g, key in enumerate(rows.symbols):
            if key not in lookup:
                continue
            idx = np.flatnonzero(rows.group == g)
            secs, vals = lookup[key]
            j = idx[np.searchsorted(rows["sec"][idx], secs)]
            assert (rows["sec"][j] == secs).all()
            incand[j] = True
            for k, v in vals.items():
                extra[k][j] = np.where(np.isnan(v), np.inf, v)
        offset = len(symbols)
        symbols += rows.symbols
        simr = Simulator(rows)
        ret = rows["ret"]
        valid = ~np.isnan(ret).any(axis=1)
        pos = np.nan_to_num(ret, nan=-1) > 0
        labels = np.column_stack(
            [valid, pos[:, :5].all(1) & valid, pos[:, :3].all(1) & valid, rows["hold5d"] & valid]
        ).astype(np.int8)
        for a, c in itertools.product(A, C):
            idx = simr.eligible(Params(a, 1.0, c, 5, 0))
            idx = idx[incand[idx]]
            cols = {k: rows[k][idx] for k in ("pre", "burst", "pool")} | {k: v[idx] for k, v in extra.items()}
            args = (rows.group[idx], rows["sec"][idx], rows["t"][idx], rows["real"][idx], rows["lchg"][idx], rows["latrm"][idx])
            for k, combo in enumerate(table):
                if combo[0] != a or combo[1] != c:
                    continue
                _, _, pool, pre, burst, span, x, cd = combo
                ok = ((cols["pool"] >> pool) & 1 == 1) & (cols["pre"] <= pre) & (cols["burst"] >= burst)
                if span is not None:
                    ok &= cols[f"opp_{span}"] <= x
                h = idx[scan(*args, ok, a, 1.0, c, cd)]
                if len(h):
                    out.append(
                        np.column_stack(
                            [np.full(len(h), k), rows.group[h] + offset, rows["t"][h], rows["dir"][h], rows.month[h],
                             labels[h], ret[h, 4]]
                        )
                    )
        print("块", len(symbols), flush=True)
    alerts = np.concatenate(out)
    np.savez(
        DATA / f"grid3_{name}.npz",
        combos=np.array([[np.nan if v is None else v for v in c] for c in table], float),
        alerts=alerts,
        symbols=np.array(symbols),
    )
    print(name, len(table), "组，提醒行", len(alerts))


if __name__ == "__main__":
    run(sys.argv[1])
