"""grid.py 的加速版：每组（涨跌幅、ATR、确认、笔数）门槛只取一次行，
72 种（合约池 × 新条件 × 冷却）在一个 numba 内核里扫完。

补齐秒和空档复核失败的行既不会提醒、也不会清零确认，预先剔除不影响结果；结果与 grid.run_chunk 逐项相同。
"""

from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

import numba as nb
import numpy as np
from common import DATA, DEV_MONTHS, MONTHS
from engine import VARIANTS
from grid import CD, GATES, GRID_COLUMNS, POOLS, A, B, C, T, chunks, combo_list
from sim import Params, Simulator, gate_mask_cols, load

K = len(MONTHS)


@nb.njit(cache=True)
def kernel(group, sec, t, poolbits, gatebits, month, labels, c, cds, pools, ngates):
    """labels 列：valid、h3、h5、h10、r5、fee、h5d；输出 [合约池, 新条件, 冷却, 8 × K]。"""
    out = np.zeros((len(pools), ngates, len(cds), 8 * K))
    n = len(sec)
    for pi in range(len(pools)):
        for gi in range(ngates):
            for di in range(len(cds)):
                cooldown = cds[di]
                current = -1
                reset_sec = -(1 << 60)
                last_alert = -1e18
                for i in range(n):
                    if (poolbits[i] >> pools[pi]) & 1 == 0 or (gatebits[i] >> gi) & 1 == 0:
                        continue
                    if group[i] != current:
                        current = group[i]
                        reset_sec = -(1 << 60)
                        last_alert = -1e18
                    if sec[i] - reset_sec < c:
                        continue
                    if t[i] - last_alert >= cooldown:
                        last_alert = t[i]
                        m = month[i]
                        out[pi, gi, di, m] += 1
                        for k in range(7):
                            out[pi, gi, di, (k + 1) * K + m] += labels[i, k]
                    reset_sec = sec[i]
    return out


def run_chunk(job: tuple[str, list]) -> tuple[str, np.ndarray]:
    variant, files = job
    rows = load(variant, DEV_MONTHS, GRID_COLUMNS, files)
    sim = Simulator(rows)
    ret = rows["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    r5 = np.where(valid, np.nan_to_num(ret[:, 4]), 0.0)
    labels = np.stack(
        [
            valid,
            pos[:, :3].all(1) & valid,
            pos[:, :5].all(1) & valid,
            pos.all(1) & valid,
            r5,
            (r5 > 0.1) & valid,
            rows["hold5d"] & valid,
        ],
        axis=1,
    ).astype(np.float64)
    cds = np.array(CD, np.float64)
    pools = np.array(POOLS, np.int64)
    stats = []
    for c in C:
        for ai, bi, ti in itertools.product(range(len(A)), range(len(B)), range(len(T))):
            a, b = A[ai], B[bi]
            pr = sim.prepare(Params(a, b, c, T[ti], 0))
            keep = pr.real & (pr.lchg >= a) & (pr.latrm >= b)
            gatebits = np.zeros(len(pr.idx), np.int64)
            for gi, gate in enumerate(GATES):
                gatebits |= gate_mask_cols(pr.features, gate).astype(np.int64) << gi
            k = np.flatnonzero(keep)
            idx = pr.idx[k]
            out = kernel(
                pr.group[k],
                pr.sec[k],
                pr.t[k],
                pr.pool[k].astype(np.int64),
                gatebits[k],
                rows.month[idx].astype(np.int64),
                labels[idx],
                c,
                cds,
                pools,
                len(GATES),
            )
            # 与 combo_list 的顺序一致：合约池 → 新条件 → 冷却
            stats.append(out.reshape(-1, 8 * K))
    return variant, np.concatenate(stats)


if __name__ == "__main__":
    variants = sys.argv[1:] or list(VARIANTS)
    jobs = [job for v in variants for job in chunks(v)]
    totals: dict[str, np.ndarray] = {}
    left = {v: sum(1 for j in jobs if j[0] == v) for v in variants}
    combos = np.array(combo_list(), np.int8)
    with Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for n, (v, stats) in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            totals[v] = totals.get(v, 0) + stats
            left[v] -= 1
            print(n, "/", len(jobs), v, flush=True)
            if not left[v]:
                np.savez(DATA / f"grid_{v}.npz", combos=combos, stats=totals.pop(v))
                print("完成", v, flush=True)
