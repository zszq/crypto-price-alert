"""第二轮网格与选参（PLAN.md 第二轮）：10 秒窗口 × 前置走势 × 成交笔数暴增 × 远离 6 小时极值，只用 2026-04~09。

输出 grid2.npz（参数与逐月汇总）与 selection2.json（按写定规则选中的一组）。
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from common import ALL_MONTHS, DATA, MONTHS
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, gate_mask_cols, load

VARIANT = "L10F10"
A = (1.0, 1.5, 2.0)
C = (1, 2)
T = (5, 20)
CD = (30, 300)
POOLS = (0, 1)
PRIOR = (("pre300_neg", (0, -2, -4, -6)), ("pre_neg", (0, -2, -4, -6)))
BURST = (6, 8, 12, 16, 24)
EXT = (None, -5, -10, -20)
DAYS = 183.0
TIE = 0.02


def summarize(rows, hit: np.ndarray) -> dict:
    ret = rows["ret"][hit]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    m = rows.month[hit]
    h5 = pos[:, :5].all(1) & valid
    k = len(ALL_MONTHS)
    return {
        "n": len(hit),
        "valid": int(valid.sum()),
        "h5": int(h5.sum()),
        "h3": int((pos[:, :3].all(1) & valid).sum()),
        "pos5": int(((ret[:, 4] > 0) & valid).sum()),
        "month_valid": np.bincount(m, weights=valid, minlength=k),
        "month_h5": np.bincount(m, weights=h5, minlength=k),
    }


def run_grid(rows) -> list[tuple]:
    sim = Simulator(rows)
    out = []
    for a, c, t in itertools.product(A, C, T):
        pr = sim.prepare(Params(a, 1.0, c, t, 0))
        for pool in POOLS:
            in_pool = (pr.pool >> pool) & 1 == 1
            for (pcol, pvals), pv, burst, ext in itertools.product(PRIOR, range(4), BURST, EXT):
                gates = [(pcol, float(pvals[pv])), ("burst", float(burst))]
                if ext is not None:
                    gates.append(("ext6h", float(ext)))
                ok = in_pool.copy()
                for g in gates:
                    ok &= gate_mask_cols(pr.features, g)
                for cd in CD:
                    p = Params(a, 1.0, c, t, cd, pool)
                    hit = sim.alerts_prepared(pr, p, ok)
                    out.append((p, tuple(gates), summarize(rows, hit)))
    return out


if __name__ == "__main__":
    rows = load(VARIANT, MONTHS, COLUMNS + EXTRA_COLUMNS)
    results = run_grid(rows)
    cands = []
    for p, gates, s in results:
        if s["n"] / DAYS >= 2 and s["n"] >= 400 and s["valid"]:
            cands.append((s["h5"] / s["valid"], s["n"] / DAYS, len([g for g in gates if g[0] != "burst"]), p, gates, s))
    print(f"{len(results)} 组，满足条数约束 {len(cands)} 组")
    best = max(c[0] for c in cands)
    tied = [c for c in cands if c[0] >= best - TIE]
    h5, per_day, _, p, gates, s = max(tied, key=lambda c: (c[1], -c[2]))
    print(f"最高 hold5 {best:.1%}；同等（2 个百分点内）{len(tied)} 组")
    for c in sorted(cands, key=lambda c: -c[0])[:15]:
        print(f"  hold5 {c[0]:.1%} 每天 {c[1]:.1f} {c[3]} {c[4]}")
    monthly = s["month_h5"] / np.maximum(s["month_valid"], 1)
    print(
        f"\n选中：{p} {gates}\n  每天 {per_day:.1f} 条，hold5 {h5:.1%}，hold3 {s['h3'] / s['valid']:.1%}，"
        f"5 分钟后顺势 {s['pos5'] / s['valid']:.1%}，逐月 hold5 "
        + " ".join(f"{ALL_MONTHS[i][4:]}月 {monthly[i]:.1%}" for i in range(len(MONTHS)))
    )
    json.dump(
        {"variant": VARIANT, "params": vars(p), "gates": [list(g) for g in gates]},
        open(DATA / "selection2.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
