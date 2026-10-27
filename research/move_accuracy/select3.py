"""第三轮按写定规则选参（PLAN.md 第三轮，只用 2025-10 ~ 2026-09），并算 CSCV 的 PBO，写 selection3.json。"""

from __future__ import annotations

import itertools
import json

import numpy as np
from common import ALL_MONTHS, DATA, FRESH_MONTHS, MONTHS
from grid3 import CURRENT, combos
from sim import month_days

DEV = MONTHS + FRESH_MONTHS
DAYS = sum(month_days(m) for m in DEV)
TIE = 0.02
TARGETS = {"A": 2.5, "B": 1.0}
NM = len(DEV)


def per_combo(alerts: np.ndarray, ncombo: int):
    """每组每月的提醒数、有效数、hold5 数。"""
    k = alerts[:, 0].astype(int)
    m = alerts[:, 4].astype(int)
    shape = (ncombo, len(ALL_MONTHS))
    count = np.zeros(shape)
    valid = np.zeros(shape)
    h5 = np.zeros(shape)
    np.add.at(count, (k, m), 1)
    np.add.at(valid, (k, m), alerts[:, 5])
    np.add.at(h5, (k, m), alerts[:, 6])
    cols = [ALL_MONTHS.index(x) for x in DEV]
    return count[:, cols], valid[:, cols], h5[:, cols]


def pbo(valid: np.ndarray, h5: np.ndarray, cand: np.ndarray) -> float:
    """CSCV：12 个月对半分成 924 种，样本内最好的一组在样本外排名落到后一半的比例。"""
    v, h = valid[cand], h5[cand]
    bad = 0
    total = 0
    for ins in itertools.combinations(range(NM), NM // 2):
        mask = np.zeros(NM, bool)
        mask[list(ins)] = True
        r_in = h[:, mask].sum(1) / np.maximum(v[:, mask].sum(1), 1)
        r_out = h[:, ~mask].sum(1) / np.maximum(v[:, ~mask].sum(1), 1)
        best = int(np.argmax(r_in))
        rank = (r_out < r_out[best]).sum() + 0.5 * ((r_out == r_out[best]).sum() - 1)
        omega = (rank + 0.5) / len(r_out)
        bad += omega <= 0.5
        total += 1
    return bad / total


def describe(c: tuple) -> str:
    a, cc, pool, pre, burst, span, x, cd = c
    opp = "不限" if span is None else f"{int(span)} 分钟 ≤ {x:g}%"
    return (
        f"涨跌幅 {a:g}%、确认 {cc}、{'800' if pool == 1 else '500'} 万池、前置 ≤ {pre:g}%、暴增 ≥ {burst:g}、"
        f"反向极值 {opp}、冷却 {cd:g}"
    )


if __name__ == "__main__":
    g = np.load(DATA / "grid3_dev.npz")
    table = combos()
    count, valid, h5 = per_combo(g["alerts"], len(table))
    per_day = count.sum(1) / DAYS
    rate = h5.sum(1) / np.maximum(valid.sum(1), 1)
    halves = [slice(0, 6), slice(6, 12)]
    half_rate = np.column_stack([h5[:, s].sum(1) / np.maximum(valid[:, s].sum(1), 1) for s in halves])
    cur = table.index(CURRENT)
    print(f"现行：每天 {per_day[cur]:.2f}，hold5 {rate[cur]:.1%}，两个半年 {half_rate[cur, 0]:.1%} / {half_rate[cur, 1]:.1%}")
    nconds = np.array([0 if c[5] is None else 1 for c in table])
    selection = {}
    for name, floor in TARGETS.items():
        ok = (per_day >= floor) & (half_rate >= half_rate[cur]).all(1)
        cand = np.flatnonzero(ok)
        best = rate[cand].max()
        tied = cand[rate[cand] >= best - TIE]
        pick = max(tied, key=lambda i: (per_day[i], -nconds[i], table[i][7]))
        p = pbo(valid, h5, cand)
        print(
            f"\n方案 {name}（每天 ≥ {floor}）：候选 {len(cand)} 组，最高 hold5 {best:.1%}，同等 {len(tied)} 组，PBO {p:.2f}\n"
            f"  选中：{describe(table[pick])}\n  每天 {per_day[pick]:.2f}，hold5 {rate[pick]:.1%}，"
            f"两个半年 {half_rate[pick, 0]:.1%} / {half_rate[pick, 1]:.1%}，逐月 "
            + " ".join(f"{h5[pick, j] / max(valid[pick, j], 1):.0%}" for j in range(NM))
        )
        for i in sorted(cand, key=lambda i: -rate[i])[:8]:
            print(f"    {rate[i]:.1%} 每天 {per_day[i]:.2f}  {describe(table[i])}")
        selection[name] = {"combo": list(table[pick]), "pbo": p, "dev_per_day": per_day[pick], "dev_hold5": rate[pick]}
    json.dump(selection, open(DATA / "selection3.json", "w"), ensure_ascii=False, indent=1)
