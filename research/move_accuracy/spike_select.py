"""第四轮按写定规则选参（PLAN.md 第四轮，只用 2024-10 ~ 2026-09），并算 CSCV 的 PBO，写 spike4_selection.json。

组合 = 短窗口（长度、涨跌幅、ATR、确认）× 新条件 × 长窗口（关闭 / 现行不加条件 / 加同一组条件），计数由 spike_grid.py 的两部分相加。
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from common import ALL_MONTHS, DATA
from sim import month_days
from spike_grid import GATES, PERIODS, SHORT_PARAMS, VARIANTS

SEL = PERIODS["sel"]
COLS = [ALL_MONTHS.index(m) for m in SEL]
DAYS = sum(month_days(m) for m in SEL)
TIE = 0.01
TARGETS = {"A": 50.0, "B": 15.0}
LONG_OPTIONS = ("off", "plain", "gated")
PLAIN = GATES.index((None, None, None))
# 指标列：条数、有效、hold5、side5、hold3、延迟 5 秒 hold5
N, V, H5, S5, H3, H5D = range(6)


def combos() -> list[tuple]:
    return list(itertools.product(VARIANTS, range(len(SHORT_PARAMS)), range(len(GATES)), LONG_OPTIONS))


def monthly(grid, combo: tuple) -> np.ndarray:
    """一组组合在全部月份上的指标计数 [月份, 指标]。"""
    length, i, g, long = combo
    out = grid[f"agg_short{length}"][i, g].astype(np.float64)
    if long != "off":
        out = out + grid[f"agg_long{length}"][0, g if long == "gated" else PLAIN]
    return out


def table(grid, months_cols=COLS):
    table_ = combos()
    data = np.stack([monthly(grid, c)[months_cols] for c in table_])
    return table_, data


def rate(data: np.ndarray, num: int, sl=slice(None)) -> np.ndarray:
    return data[:, sl, num].sum(1) / np.maximum(data[:, sl, V].sum(1), 1)


def nconds(combo: tuple) -> int:
    return sum(x is not None for x in GATES[combo[2]]) + (combo[3] == "gated")


def describe(combo: tuple) -> str:
    length, i, g, long = combo
    p = SHORT_PARAMS[i]
    pre, d, r = GATES[g]
    parts = [f"{length} 秒、{p.a:g}%、{p.b:g} ATR、确认 {p.c}"]
    parts.append("前置不限" if pre is None else f"前置 {pre[0]} 秒 ≤ {pre[1]:g}%")
    parts.append("反向极值不限" if d is None else f"反向极值 ≤ {d:g}%")
    parts.append("暴增不限" if r is None else f"暴增 ≥ {r:g}")
    parts.append({"off": "长窗口关闭", "plain": "长窗口现行", "gated": "长窗口加同一组条件"}[long])
    return "、".join(parts)


def pbo(data: np.ndarray, cand: np.ndarray, blocks: int = 12) -> float:
    """CSCV：24 个月按两个月一块切 12 块，对半分成 924 种，样本内最好的一组在样本外排名落到后一半的比例。"""
    size = data.shape[1] // blocks
    v = np.stack([data[cand, k * size : (k + 1) * size, V].sum(1) for k in range(blocks)], 1)
    h = np.stack([data[cand, k * size : (k + 1) * size, H5].sum(1) for k in range(blocks)], 1)
    bad = total = 0
    for ins in itertools.combinations(range(blocks), blocks // 2):
        mask = np.zeros(blocks, bool)
        mask[list(ins)] = True
        r_in = h[:, mask].sum(1) / np.maximum(v[:, mask].sum(1), 1)
        r_out = h[:, ~mask].sum(1) / np.maximum(v[:, ~mask].sum(1), 1)
        best = int(np.argmax(r_in))
        rank = (r_out < r_out[best]).sum() + 0.5 * ((r_out == r_out[best]).sum() - 1)
        bad += (rank + 0.5) / len(r_out) <= 0.5
        total += 1
    return bad / total


CURRENT = (30, SHORT_PARAMS.index(next(p for p in SHORT_PARAMS if (p.a, p.b, p.c) == (1.3, 1.5, 2))), PLAIN, "plain")

if __name__ == "__main__":
    grid = np.load(DATA / "spike4_grid_sel.npz")
    tab, data = table(grid)
    per_day = data[:, :, N].sum(1) / DAYS
    h5 = rate(data, H5)
    s5 = rate(data, S5)
    years = [slice(0, 12), slice(12, 24)]
    yh5 = np.column_stack([rate(data, H5, s) for s in years])
    ys5 = np.column_stack([rate(data, S5, s) for s in years])
    cur = tab.index(CURRENT)
    print(
        f"现行：每天 {per_day[cur]:.1f}，hold5 {h5[cur]:.1%}（{yh5[cur, 0]:.1%} / {yh5[cur, 1]:.1%}），"
        f"side5 {s5[cur]:.1%}（{ys5[cur, 0]:.1%} / {ys5[cur, 1]:.1%}）"
    )
    better = (yh5 > yh5[cur]).all(1) & (ys5 > ys5[cur]).all(1)
    nc = np.array([nconds(c) for c in tab])
    selection = {}
    for name, floor in TARGETS.items():
        cand = np.flatnonzero((per_day >= floor) & better)
        best = h5[cand].max()
        tied = cand[h5[cand] >= best - TIE]
        pick = max(tied, key=lambda i: (per_day[i], -nc[i]))
        p = pbo(data, cand)
        print(
            f"\n方案 {name}（每天 ≥ {floor:g}）：候选 {len(cand)} 组，最高 hold5 {best:.1%}，同等 {len(tied)} 组，PBO {p:.2f}\n"
            f"  选中：{describe(tab[pick])}\n  每天 {per_day[pick]:.1f}，hold5 {h5[pick]:.1%}（{yh5[pick, 0]:.1%} / {yh5[pick, 1]:.1%}），"
            f"side5 {s5[pick]:.1%}（{ys5[pick, 0]:.1%} / {ys5[pick, 1]:.1%}），hold3 {rate(data, H3)[pick]:.1%}，"
            f"延迟 5 秒 {rate(data, H5D)[pick]:.1%}"
        )
        for i in sorted(cand, key=lambda i: -h5[i])[:10]:
            print(f"    {h5[i]:.1%} side5 {s5[i]:.1%} 每天 {per_day[i]:6.1f}  {describe(tab[i])}")
        selection[name] = {"combo": list(tab[pick]), "pbo": p, "per_day": per_day[pick], "hold5": h5[pick], "side5": s5[pick]}
    # 不作判据的前沿：各条数下限下最高 hold5
    print("\n前沿（每天至少 → 最高 hold5 的一组，只要求两个年度都优于现行）：")
    for floor in (300, 200, 100, 50, 30, 15, 10, 5, 2):
        cand = np.flatnonzero((per_day >= floor) & better)
        if len(cand):
            i = cand[np.argmax(h5[cand])]
            print(f"  ≥ {floor:>3}: hold5 {h5[i]:.1%} side5 {s5[i]:.1%} 每天 {per_day[i]:.1f}  {describe(tab[i])}")
    json.dump(selection, open(DATA / "spike4_selection.json", "w"), ensure_ascii=False, indent=1, default=str)
