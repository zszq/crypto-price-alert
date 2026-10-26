"""按 PLAN.md 写定的规则在开发期选参（第一、二步取自 grid_*.npz，第三步与第二个窗口在这里算）。

用法：select_params.py [每天条数下限，默认 20]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from common import DATA, DEV_MONTHS, MONTHS
from grid import CD, GATES, POOLS, A, B, C, T, stats_from_alerts
from sim import Params, alert_rows

SINGLE = ["L10", "L20", "L30", "L60", "L120", "L180", "L300"]
DEV = [MONTHS.index(m) for m in DEV_MONTHS]
K = len(MONTHS)
DEV_DAYS = 122.0
TIE = 0.01
CURRENT = (("L30", Params(1.3, 1.5, 2, 15, 30, 1)), ("L180F30", Params(1.6, 1.8, 3, 30, 180, 1)))


@dataclass
class Agg:
    """可相加的开发期汇总：逐月条数、有标签条数与各标签之和。"""

    cnt: np.ndarray
    val: np.ndarray
    h3: np.ndarray
    h5: np.ndarray
    h10: np.ndarray
    r5: np.ndarray
    fee: np.ndarray
    h5d: np.ndarray

    @classmethod
    def from_stats(cls, s: np.ndarray) -> Agg:
        return cls(*(s[..., i * K : (i + 1) * K] for i in range(8)))

    def __add__(self, other: Agg) -> Agg:
        return Agg(*(getattr(self, f) + getattr(other, f) for f in self.__dataclass_fields__))

    def rate(self, name: str, months=DEV) -> np.ndarray:
        num = getattr(self, name)[..., months].sum(axis=-1)
        den = self.val[..., months].sum(axis=-1)
        return num / np.maximum(den, 1)

    def monthly(self, name: str) -> np.ndarray:
        return getattr(self, name)[..., DEV] / np.maximum(self.val[..., DEV], 1)

    def per_day(self) -> np.ndarray:
        return self.cnt[..., DEV].sum(axis=-1) / DEV_DAYS


def describe(variant: str, combo) -> str:
    ai, bi, ci, ti, di, pi, gi = (int(x) for x in combo)
    gate = GATES[gi]
    g = "无" if gate is None else f"{gate[0]} {gate[1]:g}"
    return (
        f"{variant} {A[ai]:g}% {B[bi]:g}ATR 确认{C[ci]} 笔数{T[ti]} 冷却{CD[di]} "
        f"池{[500, 800, 2000][POOLS[pi]]}万 新条件[{g}]"
    )


def params_of(combo) -> tuple[Params, tuple]:
    ai, bi, ci, ti, di, pi, gi = (int(x) for x in combo)
    gate = GATES[gi]
    return Params(A[ai], B[bi], C[ci], T[ti], CD[di], POOLS[pi]), ((gate,) if gate else ())


def summary_line(agg: Agg) -> str:
    return (
        f"每天 {agg.per_day():.1f} 条 hold5 {agg.rate('h5'):.1%} hold3 {agg.rate('h3'):.1%} "
        f"hold10 {agg.rate('h10'):.1%} 5分钟顺势均值 {agg.rate('r5'):+.3f}% >0.1% {agg.rate('fee'):.1%} "
        f"延迟5秒 hold5 {agg.rate('h5d'):.1%} 逐月 hold5 {' '.join(f'{x:.1%}' for x in agg.monthly('h5'))}"
    )


def simulate(variant: str, p: Params, gates: tuple = ()) -> Agg:
    return Agg.from_stats(stats_from_alerts(alert_rows(variant, DEV_MONTHS, p, gates)))


def current_agg() -> Agg:
    parts = [simulate(v, p) for v, p in CURRENT]
    return parts[0] + parts[1]


def load_grids() -> dict[str, tuple[np.ndarray, Agg]]:
    out = {}
    for v in SINGLE + ["L180F30"]:
        d = np.load(DATA / f"grid_{v}.npz")
        out[v] = (d["combos"], Agg.from_stats(d["stats"]))
    return out


def feasible(agg: Agg, base: Agg, floor: float) -> np.ndarray:
    ok = (agg.per_day() >= floor) & (agg.cnt[..., DEV].sum(axis=-1) >= 2000)
    return ok & (agg.monthly("h5") > base.monthly("h5")).all(axis=-1)


def pick(cands: list[tuple]) -> tuple:
    """cands: (hold5, 每天, 新条件数, 其他...)；最高 hold5 的 1 个百分点内取每天最多，再取新条件少的。"""
    best = max(c[0] for c in cands)
    tied = [c for c in cands if c[0] >= best - TIE]
    return max(tied, key=lambda c: (c[1], -c[2]))


if __name__ == "__main__":
    floor = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    base = current_agg()
    print("现行参数（开发期）：", summary_line(base))
    grids = load_grids()

    cands = []
    for v in SINGLE:
        combos, agg = grids[v]
        ok = np.flatnonzero(feasible(agg, base, floor))
        h5, pd_ = agg.rate("h5"), agg.per_day()
        for i in ok:
            cands.append((h5[i], pd_[i], int(GATES[combos[i][6]] is not None), v, i))
        if len(ok):
            j = ok[np.argmax(h5[ok])]
            print(f"  {v} 可行 {len(ok)} 组，最高 hold5 {h5[j]:.1%}（每天 {pd_[j]:.1f}）：{describe(v, combos[j])}")
    if not cands:
        sys.exit("没有满足约束的组合")
    h5, pd_, ngate, v, i = pick(cands)
    combos, agg = grids[v]
    print(f"\n第二步选中：{describe(v, combos[i])}")
    print("  ", summary_line(Agg.from_stats(np.load(DATA / f"grid_{v}.npz")["stats"][i])))
    tied = [c for c in cands if c[0] >= max(x[0] for x in cands) - TIE]
    print(f"  同等（1 个百分点内）{len(tied)} 组，窗口分布：", {w: sum(1 for c in tied if c[3] == w) for w in SINGLE})

    # 第三步：在选中组合上再加一个新条件
    p, gates = params_of(combos[i])
    sel_agg = Agg.from_stats(np.load(DATA / f"grid_{v}.npz")["stats"][i])
    third = []
    for gate in GATES[1:]:
        if gates and gate[0] == gates[0][0]:
            continue
        g2 = gates + (gate,)
        a2 = simulate(v, p, g2)
        if feasible(a2, base, floor) and a2.rate("h5") > sel_agg.rate("h5") + TIE:
            third.append((a2.rate("h5"), a2.per_day(), len(g2), g2, a2))
    if third:
        h5_3, pd3, _, g3, a3 = pick(third)
        print(f"\n第三步：加 {g3[-1]} 后 hold5 {h5_3:.1%}（高出 1 个百分点以上），采用")
        gates, sel_agg = g3, a3
    else:
        print("\n第三步：没有第二个新条件能让 hold5 再高 1 个百分点以上，不加")
    print("单窗口最终：", v, p, gates, "\n  ", summary_line(sel_agg))

    # 第二个窗口：与另一个窗口长度的组合合并（汇总可直接相加：两个窗口各自提醒、各自冷却）
    single_h5 = sel_agg.rate("h5")
    unions = []
    for w in SINGLE:
        if w == v:
            continue
        combos2, agg2 = grids[w]
        merged = Agg(*(getattr(sel_agg, f)[None, :] + getattr(agg2, f) for f in Agg.__dataclass_fields__))
        ok = np.flatnonzero(
            feasible(merged, base, floor) & (merged.rate("h5") >= single_h5 - TIE) & (combos2[:, 5] == combos[i][5])
        )
        for k in ok:
            unions.append((merged.per_day()[k], merged.rate("h5")[k], w, k))
    selection = {"floor": floor, "windows": [{"variant": v, "params": vars(p), "gates": [list(g) for g in gates]}]}
    if unions:
        pd_u, h5_u, w, k = max(unions)
        print(f"\n第二个窗口：{describe(w, grids[w][0][k])}，合并后每天 {pd_u:.1f} 条、hold5 {h5_u:.1%}")
        p2, g2 = params_of(grids[w][0][k])
        selection["windows"].append({"variant": w, "params": vars(p2), "gates": [list(g) for g in g2]})
    else:
        print("\n第二个窗口：合并后都会让 hold5 低于单窗口 1 个百分点以上，只用一个窗口")
    json.dump(selection, open(DATA / f"selection_{floor:g}.json", "w"), ensure_ascii=False, indent=1)
