"""反弹回落「增加数量」对比（报告第十七节）：在方案 B 附近放宽各项，看每天条数与 hold5 怎么换。

口径与 grid3.py 相同（候选 cand3、反向极值 opp3），只换了候选族：
反向极值回看固定 1440 分钟（检测器里 EXTREME_MINUTES 不开放配置，2880 只能改代码），
并加入 grid3 没有的更松取值（涨跌幅 1.25%、前置 −5%、暴增 10、距离 15% / 20%）。
三段都算：dev（2025-10 ~ 2026-09，选参期）、r3（2025-04 ~ 09）、r4（2024-10 ~ 2025-03）；
r3、r4 已在第三轮评估过方案 B，不再是没看过的样本外，这里只作描述，不是写定判据下的验证。
用法：loosen_reversal.py grid dev|r3|r4 → 写 loosen_<名称>.npz；loosen_reversal.py report
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from cand3 import PERIODS
from common import DATA
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, chunked_files, load, month_days, scan

A = (1.0, 1.25, 1.5)
C = (1, 2)
POOLS = (0, 1)  # 0 = 500 万，1 = 800 万（下载的成交只覆盖进过 500 万池的合约，更低的池算不了）
PRE = (-4.0, -5.0, -6.0)
BURST = (8.0, 10.0, 12.0, 16.0)
EXT = (None, 5.0, 10.0, 15.0, 20.0)
CD = 30.0
B = (1.5, 2, 1, -6.0, 16.0, 10.0)
DEFAULT = (1.5, 1, 1, -6.0, 16.0, None)
# 2026-10-31 写进 config/default/reversal.yaml 的一组，及不降合约池时的同一组
CURRENT = (1.25, 1, 0, -6.0, 12.0, 10.0)
CURRENT_800 = (1.25, 1, 1, -6.0, 12.0, 10.0)
K, SYM, T, DIR, MONTH, VALID, H5, H3, H5D, R5 = range(10)


def combos() -> list[tuple]:
    return list(itertools.product(A, C, POOLS, PRE, BURST, EXT))


def grid(name: str) -> None:
    months = PERIODS[name]
    cand = np.load(DATA / f"cand3_{name}.npz", allow_pickle=True)
    opp = np.load(DATA / f"cand3_{name}_opp.npz")
    lookup = {}
    for key in np.unique(cand["symbol"]):
        i = np.flatnonzero(cand["symbol"] == key)
        lookup[key] = (cand["sec"][i], opp["opp_1440"][i])
    table = combos()
    out, symbols = [], []
    for files in chunked_files("L10F10", months):
        rows = load("L10F10", months, COLUMNS + EXTRA_COLUMNS, files)
        n = len(rows)
        incand = np.zeros(n, bool)
        ext = np.full(n, np.inf)
        for g, key in enumerate(rows.symbols):
            if key not in lookup:
                continue
            idx = np.flatnonzero(rows.group == g)
            secs, vals = lookup[key]
            j = idx[np.searchsorted(rows["sec"][idx], secs)]
            assert (rows["sec"][j] == secs).all()
            incand[j] = True
            # 覆盖不满 24 小时时 opp 为 NaN，检测器不报，按无穷远处理
            ext[j] = np.where(np.isnan(vals), np.inf, vals)
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
            pool, pre, burst, e = rows["pool"][idx], rows["pre"][idx], rows["burst"][idx], ext[idx]
            args = (rows.group[idx], rows["sec"][idx], rows["t"][idx], rows["real"][idx], rows["lchg"][idx],
                    rows["latrm"][idx])
            for k, combo in enumerate(table):
                if combo[0] != a or combo[1] != c:
                    continue
                _, _, pl, pr, bu, x = combo
                ok = ((pool >> pl) & 1 == 1) & (pre <= pr) & (burst >= bu)
                if x is not None:
                    ok &= e <= x
                h = idx[scan(*args, ok, a, 1.0, c, CD)]
                if len(h):
                    out.append(np.column_stack([np.full(len(h), k), rows.group[h] + offset, rows["t"][h],
                                                rows["dir"][h], rows.month[h], labels[h], ret[h, 4]]))
        print(name, "块", len(symbols), flush=True)
    np.savez(DATA / f"loosen_{name}.npz", alerts=np.concatenate(out), symbols=np.array(symbols))


def stats(a: np.ndarray, days: float) -> tuple[float, float, float, int]:
    v = a[:, VALID] == 1
    if not v.any():
        return 0.0, np.nan, np.nan, 0
    return len(a) / days, a[v, H5].mean(), np.mean(a[v, R5] > 0), len(a)


def matched(a: np.ndarray, b: np.ndarray, tol: float = 10.0) -> np.ndarray:
    """a 的每条在 b 里有没有同一波急动（同合约、同方向、相差 ≤ 10 秒）。"""
    out = np.zeros(len(a), bool)
    for i, x in enumerate(a):
        same = (b[:, SYM] == x[SYM]) & (b[:, DIR] == x[DIR])
        out[i] = bool((np.abs(b[same, T] - x[T]) <= tol).any())
    return out


def report() -> None:
    table = combos()
    data, days = {}, {}
    for name in ("dev", "r3", "r4"):
        d = np.load(DATA / f"loosen_{name}.npz")
        a = d["alerts"]
        data[name] = {k: a[a[:, K] == k] for k in range(len(table))}
        days[name] = float(sum(month_days(m) for m in PERIODS[name]))

    def row(combo):
        k = table.index(combo)
        return {n: stats(data[n][k], days[n]) for n in data}

    def label(combo):
        a, c, pl, pr, bu, x = combo
        return (f"{a:g}% 确认{c} {'500万' if pl == 0 else '800万'} 前置{pr:g} 暴增{bu:g} "
                f"极值{'不限' if x is None else f'≤{x:g}%'}")

    def fmt(r):
        return "  ".join(f"{n} {r[n][0]:.2f}/天 {r[n][1]:.1%} 顺势{r[n][2]:.0%}" for n in r)

    def added(c):
        # 新增的那部分（方案 B 没报的同一波急动）本身准不准：放宽的代价看的是这部分，不是整体的平均
        parts = []
        for n in ("dev", "r3", "r4"):
            new, base = data[n][table.index(c)], data[n][table.index(B)]
            s = stats(new[~matched(new, base)], days[n])
            parts.append(f"{n} 新增 {s[3]} 条 {s[1]:.1%}")
        return "    " + "  ".join(parts)

    print("== 参照 ==")
    for name, c in (("现行（放宽四项）", CURRENT), ("前三项、800 万池", CURRENT_800), ("方案 B", B),
                    ("代码默认值", DEFAULT)):
        print(f"{name:<14}", fmt(row(c)))
        if c != B:
            print(added(c))
    print("\n== 在方案 B 上只放宽一项 ==")
    single = [("涨跌幅 1.25%", (1.25, *B[1:])), ("涨跌幅 1%", (1.0, *B[1:])),
              ("确认 1 秒", (B[0], 1, *B[2:])), ("合约池 500 万", (*B[:2], 0, *B[3:])),
              ("前置 −5%", (*B[:3], -5.0, *B[4:])), ("前置 −4%", (*B[:3], -4.0, *B[4:])),
              ("暴增 12", (*B[:4], 12.0, B[5])), ("暴增 10", (*B[:4], 10.0, B[5])), ("暴增 8", (*B[:4], 8.0, B[5])),
              ("极值 ≤15%", (*B[:5], 15.0)), ("极值 ≤20%", (*B[:5], 20.0)), ("极值不限", (*B[:5], None))]
    for name, c in single:
        print(f"{name:<14}", fmt(row(c)))

    # 按选参期（dev）选：每天条数达到目标里 dev hold5 最高的一组，再看 r3、r4
    print("\n== 按 dev 选：每天 ≥ 目标条数里 dev hold5 最高的 3 组 ==")
    dev_rows = {c: row(c) for c in table}
    for target in (2.0, 2.5, 3.0, 4.0):
        ok = [c for c in table if dev_rows[c]["dev"][0] >= target]
        best = sorted(ok, key=lambda c: -dev_rows[c]["dev"][1])[:3]
        print(f"-- ≥ {target:g} 条/天")
        for c in best:
            print(f"  {label(c)}\n    {fmt(dev_rows[c])}")
            print(added(c))


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        grid(sys.argv[2])
    else:
        report()
