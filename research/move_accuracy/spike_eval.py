"""第四轮样本外验证（PLAN.md 第四轮，2024-04 ~ 09，只运行一次）：读 spike4_selection.json，对方案 A、B 与现行逐条判据。

依赖 spike_grid.py oos 的汇总（相邻参数、分解）与 alerts_for 的明细（自助法）。
"""

from __future__ import annotations

import json

import numpy as np
from common import ALL_MONTHS, DATA, month_start
from engine4 import ROWS4
from sim import month_days
from spike_grid import GATES, PERIODS, SHORT_PARAMS, VARIANTS, A, B, C, LONG, OPP, PRE, BURST, alerts_for
from spike_select import CURRENT, H3, H5, H5D, N, PLAIN, S5, V, combos, describe, rate, table

OOS = PERIODS["oos"]
SEL = PERIODS["sel"]
DAYS = sum(month_days(m) for m in OOS)
FIRST = month_start(OOS[0])


def detail(combo: tuple) -> dict[str, np.ndarray]:
    length, i, g, long = combo
    short, long_variant = VARIANTS[length]
    parts = [alerts_for(short, OOS, SHORT_PARAMS[i], GATES[g])]
    if long != "off":
        parts.append(alerts_for(long_variant, OOS, LONG, GATES[g] if long == "gated" else GATES[PLAIN]))
    out = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    out["day"] = ((out["t"] - FIRST) // 86400).astype(np.int64)
    return out


def boot(x: dict, y: dict, col: int, n: int = 5000) -> tuple[float, float, float]:
    """按天块重抽样，x 与 y 的指标率之差与 95% 区间。"""
    nd = int(np.ceil(DAYS))

    def per_day(tab):
        return (np.bincount(tab["day"], weights=tab["lab"][:, col], minlength=nd),
                np.bincount(tab["day"], weights=tab["lab"][:, V], minlength=nd))

    xs, xv = per_day(x)
    ys, yv = per_day(y)
    pick = np.random.default_rng(0).integers(0, nd, (n, nd))
    diff = xs[pick].sum(1) / xv[pick].sum(1) - ys[pick].sum(1) / yv[pick].sum(1)
    return xs.sum() / xv.sum() - ys.sum() / yv.sum(), *np.percentile(diff, [2.5, 97.5])


def summary(tab: dict) -> str:
    lab = tab["lab"]
    v = max(lab[:, V].sum(), 1)
    ret = tab["ret"][lab[:, V] == 1]
    up = tab["dir"] > 0
    months = " ".join(
        f"{ALL_MONTHS[m][4:]}月 {lab[tab['month'] == m, H5].sum() / max(lab[tab['month'] == m, V].sum(), 1):.0%}"
        for m in sorted(set(tab["month"]))
    )
    return (
        f"每天 {len(lab) / DAYS:.1f} hold5 {lab[:, H5].sum() / v:.1%} side5 {lab[:, S5].sum() / v:.1%} "
        f"hold3 {lab[:, H3].sum() / v:.1%} 延迟5秒 {lab[:, H5D].sum() / v:.1%} "
        f"5分钟收益中位 {np.median(ret[:, 4]) if len(ret) else np.nan:+.2f}% "
        f"急涨 {lab[up, H5].sum() / max(lab[up, V].sum(), 1):.1%} 急跌 {lab[~up, H5].sum() / max(lab[~up, V].sum(), 1):.1%}"
        f"（{months}）"
    )


def random_baseline() -> str:
    rets = []
    for path in sorted(ROWS4.glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] in OOS and ".tmp" not in path.name:
            with np.load(path) as d:
                if "rand_ret" in d:
                    rets.append(d["rand_ret"][(d["rand_pool"] >> 1) & 1 == 1])
    r = np.concatenate(rets)
    r = r[~np.isnan(r).any(1)]
    pos = r > 0
    return f"随机：hold5 {pos[:, :5].all(1).mean():.1%} side5 {pos[:, 4].mean():.1%} hold3 {pos[:, :3].all(1).mean():.1%}（{len(r)} 个）"


def neighbors(combo: tuple, tab, data_oos, data_sel) -> None:
    """任一项挪一格（短窗口参数与三个新条件、长窗口选项），选参期 / 样本外 hold5、side5 与每天条数。"""
    length, i, g, long = combo
    p = SHORT_PARAMS[i]
    pre, d, r = GATES[g]
    found = []

    def add(label, c):
        if c in tab:
            found.append((label, tab.index(c)))

    for vals, idx, name in ((A, 0, "涨跌幅"), (B, 1, "ATR"), (C, 2, "确认")):
        cur = (p.a, p.b, p.c)
        k = vals.index(cur[idx])
        for j in (k - 1, k + 1):
            if 0 <= j < len(vals):
                new = list(cur)
                new[idx] = vals[j]
                ii = next(n for n, q in enumerate(SHORT_PARAMS) if (q.a, q.b, q.c) == tuple(new))
                add(f"{name} {vals[j]:g}", (length, ii, g, long))
    add(f"窗口 {40 - length} 秒", (40 - length, i, g, long))
    for vals, idx, name in ((PRE, 0, "前置"), (OPP, 1, "反向极值"), (BURST, 2, "暴增")):
        k = vals.index((pre, d, r)[idx])
        for j in sorted({k - 1, k + 1, 0}):
            if 0 <= j < len(vals) and j != k:
                new = [pre, d, r]
                new[idx] = vals[j]
                add(f"{name} {'不限' if vals[j] is None else vals[j]}", (length, i, GATES.index(tuple(new)), long))
    for lo in ("off", "plain", "gated"):
        if lo != long:
            add(f"长窗口 {lo}", (length, i, g, lo))
    days_sel = sum(month_days(m) for m in SEL)
    for label, k in found:
        print(
            f"    {label:14s} 选参期 hold5 {rate(data_sel, H5)[k]:.1%} side5 {rate(data_sel, S5)[k]:.1%}（{data_sel[k, :, N].sum() / days_sel:.1f}）"
            f" / 样本外 hold5 {rate(data_oos, H5)[k]:.1%} side5 {rate(data_oos, S5)[k]:.1%}（{data_oos[k, :, N].sum() / DAYS:.1f}）"
        )


if __name__ == "__main__":
    selection = json.load(open(DATA / "spike4_selection.json", encoding="utf-8"))
    tab, data_oos = table(np.load(DATA / "spike4_grid_oos.npz"), [ALL_MONTHS.index(m) for m in OOS])
    _, data_sel = table(np.load(DATA / "spike4_grid_sel.npz"))
    print(random_baseline())
    current = detail(CURRENT)
    print("现行：" + summary(current))
    for name, info in selection.items():
        combo = tuple(info["combo"])
        combo = (int(combo[0]), int(combo[1]), int(combo[2]), combo[3])
        tab_ = detail(combo)
        print(f"\n方案 {name}：{describe(combo)}\n  样本外 {summary(tab_)}")
        print(f"  选参期 每天 {info['per_day']:.1f} hold5 {info['hold5']:.1%} side5 {info['side5']:.1%}")
        h5 = boot(tab_, current, H5)
        s5 = boot(tab_, current, S5)
        h3 = boot(tab_, current, H3)
        hd = boot(tab_, current, H5D)
        ok = [h5[1] > 0 and h5[0] > 0, s5[1] > 0 and s5[0] > 0, h3[0] >= 0 and hd[0] >= 0, info["pbo"] < 0.5]
        print(
            f"  ① hold5 差 {h5[0]:+.1%} [{h5[1]:+.1%}, {h5[2]:+.1%}] {'✓' if ok[0] else '✗'}\n"
            f"  ② side5 差 {s5[0]:+.1%} [{s5[1]:+.1%}, {s5[2]:+.1%}] {'✓' if ok[1] else '✗'}\n"
            f"  ③ hold3 差 {h3[0]:+.1%}、延迟 5 秒 hold5 差 {hd[0]:+.1%} {'✓' if ok[2] else '✗'}\n"
            f"  ④ PBO {info['pbo']:.2f} {'✓' if ok[3] else '✗'}\n  结论：{'全部通过，推荐' if all(ok) else '未通过，不推荐'}"
        )
        print("  相邻参数：")
        neighbors(combo, tab, data_oos, data_sel)
