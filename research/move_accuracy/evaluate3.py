"""第三轮样本外验证（PLAN.md 第三轮，2025-04 ~ 09，只运行一次）：方案 A、B 对比现行，按写定判据判断是否推荐。"""

from __future__ import annotations

import json

import numpy as np
from common import ALL_MONTHS, DATA, ROUND3_MONTHS, month_start
from grid3 import A, BURST, CD, CURRENT, OPP, POOLS, PRE, C, combos
from select3 import describe
from sim import month_days

DAYS = int(sum(month_days(m) for m in ROUND3_MONTHS))
# grid3 的提醒列
K, SYM, T, DIR, MONTH, VALID, H5, H3, H5D, R5 = range(10)


def pick(alerts: np.ndarray, k: int) -> np.ndarray:
    return alerts[alerts[:, K] == k]


def stats(a: np.ndarray) -> dict:
    v = a[:, VALID].sum()
    return {
        "n": len(a),
        "per_day": len(a) / DAYS,
        "h5": a[:, H5].sum() / max(v, 1),
        "h3": a[:, H3].sum() / max(v, 1),
        "h5d": a[:, H5D].sum() / max(v, 1),
        "pos5": np.mean(a[a[:, VALID] == 1, R5] > 0) if v else np.nan,
        "med5": np.median(a[a[:, VALID] == 1, R5]) if v else np.nan,
    }


def per_day(a: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray]:
    day = ((a[:, T] - month_start(ROUND3_MONTHS[0])) // 86400).astype(int)
    return np.bincount(day, weights=a[:, col], minlength=DAYS), np.bincount(day, weights=a[:, VALID], minlength=DAYS)


def bootstrap(x: np.ndarray, y: np.ndarray, col: int = H5, n: int = 5000):
    xs, xv = per_day(x, col)
    ys, yv = per_day(y, col)
    idx = np.random.default_rng(0).integers(0, DAYS, (n, DAYS))
    diff = xs[idx].sum(1) / np.maximum(xv[idx].sum(1), 1) - ys[idx].sum(1) / np.maximum(yv[idx].sum(1), 1)
    return xs.sum() / xv.sum() - ys.sum() / yv.sum(), *np.percentile(diff, [2.5, 97.5])


def matched(a: np.ndarray, b: np.ndarray, tolerance: float = 10.0) -> np.ndarray:
    """a 的每条提醒在 b 里有没有同一波急动的提醒。"""
    out = np.zeros(len(a), bool)
    for i, x in enumerate(a):
        same = (b[:, SYM] == x[SYM]) & (b[:, DIR] == x[DIR])
        out[i] = bool((np.abs(b[same, T] - x[T]) <= tolerance).any())
    return out


def line(s: dict) -> str:
    return (
        f"每天 {s['per_day']:.2f}（{s['n']} 条） hold5 {s['h5']:.1%} hold3 {s['h3']:.1%} 延迟5秒 {s['h5d']:.1%} "
        f"5 分钟后顺势 {s['pos5']:.0%} 中位 {s['med5']:+.2f}%"
    )


def random_base() -> str:
    rets = []
    for path in sorted((DATA / "rows_L10F10").glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] in ROUND3_MONTHS:
            with np.load(path) as d:
                if "rand_ret" in d:
                    r, pool = d["rand_ret"], d["rand_pool"]
                    rets.append(r[((pool >> 1) & 1 == 1) & ~np.isnan(r).any(axis=1)])
    pos = np.concatenate(rets) > 0
    return f"hold3 {pos[:, :3].all(1).mean():.1%} hold5 {pos[:, :5].all(1).mean():.1%}（{len(pos)} 个样本）"


def neighbors(combo: tuple) -> list[tuple[str, tuple]]:
    """任一项挪一格（反向极值距离只挪上限，回看长度另列）。"""
    out = []
    grids = (A, C, POOLS, PRE, BURST, None, None, CD)
    for i, values in enumerate(grids):
        if values is None:
            continue
        j = values.index(combo[i])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(values):
                c = list(combo)
                c[i] = values[jj]
                out.append((f"第 {i} 项 → {values[jj]}", tuple(c)))
    span, x = combo[5], combo[6]
    if span is None:
        return out
    xs = sorted({o[1] for o in OPP if o[0] == span})
    j = xs.index(x)
    for jj in (j - 1, j + 1):
        if 0 <= jj < len(xs):
            out.append((f"距离上限 → {xs[jj]:g}", (*combo[:6], xs[jj], combo[7])))
    for other in (720, 2880, 1440):
        if other != span:
            out.append((f"回看 → {other} 分钟", (*combo[:5], other, x, combo[7])))
    out.append(("去掉反向极值", (*combo[:5], None, None, combo[7])))
    return out


if __name__ == "__main__":
    table = combos()
    sel = json.load(open(DATA / "selection3.json", encoding="utf-8"))
    r3 = np.load(DATA / "grid3_r3.npz")["alerts"]
    dev = np.load(DATA / "grid3_dev.npz")["alerts"]
    cur_k = table.index(CURRENT)
    cur = pick(r3, cur_k)
    print("样本外 2025-04 ~ 09（第三轮写定后只运行这一次）")
    print("随机时刻随机方向（800 万池）：", random_base())
    print("现行：", line(stats(cur)))
    for name in ("A", "B"):
        combo = tuple(None if v is None else v for v in sel[name]["combo"])
        k = table.index(combo)
        new = pick(r3, k)
        s, sc = stats(new), stats(cur)
        d5 = bootstrap(new, cur)
        print(f"\n方案 {name}：{describe(combo)}")
        print(f"  选参期 每天 {sel[name]['dev_per_day']:.2f} hold5 {sel[name]['dev_hold5']:.1%}；样本外", line(s))
        print(f"  hold5 − 现行 {d5[0]:+.1%} [{d5[1]:+.1%}, {d5[2]:+.1%}]")
        checks = {
            "① hold5 更高且区间下限 > 0": d5[0] > 0 and d5[1] > 0,
            "② 延迟 5 秒 hold5、hold3 不低于现行": s["h5d"] >= sc["h5d"] and s["h3"] >= sc["h3"],
            "③ PBO < 0.5": sel[name]["pbo"] < 0.5,
        }
        for label, ok in checks.items():
            print(f"  {label}：{'✓' if ok else '✗'}" + (f"（{sel[name]['pbo']:.2f}）" if label.startswith("③") else ""))
        print(f"  结论：{'推荐' if all(checks.values()) else '不推荐'}")
        monthly = []
        for m in ROUND3_MONTHS:
            mm = new[new[:, MONTH] == ALL_MONTHS.index(m)]
            monthly.append(f"{m[2:4]}-{m[4:]} {stats(mm)['h5']:.0%}（{len(mm)}）")
        print("  逐月", "，".join(monthly))
        for dname, dv in (("急涨", 1), ("急跌", -1)):
            print(f"  {dname}", line(stats(new[new[:, DIR] == dv])))
        # 确认 2 秒的提醒比确认 1 秒晚一秒，按同一波急动匹配：同合约、同方向、相差 ≤ 10 秒
        added = ~matched(new, cur)
        removed = ~matched(cur, new)
        print(f"  新增提醒 {added.sum()} 条 hold5 {stats(new[added])['h5']:.1%}；去掉的现行提醒 {removed.sum()} 条 "
              f"hold5 {stats(cur[removed])['h5']:.1%}")
        print("  相邻参数（选参期 / 样本外 hold5，括号内每天）：")
        for label, c in neighbors(combo):
            if c not in table:
                continue
            kk = table.index(c)
            a, b = stats(pick(dev, kk)), stats(pick(r3, kk))
            print(f"    {label:<16} {a['h5']:.1%}（{a['per_day'] * DAYS / 365:.2f}）/ {b['h5']:.1%}（{b['per_day']:.2f}）")
