"""第三轮补充检验（PLAN.md 第三轮补充检验，只运行一次）：方案 B 对比现行，以 2024-10 ~ 2025-03 为准，2024-10 ~ 2025-09 一整年合计作参考。"""

from __future__ import annotations

import json

import numpy as np
from common import ALL_MONTHS, DATA, ROUND3_MONTHS, ROUND4_MONTHS, month_start
from evaluate3 import DIR, H3, H5, H5D, MONTH, R5, SYM, T, VALID, neighbors
from grid3 import CURRENT, combos
from select3 import describe
from sim import month_days


class Period:
    """一段时间的提醒：多个 grid3 输出拼在一起，合约编号换成合约-月份名，按天编号供自助法用。"""

    def __init__(self, label: str, names: list[str], months: list[str]) -> None:
        self.label = label
        self.months = months
        self.days = int(sum(month_days(m) for m in months))
        self.start = month_start(months[0])
        parts, keys = [], []
        for name in names:
            g = np.load(DATA / f"grid3_{name}.npz")
            parts.append(g["alerts"])
            keys.append(g["symbols"][g["alerts"][:, SYM].astype(int)])
        self.alerts = np.concatenate(parts)
        self.keys = np.concatenate(keys)

    def pick(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        sel = self.alerts[:, 0] == k
        return self.alerts[sel], self.keys[sel]

    def stats(self, a: np.ndarray) -> dict:
        v = a[:, VALID].sum()
        ok = a[:, VALID] == 1
        return {
            "n": len(a),
            "per_day": len(a) / self.days,
            "h5": a[:, H5].sum() / max(v, 1),
            "h3": a[:, H3].sum() / max(v, 1),
            "h5d": a[:, H5D].sum() / max(v, 1),
            "pos5": np.mean(a[ok, R5] > 0) if v else np.nan,
            "med5": np.median(a[ok, R5]) if v else np.nan,
        }

    def per_day(self, a: np.ndarray, col: int):
        day = ((a[:, T] - self.start) // 86400).astype(int)
        return (
            np.bincount(day, weights=a[:, col], minlength=self.days),
            np.bincount(day, weights=a[:, VALID], minlength=self.days),
        )

    def bootstrap(self, x: np.ndarray, y: np.ndarray, col: int = H5, n: int = 5000):
        xs, xv = self.per_day(x, col)
        ys, yv = self.per_day(y, col)
        idx = np.random.default_rng(0).integers(0, self.days, (n, self.days))
        diff = xs[idx].sum(1) / np.maximum(xv[idx].sum(1), 1) - ys[idx].sum(1) / np.maximum(yv[idx].sum(1), 1)
        return xs.sum() / xv.sum() - ys.sum() / yv.sum(), *np.percentile(diff, [2.5, 97.5])


def line(s: dict) -> str:
    return (
        f"每天 {s['per_day']:.2f}（{s['n']} 条） hold5 {s['h5']:.1%} hold3 {s['h3']:.1%} 延迟5秒 {s['h5d']:.1%} "
        f"5 分钟后顺势 {s['pos5']:.0%} 中位 {s['med5']:+.2f}%"
    )


def matched(a, ka, b, kb, tolerance: float = 10.0) -> np.ndarray:
    out = np.zeros(len(a), bool)
    for i, x in enumerate(a):
        same = (kb == ka[i]) & (b[:, DIR] == x[DIR])
        out[i] = bool((np.abs(b[same, T] - x[T]) <= tolerance).any())
    return out


def random_base(months: list[str]) -> str:
    rets = []
    for path in sorted((DATA / "rows_L10F10").glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] in months:
            with np.load(path) as d:
                if "rand_ret" in d:
                    r, pool = d["rand_ret"], d["rand_pool"]
                    rets.append(r[((pool >> 1) & 1 == 1) & ~np.isnan(r).any(axis=1)])
    pos = np.concatenate(rets) > 0
    return f"hold3 {pos[:, :3].all(1).mean():.1%} hold5 {pos[:, :5].all(1).mean():.1%}（{len(pos)} 个样本）"


def report(period: Period, b_combo: tuple, verdict: bool) -> None:
    table = combos()
    cur, kc = period.pick(table.index(CURRENT))
    new, kn = period.pick(table.index(b_combo))
    s, sc = period.stats(new), period.stats(cur)
    print(f"\n== {period.label} ==")
    print("随机时刻随机方向（800 万池）：", random_base(period.months))
    print("现行：  ", line(sc))
    print("方案 B：", line(s))
    for name, col in (("hold5", H5), ("hold3", H3), ("延迟5秒 hold5", H5D)):
        d = period.bootstrap(new, cur, col)
        print(f"  {name} B − 现行 {d[0]:+.1%} [{d[1]:+.1%}, {d[2]:+.1%}]")
    if verdict:
        d5 = period.bootstrap(new, cur)
        checks = {
            "① hold5 更高且区间下限 > 0": d5[0] > 0 and d5[1] > 0,
            "② 延迟 5 秒 hold5、hold3 不低于现行": s["h5d"] >= sc["h5d"] and s["h3"] >= sc["h3"],
        }
        for label, ok in checks.items():
            print(f"  {label}：{'✓' if ok else '✗'}")
        print(f"  结论：{'通过，维持方案 B' if all(checks.values()) else '未通过'}")
    monthly = []
    for m in period.months:
        mm = new[new[:, MONTH] == ALL_MONTHS.index(m)]
        mc = cur[cur[:, MONTH] == ALL_MONTHS.index(m)]
        monthly.append(f"{m[2:4]}-{m[4:]} {period.stats(mm)['h5']:.0%}（{len(mm)}）/ {period.stats(mc)['h5']:.0%}（{len(mc)}）")
    print("  逐月 B / 现行：", "，".join(monthly))
    for dname, dv in (("急涨", 1), ("急跌", -1)):
        print(f"  {dname} B：", line(period.stats(new[new[:, DIR] == dv])), "| 现行：", line(period.stats(cur[cur[:, DIR] == dv])))
    added = ~matched(new, kn, cur, kc)
    removed = ~matched(cur, kc, new, kn)
    print(f"  新增提醒 {added.sum()} 条 hold5 {period.stats(new[added])['h5']:.1%}；去掉的现行提醒 {removed.sum()} 条 "
          f"hold5 {period.stats(cur[removed])['h5']:.1%}")
    print("  分解与相邻参数（hold5，括号内每天）：")
    for label, c in neighbors(b_combo):
        if c in table:
            st = period.stats(period.pick(table.index(c))[0])
            print(f"    {label:<16} {st['h5']:.1%}（{st['per_day']:.2f}）")


if __name__ == "__main__":
    sel = json.load(open(DATA / "selection3.json", encoding="utf-8"))
    b_combo = tuple(sel["B"]["combo"])
    print(f"方案 B：{describe(b_combo)}（第三轮补充检验，写定后只运行这一次）")
    report(Period("2024-10 ~ 2025-03（以此为准）", ["r4"], ROUND4_MONTHS), b_combo, verdict=True)
    report(Period("2024-10 ~ 2025-09 一整年（含已看过的 2025-04 ~ 09，只作参考）", ["r4", "r3"], ROUND4_MONTHS + ROUND3_MONTHS), b_combo, verdict=False)
