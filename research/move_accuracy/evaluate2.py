"""第二轮样本外验证（PLAN.md 第二轮，2025-10 ~ 2026-03，只运行一次）：第二轮选中参数对比第一轮推荐与随机基准。"""

from __future__ import annotations

import json

import numpy as np
from common import ALL_MONTHS, DATA, FRESH_MONTHS, month_start
from sim import COLUMNS, EXTRA_COLUMNS, Params, alert_rows, month_days

ROUND1 = [
    ("L10F10", Params(2.0, 1.0, 2, 60, 30, 2), (("pre_neg", 0.0),)),
    ("L20F10", Params(2.0, 1.0, 3, 20, 30, 2), (("pre_neg", 0.0),)),
]


def table(windows, months) -> dict[str, np.ndarray]:
    parts = []
    for variant, p, gates in windows:
        cols = COLUMNS + (EXTRA_COLUMNS if any(g[0] in ("burst", "pre300_neg", "ext6h") for g in gates) else ())
        parts.append(alert_rows(variant, months, p, gates, cols))
    ar = {k: np.concatenate([x[k] for x in parts]) for k in ("t", "dir", "ret", "month", "hold5d")}
    ret = ar["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    first = month_start(months[0])
    return {
        "t": ar["t"],
        "dir": ar["dir"],
        "month": ar["month"],
        "valid": valid,
        "ret": ret,
        "day": ((ar["t"] - first) // 86400).astype(np.int64),
        "h3": pos[:, :3].all(1) & valid,
        "h5": pos[:, :5].all(1) & valid,
        "h10": pos.all(1) & valid,
        "h5d": ar["hold5d"] & valid,
    }


def line(tab, months) -> str:
    v = tab["valid"]
    r = tab["ret"][v]
    days = sum(month_days(m) for m in months)
    monthly = []
    for m in months:
        s = tab["month"] == ALL_MONTHS.index(m)
        monthly.append(f"{m[2:4]}-{m[4:]} {tab['h5'][s].sum() / max(tab['valid'][s].sum(), 1):.0%}（{s.sum()}）")
    return (
        f"每天 {len(tab['t']) / days:.2f} 条（共 {len(tab['t'])}） hold5 {tab['h5'].sum() / max(v.sum(), 1):.1%} "
        f"hold3 {tab['h3'].sum() / max(v.sum(), 1):.1%} hold10 {tab['h10'].sum() / max(v.sum(), 1):.1%} "
        f"延迟5秒 {tab['h5d'].sum() / max(v.sum(), 1):.1%} | 顺势比例 1/3/5/10 分钟 "
        + "/".join(f"{np.mean(r[:, k - 1] > 0):.0%}" for k in (1, 3, 5, 10))
        + f" | 5 分钟中位 {np.median(r[:, 4]):+.2f}% 均值 {np.mean(np.clip(r[:, 4], -20, 20)):+.2f}%\n    逐月 "
        + "，".join(monthly)
    )


def bootstrap(x, y, ndays, key="h5", n=5000):
    def per_day(tab):
        return (
            np.bincount(tab["day"], weights=tab[key], minlength=ndays),
            np.bincount(tab["day"], weights=tab["valid"], minlength=ndays),
        )

    xs, xv = per_day(x)
    ys, yv = per_day(y)
    pick = np.random.default_rng(0).integers(0, ndays, (n, ndays))
    diff = xs[pick].sum(1) / np.maximum(xv[pick].sum(1), 1) - ys[pick].sum(1) / np.maximum(yv[pick].sum(1), 1)
    return xs.sum() / xv.sum() - ys.sum() / yv.sum(), *np.percentile(diff, [2.5, 97.5])


def single_rate(tab, key="h5"):
    pick = np.random.default_rng(1).integers(0, int(tab["day"].max()) + 1, (5000, int(tab["day"].max()) + 1))
    s = np.bincount(tab["day"], weights=tab[key])
    v = np.bincount(tab["day"], weights=tab["valid"])
    rates = s[pick].sum(1) / np.maximum(v[pick].sum(1), 1)
    return np.percentile(rates, [2.5, 97.5])


def random_base(months) -> str:
    rets = []
    for path in sorted((DATA / "rows_L10F10").glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] in months:
            with np.load(path) as d:
                if "rand_ret" in d:
                    r, pool = d["rand_ret"], d["rand_pool"]
                    rets.append(r[((pool >> 1) & 1 == 1) & ~np.isnan(r).any(axis=1)])
    r = np.concatenate(rets)
    pos = r > 0
    return (
        f"hold3 {pos[:, :3].all(1).mean():.1%} hold5 {pos[:, :5].all(1).mean():.1%} "
        f"hold10 {pos.all(1).mean():.1%}（{len(r)} 个样本）"
    )


if __name__ == "__main__":
    sel = json.load(open(DATA / "selection2.json", encoding="utf-8"))
    round2 = [(sel["variant"], Params(**sel["params"]), tuple(tuple(g) for g in sel["gates"]))]
    ndays = int(sum(month_days(m) for m in FRESH_MONTHS))
    print("样本外 2025-10 ~ 2026-03（第二轮写定后只运行这一次）")
    print("随机时刻随机方向（800 万池）：", random_base(FRESH_MONTHS))
    t1 = table(ROUND1, FRESH_MONTHS)
    t2 = table(round2, FRESH_MONTHS)
    print("第一轮推荐：", line(t1, FRESH_MONTHS))
    print("第二轮选中：", line(t2, FRESH_MONTHS), round2)
    lo, hi = single_rate(t2)
    print(f"  第二轮 hold5 的 95% 区间（按天自助法）[{lo:.1%}, {hi:.1%}]")
    for key in ("h5", "h3", "h10"):
        pt, a, b = bootstrap(t2, t1, ndays, key)
        print(f"  {key} 第二轮 − 第一轮 {pt:+.1%} [{a:+.1%}, {b:+.1%}]")
    for d in (1, -1):
        s = t2["dir"] == d
        print(f"  第二轮{'急涨' if d > 0 else '急跌'}：", line({k: v[s] for k, v in t2.items()}, FRESH_MONTHS))
