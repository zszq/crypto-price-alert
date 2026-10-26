"""留出期（8~9 月）验证：选中参数与现行参数的对比、自助法区间、随机基准、相邻参数与其他条数下限。

用法：evaluate.py [每天条数下限，默认 20]（读 select_params.py 写出的 selection_<下限>.json）
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import DATA, DEV_MONTHS, HOLD_MONTHS, MONTHS, month_start
from grid import CD, GATES, POOLS, A, B, C, T
from select_params import CURRENT, SINGLE, describe, feasible, load_grids, pick
from sim import Params, alert_rows

ALL = DEV_MONTHS + HOLD_MONTHS


def table(variant: str, p: Params, gates: tuple, months: list[str]) -> dict[str, np.ndarray]:
    ar = alert_rows(variant, months, p, gates)
    ret = ar["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    first = month_start(months[0])
    return {
        "t": ar["t"],
        "dir": ar["dir"],
        "valid": valid,
        "month": ar["month"],
        "day": ((ar["t"] - first) // 86400).astype(np.int64),
        "h3": pos[:, :3].all(axis=1) & valid,
        "h5": pos[:, :5].all(axis=1) & valid,
        "h10": pos.all(axis=1) & valid,
        "r5": np.where(valid, np.nan_to_num(ret[:, 4]), 0.0),
        "fee": (np.nan_to_num(ret[:, 4]) > 0.1) & valid,
        "h5d": ar["hold5d"] & valid,
    }


def merge(*tables: dict) -> dict:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def days_of(months: list[str]) -> float:
    from sim import month_days

    return sum(month_days(m) for m in months)


def line(tab: dict, months: list[str]) -> str:
    v = max(tab["valid"].sum(), 1)
    rate = {k: tab[k].sum() / v for k in ("h3", "h5", "h10", "fee", "h5d")}
    monthly = []
    for m in months:
        sel = tab["month"] == MONTHS.index(m)
        monthly.append(f"{m[4:]}月 {tab['h5'][sel].sum() / max(tab['valid'][sel].sum(), 1):.1%}")
    return (
        f"每天 {len(tab['t']) / days_of(months):.1f} 条 hold5 {rate['h5']:.1%} hold3 {rate['h3']:.1%} "
        f"hold10 {rate['h10']:.1%} 5分钟顺势均值 {tab['r5'].sum() / v:+.3f}% >0.1% {rate['fee']:.1%} "
        f"延迟5秒 {rate['h5d']:.1%}（{'，'.join(monthly)}）"
    )


def bootstrap(x: dict, y: dict, ndays: int, key: str = "h5", n: int = 5000) -> tuple[float, float, float]:
    """按天块重抽样，x 与 y 的 key 率之差（x − y）与 95% 区间。"""

    def per_day(tab):
        return (
            np.bincount(tab["day"], weights=tab[key], minlength=ndays),
            np.bincount(tab["day"], weights=tab["valid"], minlength=ndays),
        )

    xs, xv = per_day(x)
    ys, yv = per_day(y)
    rng = np.random.default_rng(0)
    pick_ = rng.integers(0, ndays, (n, ndays))
    diff = xs[pick_].sum(1) / xv[pick_].sum(1) - ys[pick_].sum(1) / yv[pick_].sum(1)
    point = xs.sum() / xv.sum() - ys.sum() / yv.sum()
    return point, *np.percentile(diff, [2.5, 97.5])


def random_base(months: list[str]) -> str:
    hold = []
    for path in sorted((DATA / "rows").glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] not in months:
            continue
        with np.load(path) as d:
            if "rand_ret" not in d:
                continue
            ret, pool = d["rand_ret"], d["rand_pool"]
        sel = ((pool >> 1) & 1 == 1) & ~np.isnan(ret).any(axis=1)
        hold.append(ret[sel])
    r = np.concatenate(hold)
    pos = r > 0
    return (
        f"随机时刻随机方向（800 万池，{len(r)} 个样本）：hold3 {pos[:, :3].all(1).mean():.1%} "
        f"hold5 {pos[:, :5].all(1).mean():.1%} hold10 {pos.all(1).mean():.1%}"
    )


def windows_of(selection: dict) -> list[tuple[str, Params, tuple]]:
    out = []
    for w in selection["windows"]:
        out.append((w["variant"], Params(**w["params"]), tuple(tuple(g) for g in w["gates"])))
    if len(out) == 2:
        # 长窗口的补齐上限跟随短窗口：需要 engine 单独算出 L<长>F<短> 变体
        out.sort(key=lambda x: int(x[0][1:].split("F")[0]))
        short_len = int(out[0][0][1:])
        long_len = int(out[1][0][1:].split("F")[0])
        out[1] = (f"L{long_len}F{short_len}", out[1][1], out[1][2])
    return out


def neighbors(variant: str, p: Params, gates: tuple):
    """任一现有参数在网格里挪一格（新条件换相邻档位）。"""
    grids = {"a": A, "b": B, "c": C, "trades": T, "cooldown": CD, "pool": POOLS}
    for name, values in grids.items():
        cur = getattr(p, name)
        if cur not in values:
            continue
        k = values.index(cur)
        for j in (k - 1, k + 1):
            if 0 <= j < len(values):
                yield f"{name} {values[j]:g}", variant, Params(**(vars(p) | {name: values[j]})), gates
    for gi, gate in enumerate(gates):
        same = [g for g in GATES[1:] if g[0] == gate[0]]
        k = same.index(gate)
        for j in (k - 1, k + 1):
            if 0 <= j < len(same):
                g2 = gates[:gi] + (same[j],) + gates[gi + 1 :]
                yield f"{same[j][0]} {same[j][1]:g}", variant, p, g2
        yield f"去掉 {gate[0]}", variant, p, gates[:gi] + gates[gi + 1 :]
    length = int(variant[1:].split("F")[0])
    k = SINGLE.index(f"L{length}")
    for j in (k - 1, k + 1):
        if 0 <= j < len(SINGLE):
            yield f"窗口 {SINGLE[j]}", SINGLE[j], p, gates


if __name__ == "__main__":
    floor = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    selection = json.load(open(DATA / f"selection_{floor:g}.json", encoding="utf-8"))
    chosen = windows_of(selection)
    hold_days = int(days_of(HOLD_MONTHS))

    print("== 随机基准 ==")
    print("开发期", random_base(DEV_MONTHS))
    print("留出期", random_base(HOLD_MONTHS))

    print("\n== 选中参数 vs 现行（留出期 8~9 月，只看这一次） ==")
    cur = merge(*(table(v, p, (), HOLD_MONTHS) for v, p in CURRENT))
    new = merge(*(table(v, p, g, HOLD_MONTHS) for v, p, g in chosen))
    print("现行：", line(cur, HOLD_MONTHS))
    print("选中：", line(new, HOLD_MONTHS), [(v, p, g) for v, p, g in chosen])
    for key in ("h5", "h3", "h10"):
        pt, lo, hi = bootstrap(new, cur, hold_days, key)
        print(f"  {key} 差值 {pt:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    dev_new = merge(*(table(v, p, g, DEV_MONTHS) for v, p, g in chosen))
    print("选中（开发期，对照衰减）：", line(dev_new, DEV_MONTHS))
    for d in (1, -1):
        sel = new["dir"] == d
        print(f"  留出期{'急涨' if d > 0 else '急跌'}：", line({k: v[sel] for k, v in new.items()}, HOLD_MONTHS))

    print("\n== 相邻参数（主窗口，开发期 / 留出期 hold5） ==")
    # 主窗口是第二步选中的那个；合并第二个窗口后它的变体名可能带上了补齐上限
    primary = selection["windows"][0]["variant"]
    v0, p0, g0 = next(c for c in chosen if c[0].split("F")[0] == primary)
    for label, v, p, g in neighbors(v0, p0, g0):
        d = table(v, p, g, DEV_MONTHS)
        h = table(v, p, g, HOLD_MONTHS)
        print(
            f"  {label:<14} 开发 {d['h5'].sum() / max(d['valid'].sum(), 1):.1%}（每天 {len(d['t']) / 122:.1f}） "
            f"留出 {h['h5'].sum() / max(h['valid'].sum(), 1):.1%}（每天 {len(h['t']) / hold_days:.1f}）"
        )

    print("\n== 其他条数下限（只按第二步规则：单窗口、至多一个新条件） ==")
    grids = load_grids()
    base_dev = None
    from select_params import current_agg

    base_dev = current_agg()
    for fl in (5, 10, 20, 50, 100, 200):
        cands = []
        for v in SINGLE:
            combos, agg = grids[v]
            ok = np.flatnonzero(feasible(agg, base_dev, fl))
            h5, pd_ = agg.rate("h5"), agg.per_day()
            cands += [(h5[i], pd_[i], int(GATES[combos[i][6]] is not None), v, i) for i in ok]
        if not cands:
            print(f"  ≥{fl}/天：无")
            continue
        h5, pd_, _, v, i = pick(cands)
        from select_params import params_of

        p, g = params_of(grids[v][0][i])
        h = table(v, p, g, HOLD_MONTHS)
        print(
            f"  ≥{fl}/天：{describe(v, grids[v][0][i])}\n"
            f"      开发 hold5 {h5:.1%}（每天 {pd_:.1f}） 留出 {line(h, HOLD_MONTHS)}"
        )
