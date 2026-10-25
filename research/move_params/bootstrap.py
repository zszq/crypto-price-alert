"""选中参数相对现行参数（或指定的对照参数）的变化，按天块自助法（以天为单位有放回抽样 5000 次）给 95% 置信区间。

用法：data/backtest/.venv/bin/python research/move_params/bootstrap.py [--variant s30_l180|oct_s30_l180]
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import sim

PICK = sim.ROUND1


def per_day(res, ev, params, start, days):
    """每天的提醒数、5 分钟保留数、假提醒数，以及每天大行情（3%）段数与报到段数。"""
    cols = [res[("short", params.short)][0], res[("long", params.long)][0]]
    delays = np.minimum(res[("short", params.short)][1], res[("long", params.long)][1])
    t = np.concatenate([c["t"] for c in cols])
    day = ((t - start) // 86400).astype(int)
    keep = (day >= 0) & (day < days)
    out = np.zeros((days, 5))
    for k, field in enumerate(("held", "fake")):
        values = np.concatenate([c[field] for c in cols])[keep]
        np.add.at(out[:, k + 1], day[keep], values)
    np.add.at(out[:, 0], day[keep], 1)
    big = ev[:, 4] == 3.0
    ev_day = ((ev[big, 1] - start) // 86400).astype(int)
    ok = (ev_day >= 0) & (ev_day < days)
    np.add.at(out[:, 3], ev_day[ok], 1)
    np.add.at(out[:, 4], ev_day[ok], np.isfinite(delays[big][ok]))
    return out


def ratios(a):
    return np.array(
        [
            a[..., 0].sum(-1),
            a[..., 1].sum(-1) / a[..., 0].sum(-1),
            a[..., 2].sum(-1) / a[..., 0].sum(-1),
            a[..., 4].sum(-1) / a[..., 3].sum(-1),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s30_l180")
    ns = parser.parse_args()
    data = pickle.load(open(sim.DATA / "results" / f"grid_{ns.variant}.pkl", "rb"))
    ds = sim.dataset(ns.variant)
    report(data, ds["start"], ds["days"])


def with_score(r):
    """末尾加质量分（5 分钟保留率 − 假提醒率，第三轮的选参目标）。"""
    return np.concatenate([r, [r[1] - r[2]]])


def report(data, start, days, params=PICK, base_params=sim.CURRENT):
    base = per_day(data["results"], data["events"], base_params, start, days)
    pick = per_day(data["results"], data["events"], params, start, days)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, days, size=(5000, days))
    b, p = with_score(ratios(base[idx])), with_score(ratios(pick[idx]))
    point_b, point_p = with_score(ratios(base)), with_score(ratios(pick))
    names = ("提醒数变化", "5 分钟保留率", "假提醒率", "大行情 3% 覆盖率", "质量分")
    print(f"{days} 天，每天提醒 对照 {point_b[0] / days:.0f} 条 → 选中 {point_p[0] / days:.0f} 条")
    for i, name in enumerate(names):
        diff = p[i] / b[i] - 1 if i == 0 else p[i] - b[i]
        point = point_p[i] / point_b[i] - 1 if i == 0 else point_p[i] - point_b[i]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        unit = "" if i == 0 else " 个百分点"
        scale = 100
        print(f"{name}: {point * scale:+.1f}{'%' if i == 0 else unit}，95% 区间 [{lo * scale:+.1f}, {hi * scale:+.1f}]")
    worse = sum(1 for d in range(days) if pick[d, 0] and pick[d, 1] / pick[d, 0] < base[d, 1] / max(base[d, 0], 1))
    print(f"逐日：5 分钟保留率低于对照的天数 {worse} / {days}")


if __name__ == "__main__":
    main()
