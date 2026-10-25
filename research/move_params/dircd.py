"""试验：冷却按方向分别计算（同方向冷却、反向不受限）能否在加长冷却时保住大行情覆盖。需要改检测器，不是参数。"""

import math
import pickle
from multiprocessing import Pool

import combine
import explore_lib as lib
import grid
import numpy as np
import sim

original = sim.simulate_window


def simulate_window_by_direction(rec, w):
    if len(rec) == 0:
        return []
    ok = (rec[:, sim.TRADES] >= w.trades) & (np.abs(rec[:, sim.CHG]) >= w.change) & (rec[:, sim.MATR] >= w.atr)
    alerts, last_sec, last_dir, count = [], -10.0, 0, 0
    last_alert = {1: -math.inf, -1: -math.inf}
    for i in np.flatnonzero(ok):
        r = rec[i]
        direction = 1 if r[sim.CHG] > 0 else -1
        count = count + 1 if (direction == last_dir and r[sim.SEC] - last_sec == 1) else 1
        last_dir, last_sec = direction, r[sim.SEC]
        if count < w.confirm or r[sim.CARRIED]:
            continue
        lchg = r[sim.LCHG]
        if not math.isnan(lchg) and ((lchg > 0) != (direction > 0) or abs(lchg) < w.change or r[sim.LATR] < w.atr):
            continue
        if r[sim.NOW] - last_alert[direction] >= w.cooldown:
            alerts.append(i)
            last_alert[direction] = r[sim.NOW]
        last_dir, last_sec, count = 0, -10.0, 0
    return alerts


def run(job):
    sim.simulate_window = simulate_window_by_direction if job[-1] else original
    return grid.run(job[:-1])


if __name__ == "__main__":
    data = pickle.load(open(sim.DATA / "results" / "grid_s30_l180.pkl", "rb"))
    ev = data["events"]
    ev_period = np.where(ev[:, 1] < sim.DEV_END, 0, 1)
    res = data["results"]

    def prep(key):
        return combine.prepare(*res[key])

    base_spans = [
        combine.segment_spans(
            np.sort(
                np.concatenate(
                    [prep(("short", sim.CURRENT.short))[0][p]["key"], prep(("long", sim.CURRENT.long))[0][p]["key"]]
                )
            )
        )
        for p in (0, 1)
    ]
    base = combine.combo_row(
        sim.CURRENT.short,
        sim.CURRENT.long,
        prep(("short", sim.CURRENT.short)),
        prep(("long", sim.CURRENT.long)),
        base_spans,
        ev,
        ev_period,
    )
    short = sim.Win(2.0, 2.0, 15, 3, 60)
    longs = [sim.Win(1.6, 1.8, 30, 3, cd) for cd in (180, 300, 600, 900)]
    shorts = [sim.Win(2.0, 2.0, 15, 3, cd) for cd in (60, 120, 300)]
    jobs = [("long", w, "s30_l180", ev, True) for w in longs] + [("short", w, "s30_l180", ev, True) for w in shorts]
    with Pool(8) as pool:
        out = {(n, w): (c, d) for n, w, c, d in pool.map(run, jobs)}
    lib.header()
    lib.line("现行", base, base, "dev")
    for sw in shorts:
        for lw in longs:
            for label, s_key, l_key in (("不分方向", None, None), ("按方向", ("short", sw), ("long", lw))):
                sp = combine.prepare(*out[s_key]) if s_key else prep(("short", sw))
                lp = combine.prepare(*out[l_key]) if l_key else prep(("long", lw))
                row = combine.combo_row(sw, lw, sp, lp, base_spans, ev, ev_period)
                lib.line(f"短冷{sw.cooldown}长冷{lw.cooldown}{label}"[:14], row, base, "dev")
