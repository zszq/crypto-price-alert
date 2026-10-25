"""第二轮（move_params.md 第八节）：把 4~9 月各月 grid.py 的结果拼成半年。

- 检验 A：第一轮推荐相对现行，在 4-02~8-31 整体与逐月按第二节判据判定，并给按天块自助法区间；
- 检验 B：开发期 4-02~7-31、留出期 8-01~9-30 重新组合全部候选（存 results/combos_half.pkl），
  按同一判据选参，再看相邻参数、留出期与十月；
- c：第二轮推荐与第一轮推荐、现行逐月（含九月、十月）对比，半年的按天块自助法、连续提醒与丢掉的行情段；
- d：候选族之外更严的短窗口（推荐落在族的涨跌幅、ATR 上沿），只作参考，不参与选参。

用法：data/backtest/.venv/bin/python research/move_params/half.py a|b|c|d
"""

from __future__ import annotations

import math
import pickle
import sys

import bootstrap
import combine
import explore_lib as lib
import grid
import lost
import numpy as np
import sim
import streak_count
from select_params import neighbors, passes

HALF_DEV_END = sim.month_start("202608")
CURRENT = (sim.CURRENT.short, sim.CURRENT.long)
PICK = (sim.ROUND1.short, sim.ROUND1.long)
ROUND2 = (sim.ROUND2.short, sim.ROUND2.long)
ALL = (*sim.MONTHS, "202609")


def merged(months, prefix="grid_"):
    """各月结果按同一组参数拼接；合约编号按月错开，跨月的同名合约当作不同合约（月初重新预热，与分月重放一致）。"""
    results, events = {}, []
    for k, month in enumerate(months):
        data = pickle.load(open(sim.DATA / "results" / f"{prefix}{sim.month_variant(month)}.pkl", "rb"))
        ev = data["events"].copy()
        ev[:, 0] += 1000 * k
        events.append(ev)
        for key, (cols, delay) in data["results"].items():
            cols = {**cols, "sym": cols["sym"] + 1000 * k}
            if key in results:
                old, old_delay = results[key]
                results[key] = ({f: np.concatenate([old[f], cols[f]]) for f in old}, np.concatenate([old_delay, delay]))
            else:
                results[key] = (cols, delay)
    return {"events": np.concatenate(events), "results": results}


def days_of(months):
    return sum(sim.dataset(sim.month_variant(m))["days"] for m in months)


def verdict(r, base, p="dev"):
    return "满足" if passes(r, p, base) else "不满足"


def test_a():
    lib.header()
    for months, label in [((m,), m) for m in (*sim.MONTHS, "202609")] + [(sim.MONTHS, "4-02~8-31")]:
        lib.DAYS["dev"] = days_of(months)
        base, pick = combine.build(merged(months), dev_end=math.inf, keys=[CURRENT, PICK])
        print(f"{label}（{lib.DAYS['dev']} 天）推荐{verdict(pick, base)}判据")
        lib.line("  现行", base, base, "dev")
        lib.line("  推荐", pick, base, "dev")
    data = merged(sim.MONTHS)
    bootstrap.report(data, sim.dataset(sim.month_variant(sim.MONTHS[0]))["start"], days_of(sim.MONTHS))


def test_b():
    path = sim.DATA / "results" / "combos_half.pkl"
    if not path.exists():
        rows = combine.build(merged((*sim.MONTHS, "202609")), dev_end=HALF_DEV_END)
        pickle.dump(rows, open(path, "wb"))
    rows = pickle.load(open(path, "rb"))
    index = {(r["short"], r["long"]): r for r in rows}
    base = index[CURRENT]
    oct_rows = pickle.load(open(sim.DATA / "results" / "combos_oct_s30_l180.pkl", "rb"))
    oct_index = {(r["short"], r["long"]): r for r in oct_rows}
    oct_base = oct_index[CURRENT]
    lib.DAYS.update(dev=days_of(sim.MONTHS[:4]), hold=days_of(sim.MONTHS[4:]) + 30)
    ok = sorted((r for r in rows if passes(r, "dev", base)), key=lambda r: r["dev_n"])
    print(f"满足判据 {len(ok)} / {len(rows)}")
    lib.header()
    lib.line("现行", base, base, "dev")
    lib.line("现行(留出)", base, base, "hold")
    shown = [(r["short"], r["long"]) for r in ok[:10]]
    if PICK not in shown:
        shown.append(PICK)
    for key in shown:
        r = index[key]
        nb = [index[k] for k in neighbors(key) if k in index]
        share = sum(passes(x, "dev", base) for x in nb) / max(len(nb), 1)
        print(
            lib.describe(key),
            f"{'（第一轮推荐）' if key == PICK else ''}相邻满足 {share:.0%}，",
            f"留出期{verdict(r, base, 'hold')}，十月{verdict(oct_index[key], oct_base, 'hold')}",
        )
        lib.line("  开发", r, base, "dev")
        lib.line("  留出", r, base, "hold")


def compare():
    lib.header()
    keys = [CURRENT, PICK, ROUND2]
    for months, label in [((m,), m) for m in ALL] + [(ALL, "4-02~9-30")]:
        lib.DAYS["dev"] = days_of(months)
        base, r1, r2 = combine.build(merged(months), dev_end=math.inf, keys=keys)
        print(f"{label}（{lib.DAYS['dev']} 天）第一轮{verdict(r1, base)}，第二轮{verdict(r2, base)}")
        lib.line("  现行", base, base, "dev")
        lib.line("  第一轮", r1, base, "dev")
        lib.line("  第二轮", r2, base, "dev")
    oct_index = {
        (r["short"], r["long"]): r for r in pickle.load(open(sim.DATA / "results" / "combos_oct_s30_l180.pkl", "rb"))
    }
    base = oct_index[CURRENT]
    lib.DAYS["hold"] = 9
    print(
        f"十月（9 天）第一轮{verdict(oct_index[PICK], base, 'hold')}，第二轮{verdict(oct_index[ROUND2], base, 'hold')}"
    )
    for label, key in (("  现行", CURRENT), ("  第一轮", PICK), ("  第二轮", ROUND2)):
        lib.line(label, oct_index[key], base, "hold")
    data = merged(ALL)
    start, days = sim.dataset(sim.month_variant(ALL[0]))["start"], days_of(ALL)
    print("\n第二轮推荐，按天块自助法（半年）")
    bootstrap.report(data, start, days, sim.ROUND2)
    oct_data = pickle.load(open(sim.DATA / "results" / "grid_oct_s30_l180.pkl", "rb"))
    print("\n第二轮推荐，按天块自助法（十月）")
    bootstrap.report(oct_data, sim.OCT_START, 9, sim.ROUND2)
    cases = (
        *streak_count.CASES,
        ("第二轮", sim.ROUND2.short, 3),
        ("第二轮·2 条即报", sim.ROUND2.short, 2),
    )
    print("\n连续提醒（半年）")
    streak_count.report(data["results"], days, cases)
    print("连续提醒（十月）")
    streak_count.report(oct_data["results"], 9, cases)
    print("\n丢掉的行情段（半年，第二轮）")
    lost.report(data["results"], sim.ROUND2)


OUTSIDE = [sim.Win(c, a, 15, 3, 30) for c, a in ((2.5, 2.5), (2.0, 3.0), (2.5, 3.0), (3.0, 3.0))]


def outside():
    data = merged(ALL)
    for win in OUTSIDE:
        cols, delays = [], []
        for k, month in enumerate(ALL):
            variant = sim.month_variant(month)
            ev = pickle.load(open(sim.DATA / "results" / f"grid_{variant}.pkl", "rb"))["events"]
            _, _, c, delay = grid.run(("short", win, variant, ev))
            cols.append({**c, "sym": c["sym"] + 1000 * k})
            delays.append(delay)
        data["results"][("short", win)] = (
            {f: np.concatenate([c[f] for c in cols]) for f in cols[0]},
            np.concatenate(delays),
        )
    keys = [CURRENT, ROUND2] + [(w, sim.CURRENT.long) for w in OUTSIDE]
    rows = combine.build(data, dev_end=HALF_DEV_END, keys=keys)
    lib.DAYS.update(dev=days_of(sim.MONTHS[:4]), hold=days_of(sim.MONTHS[4:]) + 30)
    lib.header()
    base = rows[0]
    for key, r in zip(keys[1:], rows[1:], strict=True):
        print(lib.describe(key), f"开发期{verdict(r, base)}，留出期{verdict(r, base, 'hold')}")
        lib.line("  开发", r, base, "dev")


if __name__ == "__main__":
    {"a": test_a, "b": test_b, "c": compare, "d": outside}[sys.argv[1]]()
