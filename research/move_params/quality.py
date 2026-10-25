"""第三轮（move_params.md 第九节）：以质量分（5 分钟保留率 − 假提醒率）为目标，在宽候选族上选参。

- s：组合 4~9 月 grid_wide 结果（首次存 results/combos_wide.pkl），按第九节约束与选参规则选出一组，
  看相邻参数、留出期与十月；
- c：规则选中、长窗口不动（sim.ROUND3）与现行、第二轮推荐逐月对比，半年与十月的按天块自助法、连续提醒、丢掉的行情段；
- f：逐条放宽约束后的最高质量分，看是哪条约束限制了质量。

用法：data/backtest/.venv/bin/python research/move_params/quality.py s|c|f
"""

from __future__ import annotations

import gc
import math
import multiprocessing
import pickle
import sys

import bootstrap
import combine
import explore_lib as lib
import grid
import half
import lost
import sim
import streak_count
from select_params import neighbors

TIE = 0.003  # 质量分相差不超过 0.3 个百分点视为同等
ROWS = sim.DATA / "results" / "combos_wide.pkl"
OCT_ROWS = sim.DATA / "results" / "combos_wide_oct.pkl"
ROUND2 = (sim.ROUND2.short, sim.ROUND2.long)
PREFIX = "grid_wide_"


def score(r, p="dev"):
    return r[f"{p}_held"] - r[f"{p}_fake"]


def feasible(r, base, p="dev"):
    return (
        r[f"{p}_big3"] >= base[f"{p}_big3"] - 0.02
        and r[f"{p}_big5"] >= base[f"{p}_big5"] - 0.01
        and r[f"{p}_big3_delay"] <= base[f"{p}_big3_delay"] + 20
        and r[f"{p}_big5_delay"] <= base[f"{p}_big5_delay"] + 20
        and r[f"{p}_segkeep"] >= 0.90
        and r[f"{p}_n"] <= base[f"{p}_n"]
    )


_ctx = None


def _rows(pairs):
    return [combine.row(_ctx, key) for key in pairs]


def build_rows():
    """约 39 万组，分块交给多个进程；各窗口的汇总先在父进程做一次，显式 fork 后共用（macOS 默认 spawn 不继承）。"""
    global _ctx
    _ctx = combine.setup(half.merged(half.ALL, PREFIX), dev_end=half.HALF_DEV_END)
    gc.collect()
    pairs = _ctx["pairs"]
    chunks = [pairs[i : i + 2000] for i in range(0, len(pairs), 2000)]
    rows = []
    with multiprocessing.get_context("fork").Pool(10) as pool:
        for i, part in enumerate(pool.imap(_rows, chunks)):
            rows.extend(part)
            if i % 20 == 0:
                print(f"{len(rows)} / {len(pairs)}", flush=True)
    return rows


def load_rows():
    if not ROWS.exists():
        pickle.dump(build_rows(), open(ROWS, "wb"))
    return pickle.load(open(ROWS, "rb"))


def oct_index(keys):
    """十月整段都是留出期（dev_end=0），只算需要的组合。"""
    cached = pickle.load(open(OCT_ROWS, "rb")) if OCT_ROWS.exists() else {}
    missing = [k for k in keys if k not in cached]
    if missing:
        data = pickle.load(open(sim.DATA / "results" / f"{PREFIX}oct_s30_l180.pkl", "rb"))
        for key, r in zip(missing, combine.build(data, dev_end=0, keys=missing), strict=True):
            cached[key] = r
        pickle.dump(cached, open(OCT_ROWS, "wb"))
    return cached


def pick(rows, base):
    ok = [r for r in rows if feasible(r, base)]
    best = max(score(r) for r in ok)
    tied = sorted((r for r in ok if score(r) >= best - TIE), key=lambda r: r["dev_n"])
    return ok, best, tied


def select():
    rows = load_rows()
    index = {(r["short"], r["long"]): r for r in rows}
    base, r2 = index[half.CURRENT], index[ROUND2]
    ok, best, tied = pick(rows, base)
    print(f"满足约束 {len(ok)} / {len(rows)}，最高质量分 {best:.1%}，同等（−{TIE:.1%} 内）{len(tied)} 组")
    lib.DAYS.update(dev=half.days_of(sim.MONTHS[:4]), hold=half.days_of(sim.MONTHS[4:]) + 30)
    shown = [(r["short"], r["long"]) for r in tied[:8]]
    top = sorted(ok, key=score, reverse=True)[:8]
    shown += [(r["short"], r["long"]) for r in top if (r["short"], r["long"]) not in shown]
    octs = oct_index([half.CURRENT, ROUND2, *shown])
    print(
        "质量分：现行",
        f"{score(base):.1%}",
        "第二轮",
        f"{score(r2):.1%}",
        "（开发）；",
        f"留出 {score(base, 'hold'):.1%} / {score(r2, 'hold'):.1%}；",
        f"十月 {score(octs[half.CURRENT], 'hold'):.1%} / {score(octs[ROUND2], 'hold'):.1%}",
    )
    lib.header()
    lib.line("现行", base, base, "dev")
    lib.line("第二轮", r2, base, "dev")
    for key in shown:
        r = index[key]
        nb = [index[k] for k in neighbors(key, grid.WIDE) if k in index]
        share = sum(feasible(x, base) and score(x) >= best - TIE for x in nb) / max(len(nb), 1)
        o = octs[key]
        hold_ok = feasible(r, base, "hold") and score(r, "hold") > score(r2, "hold")
        oct_ok = feasible(o, octs[half.CURRENT], "hold") and score(o, "hold") > score(octs[ROUND2], "hold")
        print(
            lib.describe(key),
            f"{'【选中】' if key == (tied[0]['short'], tied[0]['long']) else ''}",
            f"质量分 {score(r):.1%}，相邻同等 {share:.0%}，",
            f"留出 {score(r, 'hold'):.1%}{'满足' if hold_ok else '不满足'}，",
            f"十月 {score(o, 'hold'):.1%}{'满足' if oct_ok else '不满足'}",
        )
        lib.line("  开发", r, base, "dev")
        lib.line("  留出", r, base, "hold")
    choice = (tied[0]["short"], tied[0]["long"])
    print("\n选中参数的相邻参数（开发期）")
    for key in neighbors(choice, grid.WIDE):
        r = index[key]
        print(lib.describe(key), f"质量分 {score(r):.1%}，{'满足约束' if feasible(r, base) else '不满足约束'}")
        lib.line("  ", r, base, "dev")


def chosen():
    rows = load_rows()
    index = {(r["short"], r["long"]): r for r in rows}
    return pick(rows, index[half.CURRENT])[2][0]


def compare():
    r = chosen()
    named = {
        "现行": sim.CURRENT,
        "第二轮": sim.ROUND2,
        "规则选中": sim.Params(r["short"], r["long"]),
        "长窗口不动": sim.ROUND3,
    }
    keys = [(p.short, p.long) for p in named.values()]
    for label, key in zip(named, keys, strict=True):
        print(label, lib.describe(key))
    lib.header()
    for months, label in [((m,), m) for m in half.ALL] + [(half.ALL, "4-02~9-30")]:
        lib.DAYS["dev"] = half.days_of(months)
        rows = combine.build(half.merged(months, PREFIX), dev_end=math.inf, keys=keys)
        print(f"{label}（{lib.DAYS['dev']} 天）")
        for name, row in zip(named, rows, strict=True):
            print(f"  {name} 质量分 {score(row):.1%}，{'满足' if feasible(row, rows[0]) else '不满足'}约束")
            lib.line("", row, rows[0], "dev")
    octs = oct_index(keys)
    lib.DAYS["hold"] = 9
    print("十月（9 天）")
    for name, key in zip(named, keys, strict=True):
        print(f"  {name} 质量分 {score(octs[key], 'hold'):.1%}")
        lib.line("", octs[key], octs[keys[0]], "hold")
    data = half.merged(half.ALL, PREFIX)
    oct_data = pickle.load(open(sim.DATA / "results" / f"{PREFIX}oct_s30_l180.pkl", "rb"))
    start, days = sim.dataset(sim.month_variant(half.ALL[0]))["start"], half.days_of(half.ALL)
    for name in ("规则选中", "长窗口不动"):
        for base_name in ("现行", "第二轮"):
            print(f"\n按天块自助法：{name}相对{base_name}（半年）")
            bootstrap.report(data, start, days, named[name], named[base_name])
            print(f"按天块自助法：{name}相对{base_name}（十月）")
            bootstrap.report(oct_data, sim.OCT_START, 9, named[name], named[base_name])
    cases = [(name, p.short, 3) for name, p in named.items() if name != "规则选中"]
    cases.append(("长窗口不动·2 条即报", sim.ROUND3.short, 2))
    print("\n连续提醒（半年）")
    streak_count.report(data["results"], days, cases)
    print("连续提醒（十月）")
    streak_count.report(oct_data["results"], 9, cases)
    for name in ("规则选中", "长窗口不动"):
        print(f"\n丢掉的行情段（半年，{name}）")
        lost.report(data["results"], named[name])


def frontier():
    """放宽第九节的约束时，开发期质量分最高能到多少：看是哪条约束限制了质量。"""
    rows = load_rows()
    base = next(r for r in rows if (r["short"], r["long"]) == half.CURRENT)
    lib.DAYS.update(dev=half.days_of(sim.MONTHS[:4]), hold=half.days_of(sim.MONTHS[4:]) + 30)

    def coverage(r):
        return (
            r["dev_big3"] >= base["dev_big3"] - 0.02
            and r["dev_big5"] >= base["dev_big5"] - 0.01
            and r["dev_n"] <= base["dev_n"]
        )

    def delay(r, extra):
        return max(r["dev_big3_delay"] - base["dev_big3_delay"], r["dev_big5_delay"] - base["dev_big5_delay"]) <= extra

    cases = {
        "全部约束": lambda r: feasible(r, base),
        "段保留放宽到 70%": lambda r: coverage(r) and delay(r, 20) and r["dev_segkeep"] >= 0.7,
        "不限段保留": lambda r: coverage(r) and delay(r, 20),
        "不限延迟": lambda r: coverage(r) and r["dev_segkeep"] >= 0.9,
        "延迟放宽到 +40 秒、不限段保留": lambda r: coverage(r) and delay(r, 40),
        "只保留覆盖约束": coverage,
    }
    lib.header()
    for label, ok in cases.items():
        best = max((r for r in rows if ok(r)), key=score)
        print(
            f"{label}：质量分 {score(best):.1%}（留出 {score(best, 'hold'):.1%}）",
            lib.describe((best["short"], best["long"])),
        )
        lib.line("  开发", best, base, "dev")
        lib.line("  留出", best, base, "hold")


if __name__ == "__main__":
    {"s": select, "c": compare, "f": frontier}[sys.argv[1]]()
