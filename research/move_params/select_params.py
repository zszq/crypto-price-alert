"""按预先写定的判据在开发期选参，再看留出期。

判据（相对现行参数，开发期）：
- 大行情（5 分钟收盘价变动 ≥3%）覆盖率下降 ≤ 2 个百分点，≥5% 的下降 ≤ 1 个百分点；
- 两档大行情的首报中位延迟各增加 ≤ 20 秒；
- 现行参数的行情段（间隔 >10 分钟分段）至少 90% 仍有提醒；
- 5 分钟保留率不降、假提醒率（15 分钟内先回到基准价）不升。
满足的组合里取每天提醒最少的；稳健性看它的相邻参数（任一项挪一格）有多少同样满足。
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle

import explore_lib as lib
import sim

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="s30_l180")
parser.add_argument("--top", type=int, default=15)
ns, _ = parser.parse_known_args()
rows = pickle.load(open(sim.DATA / "results" / f"combos_{ns.variant}.pkl", "rb"))
index = {(r["short"], r["long"]): r for r in rows}
base = index[(sim.CURRENT.short, sim.CURRENT.long)]


def passes(r, p="dev", base=base):
    return (
        r[f"{p}_big3"] >= base[f"{p}_big3"] - 0.02
        and r[f"{p}_big5"] >= base[f"{p}_big5"] - 0.01
        and r[f"{p}_big3_delay"] <= base[f"{p}_big3_delay"] + 20
        and r[f"{p}_big5_delay"] <= base[f"{p}_big5_delay"] + 20
        and r[f"{p}_segkeep"] >= 0.90
        and r[f"{p}_held"] >= base[f"{p}_held"]
        and r[f"{p}_fake"] <= base[f"{p}_fake"]
    )


GRIDS = {
    "change": ([1.2, 1.5, 1.8, 2.0], [1.6, 2.0, 2.5, 3.0]),
    "atr": ([1.5, 2.0, 2.5], [1.8, 2.2, 2.6, 3.0]),
    "trades": ([15, 30], [30, 60]),
    "confirm": ([2, 3], [3, 5]),
    "cooldown": ([30, 60, 120, 300], [180, 300, 600, 900]),
}


def neighbors(key, grids=GRIDS):
    s, long = key
    for side in (0, 1):
        w = key[side]
        if w is None:
            continue
        for field, per_side in grids.items():
            values = per_side[side]
            i = values.index(getattr(w, field))
            for j in (i - 1, i + 1):
                if 0 <= j < len(values):
                    nw = dataclasses.replace(w, **{field: values[j]})
                    yield (nw, long) if side == 0 else (s, nw)


def main():
    ok = sorted((r for r in rows if passes(r)), key=lambda r: r["dev_n"])
    print(f"满足判据 {len(ok)} / {len(rows)}")
    lib.header()
    lib.line("现行", base, base, "dev")
    lib.line("现行(留出)", base, base, "hold")
    for r in ok[: ns.top]:
        key = (r["short"], r["long"])
        nb = [index[k] for k in neighbors(key) if k in index]
        share = sum(passes(x) for x in nb) / max(len(nb), 1)
        print(lib.describe(key), f"相邻满足 {share:.0%}，留出期{'满足' if passes(r, 'hold') else '不满足'}")
        lib.line("  开发", r, base, "dev")
        lib.line("  留出", r, base, "hold")


if __name__ == "__main__":
    main()
