"""第二轮稳健组合搜索：三段（4~5 月、6~7 月、8 月起）都要超过基准，按超过基准最少的那一段排序。

比只看训练期更不容易挑中靠运气的组合；各条件在达标组里的出现频率，说明哪些条件真正有用。
"""

import collections
import itertools
import pickle

import numpy as np
from paths import RESULTS
from r2_lib import Engine, Rule, fold_sums, load, wait_dip

DIPS = [None, 0.001, 0.0025, 0.004]
GRID = dict(
    hold=[300, 420, 600],
    stop=[0.06, 0.07, 0.08],
    trail=[0.12, 0.15, 0.18],
    trail_atr=[np.nan, 5],
    max_extension=[np.nan, 0.15, 0.20],
    near_high=[np.nan, 0.03, 0.05],
    trail_tighten_at=[np.nan, 0.3],
    stop_support=[-1, 2],
    entry=[(10, 0.02), (10, 0.015), (10, 0.025), (15, 0.02)],
)


def describe(rule: Rule, dip: float | None) -> str:
    parts = [f"等{rule.delay}分+{rule.confirm:.1%}"]
    if dip:
        parts.append(f"回踩≥{dip:.2%}")
    if not np.isnan(rule.near_high):
        parts.append(f"贴高{rule.near_high:.0%}")
    if not np.isnan(rule.max_extension):
        parts.append(f"不追{rule.max_extension:.0%}")
    parts.append(f"损{rule.stop:.0%}" + ("支撑" if rule.stop_support == 2 else ""))
    parts.append(f"回撤{rule.trail:.0%}" + ("" if np.isnan(rule.trail_atr) else f"/atr{rule.trail_atr:g}"))
    if not np.isnan(rule.trail_tighten_at):
        parts.append("涨30%收紧8%")
    parts.append(f"持{rule.hold // 60}h")
    return " ".join(parts)


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    base = np.array(fold_sums(engine.run(Rule())[0], first, data))
    dips = {delay: wait_dip(data, delay) for delay in (10, 15)}
    rows = []
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo, strict=True))
        delay, confirm = kw.pop("entry")
        if kw["stop_support"] == 2:
            kw.update(stop_support_gap=0.02, stop_max=kw["stop"] + 0.03)
        rule = Rule(delay=delay, confirm=confirm, **kw)
        net = engine.run(rule)[0]
        for dip in DIPS:
            mask = first if dip is None else first & (dips[delay] <= -dip)
            folds = np.array(fold_sums(net, mask, data))
            rows.append((rule, dip, folds, (folds - base).min(), folds.sum()))
    with (RESULTS / "r2_robust.pkl").open("wb") as handle:
        pickle.dump((base, rows), handle)
    good = sorted((r for r in rows if r[3] > 0), key=lambda r: -r[3])
    print(f"基准三段 {base.round()}；三段都超过基准：{len(good)}/{len(rows)}")
    print("按「超过基准最少的一段」排序前 25：")
    for rule, dip, folds, worst, total in good[:25]:
        print(
            f"  最少超{worst:+4.0f} 合{total:+5.0f} 三段 {folds[0]:+4.0f} {folds[1]:+4.0f} {folds[2]:+4.0f} | "
            f"{describe(rule, dip)}"
        )
    print("各条件在「三段都超过基准」组里的占比：")
    keys = [
        "hold",
        "stop",
        "trail",
        "trail_atr",
        "max_extension",
        "near_high",
        "trail_tighten_at",
        "stop_support",
        "delay",
        "confirm",
    ]
    for key in keys:
        total = collections.Counter(str(getattr(r[0], key)) for r in rows)
        hits = collections.Counter(str(getattr(r[0], key)) for r in good)
        print(f"  {key}: " + "  ".join(f"{v} {hits[v] / total[v]:.1%}" for v in total))
    total = collections.Counter(str(r[1]) for r in rows)
    hits = collections.Counter(str(r[1]) for r in good)
    print("  回踩: " + "  ".join(f"{v} {hits[v] / total[v]:.1%}" for v in total))


if __name__ == "__main__":
    main()
