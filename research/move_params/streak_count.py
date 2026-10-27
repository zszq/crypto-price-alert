"""短窗口提醒串成的连续提醒条数。

规则同 src/price_alert/detectors/spike/streak.py：同合约同方向、间隔 ≤ 冷却 + 30 秒，第 min_alerts 条时报一次；
这里没有模拟「累计方向不一致时先不报」，只用来比较数量级。
"""

import pickle
import sys

import numpy as np
import sim

CASES = (
    ("现行", sim.CURRENT.short, 3),
    ("选中", sim.ROUND1.short, 3),
    ("选中·2 条即报", sim.ROUND1.short, 2),
)


def streaks(results, win, need):
    cols = results[("short", win)][0]
    order = np.lexsort((cols["t"], cols["sym"]))
    gap = win.cooldown + 30
    count, prev, total = 0, None, 0
    for i in order:
        key = (cols["sym"][i], cols["d"][i])
        if prev is not None and prev[0] == key and cols["t"][i] - prev[1] <= gap:
            count += 1
        else:
            count = 1
        total += count == need
        prev = (key, cols["t"][i])
    return total


def report(results, days, cases=CASES):
    for label, win, need in cases:
        print(f"{label}：连续提醒 {streaks(results, win, need) / days:.1f} 条/天")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "s30_l180"
    data = pickle.load(open(sim.DATA / "results" / f"grid_{variant}.pkl", "rb"))
    report(data["results"], sim.dataset(variant)["days"])
