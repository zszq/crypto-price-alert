"""第二轮：另外两种入场方式——突破等待期高点追入（模式 1）、回踩支撑位挂单买（模式 2）。"""

import itertools
import pickle

import numpy as np
from paths import RESULTS
from r2_lib import Engine, Rule, evaluate, fmt, load


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    rows = []
    print("== 模式 1：突破等待期高点追入")
    for delay, window in itertools.product([5, 10, 15, 30], [10, 30, 60, 120]):
        rule = Rule(entry_mode=1, delay=delay, entry_window=window, confirm=np.nan)
        res = evaluate(engine.run(rule)[0], first, data)
        rows.append(("突破追入", rule, res))
        print(f"  观察{delay} 最多再等{window}", fmt(res))
    print("== 模式 1 + 等待期收盘贴近高点 3%")
    for delay, window in itertools.product([10, 15], [30, 60]):
        rule = Rule(entry_mode=1, delay=delay, entry_window=window, confirm=np.nan, near_high=0.03)
        res = evaluate(engine.run(rule)[0], first, data)
        rows.append(("突破追入+贴高", rule, res))
        print(f"  观察{delay} 再等{window}", fmt(res))
    print("== 模式 2：回踩支撑挂单（支撑 0 基准期高点，1 提醒价，2 等待期最低价；只列训练期 > +150 的）")
    grid = itertools.product([0, 1, 2], [-0.01, 0.0, 0.01, 0.02, 0.03], [5, 10], [30, 60, 120])
    for kind, pullback, delay, window in grid:
        rule = Rule(
            entry_mode=2, support_kind=kind, pullback=pullback, delay=delay, entry_window=window, confirm=np.nan
        )
        res = evaluate(engine.run(rule)[0], first, data)
        rows.append(("回踩挂单", rule, res))
        if res["train_sum"] > 150:
            print(f"  支撑{kind} 挂单+{pullback} 观察{delay} 再等{window}", fmt(res))
    with (RESULTS / "r2_entry_modes.pkl").open("wb") as handle:
        pickle.dump(rows, handle)


if __name__ == "__main__":
    main()
