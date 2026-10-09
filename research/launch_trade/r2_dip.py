"""第二轮：「先回踩、再涨回确认价」的阈值是否平滑，以及最稳组附近（等待分钟 × 确认涨幅 × 回踩幅度）是否平稳。

阈值从 0.1% 到 0.4% 都比基准好、效果平滑变化，才说明它是规律而不是凑巧挑中的区间。
"""

from dataclasses import replace

import numpy as np
from r2_lib import BEST, Engine, Rule, evaluate, fold_sums, load, wait_dip


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    dip = wait_dip(data)
    print("回踩阈值扫描（等待期最低价至少比提醒价低这么多）：")
    for name, rule in (("基准", Rule()), ("持仓7h", Rule(hold=420)), ("不追15%", Rule(max_extension=0.15))):
        net = engine.run(rule)[0]
        print(name)
        for x in (None, 0.0, 0.001, 0.0025, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
            mask = first if x is None else first & (dip <= -x)
            folds = np.array(fold_sums(net, mask, data))
            res = evaluate(net, mask, data)
            label = "无" if x is None else f"{x:.2%}"
            print(
                f"  回踩≥{label:6s} n={res['all_n']:3d} 三段 {folds[0]:+5.0f} {folds[1]:+5.0f} {folds[2]:+5.0f} "
                f"合{folds.sum():+5.0f} 去10{res['minus10']:+5.0f} t{res['t']:.2f}"
            )

    base = np.array(fold_sums(engine.run(Rule())[0], first, data))
    dips = (0.001, 0.0025, 0.004, 0.006)
    print("\n最稳组附近：行为等待分钟+确认涨幅，列为回踩幅度，格内为合计（* 表示三段都超过基准）")
    print(" " * 14 + "".join(f"{x:>10.2%}" for x in dips))
    for delay in (8, 10, 12):
        delay_dip = wait_dip(data, delay)
        for confirm in (0.01, 0.015, 0.02, 0.025, 0.03):
            net = engine.run(replace(BEST, delay=delay, confirm=confirm))[0]
            cells = []
            for x in dips:
                folds = np.array(fold_sums(net, first & (delay_dip <= -x), data))
                cells.append(f"{folds.sum():+8.0f}{'*' if (folds > base).all() else ' '} ")
            print(f"等{delay:2d}分+{confirm:.1%}   " + "".join(cells))


if __name__ == "__main__":
    main()
