"""第二轮：收盘价回撤（影线不算）与收盘跌破支撑位离场，按三段评估。"""

from dataclasses import replace

import numpy as np
from r2_lib import Engine, Rule, fold_line, fold_sums, load


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    base = Rule()
    reference = np.array(fold_sums(engine.run(base)[0], first, data))

    def show(label: str, rule: Rule) -> None:
        print(fold_line(label, engine.run(rule)[0], first, data, reference))

    print(f"基准三段 {reference.round()}")
    print("== 收盘价回撤")
    for trail in (0.10, 0.12, 0.15, 0.18, 0.20):
        show(f"收盘回撤{trail}", replace(base, trail_on_close=1, trail=trail))
    print("== 收盘跌破支撑离场（0 基准期高点，1 提醒价，2 等待期最低价）")
    for kind in (0, 1, 2):
        for gap in (0.0, 0.01, 0.02, 0.03):
            for after in (0, 10, 30):
                rule = replace(base, support_exit=kind, support_exit_gap=gap, support_exit_after=after)
                show(f"支撑{kind} 余量{gap} 持{after}分后", rule)
    print("== 收盘回撤 + 止损放宽")
    for trail in (0.12, 0.15, 0.18):
        for stop in (0.07, 0.08, 0.10):
            show(f"收盘回撤{trail} 止损{stop}", replace(base, trail_on_close=1, trail=trail, stop=stop))


if __name__ == "__main__":
    main()
