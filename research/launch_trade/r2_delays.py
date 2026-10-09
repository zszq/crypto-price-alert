"""第二轮：结果对等待分钟很敏感（基准等 8 分钟 +111、10 分钟 +611、12 分钟 +388），
所以把每项改进放到 12 种等待分钟 / 确认涨幅下，与同设置的基准比较。
每种设置都胜过基准，才说明改进不是 10 分钟这个点的运气。
"""

from dataclasses import replace

import numpy as np
from r2_lib import BEST_DIP, SUPPORT_STOP, Engine, Rule, load, wait_dip

DELAYS = [
    (5, 0.02),
    (7, 0.02),
    (8, 0.02),
    (10, 0.02),
    (12, 0.02),
    (15, 0.02),
    (20, 0.02),
    (30, 0.02),
    (10, 0.015),
    (10, 0.025),
    (15, 0.03),
    (20, 0.03),
]
VARIANTS = {
    "基准": (Rule(), None),
    "+回踩0.25%": (Rule(), BEST_DIP),
    "+支撑止损": (Rule(**SUPPORT_STOP), None),
    "+atr5": (Rule(trail_atr=5), None),
    "+持10h": (Rule(hold=600), None),
    "+持7h": (Rule(hold=420), None),
    "+不追20%": (Rule(max_extension=0.20), None),
    "最稳组 A": (Rule(**SUPPORT_STOP, trail_atr=5, hold=600), BEST_DIP),
    "A 不含回踩": (Rule(**SUPPORT_STOP, trail_atr=5, hold=600), None),
    "A + 涨30%平一半": (Rule(**SUPPORT_STOP, trail_atr=5, hold=600, take=0.3, take_fraction=0.5), BEST_DIP),
}


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    header = "".join(f"{f'{delay}分+{confirm:.1%}':>10s}" for delay, confirm in DELAYS)
    print(f"各等待设置下的合计（第 1 波，扣费）\n{'':22s}{header}   平均  胜基准")
    base_row = None
    for name, (rule, dip) in VARIANTS.items():
        values = []
        for delay, confirm in DELAYS:
            net = engine.run(replace(rule, delay=delay, confirm=confirm))[0]
            mask = first if dip is None else first & (wait_dip(data, delay) <= -dip)
            values.append(np.nansum(net[mask]) * 100)
        values = np.array(values)
        if base_row is None:
            base_row = values
        print(
            f"{name:20s}"
            + "".join(f"{v:+10.0f}" for v in values)
            + f"  {values.mean():+6.0f}  {(values > base_row).sum()}/{len(values)}"
        )


if __name__ == "__main__":
    main()
