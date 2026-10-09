"""第二轮其他尝试（在最稳组 A 上）：部分止盈、所有波次、成本压力、第二次入场机会、回踩守突破位、回踩缩量，
以及在 A 上再加一条单边特征过滤。"""

from dataclasses import replace

import numpy as np
from r2_lib import BEST, BEST_DIP, PRE, Engine, Rule, alert_features, fold_line, fold_sums, load, wait_dip


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    bars = data.bars
    base = np.array(fold_sums(engine.run(Rule())[0], first, data))
    dip = wait_dip(data)
    mask = first & (dip <= -BEST_DIP)
    net = engine.run(BEST)[0]

    def show(label: str, values: np.ndarray, selected: np.ndarray) -> None:
        print(fold_line(label, values, selected, data, base))

    show("最稳组 A", net, mask)
    print("== 部分止盈（先平一部分，剩下按回撤跑）")
    for take in (0.2, 0.3, 0.5, 1.0):
        for fraction in (0.3, 0.5):
            show(f"涨{take:.0%}平{fraction:.0%}", engine.run(replace(BEST, take=take, take_fraction=fraction))[0], mask)
    print("== 所有波次")
    show("所有波次", net, dip <= -BEST_DIP)
    show("第 2 波及以后", net, (data.wave >= 2) & (dip <= -BEST_DIP))
    print("== 成本压力（单边）")
    for fee in (0.001, 0.0015, 0.002):
        show(f"单边成本{fee:.2%}", engine.run(replace(BEST, fee=fee))[0], mask)
    show("基准 单边0.15%", engine.run(Rule(fee=0.0015))[0], first)

    print("== 第二次入场机会：10 分钟没确认的，20/30 分钟再看一次")
    entered = mask & ~np.isnan(net)
    for delay in (20, 30):
        later = engine.run(replace(BEST, delay=delay))[0]
        second = first & ~entered & (wait_dip(data, delay) <= -BEST_DIP) & ~np.isnan(later)
        show(f"10 分钟 + {delay} 分钟", np.where(entered, net, later), entered | second)
        show(f"  只看 {delay} 分钟才进的", later, second)
    print("== 回踩守住突破位（等待期最低 ≥ 基准期高点 × (1-x)）")
    wait_low = np.nanmin(bars[:, 2, PRE : PRE + 10], 1)
    for x in (0.0, 0.01, 0.02, 0.05):
        held = wait_low >= data.meta["breakout"] * (1 - x)
        show(f"守住突破位-{x:.0%}", net, mask & held)
        show(f"跌破突破位-{x:.0%}", net, mask & ~held)
    print("== 回踩缩量：最低价那一分钟成交额 ÷ 等待期平均")
    lowest = np.nanargmin(bars[:, 2, PRE : PRE + 10], 1)
    relative = bars[np.arange(len(lowest)), 4, PRE + lowest] / np.nanmean(bars[:, 4, PRE : PRE + 10], 1)
    for low, high in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        show(f"回踩分钟量 {low}~{high} 倍", net, mask & (relative >= low) & (relative < high))

    print("== 在 A 上再加一条单边特征过滤（只去掉一头 10%~30%），三段都比 A 好的：")
    reference = np.array(fold_sums(net, mask, data))
    traded = mask & ~np.isnan(net)
    tested = 0
    for name, x in alert_features(data).items():
        for q in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9):
            threshold = np.nanquantile(x[traded], q)
            keep = x >= threshold if q < 0.5 else x <= threshold
            tested += 1
            folds = np.array(fold_sums(net, mask & keep, data))
            if (folds > reference).all():
                side = "≥" if q < 0.5 else "≤"
                print(
                    f"  {name} {side} {threshold:.4g} 三段 {folds.round()} "
                    f"合{folds.sum():+.0f}（A 合 {reference.sum():+.0f}）"
                )
    print(f"  共试 {tested} 个过滤条件；只有一两个过关属于多次尝试下的偶然，不采用。")


if __name__ == "__main__":
    main()
