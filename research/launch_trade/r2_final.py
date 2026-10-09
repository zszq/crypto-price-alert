"""第二轮最终排名：比基准（等 10 分钟 +2% 入场、止损 7%、回撤 15%、持仓 5 小时）盈利更高的方案。

除合计外同时给出三段、去掉最好 10 笔、t 值、最大回撤，以及换 12 种等待设置后的平均（见 r2_delays.py），
后者更接近实际能期待的水平。
"""

from dataclasses import replace

import numpy as np
from r2_delays import DELAYS
from r2_lib import BEST_DIP, SUPPORT_STOP, Engine, Rule, evaluate, fold_sums, load, wait_dip

A = dict(**SUPPORT_STOP, trail_atr=5, hold=600)
SCHEMES = [
    ("基准：止损7% 回撤15% 持5h", Rule(), None, False),
    ("A 最稳组：回踩0.25% + 支撑止损 + ATR回撤 + 持10h", Rule(**A), BEST_DIP, False),
    ("B A + 涨100%先平一半", Rule(**A, take=1.0, take_fraction=0.5), BEST_DIP, False),
    ("C A 不用ATR回撤", Rule(**SUPPORT_STOP, hold=600), BEST_DIP, False),
    (
        "D 回踩0.25% + 不追20% + 止损7% + ATR回撤 + 持7h",
        Rule(max_extension=0.2, trail_atr=5, hold=420),
        BEST_DIP,
        False,
    ),
    ("E A + 涨50%先平一半", Rule(**A, take=0.5, take_fraction=0.5), BEST_DIP, False),
    ("F 回踩0.25% + 持7h（其余同基准）", Rule(hold=420), BEST_DIP, False),
    ("G A + 涨30%先平一半", Rule(**A, take=0.3, take_fraction=0.5), BEST_DIP, False),
    ("H A + 不追15%", Rule(**A, max_extension=0.15), BEST_DIP, False),
    ("I 回踩0.25%（其余同基准）", Rule(), BEST_DIP, False),
    ("J A 不要求回踩", Rule(**A), None, False),
    ("K 持10h（其余同基准）", Rule(hold=600), None, False),
    ("L 不追15%（其余同基准）", Rule(max_extension=0.15), None, False),
    ("M 所有波次 + A 的规则", Rule(**A), BEST_DIP, True),
]


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1

    def run(rule: Rule, dip: float | None, all_waves: bool, delay: int = 10, confirm: float = 0.02):
        net, _, exit_at = engine.run(replace(rule, delay=delay, confirm=confirm))
        mask = np.ones_like(first) if all_waves else first
        if dip is not None:
            mask = mask & (wait_dip(data, delay) <= -dip)
        return net, exit_at, mask

    def drawdown(net: np.ndarray, exit_at: np.ndarray, mask: np.ndarray) -> float:
        # 等额本金、按平仓时间累计盈亏，从最高点回落的最大值。
        ok = mask & ~np.isnan(net)
        order = np.argsort(data.closed_at[ok] + exit_at[ok] * 60)
        curve = np.cumsum(net[ok][order]) * 100
        return float(np.max(np.maximum.accumulate(np.concatenate([[0], curve]))[1:] - curve))

    rows = []
    base_delays = base_folds = None
    for name, rule, dip, all_waves in SCHEMES:
        net, exit_at, mask = run(rule, dip, all_waves)
        res = evaluate(net, mask, data)
        folds = np.array(fold_sums(net, mask, data))
        per_delay = []
        for delay, confirm in DELAYS:
            delay_net, _, delay_mask = run(rule, dip, all_waves, delay, confirm)
            per_delay.append(np.nansum(delay_net[delay_mask]) * 100)
        per_delay = np.array(per_delay)
        if base_delays is None:
            base_delays, base_folds = per_delay, folds
        rows.append(
            (
                name,
                res,
                folds,
                drawdown(net, exit_at, mask),
                per_delay.mean(),
                int((per_delay > base_delays).sum()),
                bool((folds > base_folds).all()),
            )
        )
    rows = [rows[0]] + sorted(rows[1:], key=lambda r: -r[1]["all_sum"])
    print(
        f"{'方案':44s} {'笔数':>4s} {'合计':>6s} {'4-5月':>6s} {'6-7月':>6s} {'8月起':>6s} {'去10':>6s} "
        f"{'t':>5s} {'回撤':>5s} {'换等待平均':>8s} {'胜基准':>6s} 三段都超"
    )
    for name, res, folds, dd, average, wins, all_folds in rows:
        print(
            f"{name:44s} {res['all_n']:4d} {res['all_sum']:+6.0f} {folds[0]:+6.0f} {folds[1]:+6.0f} {folds[2]:+6.0f} "
            f"{res['minus10']:+6.0f} {res['t']:5.2f} {dd:5.0f} {average:+8.0f} {wins:3d}/12 "
            f"{'是' if all_folds else '否'}"
        )


if __name__ == "__main__":
    main()
