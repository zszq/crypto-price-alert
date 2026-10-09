"""第二轮最终排名：基准（等 10 分钟 +2% 入场、止损 7%、回撤 15%、持仓 5 小时）与 13 个改进方案的对比。

--period 选区间：half 为选参数用的半年，year / two_years 用来检验这些方案在选参数时没用过的数据上是否还成立。
除合计外给出各段、去掉最好 10 笔、t 值、胜率、最大回撤、亏损月，以及换 12 种等待设置后的平均（见 r2_delays.py），
后者更接近实际能期待的水平。
"""

import argparse
import sys
import unicodedata
from dataclasses import replace

import numpy as np
from r2_delays import DELAYS
from r2_lib import BEST_DIP, PERIODS, SUPPORT_STOP, Engine, Rule, evaluate, fold_sums, load, wait_dip

A = dict(**SUPPORT_STOP, trail_atr=5, hold=600)
# 方案名写明与基准或 A 的区别，表里一眼能看出各方案改了什么；各条件的含义见 LEGEND。
SCHEMES = [
    ("基准：等10分钟+2%入场，止损7%，回撤15%，持5h", Rule(), None, False),
    ("A：基准 + 回踩≥0.25% + 支撑止损 + 回撤≥5倍振幅 + 持10h", Rule(**A), BEST_DIP, False),
    ("B：A + 涨100%时先平一半", Rule(**A, take=1.0, take_fraction=0.5), BEST_DIP, False),
    ("C：A 但回撤固定15%（不按振幅放宽）", Rule(**SUPPORT_STOP, hold=600), BEST_DIP, False),
    (
        "D：基准 + 回踩≥0.25% + 不追高20% + 回撤≥5倍振幅 + 持7h",
        Rule(max_extension=0.2, trail_atr=5, hold=420),
        BEST_DIP,
        False,
    ),
    ("E：A + 涨50%时先平一半", Rule(**A, take=0.5, take_fraction=0.5), BEST_DIP, False),
    ("F：基准 + 回踩≥0.25% + 持7h", Rule(hold=420), BEST_DIP, False),
    ("G：A + 涨30%时先平一半", Rule(**A, take=0.3, take_fraction=0.5), BEST_DIP, False),
    ("H：A + 不追高15%", Rule(**A, max_extension=0.15), BEST_DIP, False),
    ("I：基准 + 回踩≥0.25%", Rule(), BEST_DIP, False),
    ("J：A 去掉回踩（基准 + 支撑止损 + 回撤≥5倍振幅 + 持10h）", Rule(**A), None, False),
    ("K：基准 + 持10h", Rule(hold=600), None, False),
    ("L：基准 + 不追高15%", Rule(max_extension=0.15), None, False),
    ("M：A 的规则，所有波次都买", Rule(**A), BEST_DIP, True),
]
LEGEND = (
    "条件说明：回踩≥0.25% = 等待期最低价至少比提醒价低 0.25%；"
    "支撑止损 = 止损放在等待期最低价下方 2%，离入场价最多 9%；"
    "回撤≥5倍振幅 = 回撤幅度取 15% 与入场前 60 分钟平均振幅×5 的较大者；"
    "不追高 X% = 入场价比启动前高超过 X% 不买"
)


def display_width(text: str) -> int:
    """终端显示宽度：中文和全角符号占两格，按字符数对齐的话中文多的行会错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 方案是在半年里选出的；一年、两年用来看这些方案在选参数时没用过的数据上是否还成立。
    parser.add_argument("--period", choices=sorted(PERIODS), default="half")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = load(args.period)
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
    labels = [label for label, _ in data.period.folds]
    days = (data.closed_at.max() - data.period.start) / 86400
    print(f"区间：{data.period.name}（约 {days:.0f} 天），第 1 波提醒 {int(first.sum())} 条")
    print(LEGEND)
    width = max(display_width(name) for name, *_ in SCHEMES) + 1
    print(
        pad("方案", width)
        + f"{'笔数':>4s} {'合计':>6s} "
        + " ".join(f"{label:>10s}" for label in labels)
        + f" {'去10':>6s} {'t':>5s} {'胜率':>5s} {'回撤':>5s} {'亏损月':>5s} {'换等待平均':>8s} {'胜基准':>6s} 各段都超"
    )
    for name, res, folds, dd, average, wins, all_folds in rows:
        print(
            pad(name, width)
            + f"{res['all_n']:4d} {res['all_sum']:+6.0f} "
            + " ".join(f"{value:+10.0f}" for value in folds)
            + f" {res['minus10']:+6.0f} {res['t']:5.2f} {res['win']:5.0%} {dd:5.0f} {res['neg_months']:5d} "
            f"{average:+8.0f} {wins:3d}/12 {'是' if all_folds else '否'}"
        )


if __name__ == "__main__":
    main()
