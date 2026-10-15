"""量比按成交额（检测器口径）还是成交量（币数）算：提醒集合差异与 S5 交易结果（2026-10-15，结论见 FINDINGS.md）。

两种口径的数据集用 r5_export.py --ratio-by quote / base 导出，目录名为 <前缀>_quote、<前缀>_base；
scan 再按 <前缀>_<口径>_r<门槛> 读其余量比门槛的导出（3.0 用 <前缀>_<口径>）。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r5_ratio.py diff rv_p30 rv_p30_1440 rv_user
    & $P research/launch_trade/r5_ratio.py scan rv_p30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
from r5_studies import CANDIDATE, DEV_START, HOLDOUT_START, HOT, LONG, Book, Scheme, show, summary

PERIODS = (("样本外", HOLDOUT_START, DEV_START), ("开发期", DEV_START, 2**62))
RATIOS = ("2.5", "2.75", "3.0", "3.25", "3.5")
SIDES = (("quote", "成交额"), ("base", "成交量"))


def _keys(book: Book) -> np.ndarray:
    d = book.data
    return np.array([f"{s}|{t}" for s, t in zip(d.symbol, d.closed_at, strict=True)])


def _per_trade(book: Book, mask: np.ndarray, label: str) -> str:
    # 不限仓逐笔看，避免限仓时满仓跳过把两边的差异提醒混进别的提醒里。
    tr = book.trades(LONG)
    d = book.data
    m = mask & book.mask(Scheme("过热不买", LONG, filters=HOT)) & ~np.isnan(tr.net)
    parts = []
    for name, lo, hi in PERIODS:
        net = tr.net[m & (d.closed_at >= lo) & (d.closed_at < hi)]
        if len(net):
            parts.append(f"{name} {len(net)} 笔 胜率 {(net > 0).mean():.0%} 每笔 {net.mean() * 100:+.2f}%")
        else:
            parts.append(f"{name} 0 笔")
    return f"  {label}：" + "；".join(parts)


def study_diff(prefixes: list[str]) -> None:
    """同一门槛下两种口径的提醒差异、差异提醒的逐笔结果与 S5 组合结果。"""
    for prefix in prefixes:
        q, b = Book(prefix + "_quote"), Book(prefix + "_base")
        print(f"\n===== {prefix} =====")
        kq, kb = _keys(q), _keys(b)
        in_b, in_q = np.isin(kq, kb), np.isin(kb, kq)
        for name, lo, hi in PERIODS:
            mq = (q.data.closed_at >= lo) & (q.data.closed_at < hi)
            mb = (b.data.closed_at >= lo) & (b.data.closed_at < hi)
            print(f"{name}：共同 {(mq & in_b).sum()}，只在成交额 {(mq & ~in_b).sum()}，只在成交量 {(mb & ~in_q).sum()}")
        ok = q.data.meta["ratio"] < 1e5  # 基准期成交额为 0 时 ratio 记为 1e6
        pct = np.percentile(q.data.meta["ratio_base"][ok] / q.data.meta["ratio"][ok], [5, 25, 50, 75, 95])
        print("成交量量比 ÷ 成交额量比（成交额口径的提醒）P5/25/50/75/95：" + " ".join(f"{x:.3f}" for x in pct))
        print(_per_trade(q, ~in_b, "只在成交额口径"))
        print(_per_trade(b, ~in_q, "只在成交量口径"))
        rows = []
        for slots in (5, 10):
            for book, tag in ((q, "成交额"), (b, "成交量")):
                rows.append(summary(book, replace(CANDIDATE, name=f"{tag} S5 限{slots}", slots=slots)))
        show(rows)


def study_scan(prefixes: list[str]) -> None:
    """两种口径 × 量比门槛：提醒数与 S5 结果，在提醒数接近处比较口径（同样门槛下成交量口径更严）。"""
    for prefix in prefixes:
        rows: dict[int, list[dict]] = {5: [], 10: []}
        for side, tag in SIDES:
            for ratio in RATIOS:
                # 门槛 3.0 就是 diff 用的默认导出，不重复导出一份。
                book = Book(f"{prefix}_{side}" if ratio == "3.0" else f"{prefix}_{side}_r{ratio}")
                t = book.data.closed_at
                counts = "/".join(str(int(((t >= lo) & (t < hi)).sum())) for _, lo, hi in PERIODS)
                for slots, out in rows.items():
                    out.append(
                        summary(book, replace(CANDIDATE, name=f"{tag} ≥{ratio} 限{slots}（{counts}）", slots=slots))
                    )
        for slots, out in rows.items():
            print(f"\n--- {prefix} 限 {slots} 仓 ---")
            show(out)


STUDIES = {"diff": study_diff, "scan": study_scan}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("prefixes", nargs="+", help="数据集前缀")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    STUDIES[args.study](args.prefixes)


if __name__ == "__main__":
    main()
