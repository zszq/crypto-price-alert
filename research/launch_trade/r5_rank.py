"""历次方案统一重排（2026-10-15，结论见 FINDINGS.md「历次方案统一重排」）。

第三、四轮 r3_final.py 的 12 个方案与第五轮的 S5 等方案，按第五轮的口径放在一起比较
（含时段过滤的方案是过拟合，已从 r3_final.py 删除，这里也不测）。
提醒规则对应的数据集（r5_export.py 导出）：
- r5_base：原默认规则（基准 360、窗口 10 万、要求突破），对应第三轮 v_base；
- r5_nobreak：同上但不要求突破，对应 v_nobreak；
- r5_p：P 规则（不要求突破、池下限 0、不设插针上限），对应 v_nb_pool0_noover；
- r5_p30 / r5_p30_1440 / r5_user：第五轮的窗口 30 万规则与导出时的 YAML。

每个方案给出：不限仓（每笔 100 USDT，第三、四轮的口径）样本外 / 开发期合计；限 5、限 10 仓的样本外 / 开发期本金收益率、
开发期去最好 5 天、五年夏普与回撤、亏损半年段数；以及最近半年（持仓数据覆盖）限 5 仓加「持仓价值 ≥ 100 万」前后的收益。
排名按检验协议只用开发期：限 5 仓开发期夏普；样本外只看是否同向。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r5_export.py --out r5_nobreak --side up --set baseline_minutes=360 `
        --set min_window_quote=100000 --set require_breakout=false
    & $P research/launch_trade/r5_rank.py
    & $P research/launch_trade/r5_rank.py --dataset r5_nobreak   # 基准、窗口取代码默认值（360、10 万）
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass, replace

import numpy as np
from r2_final import pad
from r5_oi import START, _book, _book_core_hot, _period_totals
from r5_studies import DEV_START, HOLDOUT_START, HOT, LONG, X0, Book, J, Scheme, summary

X = replace(X0, confirm=0.02)
# 第三轮在 v_nb_pool0_noover 上重扫出场得到的 Y：止损余量 8%、回撤 ≥ 5 倍振幅。
Y0 = replace(X0, stop_gap=0.08, trail_atr=5)
HOT85 = (("宽度30", -np.inf, 0.85),)
HOT_CORE = (("过热J类", -np.inf, 0.5),)


@dataclass(frozen=True)
class Candidate:
    name: str
    dataset: str
    scheme: Scheme


def _c(name: str, dataset: str, rule, all_waves: bool, filters=()) -> Candidate:
    return Candidate(name, dataset, Scheme(name, rule, all_waves=all_waves, filters=filters))


CANDIDATES = [
    _c("J：J 首波", "r5_base", J, False),
    _c("J1：J 所有波次", "r5_base", J, True),
    _c("X1：出场X 首波", "r5_base", X, False),
    _c("J4：J 首波 过热85不买", "r5_base", J, False, HOT85),
    _c("N1：不突破 J 首波", "r5_nobreak", J, False),
    _c("N2：不突破 X 首波", "r5_nobreak", X, False),
    _c("N4：N2 确认0%", "r5_nobreak", X0, False),
    _c("P1：P规则 X 首波", "r5_p", X, False),
    _c("P2：P1 确认0%", "r5_p", X0, False),
    _c("P3：P2 所有波次", "r5_p", X0, True),
    _c("P4：P3 出场Y", "r5_p", Y0, True),
    _c("P5：P3 过热85时不买J类", "r5_p", X0, True, HOT_CORE),
    _c("P规则 + S5出场 过热不买", "r5_p", LONG, True, HOT),
    _c("S5：P30 S5出场 过热不买", "r5_p30", LONG, True, HOT),
    _c("S5 不过滤过热", "r5_p30", LONG, True),
    _c("P30 + P5 规则（X0 过热85时不买J类）", "r5_p30", X0, True, HOT_CORE),
    _c("S5 基准1440", "r5_p30_1440", LONG, True, HOT),
    _c("YAML(1440/20万) S5出场 过热不买", "r5_user", LONG, True, HOT),
    _c("YAML(1440/20万) S5出场", "r5_user", LONG, True),
]


def _unlimited(book: Book, scheme: Scheme) -> tuple[float, float]:
    tr = book.trades(scheme.rule)
    d = book.data
    ok = book.mask(scheme) & ~np.isnan(tr.net)
    holdout = ok & (d.closed_at >= HOLDOUT_START) & (d.closed_at < DEV_START)
    dev = ok & (d.closed_at >= DEV_START)
    return float(tr.net[holdout].sum() * 100), float(tr.net[dev].sum() * 100)


def _recent(dataset: str, schemes: list[Scheme]) -> list[tuple[float, float, float, float]]:
    """最近半年限 5 仓：不过滤与持仓价值 ≥ 100 万的收益和回撤（持仓数据只覆盖这半年）。"""
    book = _book(dataset)
    d = book.data
    value = book.features["持仓价值"]
    out = []
    for scheme in schemes:
        tr = book.trades(scheme.rule)
        mask = book.mask(scheme) & book.has_oi & (d.closed_at >= START)
        plain = _period_totals(book, tr, mask, 5)
        oi = _period_totals(book, tr, mask & (value >= 1e6), 5)
        out.append((plain[0], plain[-1], oi[0], oi[-1]))
    del book
    gc.collect()
    return out


# 同一份提醒上，12 个方案只剩交易规则的区别，相同的合并成 8 个，名字列出合并了哪些。再加两行第五轮 S5 的出场作参照。
SAME_DATA = [
    ("J 首波（J、N1）", J, False, ()),
    ("J 所有波次（J1）", J, True, ()),
    ("X 首波（X1、N2、P1）", X, False, ()),
    ("X0 首波（N4、P2）", X0, False, ()),
    ("X0 所有波次（P3）", X0, True, ()),
    ("Y0 所有波次（P4）", Y0, True, ()),
    ("J 首波 过热85不买（J4）", J, False, HOT85),
    ("X0 所有波次 过热85时不买J类（P5）", X0, True, HOT_CORE),
    ("参照：S5 出场 过热不买", LONG, True, HOT),
    ("参照：S5 出场 不过滤过热", LONG, True, ()),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        help="只用这一份提醒（如 r5_nobreak）跑 12 个方案的交易规则；不给时按各方案原来的提醒规则",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    candidates = CANDIDATES
    if args.dataset:
        candidates = [_c(name, args.dataset, rule, waves, flt) for name, rule, waves, flt in SAME_DATA]
    rows: dict[str, dict] = {}
    datasets = list(dict.fromkeys(c.dataset for c in candidates))
    for dataset in datasets:
        group = [c for c in candidates if c.dataset == dataset]
        book = Book(dataset)
        book.features["过热J类"] = _book_core_hot(book)
        for c in group:
            row = {"name": c.name, "unlimited": _unlimited(book, c.scheme)}
            for slots in (5, 10):
                s = summary(book, replace(c.scheme, slots=slots))
                row[slots] = s
            whole = row[5]["全部"]
            row["sharpe5y"] = whole.sharpe
            row["dd5y"] = whole.max_dd
            row["neg_halves"] = int((row[5]["halves"][:-1] < 0).sum())
            rows[c.name] = row
            print(f"  已算 {c.name}", file=sys.stderr, flush=True)
        del book
        gc.collect()
        for c, recent in zip(group, _recent(dataset, [c.scheme for c in group]), strict=True):
            rows[c.name]["recent"] = recent

    ordered = sorted(rows.values(), key=lambda r: -r[5]["开发期"].sharpe)
    print("排名按限 5 仓开发期夏普；收益为本金 %（不复利），不限仓为每笔 100 USDT 的合计。")
    print(
        pad("方案", 40)
        + " | 不限仓 样本外/开发期 | 限5 样本外 开发期 开发去5天 开发夏普 五年夏普 五年回撤 亏损半年"
        + " | 限10 样本外 开发期 | 近半年限5 不过滤(回撤) → 持仓≥100万(回撤)"
    )
    for r in ordered:
        u = r["unlimited"]
        s5, s10 = r[5], r[10]
        rec = r["recent"]
        print(
            pad(r["name"], 40)
            + f" | {u[0]:+7.0f} / {u[1]:+7.0f}"
            + f" | {s5['样本外'].total:+6.0f} {s5['开发期'].total:+6.0f} {s5['开发期去5天']:+6.0f}"
            + f" {s5['开发期'].sharpe:5.2f} {r['sharpe5y']:5.2f} {r['dd5y']:5.0f}"
            + f" {r['neg_halves']:3d}/{len(s5['halves']) - 1}"
            + f" | {s10['样本外'].total:+6.0f} {s10['开发期'].total:+6.0f}"
            + f" | {rec[0]:+5.0f}({rec[1]:3.0f}) → {rec[2]:+5.0f}({rec[3]:3.0f})"
        )
    print("\n各方案限 5 仓每半年（本金 %）：21/10 22/04 22/10 23/04 23/10 24/04 24/10 25/04 25/10 26/04")
    for r in ordered:
        print(pad(r["name"], 40) + " " + " ".join(f"{x:+5.0f}" for x in r[5]["halves"]))


if __name__ == "__main__":
    main()
