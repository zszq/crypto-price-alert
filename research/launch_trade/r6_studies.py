"""第六轮研究：去掉时间出场，只用止损 + 移动止盈，重新选入场、止损与移动止盈（子命令见 STUDIES）。

检验协议与第五轮相同：2024-10-10 起为开发期（选参数），2021-10-10 ~ 2024-09-30 为样本外（只看不选）；
限仓组合口径（最多同时 5 笔、每笔本金 1/5、同一合约不叠仓）、过热不买、所有波次。
提醒只取到数据结束前 15 天，持仓满 15 天按市价平仓（见 r6_lib）。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r6_studies.py <研究...> [--dataset r6_p30]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
from r5_lib import BEIJING
from r5_studies import DEV_START, HOLDOUT_START, HOT, Book, Scheme, show, summary
from r6_lib import HOLD_CAP, Exit, simulate


class Book6(Book):
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset)
        # 统计截止到数据结束前 15 天：之后的提醒没有完整的 15 天持仓窗口，不计。模拟时仍用真实截止判断数据中断。
        self.real_end = self.data.end
        self.data.end = self.real_end - HOLD_CAP * 60

    def trades(self, rule: Exit):
        key = rule.params()
        if key not in self._cache:
            self._cache[key] = simulate(replace(self.data, end=self.real_end), rule)
        return self._cache[key]

    def mask(self, scheme: Scheme) -> np.ndarray:
        return super().mask(scheme) & (self.data.closed_at < self.data.end)


S6 = Exit()  # S5 的入场与出场，去掉 18 小时平仓
BASE = Scheme("S5 无时间出场", S6, slots=5, filters=HOT)
CLOSE15 = replace(S6, close_trigger=1, close_tf=15)


def diag(book: Book6, scheme: Scheme) -> str:
    """持仓时长、持满 15 天的笔数、出场回吐（最大浮盈 − 实际收益）。"""
    tr, sel = book.selected(scheme)
    ok = sel & ~np.isnan(tr.net)
    held = (tr.t_out[ok] - tr.t_in[ok]) / 3600
    give = (tr.peak[ok] - tr.net[ok]) * 100
    win = tr.net[ok] > 0
    return (
        f"{scheme.name}：{ok.sum()} 笔，胜率 {win.mean():.0%}，持仓中位 {np.median(held):.1f}h、"
        f"平均 {held.mean():.1f}h、P95 {np.percentile(held, 95):.0f}h，持满15天 {(tr.status[ok] == 2).sum()} 笔，"
        f"盈利单回吐中位 {np.median(give[win]):.1f} 个百分点，每笔 {tr.net[ok].mean() * 100:+.2f}%"
    )


def run(book: Book6, schemes: list[Scheme], details: bool = False) -> None:
    show([summary(book, s) for s in schemes])
    if details:
        for s in schemes:
            print(diag(book, s))
    sys.stdout.flush()


def variant(name: str, **kw) -> Scheme:
    return replace(BASE, name=name, rule=replace(S6, **kw))


def study_base(book: Book6) -> None:
    """去掉 18 小时平仓的影响。"""
    run(
        book,
        [variant("S5（持18h）", hold=1080), variant("S5 持24h", hold=1440), variant("S5 持3天", hold=4320), BASE],
        details=True,
    )


def study_trail(book: Book6) -> None:
    """固定回撤比例 × 入场前振幅倍数。"""
    rows = []
    for trail in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
        for k in (np.nan, 2.0, 3.0, 5.0):
            rows.append(variant(f"回撤{trail:.0%} 振幅×{k:g}", trail=trail, trail_atr=k))
    run(book, rows)


def study_dyn(book: Book6) -> None:
    """动态振幅（吊灯式）：回撤幅度跟持仓中最近 N 分钟的平均振幅走。"""
    rows = [BASE]
    for floor in (0.05, 0.08, 0.10):
        for n in (30, 60, 240):
            for k in (3.0, 5.0, 8.0, 12.0):
                rows.append(variant(f"下限{floor:.0%} 近{n}分振幅×{k:g}", trail=floor, dyn_atr=k, dyn_n=n))
    run(book, rows)


def study_start(book: Book6) -> None:
    """浮盈达到一定幅度才启用移动止盈（之前只有初始止损）。"""
    rows = [BASE]
    for start in (0.03, 0.05, 0.10, 0.15, 0.20):
        for trail in (0.08, 0.10, 0.15):
            rows.append(variant(f"浮盈≥{start:.0%}启用 回撤{trail:.0%}", trail_start=start, trail=trail))
    run(book, rows)


def study_tight(book: Book6) -> None:
    """浮盈达到一定幅度后改变回撤幅度（收紧锁利润，或放宽让大行情跑）。"""
    rows = [BASE]
    for at in (0.10, 0.20, 0.30, 0.50, 1.00):
        for tt in (0.05, 0.07, 0.15, 0.20, 0.30):
            rows.append(variant(f"浮盈≥{at:.0%}后回撤{tt:.0%}", tight_at=at, tight_trail=tt))
    run(book, rows)


def study_lock(book: Book6) -> None:
    """锁定最大浮盈的一定比例。"""
    rows = [BASE]
    for start in (0.05, 0.10, 0.20, 0.30):
        for lock in (0.3, 0.5, 0.7):
            rows.append(variant(f"浮盈≥{start:.0%}后锁{lock:.0%}", trail_start=start, lock=lock))
    run(book, rows)


def study_be(book: Book6) -> None:
    """浮盈达到一定幅度后止损提到保本。"""
    rows = [BASE]
    for at in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        rows.append(variant(f"浮盈≥{at:.0%}保本", be_at=at))
    run(book, rows)


def study_close(book: Book6) -> None:
    """移动止盈按收盘价触发、下一分钟开盘平仓（避开插针），各回撤幅度对照。"""
    rows = []
    for trail in (0.06, 0.08, 0.10, 0.15):
        for close in (0, 1):
            label = f"回撤{trail:.0%} {'收盘触发' if close else '盘中触发'}"
            rows.append(variant(label, trail=trail, close_trigger=close))
    run(book, rows)


def study_closetf(book: Book6) -> None:
    """按时钟对齐的大周期 K 线收盘价触发移动止盈。"""
    rows = [BASE]
    for tf in (1, 5, 15, 60):
        for trail in (0.06, 0.08, 0.10, 0.12, 0.15):
            rows.append(variant(f"{tf}分钟收盘 回撤{trail:.0%}", trail=trail, close_trigger=1, close_tf=tf))
    run(book, rows, details=True)


def study_slots(book: Book6) -> None:
    """没有时间平仓后持仓变长，限仓数的影响。"""
    rows = []
    for slots in (3, 5, 8, 10):
        for name, rule in (("S5 持18h", replace(S6, hold=1080)), ("无时间", S6), ("15分收盘 回撤10%", CLOSE15)):
            rows.append(replace(BASE, name=f"{name} 限{slots}", rule=rule, slots=slots))
    run(book, rows)


TRIGGERS = {"盘中": {}, "1分收盘": {"close_trigger": 1}, "5分收盘": {"close_trigger": 1, "close_tf": 5}}
TRIGGERS["15分收盘"] = {"close_trigger": 1, "close_tf": 15}
WIDTHS = {
    "固定": {"trail_atr": np.nan},
    "入场前振幅×3": {},
    "近60分振幅×3": {"dyn_atr": 3.0, "dyn_n": 60},
    "近240分振幅×5": {"dyn_atr": 5.0, "dyn_n": 240},
}
STOPS = {"等待低点下5%": {}, "固定10%": {"stop_gap": 1.0, "stop_max": 0.10}}


def grid_rules():
    for trig, kt in TRIGGERS.items():
        for trail in (0.08, 0.10, 0.12):
            for width, kw in WIDTHS.items():
                hards = (np.nan,) if trig == "盘中" else (np.nan, 0.15, 0.20)
                for hard in hards:
                    for stop, ks in STOPS.items():
                        name = f"{trig} 回撤{trail:.0%} {width} 硬{hard:.0%} {stop}".replace("硬nan%", "无硬线")
                        yield name, replace(S6, trail=trail, hard_trail=hard, **kt, **kw, **ks)


def study_grid(book: Book6) -> None:
    """触发方式 × 回撤幅度 × 幅度放宽方式 × 盘中硬回撤 × 止损，两段分别排名，结果另存 CSV。"""
    import csv

    from paths import RESULTS

    rows = []
    for name, rule in grid_rules():
        r = summary(book, replace(BASE, name=name, rule=rule))
        rows.append(r)
    show(sorted(rows, key=lambda r: -r["开发期"].sharpe)[:30])
    print("按样本外夏普：")
    show(sorted(rows, key=lambda r: -r["样本外"].sharpe)[:30])
    with (RESULTS / "r6_grid.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        out = csv.writer(handle)
        out.writerow(
            [
                "方案",
                "样本外",
                "样本外去5天",
                "样本外夏普",
                "样本外回撤",
                "开发期",
                "开发去5天",
                "开发夏普",
                "开发回撤",
                "五年夏普",
                "亏损半年",
            ]
        )
        for r in rows:
            a, d, w = r["样本外"], r["开发期"], r["全部"]
            out.writerow(
                [
                    r["name"],
                    round(a.total),
                    round(r["样本外去5天"]),
                    round(a.sharpe, 2),
                    round(a.max_dd),
                    round(d.total),
                    round(r["开发期去5天"]),
                    round(d.sharpe, 2),
                    round(d.max_dd),
                    round(w.sharpe, 2),
                    int((r["halves"] < 0).sum()),
                ]
            )


# 网格里两段都靠前的一族：5 分钟收盘价跌破回撤线才平，盘中另设 20% 硬回撤，固定 10% 止损。
C6 = replace(S6, close_trigger=1, close_tf=5, trail=0.08, hard_trail=0.20, stop_gap=1.0, stop_max=0.10)
CAND = Scheme("C6", C6, slots=5, filters=HOT)


def study_hood(book: Book6) -> None:
    """候选的邻域：每次只改一个参数。"""
    rows = [CAND, replace(BASE, name="S5 持18h", rule=replace(S6, hold=1080))]
    for field_name, values in (
        ("delay", (3, 4, 6, 7, 10)),
        ("confirm", (0.0, 0.005, 0.015, 0.02)),
        ("close_tf", (1, 3, 10, 15)),
        ("trail", (0.06, 0.07, 0.09, 0.10, 0.12)),
        ("trail_atr", (np.nan, 2.0, 4.0, 5.0)),
        ("hard_trail", (np.nan, 0.12, 0.15, 0.25, 0.30)),
        ("stop_max", (0.06, 0.08, 0.12, 0.15)),
    ):
        for v in values:
            rows.append(replace(CAND, name=f"{field_name}={v}", rule=replace(C6, **{field_name: v})))
    for slots in (3, 4, 6, 8, 10):
        rows.append(replace(CAND, name=f"限{slots}", slots=slots))
    rows.append(replace(CAND, name="不过滤过热", filters=()))
    run(book, rows)


def study_drawdown(book: Book6) -> None:
    """候选与 S5 的最大回撤区间、期间平仓的笔数与每笔平均收益。"""
    from datetime import datetime

    for scheme in (CAND, replace(BASE, name="S5 持18h", rule=replace(S6, hold=1080))):
        tr, sel = book.selected(scheme)
        d = book.data
        for label, start, end in (("样本外", HOLDOUT_START, DEV_START), ("开发期", DEV_START, d.end)):
            ok = sel & ~np.isnan(tr.net) & (d.closed_at >= start) & (d.closed_at < end)
            order = np.argsort(tr.t_out[ok])
            t = tr.t_out[ok][order]
            curve = np.cumsum(tr.net[ok][order]) * 100 / scheme.slots
            peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
            bottom = int(np.argmax(peak - curve))
            top = int(np.argmax(curve[: bottom + 1])) if bottom > 0 else 0

            def day(x):
                return datetime.fromtimestamp(int(x), BEIJING).strftime("%Y-%m-%d")

            print(
                f"{scheme.name} {label}：最大回撤 {peak[bottom] - curve[bottom]:.0f}%，"
                f"{day(t[top])} → {day(t[bottom])}，"
                f"期间平仓 {bottom - top} 笔，每笔平均 {tr.net[ok][order][top + 1 : bottom + 1].mean() * 100:+.2f}%"
            )
        print(diag(book, scheme))


ENTRIES = [(delay, confirm) for delay in (3, 5, 7, 10) for confirm in (0.005, 0.01, 0.02)]


def robust_rules():
    """预先定义的几种简单出场，参数只取粗档位；另加 S5（持 18h）作参照。"""
    yield "S5 持18h（参照）", replace(S6, hold=1080)
    for trig, kt in TRIGGERS.items():
        if trig == "5分收盘":
            continue
        for trail in (0.08, 0.10, 0.12):
            for width, kw in WIDTHS.items():
                if width == "近240分振幅×5":
                    continue
                for hard in (np.nan,) if trig == "盘中" else (np.nan, 0.20):
                    for stop, ks in STOPS.items():
                        name = f"{trig} 回撤{trail:.0%} {width} 硬{hard:.0%} {stop}".replace("硬nan%", "无硬线")
                        yield name, replace(S6, trail=trail, hard_trail=hard, **kt, **kw, **ks)


def study_robust(book: Book6) -> None:
    """防过拟合：每种出场取 12 个入场设置（等待 × 确认）的平均，只按开发期平均排序，样本外只做核对。"""
    import csv

    from paths import RESULTS

    out = []
    for name, rule in robust_rules():
        cells = []
        for delay, confirm in ENTRIES:
            r = summary(book, replace(BASE, name=name, rule=replace(rule, delay=delay, confirm=confirm)))
            cells.append(
                (
                    r["样本外"].total,
                    r["样本外"].sharpe,
                    r["样本外"].max_dd,
                    r["开发期"].total,
                    r["开发期"].sharpe,
                    r["开发期"].max_dd,
                    r["开发期去5天"],
                    r["全部"].sharpe,
                )
            )
        c = np.array(cells)
        out.append((name, c.mean(0), (c[:, 0] > 0).mean(), c[:, 4].min()))
        book._cache.clear()  # 每种出场 12 份逐笔结果，不清会占满内存
    out.sort(key=lambda x: -x[1][4])
    print(f"{'出场':<44s} 样本外均值 夏普 回撤 为正占比 | 开发均值 夏普 最低夏普 回撤 去5天 | 五年夏普")
    for name, m, pos, worst in out:
        print(
            f"{name:<44s} {m[0]:+7.0f} {m[1]:5.2f} {m[2]:4.0f} {pos:5.0%} | {m[3]:+6.0f} {m[4]:5.2f} {worst:5.2f} "
            f"{m[5]:4.0f} {m[6]:+5.0f} | {m[7]:.2f}"
        )
    with (RESULTS / "r6_robust.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "出场",
                "样本外",
                "样本外夏普",
                "样本外回撤",
                "样本外为正占比",
                "开发期",
                "开发夏普",
                "开发最低夏普",
                "开发回撤",
                "开发去5天",
                "五年夏普",
            ]
        )
        for name, m, pos, worst in out:
            writer.writerow(
                [name, *np.round(m[:3], 2), round(pos, 2), *np.round(m[3:5], 2), round(worst, 2), *np.round(m[5:], 2)]
            )


# 推荐 T6：robust 里「15 分钟收盘 回撤 10% + 盘中硬回撤 20%」一族的 4 种写法结果相近（平原而非尖点），取最简单的：
# 回撤不按振幅放宽，入场与初始止损沿用 S5。
T6 = replace(S6, trail=0.10, trail_atr=np.nan, close_trigger=1, close_tf=15, hard_trail=0.20)
T6_SCHEME = Scheme("T6", T6, slots=5, filters=HOT)
S5_18H = replace(BASE, name="S5 持18h", rule=replace(S6, hold=1080))


def averaged(book: Book6, name: str, rule: Exit, slots: int = 5) -> str:
    cells = []
    for delay, confirm in ENTRIES:
        r = summary(book, Scheme(name, replace(rule, delay=delay, confirm=confirm), slots=slots, filters=HOT))
        cells.append(
            (
                r["样本外"].total,
                r["样本外"].sharpe,
                r["样本外"].max_dd,
                r["开发期"].total,
                r["开发期"].sharpe,
                r["开发期"].max_dd,
                r["全部"].sharpe,
            )
        )
    book._cache.clear()
    c = np.array(cells)
    m = c.mean(0)
    return (
        f"{name:<24s} 12 个入场平均：样本外 {m[0]:+5.0f}% 夏普 {m[1]:.2f} 回撤 {m[2]:.0f}% "
        f"为正 {(c[:, 0] > 0).mean():.0%} | 开发期 {m[3]:+5.0f}% 夏普 {m[4]:.2f} 回撤 {m[5]:.0f}% | 五年夏普 {m[6]:.2f}"
    )


def study_final(book: Book6) -> None:
    """T6 与 S5 对照：12 个入场平均、开发期上重选入场、限仓、成本、逐年。"""
    from r5_studies import monthly

    for name, rule in (("S5 持18h", S5_18H.rule), ("S5 无时间出场", S6), ("T6", T6)):
        print(averaged(book, name, rule))
    print("开发期上重选入场（只看开发期夏普）：")
    rows = []
    for delay in (3, 4, 5, 6, 7, 10):
        for confirm in (0.0, 0.005, 0.01, 0.015, 0.02):
            rows.append(
                replace(
                    T6_SCHEME, name=f"T6 等{delay} 确认{confirm:.1%}", rule=replace(T6, delay=delay, confirm=confirm)
                )
            )
    results = [summary(book, x) for x in rows]
    show(sorted(results, key=lambda r: -r["开发期"].sharpe)[:10])
    book._cache.clear()
    rows = [S5_18H, BASE, T6_SCHEME]
    rows += [replace(T6_SCHEME, name=f"T6 限{k}", slots=k) for k in (3, 8, 10)]
    rows += [
        replace(T6_SCHEME, name=f"T6 单边成本{fee:.2%}", rule=replace(T6, fee=fee)) for fee in (0.001, 0.0015, 0.002)
    ]
    rows += [replace(T6_SCHEME, name="T6 不过滤过热", filters=())]
    run(book, rows, details=True)
    for x in (S5_18H, T6_SCHEME, replace(T6_SCHEME, name="T6 限8", slots=8)):
        print(monthly(book, x))
    tr, sel = book.selected(T6_SCHEME)
    ok = sel & ~np.isnan(tr.net) & (book.data.closed_at >= HOLDOUT_START)
    held_hours = (tr.t_out[ok] - tr.t_in[ok]) / 3600
    for rate in (0.0001, 0.0003):
        print(f"资金费 每8h {rate:.2%}：五年合计少 {(rate * held_hours / 8).sum() * 100 / T6_SCHEME.slots:.0f}%")
    net = tr.net[ok] * 100
    print(
        f"T6 单笔收益分位（%）：P10 {np.percentile(net, 10):+.1f} 中位 {np.median(net):+.1f} "
        f"P90 {np.percentile(net, 90):+.1f} P99 {np.percentile(net, 99):+.1f}，"
        f"最大 {net.max():+.0f}，最差 {net.min():+.0f}"
    )


def study_plateau(book: Book6) -> None:
    """T6 各参数单独扰动，每个都取 12 个入场平均：看是否处在平原上。"""
    print(averaged(book, "T6", T6))
    for field_name, values in (
        ("trail", (0.08, 0.09, 0.11)),
        ("hard_trail", (0.15, 0.25, 0.30, np.nan)),
        ("close_tf", (5, 10, 30, 60)),
        ("stop_gap", (0.03, 0.08)),
    ):
        for v in values:
            print(averaged(book, f"{field_name}={v}", replace(T6, **{field_name: v})))
    sys.stdout.flush()


def study_entry(book: Book6) -> None:
    """入场：等待分钟 × 确认涨幅 × 追高上限。"""
    rows = []
    for delay in (0, 3, 5, 10, 15):
        for confirm in (np.nan, 0.0, 0.01, 0.02, 0.04):
            rows.append(variant(f"等{delay} 确认{confirm:.0%}", delay=delay, confirm=confirm))
    for chase in (0.05, 0.10, 0.20):
        rows.append(variant(f"等5 确认1% 追高≤{chase:.0%}", max_chase=chase))
    run(book, rows)


def study_stop(book: Book6) -> None:
    """初始止损：等待期最低价下方余量 × 最远距离，以及固定百分比止损。"""
    rows = []
    for gap in (0.0, 0.02, 0.05, 0.08, 0.12):
        for stop_max in (0.10, 0.15, 0.20):
            rows.append(variant(f"等待低点下{gap:.0%} 最远{stop_max:.0%}", stop_gap=gap, stop_max=stop_max))
    for fixed in (0.05, 0.08, 0.10, 0.15):
        rows.append(variant(f"固定止损{fixed:.0%}", stop_gap=1.0, stop_max=fixed))
    run(book, rows)


STUDIES = {
    "base": study_base,
    "trail": study_trail,
    "dyn": study_dyn,
    "start": study_start,
    "tight": study_tight,
    "lock": study_lock,
    "be": study_be,
    "close": study_close,
    "closetf": study_closetf,
    "slots": study_slots,
    "grid": study_grid,
    "hood": study_hood,
    "robust": study_robust,
    "final": study_final,
    "plateau": study_plateau,
    "drawdown": study_drawdown,
    "entry": study_entry,
    "stop": study_stop,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 数据要加载近 10GB，一次可以跑多个研究
    parser.add_argument("study", nargs="+", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="r6_p30")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    book = Book6(args.dataset)
    for name in args.study:
        print(f"===== {name}", flush=True)
        STUDIES[name](book)


if __name__ == "__main__":
    main()
