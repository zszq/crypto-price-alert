"""第七轮研究：超短线（持仓不超过 1 小时）能否在放量启动提醒上稳定盈利。

用户做超短线和短线，持仓最长 1 小时。S5（平均持仓 4.6h）、T6（平均 12.7h）的利润主要来自持有几个小时的大单，
这一轮在同样的提醒、过热过滤与组合口径下，只比较持仓上限 ≤ 60 分钟的入场、出场与信号过滤。

- 出场用第六轮的模拟器 r6_lib（初始止损 + 移动止盈，可按盘中或 N 分钟收盘触发），持仓上限 hold ≤ 60 分钟；
  数据用第五轮导出（提醒后 1440 分钟，r5_p30 / r5_p30_1440），够用，不必加载 15 天的 r6 导出。
- 检验协议同第五轮：2024-10-10 起两年为开发期，只按开发期选；2021-10-10 ~ 2024-09-30 三年为样本外，只核对。
  第六轮的教训是只看单点会选到尖点，所以每种出场都取 12 个入场设置（等 3/5/7/10 分钟 × 确认 0.5%/1%/2%）的平均。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r7_studies.py <研究> [--dataset r5_p30]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace

import numpy as np
from paths import RESULTS
from r5_studies import HOLDOUT_START, HOT, Book, Scheme, monthly, show, summary
from r6_lib import Exit, simulate

ENTRIES = [(delay, confirm) for delay in (3, 5, 7, 10) for confirm in (0.005, 0.01, 0.02)]
# 不设移动止盈：回撤取 100%，只剩初始止损和持仓上限。
NO_TRAIL = 1.0


class Book7(Book):
    @property
    def features(self) -> dict[str, np.ndarray]:
        # 第五轮的特征再加三个短线相关的：提醒前 1h / 24h 涨幅、提醒前 60 分钟平均振幅（都只用提醒时已收盘的 K 线）。
        base = super().features
        if "币1h" not in base:
            base["币1h"] = self.data.meta["ret_1h"]
            base["币24h"] = self.data.meta["ret_24h"]
            base["振幅60"] = self.data.meta["range_60"]
        return base

    def trades(self, rule: Exit):
        key = rule.params()
        if key not in self._cache:
            self._cache[key] = simulate(self.data, rule)
        return self._cache[key]


# 第一轮补充扫描（2026-10-16）里两份提醒集都最好的一组，作为起点；grid 跑完后按开发期平均改选。
START = Exit(hold=60, trail=0.10, trail_atr=3.0)


def scheme(name: str, rule: Exit, slots: int = 5, filters=HOT, all_waves: bool = True) -> Scheme:
    return Scheme(name, rule, slots=slots, filters=filters, all_waves=all_waves)


def averaged(book: Book7, base: Scheme, entries=ENTRIES) -> dict:
    """base 的出场在 12 个入场设置上的平均；只缓存这一组，算完就清，避免占满内存。"""
    cells = []
    for delay, confirm in entries:
        r = summary(book, replace(base, rule=replace(base.rule, delay=delay, confirm=confirm)))
        cells.append(
            (
                r["样本外"].total,
                r["样本外"].sharpe,
                r["样本外"].max_dd,
                r["样本外去5天"],
                r["开发期"].total,
                r["开发期"].sharpe,
                r["开发期"].max_dd,
                r["开发期去5天"],
                r["全部"].sharpe,
                r["样本外"].n + r["开发期"].n,
            )
        )
    book._cache.clear()
    c = np.array(cells)
    return {"name": base.name, "mean": c.mean(0), "pos": int((c[:, 0] > 0).sum()), "dev_min": c[:, 5].min()}


HEADER = f"{'方案':<40s} 样本外 夏普 回撤 去5天 为正 | 开发期 夏普 最低 回撤 去5天 | 五年夏普 笔数"


def line(r: dict) -> str:
    m = r["mean"]
    return (
        f"{r['name']:<40s} {m[0]:+5.0f} {m[1]:5.2f} {m[2]:4.0f} {m[3]:+5.0f} {r['pos']:2d}/12 | "
        f"{m[4]:+5.0f} {m[5]:5.2f} {r['dev_min']:5.2f} {m[6]:4.0f} {m[7]:+5.0f} | {m[8]:5.2f} {m[9]:5.0f}"
    )


def table(rows: list[dict], sort: bool = False) -> None:
    if sort:
        rows = sorted(rows, key=lambda r: -r["mean"][5])
    print(HEADER)
    for r in rows:
        print(line(r))
    sys.stdout.flush()


def held_minutes(book: Book7, s: Scheme) -> str:
    tr, sel = book.selected(s)
    ok = sel & ~np.isnan(tr.net) & (book.data.closed_at >= HOLDOUT_START)
    m = (tr.t_out[ok] - tr.t_in[ok]) / 60
    full = (tr.status[ok] == 2).mean()
    return (
        f"{s.name}：{ok.sum()} 笔，胜率 {(tr.net[ok] > 0).mean():.0%}，持仓中位 {np.median(m):.0f} 分钟、"
        f"平均 {m.mean():.0f} 分钟，持满上限 {full:.0%}，每笔 {tr.net[ok].mean() * 100:+.2f}%"
    )


# ---------------------------------------------------------------- 研究

TRIGGERS = {"盘中": {}, "1分收盘": {"close_trigger": 1}, "5分收盘": {"close_trigger": 1, "close_tf": 5}}
WIDTHS = {"固定": {"trail_atr": np.nan}, "≥3振": {"trail_atr": 3.0}}
STOPS = {
    "S5止损": {},
    "固定5%": {"stop_gap": 1.0, "stop_max": 0.05, "stop_fallback": 0.05},
    "固定10%": {"stop_gap": 1.0, "stop_max": 0.10, "stop_fallback": 0.10},
}


def grid_rules():
    """预先定义的粗网格：持仓上限 × 触发方式 × 回撤 × 是否按振幅放宽 × 止损。"""
    for hold in (20, 30, 45, 60):
        for stop, ks in STOPS.items():
            yield f"持{hold} 不设回撤 {stop}", replace(START, hold=hold, trail=NO_TRAIL, trail_atr=np.nan, **ks)
            for trig, kt in TRIGGERS.items():
                for trail in (0.05, 0.08, 0.10, 0.15):
                    for width, kw in WIDTHS.items():
                        name = f"持{hold} {trig} 回撤{trail:.0%}{width} {stop}"
                        yield name, replace(START, hold=hold, trail=trail, **kt, **kw, **ks)


def study_grid(book: Book7) -> None:
    """全部出场按开发期 12 入场平均夏普排序；样本外只核对。结果另存 CSV。"""
    rows = [averaged(book, scheme(name, rule)) for name, rule in grid_rules()]
    rows.append(averaged(book, scheme("S5 持18h（参照）", Exit(hold=1080))))
    table(rows, sort=True)
    path = RESULTS / f"r7_grid_{book.dataset}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "方案",
                "样本外",
                "样本外夏普",
                "样本外回撤",
                "样本外去5天",
                "样本外为正",
                "开发期",
                "开发夏普",
                "开发最低夏普",
                "开发回撤",
                "开发去5天",
                "五年夏普",
                "笔数",
            ]
        )
        for r in rows:
            m = r["mean"]
            writer.writerow(
                [
                    r["name"],
                    *np.round(m[:4], 2),
                    r["pos"],
                    *np.round(m[4:6], 2),
                    round(r["dev_min"], 2),
                    *np.round(m[6:], 2),
                ]
            )
    print(f"已写入 {path}")


def study_hold(book: Book7) -> None:
    """持仓上限的细扫：其余同 CAND。"""
    rows = [averaged(book, scheme(f"持{h}", replace(CAND, hold=h))) for h in (10, 15, 20, 25, 30, 40, 45, 50, 60)]
    table(rows)


def study_entry(book: Book7) -> None:
    """入场：等待分钟 × 确认涨幅 × 追高上限，单点，按开发期夏普排序（入场本身就是被平均掉的维度，这里看形状）。"""
    rows = []
    for delay in (0, 1, 2, 3, 5, 7, 10, 15):
        for confirm in (np.nan, 0.0, 0.005, 0.01, 0.02, 0.03):
            for chase in (np.nan, 0.05, 0.10):
                name = f"等{delay} 确认{confirm:.1%} 追高≤{chase:.0%}".replace("nan%", "不限")
                rows.append(summary(book, scheme(name, replace(CAND, delay=delay, confirm=confirm, max_chase=chase))))
        book._cache.clear()
    show(sorted(rows, key=lambda r: -r["开发期"].sharpe)[:25])
    print("--- 等待分钟的边际（各确认、追高设置的平均，样本外 / 开发期）")
    for delay in (0, 1, 2, 3, 5, 7, 10, 15):
        part = [r for r in rows if r["name"].startswith(f"等{delay} ")]
        holdout = np.mean([r["样本外"].total for r in part])
        print(f"等{delay}：{holdout:+.0f} / {np.mean([r['开发期'].total for r in part]):+.0f}")


def study_extras(book: Book7) -> None:
    """短线常用的收紧手段：保本、锁利润、浮盈后收紧、只在浮盈后启用移动止盈。"""
    rows = [averaged(book, scheme("候选", CAND))]
    for be in (0.02, 0.03, 0.05):
        rows.append(averaged(book, scheme(f"浮盈{be:.0%}保本", replace(CAND, be_at=be))))
    for lock in (0.3, 0.5):
        rows.append(averaged(book, scheme(f"锁定{lock:.0%}浮盈", replace(CAND, lock=lock))))
    for at, tight in ((0.05, 0.05), (0.10, 0.05), (0.10, 0.08)):
        rows.append(
            averaged(book, scheme(f"浮盈{at:.0%}后回撤{tight:.0%}", replace(CAND, tight_at=at, tight_trail=tight)))
        )
    for start in (0.03, 0.05):
        rows.append(averaged(book, scheme(f"浮盈{start:.0%}才启用回撤", replace(CAND, trail_start=start))))
    table(rows)


def study_signal(book: Book7) -> None:
    """信号过滤：都在 HOT 之上加，单边过滤；首波；不过滤过热。"""
    rows = [averaged(book, scheme("候选（所有波次、过热不买）", CAND))]
    rows.append(averaged(book, scheme("不过滤过热", CAND, filters=())))
    rows.append(averaged(book, scheme("只买第 1 波", CAND, all_waves=False)))
    for name, flt in (
        ("窗口成交额≥50万", ("窗口成交额", 5e5, np.inf)),
        ("窗口成交额≥100万", ("窗口成交额", 1e6, np.inf)),
        ("24h成交额≥500万", ("24h成交额", 5e6, np.inf)),
        ("24h成交额≥2000万", ("24h成交额", 2e7, np.inf)),
        ("量比≥5", ("量比", 5, np.inf)),
        ("量比≥10", ("量比", 10, np.inf)),
        ("只买已突破", ("已突破", 0.5, np.inf)),
        ("只买未突破", ("已突破", -np.inf, 0.5)),
        ("BTC/MA200≥0", ("BTC/MA200", 0.0, np.inf)),
        ("币7天涨幅≤50%", ("币7天", -np.inf, 0.5)),
        ("距30天高≥-10%", ("距30天高", -0.10, np.inf)),
    ):
        rows.append(averaged(book, scheme(name, CAND, filters=(*HOT, flt))))
    table(rows)


def study_features(book: Book7) -> None:
    """探索：短线可能相关的提醒前特征，粗档位单边过滤。前几轮这类特征多数正负交替，只认两份提醒集、两段都同向的。"""
    for name in ("币1h", "币24h", "振幅60"):
        x = book.features[name]
        tr, sel = book.selected(scheme("U7", CAND))
        v = x[sel & ~np.isnan(tr.net)]
        print(f"{name} 分位：" + " ".join(f"P{q} {np.nanpercentile(v, q):.3f}" for q in (10, 25, 50, 75, 90)))
    rows = [averaged(book, scheme("U7", CAND))]
    for label, flt in (
        ("提醒前1h涨幅≤10%", ("币1h", -np.inf, 0.10)),
        ("提醒前1h涨幅≤20%", ("币1h", -np.inf, 0.20)),
        ("提醒前1h涨幅≥5%", ("币1h", 0.05, np.inf)),
        ("提醒前24h涨幅≤20%", ("币24h", -np.inf, 0.20)),
        ("提醒前24h涨幅≤50%", ("币24h", -np.inf, 0.50)),
        ("提醒前24h涨幅≥0", ("币24h", 0.0, np.inf)),
        ("入场前振幅≤1%", ("振幅60", -np.inf, 0.01)),
        ("入场前振幅≤2%", ("振幅60", -np.inf, 0.02)),
        ("入场前振幅≥0.5%", ("振幅60", 0.005, np.inf)),
    ):
        rows.append(averaged(book, scheme(label, CAND, filters=(*HOT, flt))))
    table(rows)


def study_slots(book: Book7) -> None:
    rows = [averaged(book, scheme(f"限{k}仓", CAND, slots=k)) for k in (3, 5, 8, 10, 15)]
    table(rows)


def study_costs(book: Book7) -> None:
    """单边成本（含滑点）；超短线换手高，对成本更敏感。"""
    rows = [
        averaged(book, scheme(f"单边{fee:.2%}", replace(CAND, fee=fee)))
        for fee in (0.0005, 0.001, 0.0015, 0.002, 0.003)
    ]
    table(rows)


# 按提醒时 24h 成交额分档的单边滑点（另加 fee 的 0.05% 手续费）：(下限, 滑点)。档位是假设，不是实测，
# 用来看结论是否依赖「冷门币也能按 K 线价成交」。超短线一天进出约 2.4 次，成本的影响远大于 S5。
SLIPPAGE = {
    "无滑点": ((0, 0.0),),
    "温和": ((0, 0.003), (1e6, 0.0015), (5e6, 0.0008), (2e7, 0.0003)),
    "严苛": ((0, 0.006), (1e6, 0.003), (5e6, 0.0015), (2e7, 0.0006)),
}


class SlippedBook(Book7):
    """在模拟结果上按 24h 成交额扣滑点：进出各一次，近似为净收益减 2 × 滑点。"""

    def __init__(self, base: Book7, levels) -> None:
        self.__dict__.update(base.__dict__)
        self._cache = {}
        slip = np.zeros(len(self.data.closed_at))
        volume = self.data.meta["volume_24h"]
        for low, value in levels:
            slip[volume >= low] = value
        self.slip = slip

    def trades(self, rule: Exit):
        key = rule.params()
        if key not in self._cache:
            tr = simulate(self.data, rule)
            self._cache[key] = replace(tr, net=tr.net - 2 * self.slip)
        return self._cache[key]


def study_liquidity(book: Book7) -> None:
    """不同滑点假设下，按 24h 成交额过滤合约池的效果。"""
    for model, levels in SLIPPAGE.items():
        slipped = SlippedBook(book, levels)
        print(f"--- 滑点：{model}")
        rows = [averaged(slipped, scheme("不过滤", CAND))]
        for low in (1e6, 2e6, 5e6, 1e7):
            rows.append(
                averaged(
                    slipped, scheme(f"24h成交额≥{low / 1e4:.0f}万", CAND, filters=(*HOT, ("24h成交额", low, np.inf)))
                )
            )
        rows.append(
            averaged(slipped, scheme("24h≥500万 限8", CAND, slots=8, filters=(*HOT, ("24h成交额", 5e6, np.inf))))
        )
        table(rows)
    tr, sel = book.selected(scheme("U7", CAND))
    v = book.data.meta["volume_24h"][sel & ~np.isnan(tr.net)]
    print(
        "U7 成交的 24h 成交额分布（万）："
        + " ".join(f"P{q} {np.percentile(v, q) / 1e4:.0f}" for q in (10, 25, 50, 75, 90))
    )


def study_plateau(book: Book7) -> None:
    """候选的每个参数单独扰动，看是否在平原上。"""
    rows = [averaged(book, scheme("候选", CAND))]
    for field_name, values in PLATEAU:
        for v in values:
            rows.append(averaged(book, scheme(f"{field_name}={v}", replace(CAND, **{field_name: v}))))
    table(rows)


def study_final(book: Book7) -> None:
    """候选与 S5 的单点对照（等 5 分钟、确认 1%）、持仓时长、逐年与月度。"""
    rows = [scheme("S5 持18h", Exit(hold=1080)), scheme("候选", CAND), scheme("候选 限8", CAND, slots=8)]
    show([summary(book, s) for s in rows])
    for s in rows:
        print(held_minutes(book, s))
        print(monthly(book, s))


# 候选 U7：grid 在两份提醒集（基准 360 / 1440）上只按开发期 12 入场平均夏普排序，都进前 3 的同一个出场。
# 5 分钟收盘才判定移动止盈，避开分钟内的插针；止损固定在入场价下 10%，盘中触发。
CAND = replace(
    START, trail=0.05, trail_atr=3.0, close_trigger=1, close_tf=5, stop_gap=1.0, stop_max=0.10, stop_fallback=0.10
)
PLATEAU: list[tuple[str, tuple]] = [
    ("hold", (40, 50, 75, 90)),
    ("trail", (0.03, 0.04, 0.06, 0.08)),
    ("trail_atr", (2.0, 4.0, np.nan)),
    ("close_tf", (3, 10, 15)),
    ("stop_max", (0.07, 0.08, 0.12, 0.15)),
    ("hard_trail", (0.10, 0.15)),
]

STUDIES = {
    "grid": study_grid,
    "hold": study_hold,
    "entry": study_entry,
    "extras": study_extras,
    "signal": study_signal,
    "slots": study_slots,
    "features": study_features,
    "costs": study_costs,
    "plateau": study_plateau,
    "liquidity": study_liquidity,
    "final": study_final,
}


def compare_signals(datasets: list[str]) -> None:
    """不同提醒参数的导出（r7_exports）用同几种出场比较，都取 12 个入场平均。

    提醒参数和出场会相互作用，只用一种出场比较可能只是挑到了配合它的信号，所以同时看 SIGNAL_EXITS 里的几种；
    超短线对成本敏感，U7 另按「温和」滑点再算一遍。每份导出只加载一次（全部导出同时放进内存会太大）。
    """
    results: dict[str, list[str]] = {name: [] for name in [*SIGNAL_EXITS, "U7 温和滑点"]}
    for name in datasets:
        book = Book7(name)
        days = (book.data.end - HOLDOUT_START) / 86400
        for exit_name, rule in SIGNAL_EXITS.items():
            r = averaged(book, scheme(name, rule))
            results[exit_name].append(line(r) + f" {r['mean'][9] / days:5.1f}")
        r = averaged(SlippedBook(book, SLIPPAGE["温和"]), scheme(name, CAND))
        results["U7 温和滑点"].append(line(r) + f" {r['mean'][9] / days:5.1f}")
        print(f"已算完 {name}", file=sys.stderr, flush=True)
    for exit_name, lines in results.items():
        print(f"--- 出场：{exit_name}")
        print(HEADER + " 每天笔数")
        print("\n".join(lines))


SIGNAL_EXITS = {
    "U7": CAND,
    "起点（持60 盘中回撤10%≥3振 S5止损）": START,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="+", choices=sorted([*STUDIES, "signals"]))
    parser.add_argument("--dataset", default="r5_p30")
    parser.add_argument("--datasets", nargs="+", help="signals：要比较的导出")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.study == ["signals"]:
        compare_signals(args.datasets)
        return
    book = Book7(args.dataset)
    book.dataset = args.dataset
    for name in args.study:
        print(f"===== {name}", flush=True)
        STUDIES[name](book)


if __name__ == "__main__":
    main()
