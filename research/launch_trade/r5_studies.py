"""第五轮研究：以「稳定」为目标的方案评估（限仓组合口径），子命令见 STUDIES。

检验协议：2024-10-10 ~ 2026-10 的两年用来选参数（开发期），2021-10-10 ~ 2024-09-30 的三年（含 2022 年熊市）
完全不参与选择，只做样本外检验。所有数字是固定本金、最多同时持 slots 笔（每笔本金 ÷ slots，1 倍）的本金收益率，
不复利；同一合约同方向不叠仓，满仓跳过新提醒。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r5_studies.py <研究> [--dataset r5_p]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace

import numpy as np
from r2_final import pad
from r5_lib import Data, Market, Rule, Trades, alert_density, beijing, load, portfolio, simulate, stats, trimmed

DEV_START = beijing(2024, 10, 10)
HOLDOUT_START = beijing(2021, 10, 10)
HALVES = [beijing(y, m) for y in range(2021, 2027) for m in (4, 10)]


@dataclass
class Scheme:
    name: str
    rule: Rule
    side: int = 1
    all_waves: bool = True
    slots: int = 10
    # 过滤：提醒前最后一个已收盘小时的市场状态，(特征名, 下限, 上限)，落在区间外的提醒不买。
    filters: tuple[tuple[str, float, float], ...] = field(default_factory=tuple)


class Book:
    def __init__(self, dataset: str) -> None:
        self.data: Data = load(dataset, HOLDOUT_START - 40 * 86400)
        self.market = Market()
        self._features: dict[str, np.ndarray] | None = None
        self._cache: dict[tuple, Trades] = {}

    @property
    def features(self) -> dict[str, np.ndarray]:
        if self._features is None:
            d, mk = self.data, self.market
            t = d.closed_at
            btc = mk.at(mk.btc, t)
            up24 = alert_density(d, 86400, 1)
            dn24 = alert_density(d, 86400, -1)
            self._features = {
                "宽度30": mk.at(mk.breadth[30], t),
                "宽度7": mk.at(mk.breadth[7], t),
                "BTC/MA20": btc / mk.at(mk.btc_ma[20], t) - 1,
                "BTC/MA50": btc / mk.at(mk.btc_ma[50], t) - 1,
                "BTC/MA100": btc / mk.at(mk.btc_ma[100], t) - 1,
                "BTC/MA200": btc / mk.at(mk.btc_ma[200], t) - 1,
                "中位7天": mk.at(mk.median_ret[168], t),
                "中位30天": mk.at(mk.median_ret[720], t),
                "全市场放量": mk.at(mk.volume_surge, t),
                "拉升占比24h": up24 / np.maximum(up24 + dn24, 1),
                "拉升数24h": up24,
                "已突破": d.meta["broke_out"],
                "24h成交额": d.meta["volume_24h"],
                "窗口成交额": d.meta["window_quote"],
                "量比": d.meta["ratio"],
                "币7天": d.meta["ret_7d"],
                "距30天高": d.meta["start_price"] / d.meta["high_30d"] - 1,
            }
        return self._features

    def trades(self, rule: Rule) -> Trades:
        key = rule.params()
        if key not in self._cache:
            self._cache[key] = simulate(self.data, rule)
        return self._cache[key]

    def mask(self, scheme: Scheme) -> np.ndarray:
        d = self.data
        m = d.direction == scheme.side
        if not scheme.all_waves:
            m &= d.wave == 1
        for name, low, high in scheme.filters:
            x = self.features[name]
            # 特征缺失（市场数据没覆盖）按不过滤处理。
            m &= np.isnan(x) | ((x >= low) & (x <= high))
        return m

    def selected(self, scheme: Scheme) -> tuple[Trades, np.ndarray]:
        tr = self.trades(scheme.rule)
        return tr, portfolio(self.data, tr, self.mask(scheme), scheme.slots)


def summary(book: Book, scheme: Scheme) -> dict:
    tr, sel = book.selected(scheme)
    d = book.data
    out = {"name": scheme.name}
    for label, start, end in (("样本外", HOLDOUT_START, DEV_START), ("开发期", DEV_START, None)):
        s = stats(d, tr, sel, scheme.slots, start, end)
        out[label] = s
        out[label + "去5天"] = trimmed(d, tr, sel & (d.closed_at < (end or d.end + 1)), scheme.slots, start, 5)
    edges = [HOLDOUT_START, *[h for h in HALVES if h > HOLDOUT_START], d.end + 1]
    ok = sel & ~np.isnan(tr.net)
    unit = 100 / scheme.slots
    halves = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        part = ok & (tr.t_out >= a) & (tr.t_out < b)
        if part.any() or b <= d.end:
            halves.append(tr.net[part].sum() * unit)
    out["halves"] = np.array(halves)
    whole = stats(d, tr, sel, scheme.slots, HOLDOUT_START)
    out["全部"] = whole
    return out


def show(rows: list[dict]) -> None:
    labels = ["21/10", "22/04", "22/10", "23/04", "23/10", "24/04", "24/10", "25/04", "25/10", "26/04"]
    block = f"{'去5天':>6s} {'夏普':>5s} {'回撤':>5s} {'正季':>5s}"
    print(
        pad("方案", 40)
        + f" {'样本外':>7s} {block} | {'开发期':>7s} {block} | 半年："
        + " ".join(f"{x:>5s}" for x in labels)
    )
    for r in rows:
        cells = []
        for label, skip_last in (("样本外", False), ("开发期", True)):
            s = r[label]
            quarters = s.quarters[:-1] if skip_last else s.quarters
            cells.append(
                f"{s.total:+7.0f} {r[label + '去5天']:+6.0f} {s.sharpe:5.2f} {s.max_dd:5.0f} "
                f"{int((quarters > 0).sum()):2d}/{len(quarters):<2d}"
            )
        print(pad(r["name"], 40) + f" {cells[0]} | {cells[1]} |      " + " ".join(f"{x:+5.0f}" for x in r["halves"]))
    sys.stdout.flush()


# ---------------------------------------------------------------- 研究


LONG = Rule(delay=5, confirm=0.01, stop_gap=0.05, trail=0.10, trail_atr=3, hold=1080)
X0 = Rule(delay=10, confirm=0.0, stop_gap=0.05, stop_max=0.2, trail=0.15, trail_atr=3, hold=1080)
J = Rule(delay=10, confirm=0.02, stop_gap=0.02, stop_max=0.09, trail=0.15, trail_atr=5, hold=600)


def study_base(book: Book) -> None:
    """基础：几组出场 × 首波/所有波次 × 限仓。"""
    rows = []
    for name, rule in (("开发期最优", LONG), ("X 确认0", X0), ("J", J)):
        for waves in (False, True):
            for slots in (5, 10, 20):
                label = f"{name} {'全部' if waves else '首波'} 限{slots}"
                rows.append(summary(book, Scheme(label, rule, all_waves=waves, slots=slots)))
    show(rows)


def study_regime(book: Book) -> None:
    """单个市场状态过滤：每个特征按开发期分位数取几个阈值，跳过区间外的提醒。"""
    base = Scheme("基准", LONG)
    rows = [summary(book, base)]
    d = book.data
    dev = (d.closed_at >= DEV_START) & (d.direction == 1)
    for name in (
        "宽度30",
        "宽度7",
        "BTC/MA20",
        "BTC/MA50",
        "BTC/MA100",
        "BTC/MA200",
        "中位7天",
        "中位30天",
        "全市场放量",
        "拉升占比24h",
    ):
        x = book.features[name]
        for q in (10, 20, 80, 90):
            cut = float(np.nanpercentile(x[dev], q))
            flt = (name, cut, np.inf) if q < 50 else (name, -np.inf, cut)
            label = f"{name} {'≥' if q < 50 else '≤'}{cut:.3g}(P{q})"
            rows.append(summary(book, replace(base, name=label, filters=(flt,))))
    show(rows)


def study_slots(book: Book) -> None:
    rows = []
    for slots in (3, 5, 8, 10, 15, 20, 30):
        rows.append(summary(book, Scheme(f"限{slots}", LONG, slots=slots)))
    show(rows)


def study_exits(book: Book) -> None:
    """入场 × 出场的小网格：开发期选、样本外看是否同向。"""
    rows = []
    for delay, confirm in ((5, 0.01), (10, 0.0), (10, 0.02)):
        for trail in (0.10, 0.15):
            for hold in (600, 1080):
                rule = replace(LONG, delay=delay, confirm=confirm, trail=trail, hold=hold)
                label = f"等{delay} 确认{confirm:.0%} 回撤{trail:.0%} 持{hold // 60}h"
                rows.append(summary(book, Scheme(label, rule)))
    show(rows)


# 过热：全市场宽度（站上 30 天均线占比）或 BTC 相对 50 日均线过高时不买。
HOT = (("宽度30", -np.inf, 0.75), ("BTC/MA50", -np.inf, 0.10))


def study_filters(book: Book) -> None:
    """过热、熊市过滤及其组合。"""
    rows = [summary(book, Scheme("基准", LONG))]
    combos = {
        "宽度30≤0.85": (("宽度30", -np.inf, 0.85),),
        "宽度30≤0.75": (("宽度30", -np.inf, 0.75),),
        "宽度30≤0.65": (("宽度30", -np.inf, 0.65),),
        "BTC/MA50≤0.10": (("BTC/MA50", -np.inf, 0.10),),
        "BTC/MA50≤0.07": (("BTC/MA50", -np.inf, 0.07),),
        "中位30天≤0.10": (("中位30天", -np.inf, 0.10),),
        "过热（宽度≤0.75 且 BTC/MA50≤0.10）": HOT,
        "熊市不买（BTC/MA200≥0）": (("BTC/MA200", 0.0, np.inf),),
        "熊市不买（BTC/MA200≥-0.1）": (("BTC/MA200", -0.1, np.inf),),
        "过热 + BTC/MA200≥0": (*HOT, ("BTC/MA200", 0.0, np.inf)),
        "过热 + 只买已突破": (*HOT, ("已突破", 0.5, np.inf)),
        "过热 + 只买未突破": (*HOT, ("已突破", -np.inf, 0.5)),
        "过热 + 24h成交额>100万": (*HOT, ("24h成交额", 1e6, np.inf)),
        "过热 + 窗口成交额≥20万": (*HOT, ("窗口成交额", 2e5, np.inf)),
    }
    for name, flt in combos.items():
        rows.append(summary(book, Scheme(name, LONG, filters=flt)))
    show(rows)


# 推荐方案 S5：P30 规则（r5_p30 导出）+ 所有波次 + 过热不买 + 限 5 仓。
CANDIDATE = Scheme("S5：全部波次 过热不买 限5", LONG, slots=5, filters=HOT)


def study_hood(book: Book, scheme: Scheme | None = None) -> None:
    """邻域：等待、确认、回撤、止损余量、持仓各自扰动，看样本外与开发期是否都不塌。"""
    scheme = scheme or CANDIDATE
    rows = [summary(book, scheme)]
    r = scheme.rule
    for field_name, values in (
        ("delay", (3, 4, 6, 7, 10)),
        ("confirm", (0.0, 0.005, 0.015, 0.02)),
        ("trail", (0.08, 0.12, 0.15)),
        ("trail_atr", (2.0, 4.0, 5.0)),
        ("stop_gap", (0.03, 0.08)),
        ("hold", (600, 1440)),
    ):
        for v in values:
            rows.append(summary(book, replace(scheme, name=f"{field_name}={v}", rule=replace(r, **{field_name: v}))))
    show(rows)


def study_quality(book: Book) -> None:
    """提醒质量过滤（窗口成交额、是否突破、24h 成交额）与过热过滤的组合。"""
    combos = {
        "基准": (),
        "过热": HOT,
        "窗口≥20万": (("窗口成交额", 2e5, np.inf),),
        "窗口≥30万": (("窗口成交额", 3e5, np.inf),),
        "窗口≥50万": (("窗口成交额", 5e5, np.inf),),
        "未突破": (("已突破", -np.inf, 0.5),),
        "窗口≥30万 未突破": (("窗口成交额", 3e5, np.inf), ("已突破", -np.inf, 0.5)),
        "过热 窗口≥20万": (*HOT, ("窗口成交额", 2e5, np.inf)),
        "过热 窗口≥30万": (*HOT, ("窗口成交额", 3e5, np.inf)),
        "过热 窗口≥50万": (*HOT, ("窗口成交额", 5e5, np.inf)),
        "过热 窗口≥30万 未突破": (*HOT, ("窗口成交额", 3e5, np.inf), ("已突破", -np.inf, 0.5)),
        "过热 窗口≥30万 24h>500万": (*HOT, ("窗口成交额", 3e5, np.inf), ("24h成交额", 5e6, np.inf)),
    }
    rows = []
    for slots in (5, 10):
        for name, flt in combos.items():
            rows.append(summary(book, Scheme(f"{name} 限{slots}", LONG, slots=slots, filters=flt)))
    show(rows)


def study_core(book: Book) -> None:
    """数据集自身的门槛下：过热过滤 × 限仓数 × 首波/所有波次。"""
    rows = []
    for slots in (3, 5, 8, 10, 15):
        for flt, label in (((), "不过滤"), (HOT, "过热")):
            for waves in (True, False):
                name = f"{label} {'全部' if waves else '首波'} 限{slots}"
                rows.append(summary(book, Scheme(name, LONG, slots=slots, all_waves=waves, filters=flt)))
    show(rows)


def study_costs(book: Book) -> None:
    """成本敏感性：手续费（含滑点）提高、按持仓时间扣资金费率。"""
    rows = [summary(book, CANDIDATE)]
    for fee in (0.001, 0.0015, 0.002):
        rows.append(summary(book, replace(CANDIDATE, name=f"单边成本 {fee:.2%}", rule=replace(LONG, fee=fee))))
    show(rows)
    tr, sel = book.selected(CANDIDATE)
    held_hours = (tr.t_out - tr.t_in) / 3600
    for rate in (0.0001, 0.0003, 0.0005):
        cost = rate * held_hours / 8
        ok = sel & ~np.isnan(tr.net)
        for label, start, end in (("样本外", HOLDOUT_START, DEV_START), ("开发期", DEV_START, book.data.end + 1)):
            part = ok & (book.data.closed_at >= start) & (book.data.closed_at < end)
            print(f"资金费 每8h {rate:.2%}：{label} 少 {cost[part].sum() * 100 / CANDIDATE.slots:.0f}%")


def study_hot(book: Book) -> None:
    """过热阈值邻域。"""
    rows = []
    for breadth in (0.65, 0.70, 0.75, 0.80, 0.85, 1.0):
        for ma50 in (0.06, 0.08, 0.10, 0.12, 0.15, np.inf):
            flt = (("宽度30", -np.inf, breadth), ("BTC/MA50", -np.inf, ma50))
            rows.append(summary(book, replace(CANDIDATE, name=f"宽度≤{breadth} BTC/MA50≤{ma50}", filters=flt)))
    show(rows)


def monthly(book: Book, scheme: Scheme) -> str:
    """逐年收益、月度正收益占比、最差月、最长回撤持续时间（按平仓时间累计的已实现收益）。"""
    tr, sel = book.selected(scheme)
    d = book.data
    ok = sel & ~np.isnan(tr.net) & (d.closed_at >= HOLDOUT_START)
    unit = 100 / scheme.slots
    t = tr.t_out[ok] + 8 * 3600
    month = (t // 86400).astype("datetime64[D]").astype("datetime64[M]")
    months, inv = np.unique(month, return_inverse=True)
    pnl = np.bincount(inv, weights=tr.net[ok]) * unit
    full = pnl[:-1]  # 数据截止的当月不完整
    years = months.astype("datetime64[Y]").astype(int) + 1970
    by_year = " ".join(f"{y}:{pnl[years == y].sum():+.0f}" for y in np.unique(years))
    order = np.argsort(tr.t_out[ok])
    curve = np.cumsum(tr.net[ok][order]) * unit
    peak_at = 0.0
    longest = 0.0
    peak = -np.inf
    times = tr.t_out[ok][order]
    for value, at in zip(curve, times, strict=True):
        if value >= peak:
            peak, peak_at = value, at
        longest = max(longest, (at - peak_at) / 86400)
    held = (tr.t_out[ok] - tr.t_in[ok]) / 3600
    days = (d.end - HOLDOUT_START) / 86400
    return (
        f"{scheme.name}：{ok.sum()} 笔（每天 {ok.sum() / days:.1f}），胜率 {(tr.net[ok] > 0).mean():.0%}，"
        f"平均持仓 {held.mean():.1f}h，每笔 {tr.net[ok].mean() * 100:+.2f}%；正收益月 {(full > 0).sum()}/{len(full)}，"
        f"最差月 {full.min():+.0f}%，最好月 {full.max():+.0f}%，最长回撤 {longest:.0f} 天\n    逐年（%）：{by_year}"
    )


def study_final(book: Book) -> None:
    """候选方案的不同限仓，及与前几轮推荐在同样限仓口径下的对比（参照方案需在对应数据集上运行）。"""
    schemes = [replace(CANDIDATE, name=f"S5 限{k}", slots=k) for k in (3, 5, 8, 10)]
    rows = [summary(book, s) for s in schemes]
    show(rows)
    for s in schemes:
        print(monthly(book, s))


def study_reference(book: Book) -> None:
    """前几轮推荐在限仓口径下（在 r5_base 上运行，J5 近似：所有波次、避开 7~10 点、宽度 > 85% 不买）。"""
    hours = ((book.data.closed_at // 3600) + 8) % 24
    rows = []
    for slots in (5, 10):
        base = Scheme(f"J3 限{slots}", J, slots=slots)
        j5 = replace(base, name=f"J5 限{slots}", filters=(("宽度30", -np.inf, 0.85),))
        # 避开 7~10 点：用一个临时特征。
        book.features["北京时"] = hours.astype(float)
        for sch in (base, j5):
            rows.append(summary(book, replace(sch, filters=(*sch.filters, ("北京时", 10, 30)))))
            rows.append(summary(book, replace(sch, name=sch.name + "·含7~10点")))
    show(rows)


STUDIES = {
    "final": study_final,
    "reference": study_reference,
    "costs": study_costs,
    "hot": study_hot,
    "core": study_core,
    "quality": study_quality,
    "base": study_base,
    "regime": study_regime,
    "slots": study_slots,
    "exits": study_exits,
    "filters": study_filters,
    "hood": study_hood,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="r5_p")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    STUDIES[args.study](Book(args.dataset))


if __name__ == "__main__":
    main()
