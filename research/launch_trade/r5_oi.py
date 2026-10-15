"""持仓价值过滤：在第五轮各方案上加「提醒时的持仓价值」等持仓特征过滤，看效果大不大（2026-10-15，结论见 FINDINGS.md）。

持仓数据来自 oi_fetch.py（Gate contract_stats），REST 只保留最近 180 天，
所以只能在 2026-04-20 起约半年上评估，全部落在第五轮的开发期内，没有样本外。
为了判断方向是否稳定，把这半年拆成前后两段（A、B）分别给出；
持仓价值与 24 小时成交额高度相关，再用五年都有的「24h 成交额」按同样的保留比例做对照：两者效果相近时，
可以用 24h 成交额在五年上的表现推断持仓过滤的长期表现。

持仓特征只用提醒 K 线收盘时已知的数。接口的 time 是统计区间开头、值是区间结束时的快照
（1 小时粒度标签 X 的值等于 5 分钟粒度标签 X + 55 分钟的值），所以按「可知时刻 = 标签 + 粒度」取最后一个点。
没有持仓数据的提醒（期间已下架的合约等）在所有方案里都剔除，基准与过滤口径一致。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r5_oi.py <研究> [数据集...]   # 研究：buckets、filters、proxy、control、volume、jp
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace

import numpy as np
from oi_fetch import OI_1H, OI_5M
from r2_final import pad
from r5_lib import PRE, Market, Trades, beijing, load, portfolio, stats, trimmed
from r5_studies import CANDIDATE, HOLDOUT_START, HOT, LONG, X0, Book, J, Scheme

START = beijing(2026, 4, 20)
MID = beijing(2026, 7, 17)


class OiBook(Book):
    """只加载持仓数据覆盖的这半年，features 里多出持仓特征。"""

    def __init__(self, dataset: str) -> None:
        self.data = load(dataset, START - 2 * 86400)
        self.market = Market()
        self._features = None
        self._cache: dict[tuple, Trades] = {}
        self.dataset = dataset
        self.has_oi = np.zeros(len(self.data.closed_at), dtype=bool)

    @property
    def features(self) -> dict[str, np.ndarray]:
        f = Book.features.fget(self)
        if "持仓价值" not in f:
            f.update(self._oi_features())
        return f

    def _oi_features(self) -> dict[str, np.ndarray]:
        d = self.data
        n = len(d.closed_at)
        out = {k: np.full(n, np.nan) for k in ("持仓价值", "窗口持仓变化", "持仓1h变化", "持仓24h变化", "多空人数比")}
        for symbol in np.unique(d.symbol):
            fine, hourly = _series(OI_5M / f"{symbol}.npz", 300), _series(OI_1H / f"{symbol}.npz", 3600)
            if fine is None and hourly is None:
                continue
            idx = np.flatnonzero(d.symbol == symbol)
            t = d.closed_at[idx]
            # 提醒时刻的持仓优先用 5 分钟粒度，缺了退回 1 小时粒度（最多晚 1 小时）。
            value = _at(fine, t, "oi_usd", 900)
            out["持仓价值"][idx] = np.where(np.isnan(value), _at(hourly, t, "oi_usd", 3600), value)
            # 持仓变化用张数：持仓价值里含价格涨跌，拉升时价值自然变大。
            now = _at(fine, t, "oi", 900)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["窗口持仓变化"][idx] = now / _at(fine, t - 300, "oi", 900) - 1
                out["持仓1h变化"][idx] = now / _at(fine, t - 3600, "oi", 900) - 1
                out["持仓24h变化"][idx] = _at(hourly, t, "oi", 3600) / _at(hourly, t - 86400, "oi", 3600) - 1
                out["多空人数比"][idx] = _at(fine, t, "long_users", 900) / _at(fine, t, "short_users", 900)
        self.has_oi = ~np.isnan(out["持仓价值"]) & (out["持仓价值"] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["持仓/24h成交额"] = out["持仓价值"] / d.meta["volume_24h"]
        return out


def _series(path, step: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    s = dict(np.load(path))
    if len(s["time"]) == 0:
        return None
    # 标签是统计区间开头，区间结束（标签 + 粒度）才知道这个值。
    s["known"] = s["time"] + step
    return s


def _at(s: dict[str, np.ndarray] | None, when: np.ndarray, key: str, stale: int) -> np.ndarray:
    """when 时刻已知的最后一个值；离 when 超过 stale 秒（统计点缺失、5 分钟粒度只取了提醒附近）视为没有数据。"""
    if s is None:
        return np.full(len(when), np.nan)
    pos = np.searchsorted(s["known"], when, side="right") - 1
    safe = np.clip(pos, 0, None)
    ok = (pos >= 0) & (when - s["known"][safe] <= stale)
    return np.where(ok, s[key][safe], np.nan)


PERIODS = (("全部", START, None), ("A 4/20~7/16", START, MID), ("B 7/17~", MID, None))


@dataclass
class Row:
    name: str
    cells: list[str]


def _select(book: OiBook, tr: Trades, mask: np.ndarray, slots: int) -> tuple[np.ndarray, int]:
    """slots = 0 表示不限仓（前几轮的口径：每笔 100 USDT、同一币可叠仓），返回选中的提醒与 stats 用的 slots。

    stats 按「本金 ÷ slots」折算，不限仓时传 1，收益与回撤就是每笔 100 USDT 下的 USDT 数。
    """
    if slots == 0:
        return mask & ~np.isnan(tr.net), 1
    return portfolio(book.data, tr, mask, slots), slots


def evaluate(book: OiBook, scheme: Scheme, extra: np.ndarray | None = None) -> Row:
    """限仓组合在这半年及前后两段的本金收益（%）、去最好 3 天、回撤、笔数；不限仓时为 USDT（每笔 100）。"""
    d = book.data
    _ = book.features  # 先算持仓特征，填好 has_oi
    tr = book.trades(scheme.rule)
    mask = book.mask(scheme) & book.has_oi & (d.closed_at >= START)
    if extra is not None:
        mask &= extra
    sel, slots = _select(book, tr, mask, scheme.slots)
    cells = []
    for _, start, end in PERIODS:
        stop = end or d.end + 1
        s = stats(d, tr, sel, slots, start, stop)
        cut = trimmed(d, tr, sel & (d.closed_at < stop), slots, start, 3)
        cells.append(f"{s.total:+6.0f} {cut:+6.0f} {s.max_dd:4.0f} {s.n:4d}")
    return Row(scheme.name, cells)


def show(title: str, rows: list[Row]) -> None:
    head = " | ".join(f"{name:^25s}" for name, _, _ in PERIODS)
    print(f"\n=== {title} ===")
    print(pad("方案", 44) + " " + head)
    print(pad("", 44) + " " + " | ".join(f"{'收益':>6s} {'去3天':>6s} {'回撤':>4s} {'笔数':>4s}" for _ in PERIODS))
    for r in rows:
        print(pad(r.name, 44) + " " + " | ".join(r.cells))
    sys.stdout.flush()


# 第三、四轮的 J4、P5（r3_final.py 的定义）在第五轮数据集上重建：r5_base 是原默认提醒规则（对应 v_base），
# r5_p 是 P 规则（不要求突破、池下限 0、不设插针上限，对应 v_nb_pool0_noover）。r5 的 J、X0 与 r3 的 BEST、X0 参数相同。
# slots = 0 为不限仓。J 系列只取 J4（按时段过滤的方案是过拟合，不测）。
J4 = Scheme("J4", J, all_waves=False, slots=0, filters=(("宽度30", -np.inf, 0.85),))
P5 = Scheme("P5", X0, slots=0, filters=(("过热J类", -np.inf, 0.5),))
JP = {"r5_base": J4, "r5_p": P5}

# 各数据集上的方案：第五轮的 S5、LONG，以及 J4（限 5 仓）。
SCHEMES = {
    "r5_p30": [replace(CANDIDATE, name=f"S5 限{k}", slots=k) for k in (3, 5, 10)],
    "r5_p30_1440": [replace(CANDIDATE, name="S5（基准1440）限5", slots=5)],
    "r5_p": [Scheme("LONG 过热不买 限5（P 规则）", LONG, slots=5, filters=HOT), Scheme("LONG 限10（P 规则）", LONG)],
    "r5_user": [
        Scheme("LONG 限10（导出时 YAML）", LONG),
        Scheme("LONG 过热不买 限5（导出时 YAML）", LONG, slots=5, filters=HOT),
    ],
    "r5_base": [replace(J4, name="J4 限5", slots=5)],
}

VALUE_CUTS = (2e5, 5e5, 1e6, 2e6, 5e6)
BUCKET_FEATURES = ("持仓价值", "持仓/24h成交额", "24h成交额", "窗口持仓变化", "持仓1h变化", "持仓24h变化", "多空人数比")


def _book(dataset: str) -> OiBook:
    book = OiBook(dataset)
    book.features["过热J类"] = _book_core_hot(book)
    return book


def _money(x: float) -> str:
    return f"{x / 1e4:.0f}万" if x < 1e8 else f"{x / 1e8:.1f}亿"


def study_buckets(datasets: list[str]) -> None:
    """不限仓逐笔：按持仓特征五分位分档，每档在前后两段的笔数、胜率、每笔收益（LONG / J 出场，过热不买）。"""
    for dataset in datasets:
        book = _book(dataset)
        d = book.data
        rule = J if dataset == "r5_base" else LONG
        f = book.features
        tr = book.trades(rule)
        base = book.mask(Scheme("", rule, filters=HOT)) & book.has_oi & (d.closed_at >= START) & ~np.isnan(tr.net)
        alerts = (d.closed_at >= START) & (d.direction == 1)
        print(
            f"\n===== {dataset}：有持仓数据 {int((book.has_oi & alerts).sum())} 条，"
            f"缺 {int((~book.has_oi & alerts).sum())} 条；过热不买后入场 {int(base.sum())} 笔 ====="
        )
        for name in BUCKET_FEATURES:
            x = f[name]
            ok = base & ~np.isnan(x)
            edges = np.nanpercentile(x[ok], [20, 40, 60, 80])
            bins = np.digitize(x, edges)
            print(f"-- {name}（分位点 " + " / ".join(_fmt(name, e) for e in edges) + "）")
            for b in range(5):
                parts = []
                for label, start, end in PERIODS:
                    m = ok & (bins == b) & (d.closed_at >= start) & (d.closed_at < (end or d.end + 1))
                    net = tr.net[m] if m.any() else np.zeros(1)
                    parts.append(f"{label[:1]} {int(m.sum()):4d} 胜{(net > 0).mean():4.0%} {net.mean() * 100:+6.2f}%")
                print(f"   第{b + 1}档  " + " | ".join(parts))


def _fmt(name: str, value: float) -> str:
    if name in ("持仓价值", "24h成交额"):
        return _money(value)
    if "变化" in name:
        return f"{value:+.1%}"
    return f"{value:.2f}"


def study_filters(datasets: list[str]) -> None:
    """各方案加持仓价值下限 / 上限、持仓 ÷ 24h 成交额、持仓变化过滤后的限仓组合结果。"""
    for dataset in datasets:
        book = _book(dataset)
        f = book.features
        for scheme in SCHEMES[dataset]:
            rows = [evaluate(book, scheme)]
            for cut in VALUE_CUTS:
                rows.append(evaluate(book, scheme, f["持仓价值"] >= cut))
                rows[-1].name = f"  持仓价值 ≥ {_money(cut)}"
            for cut in (1e6, 3e6, 1e7):
                rows.append(evaluate(book, scheme, f["持仓价值"] <= cut))
                rows[-1].name = f"  持仓价值 ≤ {_money(cut)}"
            x = f["持仓/24h成交额"]
            ok = book.has_oi & (book.data.closed_at >= START)
            for q in (20, 40):
                cut = float(np.nanpercentile(x[ok], q))
                rows.append(evaluate(book, scheme, x >= cut))
                rows[-1].name = f"  持仓/24h成交额 ≥ {cut:.2f}（P{q}）"
                cut = float(np.nanpercentile(x[ok], 100 - q))
                rows.append(evaluate(book, scheme, x <= cut))
                rows[-1].name = f"  持仓/24h成交额 ≤ {cut:.2f}（P{100 - q}）"
            for name in ("窗口持仓变化", "持仓1h变化", "持仓24h变化"):
                x = f[name]
                for cut, sign in ((0.0, 1), (0.0, -1), (0.1, 1)):
                    keep = x >= cut if sign > 0 else x <= cut
                    # 变化缺失（1 小时 / 24 小时前没有统计点）不过滤，与 Book.mask 的约定一致。
                    rows.append(evaluate(book, scheme, np.isnan(x) | keep))
                    rows[-1].name = f"  {name} {'≥' if sign > 0 else '≤'} {cut:+.0%}"
            show(f"{dataset}：{scheme.name}", rows)


def study_proxy(datasets: list[str]) -> None:
    """持仓价值下限与「24h 成交额」下限按同样保留比例对照：效果相近时，可用 24h 成交额的五年结果推断长期表现。"""
    for dataset in datasets:
        book = _book(dataset)
        d = book.data
        f = book.features
        ok = book.has_oi & (d.closed_at >= START) & (d.direction == 1)
        for scheme in SCHEMES[dataset][:1]:
            rows = [evaluate(book, scheme)]
            for keep in (0.8, 0.6, 0.4):
                for name in ("持仓价值", "24h成交额"):
                    cut = float(np.nanpercentile(f[name][ok], (1 - keep) * 100))
                    rows.append(evaluate(book, scheme, f[name] >= cut))
                    rows[-1].name = f"  保留{keep:.0%}：{name} ≥ {_money(cut)}"
            show(f"{dataset}：{scheme.name}（对照）", rows)


def study_control(datasets: list[str]) -> None:
    """随机对照：按持仓价值下限去掉的比例随机去掉提醒（各 200 次），看持仓过滤的结果在随机分布里的分位。

    限仓组合里少买一部分提醒会把仓位让给别的提醒，结果本身就会变；只有明显好于随机去掉，才说明是持仓价值的作用。
    """
    rng = np.random.default_rng(0)
    for dataset in datasets:
        book = _book(dataset)
        d = book.data
        value = book.features["持仓价值"]
        for scheme in SCHEMES[dataset]:
            tr = book.trades(scheme.rule)
            mask = book.mask(scheme) & book.has_oi & (d.closed_at >= START)
            print(f"\n=== {dataset}：{scheme.name}（随机对照，各 200 次） ===")
            for cut in (1e6, 2e6):
                keep = value >= cut
                drop = float(((mask & ~keep).sum()) / mask.sum())
                real = _period_totals(book, tr, mask & keep, scheme.slots)
                draws = np.array(
                    [_period_totals(book, tr, mask & (rng.random(len(mask)) >= drop), scheme.slots) for _ in range(200)]
                )
                cells = []
                for k, (label, _, _) in enumerate(PERIODS):
                    rank = float((draws[:, k] < real[k]).mean())
                    cells.append(
                        f"{label[:1]} {real[k]:+5.0f}（随机中位 {np.median(draws[:, k]):+5.0f}，高于 {rank:4.0%}）"
                    )
                dd = float((draws[:, -1] > real[-1]).mean())
                print(
                    f"  持仓价值 ≥ {_money(cut)}（去掉 {drop:.0%}）：" + "  ".join(cells) + f"  回撤 {real[-1]:.0f}"
                    f"（随机中位 {np.median(draws[:, -1]):.0f}，低于 {dd:.0%}）"
                )
            sys.stdout.flush()


def _period_totals(book: OiBook, tr: Trades, mask: np.ndarray, slots: int) -> list[float]:
    d = book.data
    sel, slots = _select(book, tr, mask, slots)
    out = [stats(d, tr, sel, slots, start, end or d.end + 1).total for _, start, end in PERIODS]
    return [*out, stats(d, tr, sel, slots, START, d.end + 1).max_dd]


def study_jp(datasets: list[str]) -> None:
    """第三、四轮的 J4、P5 加持仓价值过滤：不限仓（每笔 100 USDT）与限 5 / 10 仓，附随机对照与五年合计核对。"""
    rng = np.random.default_rng(0)
    for dataset in [x for x in datasets if x in JP]:
        base = JP[dataset]
        # 先核对重建的方案与第四轮五年结果是否同一量级（不限仓、每笔 100 USDT，2021-10-10 起）。
        full = Book(dataset)
        if base is P5:
            full.features["过热J类"] = _book_core_hot(full)
        tr = full.trades(base.rule)
        ok = full.mask(base) & ~np.isnan(tr.net) & (full.data.closed_at >= HOLDOUT_START)
        print(f"\n##### {dataset}：{base.name} 五年不限仓 {int(ok.sum())} 笔，合计 {tr.net[ok].sum() * 100:+.0f} USDT")
        del full

        book = _book(dataset)
        value = book.features["持仓价值"]
        for slots in (0, 5, 10):
            scheme = replace(
                base, name=f"{base.name} {'不限仓（USDT）' if slots == 0 else f'限{slots}（%）'}", slots=slots
            )
            rows = [evaluate(book, scheme)]
            for cut in VALUE_CUTS:
                rows.append(evaluate(book, scheme, value >= cut))
                rows[-1].name = f"  持仓价值 ≥ {_money(cut)}"
            rows.append(evaluate(book, scheme, value < 1e6))
            rows[-1].name = "  持仓价值 < 100万 的那部分"
            show(f"{dataset}：{scheme.name}", rows)
            tr = book.trades(scheme.rule)
            d = book.data
            mask = book.mask(scheme) & book.has_oi & (d.closed_at >= START)
            for cut in (1e6, 2e6):
                keep = value >= cut
                drop = float(((mask & ~keep).sum()) / mask.sum())
                real = _period_totals(book, tr, mask & keep, slots)
                draws = np.array(
                    [_period_totals(book, tr, mask & (rng.random(len(mask)) >= drop), slots) for _ in range(200)]
                )
                above = " ".join(
                    f"{label[:1]} {(draws[:, k] < real[k]).mean():4.0%}" for k, (label, _, _) in enumerate(PERIODS)
                )
                print(
                    f"  随机对照 ≥ {_money(cut)}（去掉 {drop:.0%}）：收益高于随机 {above}；"
                    f"回撤低于随机 {(draws[:, -1] > real[-1]).mean():.0%}"
                )


def _book_core_hot(book: Book) -> np.ndarray:
    """P5 只在过热时不买 J 类（r3_lib.Book.core）：已突破、24h 成交额 > 100 万、等 10 分钟后入场开盘价比提醒价高 ≥ 2%。

    宽度缺失按不过热处理，与 r3_lib 一致。
    """
    d = book.data
    alert_price = d.bars[:, 3, PRE - 1]
    with np.errstate(invalid="ignore"):
        gain = d.bars[:, 0, PRE + 10] / alert_price - 1
        core = (d.meta["broke_out"] > 0.5) & (d.meta["volume_24h"] > 1e6) & (gain >= 0.02)
    hot = np.nan_to_num(book.features["宽度30"], nan=0.0) > 0.85
    return (hot & core).astype(float)


def study_volume(datasets: list[str]) -> None:
    """五年对照：各方案加 24h 成交额下限（持仓价值没有更早的数据），按第五轮的样本外 / 开发期口径。"""
    from r5_studies import show as show5
    from r5_studies import summary

    for dataset in datasets:
        book = Book(dataset)
        for scheme in SCHEMES[dataset]:
            rows = [summary(book, scheme)]
            for cut in (3e6, 1e7, 2e7, 5e7):
                flt = (*scheme.filters, ("24h成交额", cut, np.inf))
                rows.append(summary(book, replace(scheme, name=f"  24h成交额 ≥ {_money(cut)}", filters=flt)))
            print(f"\n=== {dataset}：{scheme.name}（五年） ===")
            show5(rows)


STUDIES = {
    "buckets": study_buckets,
    "filters": study_filters,
    "proxy": study_proxy,
    "control": study_control,
    "volume": study_volume,
    "jp": study_jp,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("datasets", nargs="*", default=list(SCHEMES))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    STUDIES[args.study](args.datasets)


if __name__ == "__main__":
    main()
