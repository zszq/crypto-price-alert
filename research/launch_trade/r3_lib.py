"""第三轮研究的公用工具：三年数据上的方案评估。

第二轮的排名在半年里选参数，三年检验后发现收益集中在少数大单上，单个参数点（如回撤 15%）的合计带有很大运气，
所以这里除了单点合计，还给出「邻域平均」：把入场（等待 × 确认）与回撤幅度一起扰动后的平均，
以及同时持仓数、资金费率敏感性等实盘约束。
数据来自 r3_export.py（--out 指定的目录），由 r2_lib.load(period, source) 读取。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from paths import RESULTS
from r2_lib import BEST, PRE, Data, Engine, Rule, evaluate, fold_sums, load

# 第三轮的出场 X：在 J（BEST）基础上止损放宽到等待期最低价下方 5%（最多 20%）、回撤 ≥ 3 倍振幅、持仓 18 小时。
# 三年里按 30 种入场的平均选出，见 FINDINGS.md「第三轮」。
EXIT_X = dict(stop_support_gap=0.05, stop_max=0.2, trail_atr=3, hold=1080)
X = replace(BEST, **EXIT_X)
# 北京时间 7~10 点（UTC 23 点到次日 2 点，跨过日线开盘与资金费率结算）：按提醒 K 线的收盘时刻判断。
# 当前提醒规则下三年里稳定亏钱、前推检验也成立；不要求突破时（v_nobreak）不成立，见 FINDINGS.md。
BAD_HOURS = (7, 8, 9)
# 邻域：入场等待 × 确认涨幅，与回撤幅度 × 倍数。
ENTRIES = [(delay, confirm) for delay in (5, 7, 8, 10, 12, 15) for confirm in (0.0, 0.005, 0.01, 0.015, 0.02)]
TRAIL_SCALE = (0.8, 0.9, 1.0, 1.1, 1.2)
WAIT_DELAYS = (3, 5, 7, 10, 12, 15, 20)
# 资金费率敏感性：按每 8 小时持仓多付这么多计（Gate 正常为 0.01%，启动中的币多头常付得更多）。
FUNDING_PER_8H = 0.0003
# 同时持仓上限（笔）：J 三年里高峰约 30 笔，持仓更久、提醒更多的方案高峰可到 100 笔，只看合计对它们偏乐观。
CAPS = (10, 20)
# 市场过热：全市场站上 30 天均线的合约占比超过这个值。三年里过热时 J 类提醒（见 Book.core）6 段全亏，
# 越热亏得越多（70%/75%/80%/85% 时每笔 -0.32%/-0.52%/-0.73%/-1.05%）；只用前 18 个月选阈值，后 18 个月也成立。
# 取 85% 而不是 80%：三年里 85% 对 J、J3、P3 的改进都更大，只用前 18 个月选也会选 85%；80% 还会滤掉 2026-09 下旬
# J 赚钱的几天。阈值附近很敏感：2026-09-23 宽度到过 84.9%。见 FINDINGS.md「亏损期分析与过热过滤」。
HOT_BREADTH = 0.85
MARKET = RESULTS / "r3_market.npz"
_breadth_cache: tuple[np.ndarray, np.ndarray] | None = None


def hourly_breadth() -> tuple[np.ndarray, np.ndarray]:
    """(每小时开盘时间, 该小时收盘时站上 30 天均线的合约占比)，由 r3_market.py 的输出算出，进程内只算一次。"""
    global _breadth_cache
    if _breadth_cache is None:
        market = np.load(MARKET)
        close = market["close"]
        with np.errstate(invalid="ignore", divide="ignore"):
            # 30 天（720 小时）均线只用有数据的小时：缺数据的小时既不计入和，也不计入个数。
            values = np.cumsum(np.nan_to_num(close), 0)
            counts = np.cumsum(~np.isnan(close), 0)
            sma = (values - np.roll(values, 720, 0)) / np.maximum(counts - np.roll(counts, 720, 0), 1)
            sma[:720] = np.nan
            above = np.where(np.isnan(close) | np.isnan(sma), np.nan, close > sma)
            valid = ~np.isnan(above)
            ratio = np.where(valid.sum(1) > 20, np.nansum(above, 1) / np.maximum(valid.sum(1), 1), np.nan)
        _breadth_cache = (market["hours"], ratio)
    return _breadth_cache


@dataclass(frozen=True)
class Scheme:
    """一个交易方案：数据集（提醒规则）+ 交易规则 + 选哪些提醒。"""

    name: str
    dataset: str  # data/backtest/results 下的目录
    rule: Rule
    all_waves: bool = False
    skip_hours: tuple[int, ...] = ()
    # 市场过热（站上 30 天均线的合约占比超过这个值）时不买；None 不过滤。
    hot_breadth: float | None = None
    # 过热时只跳过 J 类提醒（Book.core），其余照买：过热时失效的是已突破的追高类提醒，区间内放量的反而还赚钱。
    hot_core_only: bool = False


class Book:
    """一个数据集上的模拟器与提醒属性，同一数据集的多个方案共用。"""

    def __init__(self, dataset: str, period: str = "three_years") -> None:
        self.data: Data = load(period, RESULTS / dataset)
        self.engine = Engine(self.data)
        self.hour = ((self.data.closed_at // 3600) + 8) % 24
        self.first = self.data.wave == 1
        self._breadth: np.ndarray | None = None

    @property
    def breadth(self) -> np.ndarray:
        """每条提醒之前最后一个已收盘小时的市场宽度（站上 30 天均线的合约占比），只用提醒时已知的数据。"""
        if self._breadth is None:
            hours, ratio = hourly_breadth()
            slot = (self.data.closed_at.astype(np.int64) - int(hours[0])) // 3600 - 1
            inside = (slot >= 0) & (slot < len(ratio))
            self._breadth = np.where(inside, ratio[np.clip(slot, 0, len(ratio) - 1)], np.nan)
        return self._breadth

    def core(self, rule: Rule) -> np.ndarray:
        """J 类提醒：已突破基准期高点、24h 成交额 > 100 万、入场开盘价比提醒价高 ≥ 2%（按 rule 的等待分钟）。"""
        bars, meta = self.data.bars, self.data.meta
        alert_price = bars[:, 3, PRE - 1]
        with np.errstate(invalid="ignore"):
            gain = bars[:, 0, PRE + int(rule.delay)] / alert_price - 1
            return (alert_price > meta["breakout"]) & (meta["volume_24h"] > 1e6) & (gain >= 0.02)

    def mask(self, scheme: Scheme, rule: Rule | None = None) -> np.ndarray:
        """方案选中的提醒；rule 换了等待分钟时（邻域、换等待）按它判断 J 类。"""
        mask = np.ones_like(self.first) if scheme.all_waves else self.first.copy()
        if scheme.skip_hours:
            mask &= ~np.isin(self.hour, scheme.skip_hours)
        if scheme.hot_breadth is not None:
            # 宽度缺失（市场数据没覆盖到）按不过热处理，与不过滤时一致。
            hot = np.nan_to_num(self.breadth, nan=0.0) > scheme.hot_breadth
            if scheme.hot_core_only:
                hot &= self.core(rule or scheme.rule)
            mask &= ~hot
        return mask

    def run(self, rule: Rule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.engine.run(rule)


def times(data: Data, entry_at: np.ndarray, exit_at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """入场（开盘）与平仓（收盘）时刻，秒。bars 第 PRE 列是提醒后第一分钟。"""
    return data.closed_at + (entry_at - PRE) * 60, data.closed_at + (exit_at - PRE + 1) * 60


def max_concurrent(data: Data, entry_at: np.ndarray, exit_at: np.ndarray, mask: np.ndarray) -> int:
    t_in, t_out = (t[mask] for t in times(data, entry_at, exit_at))
    events = np.concatenate([np.stack([t_in, np.ones_like(t_in)], 1), np.stack([t_out, -np.ones_like(t_out)], 1)])
    # 同一时刻先平后开，免得把首尾相接的两笔算成同时持仓。
    events = events[np.lexsort((events[:, 1], events[:, 0]))]
    return int(np.cumsum(events[:, 1]).max()) if len(events) else 0


def capped(data: Data, entry_at: np.ndarray, exit_at: np.ndarray, mask: np.ndarray, limit: int) -> np.ndarray:
    """同时最多持 limit 笔：按入场时间顺序，满仓时跳过新提醒。

    每笔 100 USDT 时本金 = limit × 100，方案之间按同样本金比较。
    """
    index = np.flatnonzero(mask)
    t_in, t_out = (t[index] for t in times(data, entry_at, exit_at))
    keep = np.zeros_like(mask)
    open_until: list[float] = []
    for j in np.argsort(t_in, kind="stable"):
        open_until = [t for t in open_until if t > t_in[j]]
        if len(open_until) < limit:
            open_until.append(t_out[j])
            keep[index[j]] = True
    return keep


def drawdown(data: Data, net: np.ndarray, exit_at: np.ndarray, mask: np.ndarray) -> float:
    order = np.argsort(data.closed_at[mask] + exit_at[mask] * 60)
    curve = np.cumsum(net[mask][order]) * 100
    return float(np.max(np.maximum.accumulate(np.concatenate([[0], curve]))[1:] - curve)) if len(curve) else 0.0


def hood(book: Book, scheme: Scheme) -> tuple[float, float]:
    """邻域平均与 10 分位：入场与回撤一起扰动，排除单个参数点的运气。"""
    values = []
    for delay, confirm in ENTRIES:
        for scale in TRAIL_SCALE:
            rule = replace(scheme.rule, delay=delay, confirm=confirm, trail=scheme.rule.trail * scale)
            values.append(np.nansum(book.run(rule)[0][book.mask(scheme, rule)]) * 100)
    values = np.array(values)
    return float(values.mean()), float(np.percentile(values, 10))


def wait_sums(book: Book, scheme: Scheme) -> np.ndarray:
    """方案自己的确认涨幅不变，只换等待分钟后的合计。"""
    sums = []
    for delay in WAIT_DELAYS:
        rule = replace(scheme.rule, delay=delay)
        sums.append(np.nansum(book.run(rule)[0][book.mask(scheme, rule)]) * 100)
    return np.array(sums)


def report(book: Book, scheme: Scheme, reference_wait: np.ndarray | None = None) -> dict:
    data = book.data
    net, entry_at, exit_at = book.run(scheme.rule)
    mask = book.mask(scheme)
    ok = mask & ~np.isnan(net)
    res = evaluate(net, mask, data)
    held = (exit_at - entry_at + 1).astype(float)
    funded = net - FUNDING_PER_8H * held / 480
    caps = {limit: float(net[capped(data, entry_at, exit_at, ok, limit)].sum()) * 100 for limit in CAPS}
    waits = wait_sums(book, scheme)
    h_mean, h_p10 = hood(book, scheme)
    return dict(
        name=scheme.name,
        n=res["all_n"],
        total=res["all_sum"],
        mean=float(np.nanmean(net[ok])) * 100,
        folds=np.array(fold_sums(net, mask, data)),
        minus10=res["minus10"],
        t=res["t"],
        win=res["win"],
        neg_months=res["neg_months"],
        dd=drawdown(data, net, exit_at, ok),
        concurrent=max_concurrent(data, entry_at, exit_at, ok),
        funded=float(funded[ok].sum()) * 100,
        wait_mean=float(waits.mean()),
        wait_min=float(waits.min()),
        wait_wins=None if reference_wait is None else int((waits > reference_wait).sum()),
        hood=h_mean,
        hood_p10=h_p10,
        waits=waits,
        caps=caps,
    )
