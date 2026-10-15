"""第五轮研究的公用工具：双向（做多拉升 / 做空下跌）逐笔模拟、限仓组合与稳定性评估。

前几轮按「每笔 100 USDT、不限同时持仓」的合计排名，收益集中在少数大单、高峰同时持仓上百笔，
而且五年里 2021-10 ~ 2023-09 熊市所有方案都亏。这一轮把目标改成「稳定」：
- 固定本金、最多同时持 K 笔（每笔本金 ÷ K），满仓跳过新提醒、同一合约同方向不叠仓，按实盘能执行的口径算；
- 看每个季度的收益，要求亏损季度少、最差季度浅、最大回撤小，而不是合计最大；
- 市场状态（宽度、BTC 趋势、全市场成交额、全市场提醒密度）都只取提醒前最后一个已收盘小时，不用未来数据。
数据来自 r5_export.py。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numba as nb
import numpy as np
from paths import RESULTS

PRE = 60
BEIJING = timezone(timedelta(hours=8))
MARKET = RESULTS / "r3_market.npz"


def beijing(year: int, month: int, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=BEIJING).timestamp())


@dataclass
class Data:
    bars: np.ndarray
    meta: dict[str, np.ndarray]
    symbol: np.ndarray
    sym_id: np.ndarray
    closed_at: np.ndarray
    direction: np.ndarray
    wave: np.ndarray
    end: int  # 数据截止（最后一条提醒的时间），持仓走不完的交易不计


def load(dataset: str, start: int = 0) -> Data:
    source = RESULTS / dataset
    bars = np.load(source / "bars.npy", mmap_mode="r")
    meta_raw = np.load(source / "meta.npy")
    names = (source / "meta_names.txt").read_text(encoding="utf-8").split("\n")
    keep = meta_raw[:, 0] >= start
    order = np.flatnonzero(keep)
    order = order[np.argsort(meta_raw[order, 0], kind="stable")]
    meta = {name: meta_raw[order, k] for k, name in enumerate(names)}
    symbol = np.load(source / "symbol.npy")[order]
    _, sym_id = np.unique(symbol, return_inverse=True)
    # 按时间排好序再复制进内存：组合模拟按入场时间顺序扫描。
    return Data(
        np.ascontiguousarray(bars[order]),
        meta,
        symbol,
        sym_id.astype(np.int64),
        meta["closed_at"].astype(np.int64),
        meta["direction"].astype(np.int64),
        meta["wave"].astype(np.int64),
        int(meta_raw[:, 0].max()),
    )


@dataclass
class Rule:
    delay: int = 10  # 提醒后等这么多分钟，按下一分钟开盘价入场
    confirm: float = 0.0  # 入场价须比提醒价顺方向至少多走这么多（做多高于、做空低于）；NaN 不要求
    stop_gap: float = 0.05  # 止损放在等待期极值（做多最低价、做空最高价）外侧这么多
    stop_max: float = 0.20  # 止损离入场价最多这么远
    stop_fallback: float = 0.06  # 等待期极值贴着入场价（不到 0.5%）时改用固定止损
    trail: float = 0.15  # 从持仓期最有利价格回撤这么多平仓
    trail_atr: float = 3.0  # 回撤至少为入场前 60 分钟平均振幅 × 这个倍数；NaN 不放宽
    hold: int = 1080  # 最长持仓分钟
    fee: float = 0.0005  # 单边手续费
    max_chase: float = np.nan  # 入场价相对提醒价顺方向最多走这么多（不追太远）；NaN 不限
    take: float = np.nan  # 固定止盈；NaN 不设

    def params(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, f.name)) for f in fields(self))


@nb.njit(cache=True)
def _simulate(bars, direction, p):
    delay, confirm, stop_gap, stop_max, stop_fallback, trail, trail_atr, hold, fee, max_chase, take = p
    n = bars.shape[0]
    width = bars.shape[2]
    net = np.full(n, np.nan)
    entry_at = np.full(n, -1)
    exit_at = np.full(n, -1)
    # 0 正常平仓，1 数据中断（下架、停牌、数据截止），按最后一个收盘价平仓
    broken = np.zeros(n, dtype=np.int8)
    d = int(delay)
    for a in range(n):
        o = bars[a, 0]
        h = bars[a, 1]
        lo = bars[a, 2]
        c = bars[a, 3]
        s = direction[a]
        ap = c[PRE - 1]
        e = PRE + d
        if e >= width or np.isnan(o[e]):
            continue
        price = o[e]
        move = (price / ap - 1) * s
        if not np.isnan(confirm) and move < confirm:
            continue
        if not np.isnan(max_chase) and move > max_chase:
            continue
        # 等待期极值：等 0 分钟时用提醒 K 线自身。
        lo_k = PRE if d > 0 else PRE - 1
        ext = lo[lo_k] if s > 0 else h[lo_k]
        for k in range(lo_k, e):
            if s > 0:
                ext = min(ext, lo[k])
            else:
                ext = max(ext, h[k])
        if s > 0:
            stop = max(ext * (1 - stop_gap), price * (1 - stop_max))
            if stop >= price * 0.995:
                stop = price * (1 - stop_fallback)
        else:
            stop = min(ext * (1 + stop_gap), price * (1 + stop_max))
            if stop <= price * 1.005:
                stop = price * (1 + stop_fallback)
        tr = trail
        if not np.isnan(trail_atr):
            rng = 0.0
            cnt = 0
            for k in range(e - 60, e):
                if k >= 0 and not np.isnan(c[k]) and c[k] > 0:
                    rng += (h[k] - lo[k]) / c[k]
                    cnt += 1
            if cnt > 0:
                tr = max(trail, trail_atr * rng / cnt)
        tp = np.nan
        if not np.isnan(take):
            tp = price * (1 + take * s)
        best = price
        px = np.nan
        k_out = -1
        last = e
        for k in range(e, min(e + int(hold), width)):
            if np.isnan(c[k]):
                break
            last = k
            if s > 0:
                line = max(stop, best * (1 - tr))
                if not np.isnan(tp) and o[k] >= tp:
                    px = o[k]
                elif o[k] <= line:
                    px = o[k]
                elif lo[k] <= line:
                    px = line
                elif not np.isnan(tp) and h[k] >= tp:
                    px = tp
                best = max(best, h[k])
            else:
                line = min(stop, best * (1 + tr))
                if not np.isnan(tp) and o[k] <= tp:
                    px = o[k]
                elif o[k] >= line:
                    px = o[k]
                elif h[k] >= line:
                    px = line
                elif not np.isnan(tp) and lo[k] <= tp:
                    px = tp
                best = min(best, lo[k])
            if not np.isnan(px):
                k_out = k
                break
        if k_out < 0:
            k_out = last
            px = c[last]
            if last < e + int(hold) - 1:
                broken[a] = 1
        r = px / price
        net[a] = (r - 1) * s - fee * (1 + r)
        entry_at[a] = e
        exit_at[a] = k_out
    return net, entry_at, exit_at, broken


@dataclass
class Trades:
    net: np.ndarray
    t_in: np.ndarray  # 入场时刻（秒）
    t_out: np.ndarray  # 平仓时刻（秒，平仓那一分钟收盘）
    broken: np.ndarray


def simulate(data: Data, rule: Rule) -> Trades:
    net, entry_at, exit_at, broken = _simulate(data.bars, data.direction, rule.params())
    t_in = data.closed_at + (entry_at - PRE) * 60
    t_out = data.closed_at + (exit_at - PRE + 1) * 60
    # 数据截止前持仓没走完的不算（结果未知）；其余数据中断的按最后收盘价平仓，代表下架、长时间停牌的真实损失。
    unfinished = (broken == 1) & (t_out >= data.end - 60 * 60)
    net = np.where(unfinished, np.nan, net)
    return Trades(net, t_in, t_out, broken)


@nb.njit(cache=True)
def _portfolio(order, t_in, t_out, sym, direction, slots, no_stack, n_sym):
    keep = np.zeros(len(t_in), dtype=np.bool_)
    open_out = np.full(slots, -1, dtype=np.int64)
    # 同一合约同方向持仓到什么时候（不叠仓）
    busy = np.full((n_sym, 2), -1, dtype=np.int64)
    for j in order:
        free = -1
        for k in range(slots):
            if open_out[k] <= t_in[j]:
                free = k
                break
        if free < 0:
            continue
        side = 0 if direction[j] > 0 else 1
        if no_stack and busy[sym[j], side] > t_in[j]:
            continue
        open_out[free] = t_out[j]
        busy[sym[j], side] = t_out[j]
        keep[j] = True
    return keep


def portfolio(data: Data, trades: Trades, mask: np.ndarray, slots: int, no_stack: bool = True) -> np.ndarray:
    """最多同时持 slots 笔：按入场时间先到先买，满仓或同一合约同方向已有持仓时跳过。"""
    ok = mask & ~np.isnan(trades.net)
    index = np.flatnonzero(ok)
    order = index[np.argsort(trades.t_in[index], kind="stable")]
    n_sym = int(data.sym_id.max()) + 1
    return _portfolio(order, trades.t_in, trades.t_out, data.sym_id, data.direction, slots, no_stack, n_sym)


QUARTER_STARTS = [beijing(y, m) for y in range(2021, 2027) for m in (1, 4, 7, 10)]


@dataclass
class Stats:
    n: int
    total: float  # 本金收益率（%），不复利：每笔占本金 1/slots
    quarters: np.ndarray  # 各季度收益率（%）
    neg_q: int
    worst_q: float
    max_dd: float  # 按平仓时间累计的已实现收益曲线的最大回撤（本金 %）
    sharpe: float  # 按日已实现收益算的年化夏普
    win: float
    t: float
    top10_share: float  # 最好 10 笔占总收益的比例


def stats(data: Data, trades: Trades, selected: np.ndarray, slots: int, start: int, end: int | None = None) -> Stats:
    end = end or data.end
    sel = selected & (data.closed_at >= start) & (data.closed_at < end) & ~np.isnan(trades.net)
    net = trades.net[sel]
    unit = 100.0 / slots
    t_out = trades.t_out[sel]
    edges = [q for q in QUARTER_STARTS if start <= q < end]
    bounds = [start, *[q for q in edges if q > start], end]
    quarters = np.array(
        [net[(t_out >= a) & (t_out < b)].sum() * unit for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    )
    order = np.argsort(t_out)
    curve = np.cumsum(net[order]) * unit
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:] - curve)) if len(curve) else 0.0
    day = (t_out - start) // 86400
    days = int((end - start) // 86400) + 1
    daily = np.bincount(np.clip(day, 0, days - 1).astype(np.int64), weights=net * unit, minlength=days)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else np.nan
    total = float(net.sum() * unit)
    top = np.sort(net)[-10:].sum() * unit if len(net) else 0.0
    # 最后一个季度通常不完整，不算进亏损季度数。
    full = quarters[:-1] if len(quarters) > 1 else quarters
    return Stats(
        int(sel.sum()),
        total,
        quarters,
        int((full < 0).sum()),
        float(full.min()) if len(full) else np.nan,
        dd,
        sharpe,
        float((net > 0).mean()) if len(net) else np.nan,
        float(net.mean() / net.std(ddof=1) * np.sqrt(len(net))) if len(net) > 2 else np.nan,
        float(top / total) if total > 0 else np.nan,
    )


def fmt(s: Stats) -> str:
    return (
        f"n={s.n:5d} 合计{s.total:+7.0f}% 夏普{s.sharpe:5.2f} 回撤{s.max_dd:5.0f}% "
        f"亏损季{s.neg_q:2d}/{len(s.quarters) - 1} 最差季{s.worst_q:+5.0f}% "
        f"胜{s.win:.0%} t{s.t:5.2f} 前10占{s.top10_share:4.0%}"
    )


# ---------------------------------------------------------------- 市场状态


class Market:
    """全市场逐小时状态，由 r3_market.py 的输出算出；alert_at(values, closed_at) 取提醒前最后一个已收盘小时。"""

    def __init__(self, path: Path = MARKET) -> None:
        market = np.load(path)
        self.hours = market["hours"].astype(np.int64)
        close = market["close"]
        quote = market["quote"]
        symbols = list(market["symbols"])
        self.symbols = symbols
        self.n_hours = len(self.hours)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.breadth = {days: self._breadth(close, days * 24) for days in (7, 30)}
            btc = close[:, symbols.index("BTC_USDT")]
            btc = _ffill(btc)
            self.btc = btc
            self.btc_ma = {days: _rolling_mean(btc, days * 24) for days in (20, 50, 100, 200)}
            # 全市场中位涨跌：各合约 N 小时涨跌的中位数（只算有数据的合约）。
            self.median_ret = {}
            for hours in (24, 168, 720):
                prev = np.roll(close, hours, 0)
                prev[:hours] = np.nan
                self.median_ret[hours] = np.nanmedian(close / prev - 1, axis=1)
            total = quote.sum(1)
            day = _rolling_sum(total, 24)
            self.market_quote_24h = day
            # 全市场放量：最近 24 小时总成交额 ÷ 之前 30 天的日均。
            self.volume_surge = day / (_rolling_sum(np.roll(total, 24), 720) / 30)

    @staticmethod
    def _breadth(close: np.ndarray, window: int) -> np.ndarray:
        values = np.cumsum(np.nan_to_num(close), 0)
        counts = np.cumsum(~np.isnan(close), 0)
        sma = (values - np.roll(values, window, 0)) / np.maximum(counts - np.roll(counts, window, 0), 1)
        sma[:window] = np.nan
        above = np.where(np.isnan(close) | np.isnan(sma), np.nan, close > sma)
        valid = ~np.isnan(above)
        return np.where(valid.sum(1) > 20, np.nansum(above, 1) / np.maximum(valid.sum(1), 1), np.nan)

    def at(self, values: np.ndarray, closed_at: np.ndarray) -> np.ndarray:
        slot = (closed_at - self.hours[0]) // 3600 - 1
        inside = (slot >= 0) & (slot < self.n_hours)
        return np.where(inside, values[np.clip(slot, 0, self.n_hours - 1)], np.nan)


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(x), 0, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan)
    out[window - 1 :] = c[window:] - c[:-window]
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(x, window) / window


def alert_density(data: Data, window: int, direction: int | None = None) -> np.ndarray:
    """每条提醒之前 window 秒内全市场的提醒数（不含自己），direction 为 None 时两个方向都算。"""
    t = data.closed_at if direction is None else data.closed_at[data.direction == direction]
    t = np.sort(t)
    return (np.searchsorted(t, data.closed_at, side="left") - np.searchsorted(t, data.closed_at - window)).astype(float)


def trimmed(data: Data, trades: Trades, selected: np.ndarray, slots: int, start: int, days: int = 5) -> float:
    """去掉组合最好 days 天（按平仓日的已实现收益）后的本金收益率（%）：收益是否只靠少数几天。"""
    sel = selected & (data.closed_at >= start) & ~np.isnan(trades.net)
    day = (trades.t_out[sel] + 8 * 3600) // 86400
    _, inv = np.unique(day, return_inverse=True)
    daily = np.bincount(inv, weights=trades.net[sel]) * 100.0 / slots
    return float(daily.sum() - np.sort(daily)[::-1][:days].sum())
