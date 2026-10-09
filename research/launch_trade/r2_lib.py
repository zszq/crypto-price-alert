"""第二版研究：在 data/backtest/results/r2 上用 numba 逐笔模拟带量价确认的入场与出场规则。

bars 第 PRE-1 列是提醒所在分钟（提醒价 = 它的收盘价），第 PRE 列是提醒后第一分钟。
所有规则只用当时已经能看到的数据：入场判断只看入场那一分钟开盘之前收盘的 K 线和这一分钟的开盘价，
持仓中每分钟先按开盘价/最低价检查亏损的线、再检查止盈，最后才用这一分钟的高点更新最高价。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone

import numba as nb
import numpy as np
from paths import RESULTS

R2 = RESULTS / "r2"
PRE = 1440
POST = 720
BEIJING = timezone(timedelta(hours=8))
START = int(datetime(2026, 4, 10, tzinfo=BEIJING).timestamp())
SPLIT = int(datetime(2026, 8, 1, tzinfo=BEIJING).timestamp())


@dataclass
class Data:
    bars: np.ndarray
    meta: dict[str, np.ndarray]
    symbol: np.ndarray
    closed_at: np.ndarray
    wave: np.ndarray
    train: np.ndarray


def load() -> Data:
    bars = np.load(R2 / "bars.npy", mmap_mode="r")
    meta_raw = np.load(R2 / "meta.npy")
    names = (R2 / "meta_names.txt").read_text(encoding="utf-8").split("\n")
    keep = meta_raw[:, 0] >= START
    bars = np.ascontiguousarray(bars[keep])
    meta = {name: meta_raw[keep, k] for k, name in enumerate(names)}
    symbol = np.load(R2 / "symbol.npy")[keep]
    return Data(bars, meta, symbol, meta["closed_at"], meta["wave"], meta["closed_at"] < SPLIT)


@dataclass
class Rule:
    # ---- 入场 ----
    entry_mode: int = 0  # 0 固定等待后确认；1 突破等待期高点追入；2 回踩支撑位挂单买
    delay: int = 10  # 等待分钟数（模式 1/2 为观察期长度）
    confirm: float = 0.02  # 入场价相对提醒价的最低涨幅（模式 0）；NaN 不要求
    entry_window: int = 30  # 模式 1/2：观察期结束后最多再等这么多分钟
    pullback: float = 0.0  # 模式 2：挂单价 = 支撑位 × (1 + pullback)
    support_kind: int = 0  # 支撑位：0 基准期高点（被突破的压力位）；1 提醒价；2 等待期最低价
    # 等待期成交额 ÷ (等待分钟数 × 基准期每分钟平均成交额) 的下限：放量是否持续；0 不要求
    wait_volume_ratio: float = 0.0
    # 等待期成交额 ÷ 提醒窗口每分钟成交额 × 窗口分钟数 的下限：等待期的量是否跟得上启动时的量；0 不要求
    wait_vs_window: float = 0.0
    above_support: float = np.nan  # 入场价 ≥ 支撑位（基准期高点）× (1 + x)；NaN 不要求
    near_high: float = np.nan  # 等待期最后一根收盘 ≥ 等待期最高价 × (1 - x)；NaN 不要求
    # 上方压力位：0 不看；1 24h 高点；2 7 天高点；3 30 天高点。room_min：压力位在入场价上方至少这么远，或已突破
    resistance_kind: int = 0
    room_min: float = np.nan
    blue_sky: int = 0  # 1：只买已在压力位之上的（没有上方压力）；-1：只买还在压力位之下的
    btc_min: float = np.nan  # 提醒前 1 小时 BTC 涨跌下限
    max_extension: float = np.nan  # 入场价相对窗口起点的最大涨幅（不追太高）
    # ---- 出场 ----
    hold: int = 300
    stop: float = 0.07
    stop_support: int = -1  # 止损放在支撑位下方：-1 不用；0/1/2 同 support_kind
    stop_support_gap: float = 0.01  # 支撑位下方留的余量
    stop_max: float = 0.10  # 支撑止损最多亏这么多（太远就用这个）
    trail: float = 0.15
    trail_atr: float = np.nan  # 用入场前 60 分钟平均振幅 × k 作为回撤幅度（取代 trail，按 trail 设下限）
    trail_after: float = 0.0  # 最高价涨过入场价这么多之后才启用回撤止损
    trail_tighten_at: float = np.nan  # 最高价涨过这么多之后，回撤收紧到 trail_tight
    trail_tight: float = 0.08
    breakeven_at: float = np.nan  # 最高价涨过这么多后，止损抬到保本（入场价 + 0.2%）
    time_stop: int = 0  # 持仓这么多分钟后，最高价还没涨过 time_gain 就按收盘离场；0 不用
    time_gain: float = 0.03
    vol_exit_ratio: float = 0.0  # 最近 vol_exit_window 分钟成交额 < 入场时同长度成交额 × 比例，且价格低于最高价，就离场
    vol_exit_window: int = 15
    vol_exit_min_hold: int = 30
    take: float = np.nan  # 固定止盈
    take_resistance: int = 0  # 1：止盈放在上方压力位（resistance_kind）下方 0.5%，只在压力位高于入场价 3% 以上时
    take_fraction: float = 1.0  # 止盈时平掉的比例，剩下的继续按回撤止损跑
    fee: float = 0.0005
    # 用收盘价判断回撤止损（影线刺破不算，收盘跌破才按收盘价离场）；固定止损仍按盘中最低价，防止急跌失控。
    trail_on_close: int = 0
    # 收盘跌破支撑位 × (1 - support_exit_gap) 就按收盘离场：-1 不用；0/1/2 同 support_kind。
    support_exit: int = -1
    support_exit_gap: float = 0.0
    # 只在持仓这么多分钟后才检查支撑位，刚入场时的来回不算。
    support_exit_after: int = 0

    def key(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))


@nb.njit(cache=True)
def _simulate(bars, alert_price, support0, resist, btc1h, start_price, base_q, window_q, p):
    (
        entry_mode,
        delay,
        confirm,
        entry_window,
        pullback,
        support_kind,
        wait_volume_ratio,
        wait_vs_window,
        above_support,
        near_high,
        room_min,
        blue_sky,
        btc_min,
        max_extension,
        hold,
        stop,
        stop_support,
        stop_support_gap,
        stop_max,
        trail,
        trail_atr,
        trail_after,
        trail_tighten_at,
        trail_tight,
        breakeven_at,
        time_stop,
        time_gain,
        vol_exit_ratio,
        vol_exit_window,
        vol_exit_min_hold,
        take,
        take_resistance,
        take_fraction,
        fee,
        trail_on_close,
        support_exit,
        support_exit_gap,
        support_exit_after,
    ) = p
    n = bars.shape[0]
    net = np.full(n, np.nan)
    entry_at = np.full(n, -1)
    exit_at = np.full(n, -1)
    for a in range(n):
        o = bars[a, 0]
        h = bars[a, 1]
        lo = bars[a, 2]
        c = bars[a, 3]
        q = bars[a, 4]
        ap = alert_price[a]
        w0 = PRE  # 提醒后第一分钟
        d = int(delay)
        if np.isnan(c[w0 + d - 1]):
            continue
        # 等待期统计（只用已收盘的 K 线）
        wait_high = -1.0
        wait_low = 1e300
        wait_q = 0.0
        for k in range(w0, w0 + d):
            wait_high = max(wait_high, h[k])
            wait_low = min(wait_low, lo[k])
            wait_q += q[k]
        if wait_volume_ratio > 0 and (base_q[a] <= 0 or wait_q / (d * base_q[a]) < wait_volume_ratio):
            continue
        if wait_vs_window > 0 and wait_q / (window_q[a] / 5.0 * d) < wait_vs_window:
            continue
        if not np.isnan(near_high) and c[w0 + d - 1] < wait_high * (1 - near_high):
            continue
        if not np.isnan(btc_min) and (np.isnan(btc1h[a]) or btc1h[a] < btc_min):
            continue
        if int(support_kind) == 0:
            support = support0[a]
        elif int(support_kind) == 1:
            support = ap
        else:
            support = wait_low
        # 入场
        e = -1
        price = 0.0
        if int(entry_mode) == 0:
            e = w0 + d
            if np.isnan(o[e]):
                continue
            price = o[e]
            if not np.isnan(confirm) and price < ap * (1 + confirm):
                continue
        elif int(entry_mode) == 1:
            level = wait_high * 1.001
            for k in range(w0 + d, min(w0 + d + int(entry_window), PRE + POST)):
                if np.isnan(c[k]):
                    break
                if h[k] >= level:
                    e = k
                    price = max(o[k], level)
                    break
            if e < 0:
                continue
        else:
            level = support * (1 + pullback)
            for k in range(w0 + d, min(w0 + d + int(entry_window), PRE + POST)):
                if np.isnan(c[k]):
                    break
                if lo[k] <= level:
                    e = k
                    price = min(o[k], level)
                    break
            if e < 0:
                continue
        if not np.isnan(above_support) and price < support0[a] * (1 + above_support):
            continue
        if not np.isnan(max_extension) and price > start_price[a] * (1 + max_extension):
            continue
        r = resist[a]
        if int(blue_sky) == 1 and not (price > r):
            continue
        if int(blue_sky) == -1 and not (price <= r):
            continue
        if not np.isnan(room_min) and price <= r and r / price - 1 < room_min:
            continue
        # 止损线
        stop_line = price * (1 - stop)
        if int(stop_support) >= 0:
            if int(stop_support) == 0:
                s = support0[a]
            elif int(stop_support) == 1:
                s = ap
            else:
                s = wait_low
            stop_line = max(s * (1 - stop_support_gap), price * (1 - stop_max))
            if stop_line >= price * 0.995:
                stop_line = price * (1 - stop)
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
            tp = price * (1 + take)
        if int(take_resistance) == 1 and r > price * 1.03:
            tp = r * 0.995
        # 入场时的成交额基准（量能衰竭出场用）
        vw = int(vol_exit_window)
        entry_vol = 0.0
        for k in range(e - vw, e):
            if k >= 0 and not np.isnan(q[k]):
                entry_vol += q[k]
        peak = price
        remaining = 1.0
        realized = 0.0
        exit_k = -1
        tight_hit = False
        be_hit = False
        for k in range(e, e + int(hold)):
            if k >= PRE + POST or np.isnan(c[k]):
                exit_k = -2
                break
            line = stop_line
            if be_hit:
                line = max(line, price * 1.002)
            cur_tr = trail_tight if tight_hit else tr
            trail_line = 0.0
            if peak >= price * (1 + trail_after):
                trail_line = peak * (1 - cur_tr)
                if int(trail_on_close) == 0:
                    line = max(line, trail_line)
            if o[k] <= line:
                realized += remaining * (o[k] / price - 1 - fee * o[k] / price)
                remaining = 0.0
                exit_k = k
                break
            if lo[k] <= line:
                realized += remaining * (line / price - 1 - fee * line / price)
                remaining = 0.0
                exit_k = k
                break
            if not np.isnan(tp) and h[k] >= tp and remaining > 0:
                part = take_fraction if remaining == 1.0 else 0.0
                if part > 0:
                    px = max(o[k], tp)
                    realized += part * (px / price - 1 - fee * px / price)
                    remaining -= part
                    if remaining <= 1e-9:
                        exit_k = k
                        break
            held0 = k - e + 1
            if int(trail_on_close) == 1 and trail_line > 0 and c[k] <= trail_line:
                realized += remaining * (c[k] / price - 1 - fee * c[k] / price)
                remaining = 0.0
                exit_k = k
                break
            if int(support_exit) >= 0 and held0 >= int(support_exit_after):
                if int(support_exit) == 0:
                    sl = support0[a]
                elif int(support_exit) == 1:
                    sl = ap
                else:
                    sl = wait_low
                if c[k] < sl * (1 - support_exit_gap):
                    realized += remaining * (c[k] / price - 1 - fee * c[k] / price)
                    remaining = 0.0
                    exit_k = k
                    break
            peak = max(peak, h[k])
            if not np.isnan(trail_tighten_at) and peak >= price * (1 + trail_tighten_at):
                tight_hit = True
            if not np.isnan(breakeven_at) and peak >= price * (1 + breakeven_at):
                be_hit = True
            held = k - e + 1
            if time_stop > 0 and held == int(time_stop) and peak < price * (1 + time_gain):
                realized += remaining * (c[k] / price - 1 - fee * c[k] / price)
                remaining = 0.0
                exit_k = k
                break
            if vol_exit_ratio > 0 and held >= int(vol_exit_min_hold) and c[k] < peak:
                recent = 0.0
                for j in range(k - vw + 1, k + 1):
                    recent += q[j]
                if recent < entry_vol * vol_exit_ratio:
                    realized += remaining * (c[k] / price - 1 - fee * c[k] / price)
                    remaining = 0.0
                    exit_k = k
                    break
        if exit_k == -2:
            continue
        if remaining > 0:
            k = e + int(hold) - 1
            realized += remaining * (c[k] / price - 1 - fee * c[k] / price)
            exit_k = k
        net[a] = realized - fee
        entry_at[a] = e
        exit_at[a] = exit_k
    return net, entry_at, exit_at


class Engine:
    def __init__(self, data: Data) -> None:
        self.data = data
        m = data.meta
        self.alert_price = np.ascontiguousarray(data.bars[:, 3, PRE - 1])
        self.support0 = m["breakout"]
        self.start_price = m["start_price"]
        self.base_q = m["baseline_quote_per_min"]
        self.window_q = m["window_quote"]
        self.btc1h = m["btc_1h"]
        self.resist = {0: np.full(len(self.alert_price), np.inf), 1: m["high_24h"], 2: m["high_7d"], 3: m["high_30d"]}

    def run(self, rule: Rule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # resistance_kind 在外面换成对应的压力位数组，不进 numba 参数。
        params = tuple(float(getattr(rule, f.name)) for f in fields(rule) if f.name != "resistance_kind")
        return _simulate(
            self.data.bars,
            self.alert_price,
            self.support0,
            self.resist[rule.resistance_kind],
            self.btc1h,
            self.start_price,
            self.base_q,
            self.window_q,
            params,
        )


def evaluate(net: np.ndarray, mask: np.ndarray, data: Data, stake: float = 100.0) -> dict:
    out = {}
    for name, period in (("train", data.train), ("test", ~data.train), ("all", np.ones_like(mask))):
        x = net[mask & period & ~np.isnan(net)]
        out[name + "_n"] = len(x)
        out[name + "_mean"] = float(x.mean()) if len(x) else np.nan
        out[name + "_sum"] = float(x.sum() * stake)
    x = np.sort(net[mask & ~np.isnan(net)])
    out["minus5"] = float(x[:-5].sum() * stake) if len(x) > 5 else np.nan
    out["minus10"] = float(x[:-10].sum() * stake) if len(x) > 10 else np.nan
    out["t"] = float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan
    out["win"] = float((x > 0).mean()) if len(x) else np.nan
    months = np.array([datetime.fromtimestamp(t, BEIJING).month for t in data.closed_at])
    sel = mask & ~np.isnan(net)
    out["neg_months"] = int(sum(net[sel & (months == mo)].sum() < 0 for mo in range(4, 10)))
    return out


FOLDS = (
    int(datetime(2026, 6, 1, tzinfo=BEIJING).timestamp()),
    int(datetime(2026, 8, 1, tzinfo=BEIJING).timestamp()),
)


def fold_sums(net: np.ndarray, mask: np.ndarray, data: Data, stake: float = 100.0) -> tuple[float, float, float]:
    """4~5 月、6~7 月、8 月以后三段各自的合计：要求每段都好，比只看训练/检验两段更不容易被噪声骗。"""
    t = data.closed_at
    ok = mask & ~np.isnan(net)
    parts = (t < FOLDS[0], (t >= FOLDS[0]) & (t < FOLDS[1]), t >= FOLDS[1])
    return tuple(float(net[ok & part].sum() * stake) for part in parts)


def fmt(r: dict) -> str:
    return (
        f"训{r['train_n']:4d}笔 均{r['train_mean']:+.2%} 合{r['train_sum']:+5.0f} | "
        f"检{r['test_n']:4d}笔 均{r['test_mean']:+.2%} 合{r['test_sum']:+5.0f} | "
        f"全{r['all_n']:4d}笔 合{r['all_sum']:+5.0f} 去5{r['minus5']:+5.0f} 去10{r['minus10']:+5.0f} "
        f"t{r['t']:.2f} 胜{r['win']:.0%} 负月{r['neg_months']}"
    )


def wait_dip(data: Data, delay: int = 10) -> np.ndarray:
    """等待期（提醒后 delay 分钟）最低价相对提醒价的涨跌，负数表示回踩到提醒价下方。"""
    alert_price = data.bars[:, 3, PRE - 1]
    return np.nanmin(data.bars[:, 2, PRE : PRE + delay], 1) / alert_price - 1


# 止损放在等待期最低价下方 2%（支撑位止损），离入场价最多 9%；6% 只在支撑位贴着入场价时兜底。
SUPPORT_STOP = dict(stop=0.06, stop_support=2, stop_support_gap=0.02, stop_max=0.09)
# 第二轮的最稳组 A（需另加 wait_dip ≤ -0.25% 的回踩条件）；完整规则与选出过程见 FINDINGS.md「最稳方案 A 的完整规则」。
BEST = Rule(**SUPPORT_STOP, trail_atr=5, hold=600)
BEST_DIP = 0.0025


def fold_line(label: str, net: np.ndarray, mask: np.ndarray, data: Data, reference: np.ndarray | None = None) -> str:
    """一行输出：三段合计、去掉最好 10 笔、t 值，以及三段是否都超过参照。"""
    folds = np.array(fold_sums(net, mask, data))
    res = evaluate(net, mask, data)
    line = (
        f"  {label:34s} n={res['all_n']:3d} 三段 {folds[0]:+5.0f} {folds[1]:+5.0f} {folds[2]:+5.0f} "
        f"合{folds.sum():+5.0f} 去10{res['minus10']:+5.0f} t{res['t']:.2f}"
    )
    if reference is not None:
        line += f" 三段都超参照:{'是' if (folds > reference).all() else '否'}"
    return line


def alert_features(data: Data) -> dict[str, np.ndarray]:
    """提醒与等待期的量价特征，供过滤扫描使用；只用入场（提醒后第 10 分钟开盘）之前能看到的数据。"""
    b = data.bars
    m = data.meta
    alert_price = b[:, 3, PRE - 1]
    wait = slice(PRE, PRE + 10)
    t = data.closed_at
    ordered = np.sort(t)
    return {
        "量比": m["ratio"],
        "窗口成交额": m["window_quote"],
        "24h成交额": m["volume_24h"],
        "启动涨幅%": m["change"],
        "基准振幅": m["breakout"] / m["baseline_low"] - 1,
        "入场相对24h高": b[:, 0, PRE + 10] / m["high_24h"] - 1,
        "入场相对7d高": b[:, 0, PRE + 10] / m["high_7d"] - 1,
        "提醒K上影": (b[:, 1, PRE - 1] - b[:, 3, PRE - 1]) / np.maximum(b[:, 1, PRE - 1] - b[:, 2, PRE - 1], 1e-12),
        "等待期量/窗口量": np.nansum(b[:, 4, wait], 1) / (m["window_quote"] * 2),
        "等待期后5分/前5分量": np.nansum(b[:, 4, PRE + 5 : PRE + 10], 1)
        / np.maximum(np.nansum(b[:, 4, PRE : PRE + 5], 1), 1),
        "等待期涨幅": b[:, 0, PRE + 10] / alert_price - 1,
        "等待期回踩": wait_dip(data),
        "等待期上影": (np.nanmax(b[:, 1, wait], 1) - b[:, 3, PRE + 9]) / alert_price,
        "提醒前1h涨幅": alert_price / b[:, 3, PRE - 61] - 1,
        "提醒前24h涨幅": alert_price / b[:, 3, 0] - 1,
        "BTC1h": m["btc_1h"],
        "BTC4h": m["btc_4h"],
        "BTC24h": m["btc_24h"],
        "北京时": ((t // 3600) + 8) % 24,
        # 市场热度：前 60 分钟内全市场（全部波次）的提醒数。
        "前1h全市场提醒数": (np.searchsorted(ordered, t) - np.searchsorted(ordered, t - 3600)).astype(float),
        "价格": alert_price,
    }
