"""第三轮探索：给宽松反弹回落候选（cand3_dev.npz 按冷却 30 秒取提醒）补算更多只用提醒前数据的特征，写 feat3.npz。

候选：10 秒窗口、涨跌幅 ≥ 1%、1 ATR、确认 1、5 笔、500 万池、前置走势 ≤ −2%、成交暴增 ≥ 6，冷却 30 秒。
"""

from __future__ import annotations

import gzip
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA, ROOT, contracts, multiplier
from engine import load_trades, price_at

FUNDING = ROOT / "data" / "backtest" / "funding"
PRIOR = (30, 60, 120, 300, 900, 1800, 3600, 4 * 3600, 12 * 3600)
FLOW = (10, 60, 300, 900)


def cooldown(symbol: np.ndarray, t: np.ndarray, mask: np.ndarray, cd: float) -> np.ndarray:
    out, last_sym, last_t = [], None, -1e18
    for i in np.flatnonzero(mask):
        if symbol[i] != last_sym:
            last_sym, last_t = symbol[i], -1e18
        if t[i] - last_t >= cd:
            out.append(i)
            last_t = t[i]
    return np.array(out, int)


def funding_rate(symbol: str, month: str, tt: np.ndarray) -> np.ndarray:
    rows = []
    for m in (ALL_MONTHS[max(ALL_MONTHS.index(month) - 1, 0)], month) if month != "202510" else ("202509", month):
        path = FUNDING / m / f"{symbol}-{m}.csv.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                rows += [tuple(map(float, line.split(","))) for line in f.read().split() if line]
    if not rows:
        return np.full(len(tt), np.nan)
    a = np.array(sorted(rows))
    k = np.searchsorted(a[:, 0], tt, side="right") - 1
    return np.where(k >= 0, a[np.clip(k, 0, None), 1], np.nan)


def group_features(args):
    key, idx, tt, direction = args
    symbol, month = key.rsplit("-", 1)
    t, p, s = load_trades(symbol, month)
    mult = multiplier(contracts().get(symbol)) or 1.0
    q = np.abs(s).astype(np.float64) * mult * p
    cum_q = np.r_[0.0, np.cumsum(q)]
    cum_signed = np.r_[0.0, np.cumsum(np.where(s > 0, q, -q))]
    cum_n = np.arange(len(t) + 1, dtype=np.float64)
    entry = price_at(t, p, tt)
    d = direction.astype(np.float64)
    out: dict[str, np.ndarray] = {}
    for k in PRIOR:
        out[f"ret_{k}"] = d * (entry / price_at(t, p, tt - k) - 1) * 100

    def window_sum(cum, lo, hi):
        return cum[np.searchsorted(t, hi, side="right")] - cum[np.searchsorted(t, lo, side="right")]

    hour_q = window_sum(cum_q, tt - 3600, tt) / 60
    hour_n = window_sum(cum_n, tt - 3600, tt) / 60
    for k in FLOW:
        qk = window_sum(cum_q, tt - k, tt)
        out[f"quote_{k}"] = np.log10(qk + 1)
        out[f"qratio_{k}"] = qk / (k / 60) / np.maximum(hour_q, 1e-9)
        out[f"imb_{k}"] = d * window_sum(cum_signed, tt - k, tt) / np.maximum(qk, 1e-9)
        out[f"nratio_{k}"] = window_sum(cum_n, tt - k, tt) / (k / 60) / np.maximum(hour_n, 1e-9)
    # 急动之前（不含窗口）的成交：区分「急动本身带来的成交」与「急动前已经涌入」
    pre_n = window_sum(cum_n, tt - 70, tt - 10)
    out["nratio_pre60"] = pre_n / np.maximum(hour_n, 1e-9)
    out["imb_pre300"] = d * window_sum(cum_signed, tt - 310, tt - 10) / np.maximum(window_sum(cum_q, tt - 310, tt - 10), 1e-9)
    out["quote_hour"] = np.log10(hour_q * 60 + 1)
    out["quote_day"] = np.log10(window_sum(cum_q, tt - 86400, tt) + 1)
    for span in (900, 3600):
        grid = tt[:, None] - span + 10.0 * np.arange(span // 10 + 1)[None, :]
        px = price_at(t, p, grid.ravel()).reshape(grid.shape)
        out[f"vol_{span}"] = np.nanstd(np.diff(np.log(px), axis=1), axis=1) * 100
    # 前置期（窗口开始前 15 分钟、60 分钟）的反向极值：离它多远、多久之前，即急动前是否已经开始回头
    for span in (900, 3600):
        lowd = np.empty(len(tt))
        age = np.empty(len(tt))
        rng = np.empty(len(tt))
        for i, x in enumerate(tt):
            a = np.searchsorted(t, x - 10 - span, side="left")
            b = np.searchsorted(t, x - 10, side="right")
            seg = p[a:b]
            if not len(seg):
                lowd[i] = age[i] = rng[i] = np.nan
                continue
            j = int(np.argmin(seg) if d[i] > 0 else np.argmax(seg))
            base = p[b - 1]
            lowd[i] = d[i] * (base / seg[j] - 1) * 100
            age[i] = x - 10 - t[a + j]
            rng[i] = (seg.max() / seg.min() - 1) * 100
        out[f"offext_{span}"] = lowd
        out[f"extage_{span}"] = age
        out[f"range_{span}"] = rng
    hi = np.empty(len(tt))
    lo = np.empty(len(tt))
    for i, x in enumerate(tt):
        a = np.searchsorted(t, x - 24 * 3600, side="left")
        b = np.searchsorted(t, x - 60, side="right")
        seg = p[a:b]
        hi[i] = seg.max() if len(seg) else np.nan
        lo[i] = seg.min() if len(seg) else np.nan
    out["ext24h"] = np.where(d > 0, entry / hi - 1, 1 - entry / lo) * 100
    out["opp24h"] = np.where(d > 0, entry / lo - 1, 1 - entry / hi) * 100
    out["hour"] = ((tt % 86400) // 3600).astype(np.float64)
    out["logprice"] = np.log10(entry)
    out["funding"] = -d * funding_rate(symbol, month, tt) * 1e4
    for span in (900, 3600, 86400):
        same = np.zeros(len(tt))
        opp = np.zeros(len(tt))
        for i in range(len(tt)):
            j0 = np.searchsorted(tt, tt[i] - span, side="left")
            prev = direction[j0:i]
            same[i] = (prev == direction[i]).sum()
            opp[i] = (prev != direction[i]).sum()
        out[f"same_{span}"] = same
        out[f"opp_{span}"] = opp
    return idx, out


def market_features(tt, direction, months):
    out = {f"{c}_{k}": np.full(len(tt), np.nan) for c in ("btc", "eth") for k in (60, 300, 900, 3600)}
    for m in np.unique(months):
        sel = np.flatnonzero(months == m)
        for c, sym in (("btc", "BTC_USDT"), ("eth", "ETH_USDT")):
            t, p, _ = load_trades(sym, ALL_MONTHS[m])
            now = price_at(t, p, tt[sel])
            for k in (60, 300, 900, 3600):
                out[f"{c}_{k}"][sel] = direction[sel] * (now / price_at(t, p, tt[sel] - k) - 1) * 100
    return out


if __name__ == "__main__":
    c = dict(np.load(DATA / "cand3_dev.npz", allow_pickle=True))
    order = np.lexsort((c["t"], c["symbol"]))
    c = {k: v[order] for k, v in c.items()}
    mask = (c["lchg"] >= 1.0) & (c["latrm"] >= 1.0) & c["real"]
    keep = cooldown(c["symbol"], c["t"], mask, 30)
    c = {k: v[keep] for k, v in c.items()}
    n = len(keep)
    jobs = []
    for key in np.unique(c["symbol"]):
        idx = np.flatnonzero(c["symbol"] == key)
        jobs.append((key, idx, c["t"][idx], c["dir"][idx]))
    cols: dict[str, np.ndarray] = {}
    with Pool(10) as pool:
        for idx, feats in pool.imap_unordered(group_features, jobs):
            for k, v in feats.items():
                cols.setdefault(k, np.full(n, np.nan))[idx] = v
    cols |= market_features(c["t"], c["dir"], c["month"])
    np.savez(DATA / "feat3.npz", **c, **{f"f_{k}": v for k, v in cols.items()})
    print(n, "条，特征", len(cols), flush=True)
