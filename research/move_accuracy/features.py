"""第二轮：给一批宽松参数下的提醒补算更多特征（只用提醒时刻之前的成交），供 model.py 估计可预测性的上限。

特征都能在实时检测器里算出（只需保留最近一段逐笔价格与成交额），标签沿用引擎的提醒后 1~10 分钟收益。
用法：features.py 名称 变体 涨跌幅 ATR 确认 笔数 冷却 合约池档 [月份 ...]，输出 feat_<名称>.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
from common import DATA, MONTHS, contracts, multiplier
from engine import load_trades, price_at
from sim import Params, alert_rows

PRIOR = (10, 30, 60, 300, 900, 1800, 3600, 4 * 3600)
FLOW = (60, 300, 900)


def group_features(args: tuple[str, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
        a = np.searchsorted(t, lo, side="right")
        b = np.searchsorted(t, hi, side="right")
        return cum[b] - cum[a]

    hour_q = window_sum(cum_q, tt - 3600, tt) / 60
    hour_n = window_sum(cum_n, tt - 3600, tt) / 60
    for k in FLOW:
        qk = window_sum(cum_q, tt - k, tt)
        out[f"quote_{k}"] = np.log10(qk + 1)
        out[f"qratio_{k}"] = qk / (k / 60) / np.maximum(hour_q, 1e-9)
        out[f"imb_{k}"] = d * window_sum(cum_signed, tt - k, tt) / np.maximum(qk, 1e-9)
        out[f"nratio_{k}"] = window_sum(cum_n, tt - k, tt) / (k / 60) / np.maximum(hour_n, 1e-9)
    out["quote_hour"] = np.log10(hour_q * 60 + 1)
    # 波动率：此前 15 分钟、60 分钟每 10 秒收益的标准差
    for span in (900, 3600):
        grid = tt[:, None] - span + 10.0 * np.arange(span // 10 + 1)[None, :]
        px = price_at(t, p, grid.ravel()).reshape(grid.shape)
        out[f"vol_{span}"] = np.nanstd(np.diff(np.log(px), axis=1), axis=1) * 100
    # 离此前 6 小时最高价 / 最低价的距离（顺提醒方向：急涨看最高、急跌看最低，正数表示已越过）
    hi = np.empty(len(tt))
    lo = np.empty(len(tt))
    for i, x in enumerate(tt):
        a = np.searchsorted(t, x - 6 * 3600, side="left")
        b = np.searchsorted(t, x - 60, side="right")
        seg = p[a:b]
        hi[i] = seg.max() if len(seg) else np.nan
        lo[i] = seg.min() if len(seg) else np.nan
    out["ext6h"] = np.where(d > 0, entry / hi - 1, 1 - entry / lo) * 100
    out["hour"] = ((tt % 86400) // 3600).astype(np.float64)
    # 同一合约此前 15、60 分钟内同向 / 反向提醒次数（在这批宽松提醒里数）
    for span in (900, 3600):
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


def btc_features(tt: np.ndarray, direction: np.ndarray, months: np.ndarray) -> dict[str, np.ndarray]:
    out = {f"btc_{k}": np.full(len(tt), np.nan) for k in (10, 60, 300, 900)}
    for m in np.unique(months):
        sel = np.flatnonzero(months == m)
        t, p, _ = load_trades("BTC_USDT", MONTHS[m])
        now = price_at(t, p, tt[sel])
        for k in (10, 60, 300, 900):
            out[f"btc_{k}"][sel] = direction[sel] * (now / price_at(t, p, tt[sel] - k) - 1) * 100
    return out


if __name__ == "__main__":
    name, variant = sys.argv[1], sys.argv[2]
    a, b, c, trades, cd, pool = (float(x) for x in sys.argv[3:9])
    months = sys.argv[9:] or MONTHS
    ar = alert_rows(variant, months, Params(a, b, int(c), int(trades), cd, int(pool)))
    n = len(ar["t"])
    jobs = []
    for key in np.unique(ar["symbol"]):
        idx = np.flatnonzero(ar["symbol"] == key)
        idx = idx[np.argsort(ar["t"][idx], kind="stable")]
        jobs.append((key, idx, ar["t"][idx], ar["dir"][idx]))
    cols: dict[str, np.ndarray] = {}
    with Pool(10) as pool_:
        for idx, feats in pool_.imap_unordered(group_features, jobs):
            for k, v in feats.items():
                cols.setdefault(k, np.full(n, np.nan))[idx] = v
    cols |= btc_features(ar["t"], ar["dir"], ar["month"])
    keep = {k: v for k, v in ar.items() if k != "symbol"}
    np.savez(DATA / f"feat_{name}.npz", symbol=ar["symbol"].astype(str), **keep, **cols)
    print(name, n, "条，特征", len(cols) + len(keep), flush=True)
