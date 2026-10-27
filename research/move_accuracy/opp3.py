"""第三轮：给 cand3_<名称>.npz 的每一行补算反向极值距离，写 cand3_<名称>_opp.npz（行对齐）。

定义（PLAN.md 第三轮）：入场价（触发结算的成交价）相对此前 N 分钟已收盘 1 分钟 K 线（不含被判定那一秒所在的分钟）
的最低价（急拉）/ 最高价（急砸）的涨跌幅，顺提醒方向为正；K 线最早一根晚于 N 分钟前（覆盖不满）时为 NaN，不提醒。
K 线取下载站数据（含前两个月）；下架后重新上线的空档不算覆盖（见 group 内注释）。用法：opp3.py dev|r3
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
from common import DATA, load_klines, previous_month
from engine import load_trades, price_at

SPANS = (720, 1440, 2880)


def group(args):
    key, idx, sec, tt, direction = args
    symbol, month = key.rsplit("-", 1)
    t, p, _ = load_trades(symbol, month)
    entry = price_at(t, p, tt)
    k = [load_klines(symbol, m) for m in (previous_month(previous_month(month)), previous_month(month), month)]
    k = np.concatenate([x for x in k if len(x)])
    k = k[np.argsort(k[:, 0], kind="stable")]
    minute = k[:, 0].astype(np.int64) // 60
    hi, lo = k[:, 3], k[:, 4]
    now = sec.astype(np.int64) // 60
    out = {"entry": entry}
    for span in SPANS:
        a = np.searchsorted(minute, now - span, side="left")
        b = np.searchsorted(minute, now, side="left")
        res = np.full(len(sec), np.nan)
        for i in range(len(sec)):
            # 覆盖：N 分钟前那一刻要有交易中的 K 线。检测器用 REST 预热，无成交的分钟也有平线；下载站只有成交过的分钟，
            # 所以只要 N 分钟前最后一根 K 线距那一刻不到 1 天就算覆盖。更久的空档是下架后重新上线，检测器那边没有这段 K 线
            at_start = a[i] < len(minute) and minute[a[i]] == now[i] - span
            recent = a[i] > 0 and minute[a[i] - 1] >= now[i] - span - 1440
            if b[i] <= a[i] or not (at_start or recent):
                continue
            if direction[i] > 0:
                res[i] = (entry[i] / lo[a[i] : b[i]].min() - 1) * 100
            else:
                res[i] = (1 - entry[i] / hi[a[i] : b[i]].max()) * 100
        out[f"opp_{span}"] = res
    return idx, out


if __name__ == "__main__":
    name = sys.argv[1]
    d = np.load(DATA / f"cand3_{name}.npz", allow_pickle=True)
    sym = d["symbol"]
    jobs = []
    for key in np.unique(sym):
        idx = np.flatnonzero(sym == key)
        jobs.append((key, idx, d["sec"][idx], d["t"][idx], d["dir"][idx]))
    cols: dict[str, np.ndarray] = {}
    with Pool(int(os.environ.get("WORKERS", "10"))) as pool:
        for idx, f in pool.imap_unordered(group, jobs):
            for k, v in f.items():
                cols.setdefault(k, np.full(len(sym), np.nan))[idx] = v
    np.savez(DATA / f"cand3_{name}_opp.npz", **cols)
    print(name, "完成", {k: int(np.isfinite(v).sum()) for k, v in cols.items()}, "/", len(sym))
