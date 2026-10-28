"""第四轮：给逐秒行补算反向极值距离（PLAN.md 第四轮），写 opp4/<变体>/<合约>-<月>.npy（与该文件该变体的行对齐）。

只算可能提醒的行（真实秒、涨跌幅 ≥ 1%、1 ATR、15 笔、在 800 万池），其余为 NaN（候选族任何一组都不会在这些行提醒）。
口径与第三轮 opp3.py 相同：入场价相对此前 1440 分钟已收盘 1 分钟 K 线（不含被判定那一秒所在分钟）的最低价（急涨）/ 最高价（急跌），
N 分钟前那一刻有 K 线、或之前一天内有 K 线才算覆盖。用法：spike_feat.py [月份 ...]（默认选参 24 个月与样本外）
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numba as nb
import numpy as np
from common import ALL_MONTHS, DATA, load_klines, previous_month
from engine import load_trades, price_at, rows_dir
from engine4 import ROWS4, ROWS4_LONG10

SPAN = 1440
OPP4 = DATA / "opp4"
# 变体 -> 行所在目录
FOLDERS = {"L10F10": rows_dir("L10F10"), "L30": ROWS4, "L180F30": ROWS4, "L180F10": ROWS4_LONG10}


@nb.njit(cache=True)
def _opp(minute, hi, lo, now, direction, entry, span):
    out = np.full(len(now), np.nan)
    for i in range(len(now)):
        a = np.searchsorted(minute, now[i] - span)
        b = np.searchsorted(minute, now[i])
        if b <= a:
            continue
        at_start = minute[a] == now[i] - span
        recent = a > 0 and minute[a - 1] >= now[i] - span - 1440
        if not (at_start or recent):
            continue
        if direction[i] > 0:
            out[i] = (entry[i] / lo[a:b].min() - 1) * 100
        else:
            out[i] = (1 - entry[i] / hi[a:b].max()) * 100
    return out


def one(args: tuple[str, str]) -> tuple[str, str, int]:
    variant, stem = args
    out_path = OPP4 / variant / f"{stem}.npy"
    if out_path.exists():
        return variant, stem, -1
    with np.load(FOLDERS[variant] / f"{stem}.npz") as d:
        if f"{variant}_sec" not in d:
            np.save(out_path, np.empty(0, np.float32))
            return variant, stem, 0
        cols = {c: d[f"{variant}_{c}"] for c in ("sec", "t", "dir", "real", "chg", "atrm", "trades", "pool")}
    cand = cols["real"] & (cols["chg"] >= 1.0) & (cols["atrm"] >= 1.0) & (cols["trades"] >= 15) & ((cols["pool"] >> 1) & 1 == 1)
    res = np.full(len(cols["sec"]), np.nan, np.float32)
    idx = np.flatnonzero(cand)
    if len(idx):
        symbol, month = stem.rsplit("-", 1)
        t, p, _ = load_trades(symbol, month)
        entry = price_at(t, p, cols["t"][idx])
        k = [load_klines(symbol, m) for m in (previous_month(previous_month(month)), previous_month(month), month)]
        k = np.concatenate([x for x in k if len(x)])
        k = k[np.argsort(k[:, 0], kind="stable")]
        minute = k[:, 0].astype(np.int64) // 60
        now = cols["sec"][idx].astype(np.int64) // 60
        res[idx] = _opp(minute, k[:, 3].copy(), k[:, 4].copy(), now, cols["dir"][idx].astype(np.int64), entry, SPAN)
    tmp = out_path.with_name(out_path.stem + ".tmp.npy")
    np.save(tmp, res)
    os.replace(tmp, out_path)
    return variant, stem, len(idx)


if __name__ == "__main__":
    months = sys.argv[1:] or ALL_MONTHS
    jobs = []
    for variant, folder in FOLDERS.items():
        (OPP4 / variant).mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.npz")):
            if ".tmp" not in path.name and path.stem.rsplit("-", 1)[1] in months:
                jobs.append((variant, path.stem))
    with Pool(int(os.environ.get("WORKERS", "10"))) as pool:
        for n, (variant, stem, k) in enumerate(pool.imap_unordered(one, jobs), 1):
            if n % 500 == 0 or n == len(jobs):
                print(n, "/", len(jobs), variant, stem, k, flush=True)
