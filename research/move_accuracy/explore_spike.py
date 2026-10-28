"""第四轮探索（只用 2024-10 ~ 2026-09）：现行瞬时异动参数的提醒补算特征，分档看哪些特征让 hold5 在两个年度都更高。

现行 = config/default/spike.yaml：短窗口 30 秒 / 1.3% / 1.5 ATR / 确认 2 / 15 笔 / 冷却 30，
长窗口 180 秒 / 1.6% / 1.8 ATR / 确认 3 / 30 笔 / 冷却 180，合约池 800 万。
用法：explore_spike.py collect（收集提醒并补算特征，写 spike4_alerts.npz）| bins（分档）
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA, FRESH_MONTHS, MONTHS, ROUND3_MONTHS, ROUND4_MONTHS, load_klines, previous_month
from engine import load_trades, price_at
from engine4 import ROWS4
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, load, row_count

SELECT_MONTHS = ROUND4_MONTHS + ROUND3_MONTHS + FRESH_MONTHS + MONTHS
YEAR1 = ROUND4_MONTHS + ROUND3_MONTHS  # 2024-10 ~ 2025-09
CURRENT = {"L30": Params(1.3, 1.5, 2, 15, 30, 1), "L180F30": Params(1.6, 1.8, 3, 30, 180, 1)}
KEEP = COLUMNS + EXTRA_COLUMNS
OUT = DATA / "spike4_alerts.npz"


def rows4_files(months: list[str], variant: str, limit: int = 8_000_000) -> list[list]:
    chunks, cur, size = [], [], 0
    for path in sorted(ROWS4.glob("*.npz")):
        if ".tmp" in path.name or path.stem.rsplit("-", 1)[1] not in months:
            continue
        n = row_count(path, variant)
        if not n:
            continue
        if cur and size + n > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(path)
        size += n
    return chunks + ([cur] if cur else [])


def collect_alerts(variant: str, p: Params, gates: tuple = (), months: list[str] = SELECT_MONTHS) -> dict:
    parts = []
    for files in rows4_files(months, variant):
        rows = load(variant, months, KEEP, files)
        hit = Simulator(rows).alerts(p, gates)
        part = {c: rows[c][hit] for c in KEEP}
        part["month"] = rows.month[hit]
        part["symbol"] = np.array([rows.symbols[g] for g in rows.group[hit]])
        parts.append(part)
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


def group_features(args):
    key, idx, sec, tt, direction, window = args
    symbol, month = key.rsplit("-", 1)
    t, p, _ = load_trades(symbol, month)
    d = direction.astype(np.float64)
    entry = price_at(t, p, tt)
    k = [load_klines(symbol, m) for m in (previous_month(month), month)]
    k = np.concatenate([x for x in k if len(x)])
    k = k[np.argsort(k[:, 0], kind="stable")]
    minute = k[:, 0].astype(np.int64) // 60
    hi, lo, close = k[:, 3], k[:, 4], k[:, 2]
    now = sec.astype(np.int64) // 60
    a = np.searchsorted(minute, now - 1440, side="left")
    b = np.searchsorted(minute, now, side="left")
    hi24 = np.full(len(sec), np.nan)
    lo24 = np.full(len(sec), np.nan)
    for i in range(len(sec)):
        # 与第三轮同一口径：此前 1440 分钟已收盘 K 线，不含被判定那一秒所在的分钟；覆盖不满 20 小时视为缺失
        if b[i] > a[i] and minute[a[i]] <= now[i] - 1200:
            hi24[i] = hi[a[i] : b[i]].max()
            lo24[i] = lo[a[i] : b[i]].min()
    out = {
        # 反向极值距离：急涨时离 24 小时最低价、急跌时离最高价，顺提醒方向为正
        "opp24h": np.where(d > 0, entry / lo24 - 1, 1 - entry / hi24) * 100,
        # 同向极值距离：急涨时离 24 小时最高价（> 0 即创新高）
        "ext24h": np.where(d > 0, entry / hi24 - 1, 1 - entry / lo24) * 100,
        # 24 小时区间里的位置，顺提醒方向：1 = 已在提醒方向的尽头
        "pos24h": np.where(d > 0, (entry - lo24), (hi24 - entry)) / np.where(hi24 > lo24, hi24 - lo24, np.nan),
    }
    wstart = tt - window
    for span in (3600, 4 * 3600):
        ci = np.searchsorted(minute, (wstart - span) // 60, side="right") - 1
        past = np.where(ci >= 0, close[np.clip(ci, 0, None)], np.nan)
        out[f"pre{span}"] = d * (price_at(t, p, wstart) / past - 1) * 100
    # 提醒后同一合约此前 1 小时内的同向 / 反向急动（行情在这一段里反复抽动还是第一下）
    same = np.zeros(len(tt))
    opp = np.zeros(len(tt))
    for i in range(len(tt)):
        j0 = np.searchsorted(tt, tt[i] - 3600, side="left")
        prev = direction[j0:i]
        same[i] = (prev == direction[i]).sum()
        opp[i] = (prev != direction[i]).sum()
    out["same1h"] = same
    out["opp1h"] = opp
    return idx, out


def market(tt, direction, months):
    out = {f"btc_{k}": np.full(len(tt), np.nan) for k in (60, 300, 900)}
    for m in np.unique(months):
        sel = np.flatnonzero(months == m)
        t, p, _ = load_trades("BTC_USDT", ALL_MONTHS[m])
        now = price_at(t, p, tt[sel])
        for k in (60, 300, 900):
            out[f"btc_{k}"][sel] = direction[sel] * (now / price_at(t, p, tt[sel] - k) - 1) * 100
    return out


def collect() -> None:
    tables = []
    for variant, p in CURRENT.items():
        a = collect_alerts(variant, p)
        a["window"] = np.full(len(a["t"]), 30 if variant == "L30" else 180, np.int16)
        tables.append(a)
        print(variant, len(a["t"]), flush=True)
    c = {k: np.concatenate([x[k] for x in tables]) for k in tables[0]}
    jobs = []
    for key in np.unique(c["symbol"]):
        for w in (30, 180):
            idx = np.flatnonzero((c["symbol"] == key) & (c["window"] == w))
            if len(idx):
                idx = idx[np.argsort(c["t"][idx], kind="stable")]
                jobs.append((key, idx, c["sec"][idx], c["t"][idx], c["dir"][idx], w))
    n = len(c["t"])
    cols: dict[str, np.ndarray] = {}
    with Pool(int(os.environ.get("WORKERS", "10"))) as pool:
        for idx, f in pool.imap_unordered(group_features, jobs):
            for k, v in f.items():
                cols.setdefault(k, np.full(n, np.nan))[idx] = v
    cols |= market(c["t"], c["dir"], c["month"])
    np.savez(OUT, **c, **{f"f_{k}": v for k, v in cols.items()})
    print("提醒", n, "特征", len(cols), flush=True)


def bins() -> None:
    d = dict(np.load(OUT, allow_pickle=True))
    ret = d["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    pos = np.nan_to_num(ret, nan=-1) > 0
    h5 = pos[:, :5].all(axis=1)
    side5 = pos[:, 4]
    year1 = np.isin(d["month"], [ALL_MONTHS.index(m) for m in YEAR1])
    features = ["chg", "atrm", "trades", "imb", "vr", "retr", "pre", "pre300", "burst", "brk", "eff", "ext6h", "atrpct"]
    features += [k for k in d if k.startswith("f_")]
    days = len(SELECT_MONTHS) * 30.4
    for w in (30, 180):
        base = valid & (d["window"] == w)
        print(f"\n=== {w} 秒窗口：每天 {base.sum() / days:.0f} 条，hold5 {h5[base].mean():.1%}，5 分钟后顺向 {side5[base].mean():.1%}"
              f"（第一年 {h5[base & year1].mean():.1%} / 第二年 {h5[base & ~year1].mean():.1%}）")
        for name in features:
            x = d[name].astype(np.float64)
            ok = base & np.isfinite(x)
            if ok.sum() < 1000:
                continue
            edges = np.unique(np.nanpercentile(x[ok], np.linspace(0, 100, 11)))
            if len(edges) < 3:
                continue
            k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            cells = []
            for b in range(len(edges) - 1):
                s = ok & (k == b)
                if s.sum() < 200:
                    continue
                cells.append(f"[{edges[b]:.3g},{edges[b + 1]:.3g}) {h5[s & year1].mean():.0%}/{h5[s & ~year1].mean():.0%}")
            print(f"{name:12s}", " | ".join(cells))


if __name__ == "__main__":
    {"collect": collect, "bins": bins}[sys.argv[1]]()
