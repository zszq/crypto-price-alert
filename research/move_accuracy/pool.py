"""按 1 分钟 K 线估算每个合约每 10 分钟刷新时刻的滚动 24 小时成交额，供任意门槛重建合约池。

实时服务每 10 分钟用 volume_24h_quote 刷新合约池，这里用 K 线张数 × 面值 × 收盘价近似。
输出 volume_<月>.npz（marks：刷新时刻；vols：合约 × 刷新时刻），与 sizes_<月>.json（下载站逐笔成交文件大小）。

用法：pool.py [2026MM ...]，默认全部月份。
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
from common import (
    DATA,
    DOWNLOAD_POOL_QUOTE,
    MONTHS,
    RAW,
    contracts,
    load_klines,
    month_end,
    month_start,
    multiplier,
    previous_month,
)

BUCKET = "https://gateio-public-data.s3.amazonaws.com/?list-type=2&prefix=futures_usdt/trades/{m}/"


def list_sizes(month: str) -> dict[str, int]:
    # download.gatedata.org 会丢掉列表查询参数，只能直连存储桶；大小用于校验下载是否完整
    sizes, token = {}, None
    while True:
        url = BUCKET.format(m=month) + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        page = urllib.request.urlopen(url, timeout=60).read().decode()
        for name, size in re.findall(
            r"<Key>futures_usdt/trades/\d+/(\w+?)-\d+\.csv\.gz</Key>.*?<Size>(\d+)</Size>", page
        ):
            sizes[name] = int(size)
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", page)
        if not match:
            return sizes
        token = match.group(1)


def month_volumes(month: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    start, end = month_start(month), month_end(month)
    origin = start - 86400
    n = (end - origin) // 60
    info = contracts()
    names = sorted({p.name.split("-")[0] for m in (month, previous_month(month)) for p in (RAW / m).glob("*.csv.gz")})
    marks = np.arange(start, end, 600)
    idx = (marks - origin) // 60
    symbols, rows = [], []
    for sym in names:
        mult = multiplier(info.get(sym))
        if mult <= 0:
            continue
        parts = [load_klines(sym, m) for m in (previous_month(month), month)]
        a = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.empty((0, 6))
        if not len(a):
            continue
        minute = (a[:, 0].astype(np.int64) - origin) // 60
        ok = (minute >= 0) & (minute < n)
        per = np.zeros(n)
        np.add.at(per, minute[ok], np.abs(a[ok, 1]) * mult * a[ok, 2])
        cs = np.concatenate([[0.0], np.cumsum(per)])
        # 刷新时刻只能看到它之前的 24 小时
        vol = cs[idx] - cs[idx - 1440]
        if vol.max() > DOWNLOAD_POOL_QUOTE * 0.5:
            symbols.append(sym)
            rows.append(vol.astype(np.float32))
    return marks, symbols, np.array(rows)


def membership(vols: np.ndarray, threshold: float, exit_ratio: float = 0.8) -> np.ndarray:
    """与实时服务相同的滞回：不在池时 > 门槛才进，在池时 ≤ 门槛 × 比例才出。"""
    member = np.zeros(vols.shape, bool)
    inside = np.zeros(vols.shape[0], bool)
    for j in range(vols.shape[1]):
        inside = np.where(inside, vols[:, j] > threshold * exit_ratio, vols[:, j] > threshold)
        member[:, j] = inside
    return member


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    for month in sys.argv[1:] or MONTHS:
        marks, symbols, vols = month_volumes(month)
        np.savez(DATA / f"volume_{month}.npz", marks=marks, symbols=np.array(symbols), vols=vols)
        sizes = list_sizes(month)
        json.dump(sizes, open(DATA / f"sizes_{month}.json", "w"))
        member = membership(vols, DOWNLOAD_POOL_QUOTE)
        ever = [s for s, m in zip(symbols, member.any(axis=1), strict=True) if m]
        have = [s for s in ever if s in sizes]
        m8 = membership(vols, 8e6)
        print(
            month,
            f"5M 曾进池 {len(ever)}（下载站有 {len(have)}，{sum(sizes[s] for s in have) / 1e9:.1f}GB）",
            f"平均在池 5M {member.sum(axis=0).mean():.1f} / 8M {m8.sum(axis=0).mean():.1f}",
            flush=True,
        )
