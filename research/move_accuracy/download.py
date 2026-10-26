"""下载各月曾进 5M 合约池的合约的逐笔成交月度文件（已完整下载的跳过）。

用法：download.py [2026MM ...]，默认全部月份；THREADS 环境变量控制并发。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from common import DATA, DOWNLOAD_POOL_QUOTE, MONTHS, TRADES
from pool import membership

URL = "https://download.gatedata.org/futures_usdt/trades/{m}/{s}-{m}.csv.gz"


def fetch(job: tuple[str, str, int]) -> None:
    symbol, month, size = job
    out = TRADES / f"{symbol}-{month}.csv.gz"
    # 下载站常截断长连接且 read() 可能静默返回空，只能按 S3 列表里的大小判断是否完整
    if out.exists() and out.stat().st_size == size:
        return
    tmp = out.with_suffix(".part")
    for _ in range(20):
        try:
            done = tmp.stat().st_size if tmp.exists() else 0
            if done > size:
                tmp.unlink()
                done = 0
            if done < size:
                request = urllib.request.Request(
                    URL.format(s=urllib.parse.quote(symbol), m=month), headers={"Range": f"bytes={done}-"}
                )
                with urllib.request.urlopen(request, timeout=300) as r, open(tmp, "ab" if done else "wb") as f:
                    if done and r.status != 206:
                        raise OSError(f"不支持续传 {r.status}")
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size != size:
                raise OSError(f"大小不符 {tmp.stat().st_size} != {size}")
            os.replace(tmp, out)
            return
        except Exception as exc:  # noqa: BLE001 - 网络错误种类很多，统一续传重试
            print(symbol, month, "重试", exc, flush=True)
    print(symbol, month, "失败", flush=True)


def jobs_for(month: str) -> list[tuple[str, str, int]]:
    data = np.load(DATA / f"volume_{month}.npz")
    sizes = json.load(open(DATA / f"sizes_{month}.json"))
    ever = membership(data["vols"], DOWNLOAD_POOL_QUOTE).any(axis=1)
    return [(str(s), month, sizes[str(s)]) for s, e in zip(data["symbols"], ever, strict=True) if e and str(s) in sizes]


if __name__ == "__main__":
    TRADES.mkdir(parents=True, exist_ok=True)
    jobs = [job for month in sys.argv[1:] or MONTHS for job in jobs_for(month)]
    # 大文件先下，避免最后只剩一两个大文件单线程慢慢拖
    jobs.sort(key=lambda job: -job[2])
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "16"))) as pool:
        list(pool.map(fetch, jobs))
    missing = [j for j in jobs if not (TRADES / f"{j[0]}-{j[1]}.csv.gz").exists()]
    print("完成", len(jobs) - len(missing), "/", len(jobs), "缺", missing, flush=True)
