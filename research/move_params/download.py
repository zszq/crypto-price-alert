"""下载某月逐笔成交月度文件（下载过的跳过）。

用法：download.py [2026MM ...]，默认九月。
其他月份的大小清单 trade_sizes_<月>.json 来自 list_sizes.py，合约来自 pool.py 的 symbols_<月>.txt。
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import sim

URL = "https://download.gatedata.org/futures_usdt/trades/{m}/{s}-{m}.csv.gz"


def fetch(job):
    symbol, month, size = job
    out = sim.DATA / "trades" / f"{symbol}-{month}.csv.gz"
    # 连接中途断开时 read() 可能静默返回空，按 S3 列表里的大小校验，不完整的重下
    if out.exists() and out.stat().st_size == size:
        return
    tmp = out.with_suffix(".part")
    for _ in range(10):
        try:
            # 并发多时下载站常把长连接截断，从已下载的位置续传
            done = tmp.stat().st_size if tmp.exists() else 0
            if done > size:
                tmp.unlink()
                done = 0
            if done < size:
                request = urllib.request.Request(URL.format(s=symbol, m=month), headers={"Range": f"bytes={done}-"})
                with urllib.request.urlopen(request, timeout=600) as r, open(tmp, "ab" if done else "wb") as f:
                    if done and r.status != 206:
                        raise OSError(f"不支持续传 {r.status}")
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size != size:
                raise OSError(f"大小不符 {tmp.stat().st_size} != {size}")
            os.replace(tmp, out)
            print(symbol, month, out.stat().st_size // 1_000_000, "MB", flush=True)
            return
        except Exception as exc:
            print(symbol, month, "重试", exc, flush=True)
    print(symbol, month, "失败", flush=True)


jobs = []
for month in sys.argv[1:] or ["202609"]:
    sizes = json.load(open(sim.DATA / ("trade_sizes.json" if month == "202609" else f"trade_sizes_{month}.json")))
    names = sim.DATA / sim.dataset(sim.month_variant(month))["symbols"]
    jobs += [(s, month, sizes[s]) for s in (line.strip() for line in open(names)) if s]
(sim.DATA / "trades").mkdir(exist_ok=True)
# 下载站单连接约 0.5MB/s，并发多一些才能用满带宽
with ThreadPoolExecutor(int(os.environ.get("THREADS", "12"))) as pool:
    list(pool.map(fetch, jobs))
print("done", len(jobs))
