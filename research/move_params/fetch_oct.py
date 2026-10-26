"""用 Gate REST 取 10-02 12:00 ~ 10-12 00:00（UTC）的逐笔成交，作为第二个留出期。

只取实时日志里 10-03~10-11 有过秒级提醒的合约：更严的参数只会在现行参数提醒的附近触发，
其余合约在更严参数下也不会提醒。按天分段保存，格式与下载站一致（时间,id,价格,张数），下载过的跳过。
"""

import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from price_alert.exchange.gate import GateRestClient, RateLimiter

# 行情、记录与结果只放本机（data/backtest/move/，不进版本库），代码在这里
DATA = Path(__file__).resolve().parents[2] / "data" / "backtest" / "move"
LOG = DATA.parents[1] / "alerts" / "alerts.jsonl"
START = datetime(2026, 10, 2, 12, tzinfo=UTC)
DAYS = [datetime(2026, 10, d, tzinfo=UTC) for d in range(2, 13)]

client = GateRestClient("https://api.gateio.ws/api/v4", "usdt", 30, 5, RateLimiter(8, 10))
symbols = sorted(
    {
        a["symbol"]
        for a in map(json.loads, open(LOG))
        if a.get("kind", "move") == "move" and "2026-10-03" <= a["timestamp"][:10] <= "2026-10-11"
    }
)
json.dump(symbols, open(DATA / "oct_symbols.json", "w"))
(DATA / "oct").mkdir(exist_ok=True)


def fetch(job):
    symbol, day_start, day_end = job
    out = DATA / "oct" / f"{symbol}-{day_start:%m%d}.csv.gz"
    if out.exists():
        return
    try:
        ticks = client.fetch_trades(symbol, max(day_start, START), day_end, max_trades=20_000_000)
    except Exception as exc:
        print(symbol, day_start.date(), "失败", exc, flush=True)
        return
    tmp = out.with_suffix(".part")
    with gzip.open(tmp, "wt") as f:
        for t in ticks:
            f.write(f"{t.timestamp.timestamp():.6f},{t.trade_id or 0},{t.price!r},{t.size!r}\n")
    os.replace(tmp, out)
    print(symbol, day_start.date(), len(ticks), flush=True)


jobs = [(s, DAYS[i], DAYS[i + 1]) for s in symbols for i in range(len(DAYS) - 1)]
with ThreadPoolExecutor(20) as pool:
    list(pool.map(fetch, jobs))
print("done", len(symbols))
