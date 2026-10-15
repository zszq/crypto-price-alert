"""下载提醒涉及合约的历史持仓（Gate contract_stats），供 r5_oi.py 做持仓价值过滤研究。

下载站（download.gatedata.org）没有持仓数据，只能用 REST `/futures/usdt/contract_stats`，而它只保留最近 180 天：
所以只能在最近约半年上检验，更早的提醒没有持仓数据。隔太久不更新，最早的部分会过期，已下载的会保留（与新数据合并）。

接口不支持 gzip，5 分钟粒度每条约 600 字节，全部合约取满 180 天约 15GB，所以分两种粒度取：
- oi_1h/<合约>.npz：1 小时粒度的整段（持仓水平、24 小时变化用），每个合约约 3 次请求；
- oi_5m/<合约>.npz：5 分钟粒度只取每条提醒前 WINDOW 秒（提醒时刻的精确持仓、1 小时变化用），相邻提醒的区间合并后请求。
字段：time（统计区间开头，秒；值是区间结束时的快照）、oi_usd（持仓价值，USDT）、oi（持仓张数）、mark（标记价格）、
long_users / short_users（多空持仓人数）。
合约与提醒时刻取自给定数据集里最近 180 天的上涨提醒。

用法（在仓库根目录）：
    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/oi_fetch.py r5_p30 r5_p30_1440 r5_p r5_user r5_base r5_nobreak
并发高时 Gate 会断开 TLS 连接（实测 24 并发约四分之一失败），默认 8 并发、每秒 8 次；失败的用 --only-missing 补。
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, RESULTS  # noqa: E402

from price_alert.gate import GateRestClient, RateLimiter

OI_1H = DATA / "oi_1h"
OI_5M = DATA / "oi_5m"
REST_URL = "https://api.gateio.ws/api/v4"
# 接口 limit 上限在 2000 与 5000 之间（5000 被拒绝），返回 limit + 1 条（两端都含）。
PAGE = 2000
# 接口拒绝早于 180 天的 from；留一点余量，避免请求发出时刚好越界。
KEEP_SECONDS = 180 * 86400 - 3600
# 5 分钟粒度取提醒前多久：1 小时变化要 1 小时前的点，多留半小时容忍统计点缺失。
WINDOW = 5400
FIELDS = ("oi_usd", "oi", "mark", "long_users", "short_users")
SOURCE = {"oi_usd": "open_interest_usd", "oi": "open_interest", "mark": "mark_price"}


def alerts_from(datasets: list[str], since: int) -> dict[str, np.ndarray]:
    """各合约最近 180 天上涨提醒的收盘时刻（多个数据集取并集）。"""
    found: dict[str, set[int]] = {}
    for name in datasets:
        meta = np.load(RESULTS / name / "meta.npy")
        names = (RESULTS / name / "meta_names.txt").read_text(encoding="utf-8").split("\n")
        symbol = np.load(RESULTS / name / "symbol.npy")
        t = meta[:, names.index("closed_at")].astype(np.int64)
        keep = (t >= since) & (meta[:, names.index("direction")] == 1)
        for s, at in zip(symbol[keep].tolist(), t[keep].tolist(), strict=True):
            found.setdefault(s, set()).add(at)
    return {s: np.array(sorted(v), dtype=np.int64) for s, v in found.items()}


def merge_windows(times: np.ndarray, step: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for t in times.tolist():
        start = max(int(t) - WINDOW, 0)
        if spans and start <= spans[-1][1] + step:
            spans[-1] = (spans[-1][0], int(t))
        else:
            spans.append((start, int(t)))
    return spans


def listed_at(rest: GateRestClient, symbol: str) -> int | None:
    """合约上线时间（秒）；查不到（已下架等）为 None。"""
    try:
        return int(float(rest.fetch_contract(symbol)["create_time"]))
    except (urllib.error.HTTPError, KeyError, TypeError, ValueError):
        return None


def fetch_span(rest: GateRestClient, symbol: str, interval: str, step: int, start: int, end: int) -> list[dict]:
    rows: list[dict] = []
    listed_checked = False
    while start <= end:
        limit = min(PAGE, (end - start) // step + 1)
        try:
            page = rest._get(
                "/futures/usdt/contract_stats",
                {"contract": symbol, "interval": interval, "from": start, "limit": limit},
            )
        except urllib.error.HTTPError as exc:
            # 已下架的合约接口返回 4xx：没有持仓数据。
            if 400 <= exc.code < 500 and exc.code != 429:
                return rows
            raise
        if not isinstance(page, list) or not page:
            # from 早于合约上线时间时接口返回空列表、不报错（180 天内新上线的合约），改从上线后的第一个统计点重取一次。
            if not rows and not listed_checked:
                listed_checked = True
                listed = listed_at(rest, symbol)
                if listed is not None and listed > start:
                    start = -(-listed // step) * step
                    continue
            return rows
        rows.extend(item for item in page if isinstance(item, dict) and start <= int(item["time"]) <= end)
        last = int(page[-1]["time"])
        if last < start:
            return rows
        start = last + step
    return rows


def save(path: Path, rows: list[dict]) -> None:
    new = {"time": np.array([int(r["time"]) for r in rows], dtype=np.int64)}
    for name in FIELDS:
        key = SOURCE.get(name, name)
        new[name] = np.array([float(r.get(key) or np.nan) for r in rows], dtype=np.float64)
    if path.exists():
        old = np.load(path)
        new = {k: np.concatenate([old[k], v]) for k, v in new.items()}
    # 同一时刻以新取的为准：倒序后 unique 取到的是最后一次出现。
    reverse = {k: v[::-1] for k, v in new.items()}
    _, first = np.unique(reverse["time"], return_index=True)
    np.savez(path, **{k: v[first] for k, v in reverse.items()})


def update(rest: GateRestClient, symbol: str, times: np.ndarray, now: int) -> tuple[int, int]:
    since = now - KEEP_SECONDS
    hourly = fetch_span(rest, symbol, "1h", 3600, since, now)
    save(OI_1H / f"{symbol}.npz", hourly)
    fine: list[dict] = []
    for start, end in merge_windows(times[times >= since + WINDOW], 300):
        fine.extend(fetch_span(rest, symbol, "5m", 300, start, end))
    save(OI_5M / f"{symbol}.npz", fine)
    return len(hourly), len(fine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="取这些导出里最近 180 天的上涨提醒")
    parser.add_argument("--rate", type=float, default=8.0, help="每秒请求数")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--only-missing", action="store_true", help="只取上次失败（没有 5 分钟数据文件）或 1 小时数据为空的合约"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OI_1H.mkdir(parents=True, exist_ok=True)
    OI_5M.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    alerts = alerts_from(args.datasets, now - KEEP_SECONDS)
    if args.only_missing:
        # 5 分钟粒度最后保存，文件存在就说明这个合约上次取完了；1 小时为空的再试一次
        # （已下架的合约每次都会重试，只多几个请求）。
        alerts = {s: t for s, t in alerts.items() if not (OI_5M / f"{s}.npz").exists() or _empty(OI_1H / f"{s}.npz")}
    rest = GateRestClient(REST_URL, retries=5, rate_limiter=RateLimiter(args.rate, int(args.rate)))
    print(f"{len(alerts)} 个合约，{sum(len(v) for v in alerts.values())} 个提醒时刻", flush=True)
    done = 0
    empty = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(update, rest, s, t, now): s for s, t in alerts.items()}
        for future in as_completed(futures):
            symbol = futures[future]
            done += 1
            try:
                hourly, fine = future.result()
            except Exception as exc:  # noqa: BLE001 - 单个合约失败不影响其他，重跑会补
                print(f"[{done}/{len(alerts)}] {symbol} 失败：{exc}", flush=True)
                continue
            if hourly == 0:
                empty.append(symbol)
            if done % 50 == 0:
                print(f"[{done}/{len(alerts)}] {symbol} 1h +{hourly} 5m +{fine}", flush=True)
    print(f"完成；1 小时持仓为空的合约 {len(empty)} 个：{' '.join(sorted(empty))}", flush=True)


def _empty(path: Path) -> bool:
    return not path.exists() or len(np.load(path)["time"]) == 0


if __name__ == "__main__":
    main()
