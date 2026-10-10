"""下载 / 增量更新回测用的 Gate USDT 永续 1 分钟 K 线。

数据来源分两段：
- 已结束的月份：Gate 历史数据下载站（download.gatedata.org）的月度压缩包，原样保存在 raw/，下载过就不再请求；
  文件只有张数没有计价成交额，读取时按合约面值 × 典型价换算（见 load_data.py）。
- 下载站还没有的月份（通常是当月）：REST 1 分钟 K 线（带精确计价成交额 sum），追加到 recent/<合约>.csv，
  每次只拉上次之后的新分钟。REST 只保留最近 10000 根（约 6.9 天），隔太久没更新会留下缺口，回测按无数据处理。

用法（在仓库根目录）：
    data/backtest/.venv/Scripts/python.exe research/launch_trade/update_data.py [--start 2026-04]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA  # noqa: E402

from price_alert.gate import GATE_MINUTE_HISTORY, GateRestClient, RateLimiter, fetch_minute_candles

RAW = DATA / "raw"
RECENT = DATA / "recent"
CONTRACTS = DATA / "contracts.json"
# 已确认下载站没有的（合约, 月份），避免每次更新都对几百个未上线月份重复请求 404。
MISSING = DATA / "raw_missing.json"
DOWNLOAD_URL = "https://download.gatedata.org/futures_usdt/candlesticks_1m/{month}/{symbol}-{month}.csv.gz"
REST_URL = "https://api.gateio.ws/api/v4"
# 下载站背后的 S3 存储桶：download.gatedata.org 会丢掉列表查询参数，直连存储桶才能按月份列出当月存在的全部合约。
LISTING_URL = "https://gateio-public-data.s3.amazonaws.com/"
# 下载站在月初过几天才发布上月文件；结束不满这么多天的月份 404 不记入 MISSING，下次再试。
PUBLISH_GRACE = timedelta(days=20)


def month_start(month: str) -> datetime:
    return datetime(int(month[:4]), int(month[4:]), 1, tzinfo=UTC)


def next_month(month: str) -> str:
    year, mon = int(month[:4]), int(month[4:])
    return f"{year + mon // 12}{mon % 12 + 1:02d}"


def months_between(first: str, last: str) -> list[str]:
    months = [first]
    while months[-1] < last:
        months.append(next_month(months[-1]))
    return months


def raw_path(symbol: str, month: str) -> Path:
    return RAW / month / f"{symbol}-{month}.csv.gz"


def fetch_contracts(rest: GateRestClient) -> list[dict]:
    # 与 universe.py 同样只保留虚拟币合约：contract_type 必须存在且为空，股票、指数等有分类值。
    contracts = [
        item
        for item in rest.fetch_contracts()
        if item.get("contract_type", None) == "" and str(item.get("name", "")).endswith("_USDT")
    ]
    # 合并而不是覆盖：之后下架的合约仍保留面值，已下载的历史数据还能用。
    known: dict[str, dict] = {}
    if CONTRACTS.exists():
        known = {item["name"]: item for item in json.loads(CONTRACTS.read_text(encoding="utf-8"))}
    for item in contracts:
        known[item["name"]] = {
            "name": item["name"],
            "quanto_multiplier": item.get("quanto_multiplier"),
            "order_price_round": item.get("order_price_round"),
            "status": item.get("status"),
            "create_time": item.get("create_time"),
        }
    merged = sorted(known.values(), key=lambda item: item["name"])
    CONTRACTS.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    return merged


def listed_symbols(month: str) -> set[str]:
    """下载站某个月有 1 分钟 K 线文件的全部 USDT 合约（含之后已下架的，以及股票等非币类合约）。"""
    found: set[str] = set()
    token = None
    while True:
        query = {"list-type": "2", "prefix": f"futures_usdt/candlesticks_1m/{month}/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(LISTING_URL + "?" + urllib.parse.urlencode(query), timeout=60) as response:
            xml = response.read().decode("utf-8")
        for key in re.findall(r"<Key>([^<]+)</Key>", xml):
            symbol = key.rsplit("/", 1)[-1].rsplit("-", 1)[0]
            if symbol.endswith("_USDT"):
                found.add(symbol)
        next_token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if next_token is None:
            return found
        token = next_token.group(1)


def discover_delisted(rest: GateRestClient, months: list[str]) -> int:
    """把这些月份下载站有、合约列表里没有的币类合约补进 contracts.json，返回新增个数。

    fetch_contracts 只能取到当前还在交易的合约，期间下架的币从来不会进样本，回测有幸存者偏差：
    只算币类合约，2021-10 ~ 2025-04 每个月有 39%~47% 不在样本里，2025-10 约 28%。合约详情接口对已下架合约仍返回面值，
    只保留 contract_type 为空的（股票、指数等有分类值）。面值按当前值，期间改过面值的合约成交额会算错，
    用 check_jumps.py 检查价格跳变。
    """
    known = {item["name"]: item for item in json.loads(CONTRACTS.read_text(encoding="utf-8"))}
    listed: set[str] = set()
    for month in months:
        listed |= listed_symbols(month)
    unknown = sorted(listed - set(known))
    print(f"下载站列表：{months[0]}~{months[-1]} 共 {len(listed)} 个 USDT 合约，合约列表里没有的 {len(unknown)} 个")
    added = 0
    for symbol in unknown:
        try:
            item = rest.fetch_contract(symbol)
        except Exception as exc:  # 查不到的合约跳过，下次运行再试
            print(f"  {symbol} 合约详情获取失败：{exc}")
            continue
        if item.get("contract_type", None) != "":
            continue
        known[symbol] = {
            "name": symbol,
            "quanto_multiplier": item.get("quanto_multiplier"),
            "order_price_round": item.get("order_price_round"),
            "status": item.get("status"),
            "create_time": item.get("create_time"),
        }
        added += 1
    merged = sorted(known.values(), key=lambda item: item["name"])
    CONTRACTS.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"补进 {added} 个已下架或此前未收录的币类合约（其余是股票等非币类合约）")
    return added


def download_month(symbol: str, month: str) -> str:
    target = raw_path(symbol, month)
    url = DOWNLOAD_URL.format(month=month, symbol=urllib.parse.quote(symbol))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再改名：中途中断不会留下半个压缩包被当成已下载。
            partial = target.with_suffix(".part")
            partial.write_bytes(body)
            partial.replace(target)
            return "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "missing"
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc
        time.sleep(2**attempt)
    return f"error: {error}"


def last_recent_minute(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        # 只读文件末尾，不必把整个文件读进来。
        handle.seek(max(0, path.stat().st_size - 4096))
        lines = handle.read().decode("utf-8").strip().splitlines()
    return int(lines[-1].split(",")[0]) if lines else None


def update_recent(rest: GateRestClient, symbol: str, since: datetime) -> int:
    # 每个合约请求前现取当前时间：前面下载月度文件可能要几十分钟，用启动时的时间算保留范围会越界被拒（400）。
    now = datetime.now(UTC)
    path = RECENT / f"{symbol}.csv"
    last = last_recent_minute(path)
    start = since if last is None else max(since, datetime.fromtimestamp(last + 60, tz=UTC))
    # 留几分钟余量：分段请求要好几次，起点正好落在保留范围边缘时，Gate 会直接返回 400。
    earliest = now - timedelta(minutes=GATE_MINUTE_HISTORY - 5)
    start = max(start, earliest.replace(second=0, microsecond=0))
    # 只要已收盘的分钟：end 取当前分钟的开盘时间，[start, end) 不含正在走的这一根。
    end = now.replace(second=0, microsecond=0)
    if start >= end:
        return 0
    candles = fetch_minute_candles(rest, symbol, start, end)
    RECENT.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for candle in candles:
            writer.writerow(
                [
                    int(candle.timestamp.timestamp()),
                    repr(candle.open),
                    repr(candle.high),
                    repr(candle.low),
                    repr(candle.close),
                    repr(candle.quote_volume),
                ]
            )
    # 记下这次实际覆盖的区间：没有返回的分钟是无成交，区间之外的缺口（隔太久没更新）是无数据，加载时据此区分。
    spans = RECENT / f"{symbol}.spans"
    if not spans.exists() and last is not None:
        # 早期版本没有记录区间：把已有数据按首尾一段补记上，否则加载时只认新区间、旧数据会被当作无数据。
        with path.open(encoding="utf-8") as existing:
            first = int(existing.readline().split(",")[0])
        spans.write_text(f"{first},{last + 60}\n", encoding="utf-8")
    with spans.open("a", encoding="utf-8") as handle:
        handle.write(f"{int(start.timestamp())},{int(end.timestamp())}\n")
    return len(candles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    now = datetime.now(UTC)
    default_start = (now - timedelta(days=190)).strftime("%Y%m")
    parser.add_argument("--start", default=default_start, help="最早的月份 YYYYMM，默认约半年前")
    parser.add_argument("--rate", type=float, default=15.0, help="REST 每秒请求数，给正在运行的监控留出余量")
    parser.add_argument("--workers", type=int, default=8, help="下载站并发数；补很多月份时可调高")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="按下载站的月度文件列表补进期间已下架的币类合约，减少幸存者偏差（每个合约查一次详情）",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rest = GateRestClient(REST_URL, rate_limiter=RateLimiter(args.rate, int(args.rate)))
    contracts = fetch_contracts(rest)
    last_complete = (now.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
    months = months_between(args.start.replace("-", ""), last_complete)
    if args.discover:
        discover_delisted(rest, months)
        contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    symbols = [item["name"] for item in contracts]
    print(f"合约 {len(symbols)} 个（含已下架但此前下载过的）")

    missing: set[str] = set(json.loads(MISSING.read_text(encoding="utf-8"))) if MISSING.exists() else set()
    jobs = [
        (symbol, month)
        for month in months
        for symbol in symbols
        if not raw_path(symbol, month).exists() and f"{month}/{symbol}" not in missing
    ]
    print(f"下载站：月份 {months[0]}~{months[-1]}，待下载 {len(jobs)} 个文件")
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_month, symbol, month): (symbol, month) for symbol, month in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            symbol, month = futures[future]
            status = future.result()
            key = status.split(":")[0]
            results[key] = results.get(key, 0) + 1
            if status == "missing" and now - month_start(next_month(month)) > PUBLISH_GRACE:
                missing.add(f"{month}/{symbol}")
            elif status.startswith("error"):
                print(f"  {symbol} {month} {status}")
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} {results}")
    MISSING.write_text(json.dumps(sorted(missing), ensure_ascii=False), encoding="utf-8")
    print(f"下载站完成：{results}")

    # 下载站之后的月份用 REST 补；下载站发布了新月份后，load_data 以月度文件为准，recent 里的旧行自动被覆盖。
    recent_since = month_start(next_month(last_complete))
    # 已下架的合约没有新数据，REST 也取不到，不必为它们各发一轮请求、刷一屏失败。
    trading = [item["name"] for item in contracts if item.get("status") != "delisted"]
    print(f"REST：补 {recent_since:%Y-%m-%d} 之后的已收盘分钟（{len(trading)} 个未下架的合约）")
    total = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(update_recent, rest, symbol, recent_since): symbol for symbol in trading}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                total += future.result()
            except Exception as exc:  # 单个合约失败（如已下架）不影响其他合约，下次运行会接着补。
                failures += 1
                print(f"  {futures[future]} REST 失败：{exc}")
            if done % 100 == 0:
                print(f"  {done}/{len(trading)}，新增 {total} 根")
    print(f"REST 完成：新增 {total} 根，失败 {failures} 个合约")


if __name__ == "__main__":
    main()
