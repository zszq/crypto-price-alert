"""把 raw/ 月度文件与 recent/ REST 数据合并成每个合约一条连续的分钟序列，并缓存为 cache/<合约>.npz。

序列从该合约第一根有数据的分钟开始，到最后一根结束，每分钟一个下标：
- 有成交的分钟取原始 K 线；
- 数据源覆盖范围内、没有成交的分钟按上一收盘价补平线、成交额 0（下载站和 REST 都省略这类分钟，与检测器口径一致）；
- 数据源没有覆盖的分钟（月度文件缺失、REST 保留范围之外的缺口）covered=False，回测不在这里判定、不计持仓；
  连续 DEAD_MINUTES（一天）以上完全没有成交的空档（停牌、下架后重新上线）同样 covered=False。
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from paths import DATA

RAW = DATA / "raw"
RECENT = DATA / "recent"
CACHE = DATA / "cache"
CONTRACTS = DATA / "contracts.json"
# 加载规则变了就改这个版本号，旧缓存按来源文件签名判断不出来，必须整体重建。
CACHE_VERSION = 2
# 连续这么多分钟完全没有成交，视为停牌或下架后重新上线的空档，按无数据处理（见 _build）。
DEAD_MINUTES = 1440


@dataclass(slots=True)
class Series:
    symbol: str
    t0: int  # 第一根 K 线的开盘时间（秒）
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote: np.ndarray
    covered: np.ndarray

    def __len__(self) -> int:
        return len(self.close)


def load_contracts() -> dict[str, dict]:
    return {item["name"]: item for item in json.loads(CONTRACTS.read_text(encoding="utf-8"))}


def _month_bounds(month: str) -> tuple[int, int]:
    from datetime import UTC, datetime

    year, mon = int(month[:4]), int(month[4:])
    start = datetime(year, mon, 1, tzinfo=UTC)
    end = datetime(year + mon // 12, mon % 12 + 1, 1, tzinfo=UTC)
    return int(start.timestamp()), int(end.timestamp())


def _sources(symbol: str) -> list[Path]:
    files = sorted(RAW.glob(f"*/{symbol}-*.csv.gz"))
    files.extend(path for path in (RECENT / f"{symbol}.csv", RECENT / f"{symbol}.spans") if path.exists())
    return files


def _signature(files: list[Path]) -> str:
    # 任何来源文件新增或变化就重建缓存；recent 每次更新都会变大。
    parts = (f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}" for path in files)
    return f"v{CACHE_VERSION}|" + "|".join(parts)


def _build(symbol: str, multiplier: float | None) -> Series | None:
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    # 每段数据源覆盖的 [起, 止) 时间：月度文件覆盖整月（从其第一根算起，之前视为未上线），REST 覆盖拉取的区间。
    spans: list[tuple[int, int]] = []
    raw_months: list[tuple[int, int]] = []
    for path in sorted(RAW.glob(f"*/{symbol}-*.csv.gz")):
        month = path.parent.name
        start, end = _month_bounds(month)
        raw_months.append((start, end))
        first = None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for record in csv.reader(handle):
                # 下载站列顺序：时间、张数、收、高、低、开。
                t = int(record[0])
                # 下载站个别文件有字段为空的行（2021~2023 年 24 个文件各一行，缺高低价或张数），
                # 无法还原，按这一分钟无成交跳过；只影响一分钟，不值得为它猜补数值。
                if "" in record[1:6]:
                    continue
                volume, close, high, low, open_ = (float(value) for value in record[1:6])
                # 没有计价成交额，按面值 × 典型价估算；面值缺失的合约无法换算，与实时检测器一样不判定。
                quote = volume * multiplier * (high + low + close) / 3 if multiplier else 0.0
                rows[t] = (open_, high, low, close, quote)
                first = t if first is None else min(first, t)
        if first is not None:
            spans.append((start, end))
    recent = RECENT / f"{symbol}.csv"
    if recent.exists():
        recent_rows = []
        with recent.open(encoding="utf-8") as handle:
            for record in csv.reader(handle):
                t = int(record[0])
                # 月度文件覆盖的月份以月度文件为准（下载站发布后 REST 那份就不用了）。
                if any(start <= t < end for start, end in raw_months):
                    continue
                recent_rows.append(t)
                rows[t] = tuple(float(value) for value in record[1:6])  # type: ignore[assignment]
        span_file = RECENT / f"{symbol}.spans"
        if span_file.exists():
            # 每次 REST 拉取的 [起, 止)：区间内没有返回的分钟就是无成交；两次拉取之间超出保留范围的缺口不算覆盖。
            for line in span_file.read_text(encoding="utf-8").split():
                start, end = (int(value) for value in line.split(","))
                spans.append((start, end))
        elif recent_rows:
            spans.append((min(recent_rows), max(recent_rows) + 60))
    if not rows or multiplier is None:
        return None
    times = sorted(rows)
    t0, t_last = times[0], times[-1]
    size = (t_last - t0) // 60 + 1
    data = np.full((5, size), np.nan)
    present = np.zeros(size, dtype=bool)
    index = (np.array(times) - t0) // 60
    data[:, index] = np.array([rows[t] for t in times]).T
    present[index] = True
    covered = np.zeros(size, dtype=bool)
    minute_times = t0 + np.arange(size) * 60
    for start, end in spans:
        covered |= (minute_times >= start) & (minute_times < end)
    # 月度文件存在时整月都算覆盖，合约停牌几周也会被补成平线；重新上线后检测器会把这段假平线当作基准期
    # （基准成交额为 0 视为放量、旧价格当作要突破的高点）。实盘中交易所给不出这段历史，要像新上线一样
    # 攒够 K 线才判定，所以整天以上没有成交的空档按无数据处理。合约池要求 24 小时成交额 > 100 万，
    # 正常交易中的币不会整天零成交，这只影响停牌后重新上线的情况。
    edges = np.diff(np.concatenate([[0], (~present).astype(np.int8), [0]]))
    for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if end - begin >= DEAD_MINUTES:
            covered[begin:end] = False
    # 平线：开高低收都等于上一根收盘价（缺口之后的第一根若也无成交，同样沿用缺口前的收盘价，价格不会编造）。
    close = data[3]
    filled_index = np.where(present, np.arange(size), 0)
    np.maximum.accumulate(filled_index, out=filled_index)
    last_close = close[filled_index]
    for row in range(4):
        data[row] = np.where(present, data[row], last_close)
    data[4] = np.where(present, data[4], 0.0)
    return Series(symbol, t0, data[0], data[1], data[2], data[3], data[4], covered)


def load_series(symbol: str, multiplier: float | None) -> Series | None:
    files = _sources(symbol)
    if not files:
        return None
    signature = _signature(files)
    cache = CACHE / f"{symbol}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["signature"]) == signature:
                if int(stored["t0"]) < 0:
                    return None
                return Series(
                    symbol,
                    int(stored["t0"]),
                    stored["open"],
                    stored["high"],
                    stored["low"],
                    stored["close"],
                    stored["quote"],
                    stored["covered"],
                )
    series = _build(symbol, multiplier)
    CACHE.mkdir(parents=True, exist_ok=True)
    if series is None:
        np.savez(cache, signature=signature, t0=-1)
        return None
    np.savez_compressed(
        cache,
        signature=signature,
        t0=series.t0,
        open=series.open,
        high=series.high,
        low=series.low,
        close=series.close,
        quote=series.quote,
        covered=series.covered,
    )
    return series
