"""把 raw/ 月度文件与 recent/ REST 数据合并成每个合约一条连续的分钟序列，并缓存为 cache/<合约>.npz。

序列从该合约第一根有数据的分钟开始，到最后一根结束，每分钟一个下标：
- 有成交的分钟取原始 K 线；
- 数据源覆盖范围内、没有成交的分钟按上一收盘价补平线、成交额 0（下载站和 REST 都省略这类分钟，与检测器口径一致）；
- 数据源没有覆盖的分钟（月度文件缺失、REST 保留范围之外的缺口）covered=False，回测不在这里判定、不计持仓。
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
    return "|".join(f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}" for path in files)


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
