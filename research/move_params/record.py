"""用项目里的秒级检测器逐笔重放九月成交，记录每个窗口每一秒通过「宽松基准门槛」的判定指标。

更严的参数（涨跌幅、ATR 倍数、笔数、确认秒数、冷却）只会在这些秒里触发，所以确认与冷却可以
之后在记录上离线重算（sim.py），不必每组参数都重放一遍成交。同时保存逐秒价格，用来评估提醒之后的走势。

用法（仓库根目录）：
data/backtest/.venv/bin/python research/move_params/record.py [--source 2026MM|oct] [--short 30] [--long 180] [合约 ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sim

from price_alert.detectors.move.detector import AtrMoveDetector
from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.move.windows import MoveWindow, Outcome

# 行情、记录与结果只放本机（data/backtest/move/，不进版本库），代码在这里
DATA = Path(__file__).resolve().parents[2] / "data" / "backtest" / "move"
HIST = DATA.parent / "hist"
PASSED = {Outcome.CONFIRMING, Outcome.CARRIED, Outcome.MOVE_FADED, Outcome.ALERT}
# 宽松基准：低于现行门槛足够多，候选参数只在这之上收紧。
BASE = {"short": (0.8, 1.0), "long": (1.0, 1.2)}


class RecordingDetector(AtrMoveDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = {window.name: [] for window in self.windows}
        self._now = 0.0

    def _evaluate(self, symbol, state, current, timestamp, latest_price, atr_state):
        self._now = timestamp.timestamp()
        return super()._evaluate(symbol, state, current, timestamp, latest_price, atr_state)

    def _evaluate_window(self, symbol, state, window, candidate, current, latest_price, atr_state):
        result = super()._evaluate_window(symbol, state, window, candidate, current, latest_price, atr_state)
        if result.outcome in PASSED:
            base = result.baseline_price
            if latest_price is None:
                latest_chg = latest_atr = np.nan
            else:
                latest_chg = (latest_price - base) / base * 100.0
                latest_atr = abs(latest_price - base) / result.atr
            self.records[window.name].append(
                (
                    current.timestamp.timestamp(),
                    self._now,
                    not current.has_trades,
                    result.change_percent,
                    result.move_atr,
                    result.trade_count,
                    result.price,
                    base,
                    latest_chg,
                    latest_atr,
                )
            )
        return result


def load_trades(symbol: str, source: str = "202609") -> np.ndarray:
    """source 为 oct（REST 十月成交）或下载站月份（2026MM）。"""
    if source == "oct":
        raw = b"".join(gzip.open(p, "rb").read() for p in sorted((DATA / "oct").glob(f"{symbol}-*.csv.gz")))
    else:
        with gzip.open(DATA / "trades" / f"{symbol}-{source}.csv.gz", "rb") as f:
            raw = f.read()
    # 列：时间（秒，带小数）、成交 id、价格、张数（带符号表示方向）
    data = np.array(raw.replace(b"\n", b",").split(b",")[:-1], dtype=np.float64) if raw else np.zeros(0)
    data = data.reshape(-1, 4)
    order = np.lexsort((data[:, 1], data[:, 0]))
    data = data[order]
    return data[data[:, 2] > 0]


def seed_candles(symbol: str, month: str) -> list[Candle]:
    path = HIST / f"{symbol}-{sim.previous_month(month)}.csv.gz"
    if not path.exists():
        return []
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            t, _, close, high, low, opn = line.strip().split(",")
            rows.append((int(t), float(opn), float(high), float(low), float(close)))
    rows.sort()
    # 只有有成交的分钟，空分钟按上一收盘价补平线，与 Gate REST K 线口径一致。
    candles: list[Candle] = []
    for t, o, h, lo, c in rows[-200:]:
        if candles:
            prev = candles[-1]
            gap_t = int(prev.timestamp.timestamp()) + 60
            while gap_t < t:
                candles.append(
                    Candle(datetime.fromtimestamp(gap_t, UTC), prev.close, prev.close, prev.close, prev.close)
                )
                gap_t += 60
        candles.append(Candle(datetime.fromtimestamp(t, UTC), o, h, lo, c))
    return candles[-49:]


def run(args: tuple[str, int, int, str]) -> str:
    symbol, short_lb, long_lb, source = args
    suffix = f"s{short_lb}_l{long_lb}"
    variant = f"oct_{suffix}" if source == "oct" else sim.month_variant(source, suffix)
    out = DATA / "rec" / variant / f"{symbol}.npz"
    if out.exists():
        return f"{symbol} 已存在"
    started = time.time()
    trades = load_trades(symbol, source)
    windows = [
        MoveWindow("short", short_lb, BASE["short"][1], BASE["short"][0], 1, 1, 0),
        MoveWindow("long", long_lb, BASE["long"][1], BASE["long"][0], 1, 1, 0),
    ]
    detector = RecordingDetector(
        atr_period=14, candle_interval_seconds=60, max_atr_age_seconds=180, windows=windows, streak=None
    )
    # 10 月数据从 10-02 12:00 开始、评估从 10-03 起，ATR 靠前 12 小时的实时 K 线自行就绪；
    # 4 月没有 3 月 K 线可预热，同样靠 4 月 1 日的实时 K 线就绪（4 月从 2 日起评估）
    detector.add_symbol(symbol, seed_candles(symbol, source) if source != "oct" else [], 0.0)
    for ts, _, price, size in trades:
        detector.add_tick(PriceTick(symbol, price, abs(size), datetime.fromtimestamp(ts, UTC)))
    # 逐秒价格：最后成交价与秒内高低价，供评估提醒之后的走势。
    secs = np.floor(trades[:, 0]).astype(np.int64)
    uniq, start = np.unique(secs, return_index=True)
    end = np.append(start[1:], len(secs))
    prices = trades[:, 2]
    last = prices[end - 1]
    hi = np.maximum.reduceat(prices, start) if len(start) else np.zeros(0)
    lo = np.minimum.reduceat(prices, start) if len(start) else np.zeros(0)
    count = (end - start).astype(np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        short=np.array(detector.records["short"], dtype=np.float64).reshape(-1, 10),
        long=np.array(detector.records["long"], dtype=np.float64).reshape(-1, 10),
        sec=uniq,
        last=last,
        hi=hi,
        lo=lo,
        count=count,
    )
    os.replace(tmp, out)
    return f"{symbol} {len(trades):,} 笔 {time.time() - started:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=int, default=30)
    parser.add_argument("--long", type=int, default=180)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--source", default="202609", help="oct 或下载站月份 2026MM")
    parser.add_argument("symbols", nargs="*")
    ns = parser.parse_args()
    if ns.source == "oct":
        symbols = ns.symbols or json.load(open(DATA / "oct_symbols.json"))
        with Pool(ns.workers) as pool:
            for message in pool.imap_unordered(run, [(s, ns.short, ns.long, "oct") for s in symbols]):
                print(message, flush=True)
        return
    month = ns.source
    names = DATA / sim.dataset(sim.month_variant(month))["symbols"]
    symbols = ns.symbols or [line.strip() for line in open(names) if line.strip()]
    # 只处理已下载完的合约，下载未完成时可以反复运行
    expected = json.load(open(DATA / ("trade_sizes.json" if month == "202609" else f"trade_sizes_{month}.json")))
    files = {s: DATA / "trades" / f"{s}-{month}.csv.gz" for s in symbols}
    symbols = [s for s in symbols if files[s].exists() and files[s].stat().st_size == expected[s]]
    symbols.sort(key=lambda s: -files[s].stat().st_size)
    with Pool(ns.workers) as pool:
        for message in pool.imap_unordered(run, [(s, ns.short, ns.long, month) for s in symbols]):
            print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
