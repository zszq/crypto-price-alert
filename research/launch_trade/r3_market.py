"""第三轮：全市场逐小时状态，用来让交易规则随市场环境动态调整。

市场深度、交易活跃度、上涨币占比在不同时期差别很大，同一条固定过滤在不同时期方向可能相反，
所以先把这些环境量算成逐小时序列，提醒时只取提醒之前已经收盘的那个小时，不会用到未来数据。
输出 data/backtest/results/r3_market.npz：
- hours：每小时的开盘时间（秒）；close / quote：(小时数, 合约数) 每小时收盘价与成交额，无数据为 NaN / 0。
hourly_breadth() 由它算出每小时站上 30 天均线的合约占比（市场宽度），供过热过滤使用。
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_contracts, load_series  # noqa: E402
from paths import RESULTS  # noqa: E402

START = 1630454400  # 2021-09-01 UTC：五年区间（2021-10-10 起）之前留出 30 天以上，30 天均线才算得出
MARKET = RESULTS / "r3_market.npz"
_breadth_cache: tuple[np.ndarray, np.ndarray] | None = None


def hourly_breadth() -> tuple[np.ndarray, np.ndarray]:
    """(每小时开盘时间, 该小时收盘时站上 30 天均线的合约占比)，由 main() 的输出算出，进程内只算一次。

    过热过滤（r3_lib 与 launch_trade_backtest.py --hot-breadth）都用它，两边判断一致。
    """
    global _breadth_cache
    if _breadth_cache is None:
        market = np.load(MARKET)
        close = market["close"]
        with np.errstate(invalid="ignore", divide="ignore"):
            # 30 天（720 小时）均线只用有数据的小时：缺数据的小时既不计入和，也不计入个数。
            values = np.cumsum(np.nan_to_num(close), 0)
            counts = np.cumsum(~np.isnan(close), 0)
            sma = (values - np.roll(values, 720, 0)) / np.maximum(counts - np.roll(counts, 720, 0), 1)
            sma[:720] = np.nan
            above = np.where(np.isnan(close) | np.isnan(sma), np.nan, close > sma)
            valid = ~np.isnan(above)
            ratio = np.where(valid.sum(1) > 20, np.nansum(above, 1) / np.maximum(valid.sum(1), 1), np.nan)
        _breadth_cache = (market["hours"], ratio)
    return _breadth_cache


def _hourly(args):
    symbol, multiplier, end = args
    series = load_series(symbol, multiplier)
    n_hours = (end - START) // 3600
    close = np.full(n_hours, np.nan)
    quote = np.zeros(n_hours)
    if series is None:
        return close, quote
    minute = series.t0 + np.arange(len(series)) * 60
    ok = series.covered & (minute >= START) & (minute < end)
    slot = (minute[ok] - START) // 3600
    np.add.at(quote, slot, series.quote[ok])
    # 每小时最后一根有数据的分钟的收盘价：slot 单调不减，按 slot 取最后一个位置。
    last = np.flatnonzero(np.diff(np.append(slot, n_hours + 1)) != 0)
    close[slot[last]] = series.close[ok][last]
    return close, quote


def main() -> None:
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    symbols = sorted(contracts)
    # 取到运行时的整点：数据更新后重跑就包含新的小时。由主进程算好传给子进程，
    # 子进程各自取时间的话跨过整点时小时数会不一致。
    end = int(time.time()) // 3600 * 3600
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_hourly, [(s, multiplier(s), end) for s in symbols], chunksize=4))
    close = np.stack([r[0] for r in results], axis=1)
    quote = np.stack([r[1] for r in results], axis=1)
    hours = START + np.arange(close.shape[0]) * 3600
    np.savez(MARKET, hours=hours, close=close, quote=quote, symbols=np.array(symbols))
    print(f"{len(symbols)} 个合约，{close.shape[0]} 小时")


if __name__ == "__main__":
    main()
