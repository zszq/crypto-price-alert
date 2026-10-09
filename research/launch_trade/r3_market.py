"""第三轮：全市场逐小时状态，用来让交易规则随市场环境动态调整。

市场深度、交易活跃度、上涨币占比在不同时期差别很大，同一条固定过滤在不同时期方向可能相反，
所以先把这些环境量算成逐小时序列，提醒时只取提醒之前已经收盘的那个小时，不会用到未来数据。
输出 data/backtest/results/r3_market.npz：
- hours：每小时的开盘时间（秒）；close / quote：(小时数, 合约数) 每小时收盘价与成交额，无数据为 NaN / 0。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_contracts, load_series  # noqa: E402
from paths import RESULTS  # noqa: E402

START = 1688169600  # 2023-07-01 UTC：三年区间起点之前留出 30 天以上的预热
END = 1791590400  # 2026-10-10 UTC


def _hourly(args):
    symbol, multiplier = args
    series = load_series(symbol, multiplier)
    n_hours = (END - START) // 3600
    close = np.full(n_hours, np.nan)
    quote = np.zeros(n_hours)
    if series is None:
        return close, quote
    minute = series.t0 + np.arange(len(series)) * 60
    ok = series.covered & (minute >= START) & (minute < END)
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
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_hourly, [(s, multiplier(s)) for s in symbols], chunksize=4))
    close = np.stack([r[0] for r in results], axis=1)
    quote = np.stack([r[1] for r in results], axis=1)
    hours = START + np.arange(close.shape[0]) * 3600
    np.savez(RESULTS / "r3_market.npz", hours=hours, close=close, quote=quote, symbols=np.array(symbols))
    print(f"{len(symbols)} 个合约，{close.shape[0]} 小时")


if __name__ == "__main__":
    main()
