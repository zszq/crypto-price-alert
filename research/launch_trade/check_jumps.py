"""检查价格的持久跳变：合约改过面值或拆合并时，价格会在某一分钟整体跳几倍并保持在新水平。

月度文件只有张数，成交额按「张数 × 当前面值 × 价格」估算；面值改过的合约，改之前的成交额会算错，
量比、窗口成交额门槛都会失真。这里列出前后各一天的收盘价中位数相差超过 RATIO 倍的位置，供人工判断。

用法：data/backtest/.venv/Scripts/python.exe research/launch_trade/check_jumps.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
from load_data import load_contracts, load_series

RATIO = 3.0
DAY = 1440
BEIJING = timezone(timedelta(hours=8))


def scan(args: tuple[str, float | None]) -> list[tuple[str, int, float, float]]:
    symbol, multiplier = args
    series = load_series(symbol, multiplier)
    if series is None or len(series) < 2 * DAY:
        return []
    close = series.close
    step = np.abs(np.diff(np.log(close)))
    found = []
    # 只看单分钟跳变超过 RATIO 倍的位置，再用前后一天的中位数确认是持久的水平变化，不是插针。
    for k in np.flatnonzero(step > np.log(RATIO)) + 1:
        if k < DAY or k + DAY > len(close):
            continue
        before = float(np.median(close[k - DAY : k]))
        after = float(np.median(close[k : k + DAY]))
        if max(before, after) / min(before, after) > RATIO:
            found.append((symbol, series.t0 + int(k) * 60, before, after))
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    jobs = [(symbol, multiplier(symbol)) for symbol in sorted(contracts)]
    hits = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for found in executor.map(scan, jobs, chunksize=4):
            hits.extend(found)
    print(f"检查 {len(jobs)} 个合约，持久跳变 {len(hits)} 处：")
    for symbol, at, before, after in hits:
        when = datetime.fromtimestamp(at, BEIJING).strftime("%Y-%m-%d %H:%M")
        print(f"  {symbol} {when} 前一天中位数 {before:.6g} → 后一天 {after:.6g}（×{after / before:.3g}）")


if __name__ == "__main__":
    main()
