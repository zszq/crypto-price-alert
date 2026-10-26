"""对拍：反弹回落检测器（src/price_alert/reversal）与研究引擎（engine.py --extras 的 L10F10 行）逐条比较，不限合约池。

用法：verify_reversal.py 合约 月份 —— 单个合约-月份；verify_hold2.py 对样本外全量对拍。
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
from common import load_klines, month_start, previous_month
from engine import EXTRA_WARM, load_trades
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, files_for, load, row_count

from price_alert.models import Candle, PriceTick
from price_alert.reversal.config import ReversalConfig
from price_alert.reversal.detector import ReversalDetector

# 研究里第二轮选中的一组（= ReversalConfig 的代码默认值），引擎侧用同样的门槛与新条件。
P = Params(1.5, 1.0, 1, 5, 30)
GATES = (("pre_neg", -6.0), ("burst", 16.0))


def detector_alerts(symbol: str, month: str) -> set[tuple[int, int]]:
    start = month_start(month)
    prev = load_klines(symbol, previous_month(month))
    prev = prev[prev[:, 0] >= start - EXTRA_WARM * 60] if len(prev) else prev
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    detector = ReversalDetector(ReversalConfig().to_rule())
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    out = set()
    for ts, px, sz in zip(t, price, size, strict=True):
        for alert in detector.add_tick(
            PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))
        ):
            out.add((int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1))
    return out


def engine_alerts(symbol: str, month: str) -> set[tuple[int, int]]:
    # 整月没有一秒过宽松门槛的合约没有 L10F10 的行，引擎自然是 0 条
    files = [f for f in files_for([month], "L10F10") if f.stem == f"{symbol}-{month}" and row_count(f, "L10F10")]
    if not files:
        return set()
    rows = load("L10F10", [month], COLUMNS + EXTRA_COLUMNS, files)
    rows.cols["pool"] = np.full(len(rows), 0xFF, np.uint8)
    return {(int(np.floor(rows["t"][i])), int(rows["dir"][i])) for i in Simulator(rows).alerts(P, GATES)}


if __name__ == "__main__":
    symbol, month = sys.argv[1], sys.argv[2]
    a, b = detector_alerts(symbol, month), engine_alerts(symbol, month)
    print(
        f"{symbol} {month}：检测器 {len(a)} 条，引擎 {len(b)} 条，一致 {len(a & b)}，"
        f"一致率 {len(a & b) / max(len(a | b), 1):.2%}"
    )
    for x in sorted(a ^ b)[:8]:
        print("   差异", datetime.fromtimestamp(x[0], UTC), x[1], "仅检测器" if x in a else "仅引擎")
