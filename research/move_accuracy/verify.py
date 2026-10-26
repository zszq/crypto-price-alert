"""对拍：同一段逐笔成交分别交给真实 AtrMoveDetector 与本研究的引擎，逐条比较提醒（秒、方向）。

用法：verify.py 合约 月份 [天数]，对几组参数（含宽松的一组）逐条比较。
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
from common import load_klines, month_start, previous_month
from engine import load_trades, process
from sim import Params, Rows, Simulator

from price_alert.detectors.move.detector import AtrMoveDetector
from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.move.windows import MoveWindow

CASES = [
    (30, Params(1.3, 1.5, 2, 15, 30)),
    (10, Params(0.8, 1.0, 1, 5, 30)),
    (60, Params(1.0, 1.5, 3, 20, 120)),
    (180, Params(1.6, 2.0, 3, 60, 180)),
    (300, Params(2.0, 2.5, 5, 60, 300)),
]


def real_alerts(symbol: str, month: str, days: float, window: int, p: Params) -> list[tuple[int, int]]:
    start = month_start(month)
    prev = load_klines(symbol, previous_month(month))
    prev = prev[prev[:, 0] >= start - 3600]
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    detector = AtrMoveDetector(
        atr_period=14,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[MoveWindow("short", window, p.b, p.a, p.c, p.trades, int(p.cooldown))],
    )
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    cut = t < start + days * 86400
    out = []
    for ts, px, sz in zip(t[cut], price[cut], size[cut], strict=True):
        tick = PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))
        for alert in detector.add_tick(tick):
            # 提醒时间是触发结算的成交时间；比较被结算的秒需要回推，这里直接比较提醒时刻所在秒与方向
            out.append((int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1))
    return out


def engine_alerts(symbol: str, month: str, days: float, window: int, p: Params) -> list[tuple[int, int]]:
    out = process((symbol, month), days=days, variants={"X": (window, window)})
    cols = {k[2:]: v for k, v in out.items() if k.startswith("X_")}
    n = len(cols["sec"])
    rows = Rows("X", cols, np.zeros(n, np.int32), np.zeros(n, np.int8), [symbol])
    # 对拍不限合约池
    rows.cols["pool"] = np.full(n, 0xFF, np.uint8)
    hit = Simulator(rows).alerts(p)
    return [(int(np.floor(cols["t"][i])), int(cols["dir"][i])) for i in hit]


def dual_case(symbol: str, month: str, days: float) -> None:
    """现行双窗口：长窗口的补齐上限跟随短窗口（30 秒），对应引擎的 L180F30 变体。"""
    from datetime import UTC, datetime

    start = month_start(month)
    prev = load_klines(symbol, previous_month(month))
    prev = prev[prev[:, 0] >= start - 3600]
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    short, long_ = Params(1.3, 1.5, 2, 15, 30), Params(1.6, 1.8, 3, 30, 180)
    detector = AtrMoveDetector(
        atr_period=14,
        candle_interval_seconds=60,
        max_atr_age_seconds=180,
        windows=[MoveWindow("short", 30, 1.5, 1.3, 2, 15, 30), MoveWindow("long", 180, 1.8, 1.6, 3, 30, 180)],
    )
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    cut = t < start + days * 86400
    real = set()
    for ts, px, sz in zip(t[cut], price[cut], size[cut], strict=True):
        for alert in detector.add_tick(
            PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))
        ):
            real.add((alert.window, int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1))
    out = process((symbol, month), days=days, variants={"S": (30, 30), "G": (180, 30)})
    eng = set()
    for name, label, p in (("S", "short", short), ("G", "long", long_)):
        cols = {k[2:]: v for k, v in out.items() if k.startswith(name + "_")}
        n = len(cols["sec"])
        rows = Rows(name, cols, np.zeros(n, np.int32), np.zeros(n, np.int8), [symbol])
        rows.cols["pool"] = np.full(n, 0xFF, np.uint8)
        eng |= {(label, int(np.floor(cols["t"][i])), int(cols["dir"][i])) for i in Simulator(rows).alerts(p)}
    print(
        f"双窗口：检测器 {len(real)} 条，引擎 {len(eng)} 条，一致 {len(real & eng)}，"
        f"一致率 {len(real & eng) / max(len(real | eng), 1):.2%}"
    )
    for x in sorted(real ^ eng)[:8]:
        print("   差异", x, "仅检测器" if x in real else "仅引擎")


if __name__ == "__main__":
    symbol, month = sys.argv[1], sys.argv[2]
    days = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    for window, p in CASES:
        a = real_alerts(symbol, month, days, window, p)
        b = engine_alerts(symbol, month, days, window, p)
        sa, sb = set(a), set(b)
        both = len(sa & sb)
        print(
            f"L{window} {p}: 检测器 {len(a)} 条，引擎 {len(b)} 条，一致 {both}，"
            f"一致率 {both / max(len(sa | sb), 1):.2%}",
            flush=True,
        )
        for x in sorted(sa ^ sb)[:8]:
            print("   差异", datetime.fromtimestamp(x[0], UTC), x[1], "仅检测器" if x in sa else "仅引擎")
    dual_case(symbol, month, days)
