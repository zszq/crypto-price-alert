"""对拍：真实检测器直接按给定参数重放，与 sim.py 在记录上重算的提醒逐条比较。

用法：verify.py [--month 2026MM] 合约 ...
"""

import argparse
from datetime import UTC, datetime

import sim
from record import load_trades, seed_candles

from price_alert.detectors.move.detector import AtrMoveDetector
from price_alert.common.models import PriceTick
from price_alert.detectors.move.windows import MoveWindow

PARAM_SETS = [
    sim.CURRENT,
    sim.Params(sim.Win(1.5, 2.0, 30, 3, 120), sim.Win(2.0, 2.5, 60, 5, 600)),
    sim.Params(sim.Win(1.0, 1.2, 5, 1, 0), sim.Win(1.2, 1.5, 5, 1, 0)),
    # 第三轮：确认 5 秒、长窗口 120 笔
    sim.ROUND3,
    sim.Params(sim.Win(3.0, 2.5, 60, 5, 30), sim.Win(1.6, 1.8, 120, 3, 180)),
]

parser = argparse.ArgumentParser()
parser.add_argument("--month", default="202609")
parser.add_argument("symbols", nargs="+")
ns = parser.parse_args()
for symbol in ns.symbols:
    trades = load_trades(symbol, ns.month)
    data = sim.load(symbol, sim.month_variant(ns.month))
    for params in PARAM_SETS:
        windows = [
            MoveWindow(
                "short",
                30,
                params.short.atr,
                params.short.change,
                params.short.confirm,
                params.short.trades,
                params.short.cooldown,
            ),
            MoveWindow(
                "long",
                180,
                params.long.atr,
                params.long.change,
                params.long.confirm,
                params.long.trades,
                params.long.cooldown,
            ),
        ]
        det = AtrMoveDetector(
            atr_period=14, candle_interval_seconds=60, max_atr_age_seconds=180, windows=windows, streak=None
        )
        det.add_symbol(symbol, seed_candles(symbol, ns.month), 0.0)
        real = []
        for ts, _, price, size in trades:
            for a in det.add_tick(PriceTick(symbol, price, abs(size), datetime.fromtimestamp(ts, UTC))):
                real.append((round(a.timestamp.timestamp(), 6), a.window, round(a.change_percent, 9)))
        mine = []
        for name, w in (("short", params.short), ("long", params.long)):
            for i in sim.simulate_window(data[name], w):
                r = data[name][i]
                mine.append((round(r[sim.NOW], 6), name, round(r[sim.CHG], 9)))
        real.sort()
        mine.sort()
        print(symbol, "real", len(real), "sim", len(mine), "一致" if real == mine else "不一致")
        if real != mine:
            print(" 只在 real:", sorted(set(real) - set(mine))[:5])
            print(" 只在 sim:", sorted(set(mine) - set(real))[:5])
