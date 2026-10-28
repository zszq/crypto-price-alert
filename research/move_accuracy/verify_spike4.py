"""第四轮对拍：日常配置（方案 A，config/default/spike.yaml）的真实短时异动检测器与研究引擎逐条比较，不限合约池。

检测器按 assembly.build_detector 组装（10 秒短窗口与 180 秒长窗口共用秒桶，补齐上限跟随 10 秒），不追加连续提醒；
引擎取 rows_L10F10 与 rows4_L180F10 的行，按 spike_grid 同一套门槛与附加条件重算。比较（提醒所在秒、方向、窗口）。
用法：verify_spike4.py 进程数 [合约-月份 ...]（不给时取样本外方案 A 提醒最多的 40 个合约-月份）
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from common import load_klines, month_start, previous_month
from engine import EXTRA_WARM, load_trades
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, load, row_count
from spike_grid import FOLDERS, LONG, PERIODS, alerts_for, hits

from price_alert.assembly import build_detector
from price_alert.common.models import Candle, PriceTick
from price_alert.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SHORT = Params(1.0, 1.0, 1, 15, 30.0, 0)
GATE = ((300, -2.0), None, 4.0)
LONG_ANY = Params(LONG.a, LONG.b, LONG.c, LONG.trades, LONG.cooldown, 0)


def detector_alerts(key: str) -> set[tuple[int, int, str]]:
    symbol, month = key.rsplit("-", 1)
    spike = load_config(ROOT / "config" / "default").spike
    start = month_start(month)
    prev = load_klines(symbol, previous_month(month))
    prev = prev[prev[:, 0] >= start - EXTRA_WARM * 60] if len(prev) else prev
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    detector = build_detector(spike, streak=False)
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    out = set()
    for ts, px, sz in zip(t, price, size, strict=True):
        for alert in detector.add_tick(PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))):
            out.add((int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1, alert.window))
    return out


def engine_alerts(key: str) -> set[tuple[int, int, str]]:
    month = key.rsplit("-", 1)[1]
    out = set()
    for variant, p, name in (("L10F10", SHORT, "short"), ("L180F10", LONG_ANY, "long")):
        path = FOLDERS[variant] / f"{key}.npz"
        if not path.exists() or not row_count(path, variant):
            continue
        rows = load(variant, [month], COLUMNS + EXTRA_COLUMNS, [path])
        # 对拍不限合约池
        rows.cols["pool"] = np.full(len(rows), 0xFF, np.uint8)
        opp = np.full(len(rows), np.nan)
        sim = Simulator(rows)
        for i in hits(rows, opp, sim, p, sim.eligible(p), GATE):
            out.add((int(np.floor(rows["t"][i])), int(rows["dir"][i]), name))
    return out


def compare(key: str) -> tuple[str, int, int, int, list]:
    a, b = detector_alerts(key), engine_alerts(key)
    return key, len(a), len(b), len(a & b), sorted(a ^ b)[:5]


if __name__ == "__main__":
    workers = int(sys.argv[1])
    keys = sys.argv[2:]
    if not keys:
        oos = PERIODS["oos"]
        tab = alerts_for("L10F10", oos, Params(1.0, 1.0, 1, 15, 30.0, 1), GATE)
        keys = [k for k, _ in Counter(tab["symbol"]).most_common(40)]
    total = Counter()
    with Pool(workers) as pool:
        for key, na, nb, both, diff in pool.imap_unordered(compare, keys):
            total.update(det=na, eng=nb, both=both)
            print(f"{key}：检测器 {na}，引擎 {nb}，一致 {both}", *(f"\n   差异 {datetime.fromtimestamp(x[0], UTC)} {x[1:]}" for x in diff), flush=True)
    print(f"合计：检测器 {total['det']}，引擎 {total['eng']}，一致 {total['both']}，"
          f"一致率 {total['both'] / max(total['det'] + total['eng'] - total['both'], 1):.2%}")
