"""第七轮对拍：方案 C / D 写成瞬时异动配置后的真实检测器与研究引擎逐条比较，不限合约池。

配置取 config/default/instant.yaml，只按方案改窗口参数与两个附加条件（检测器已支持，不改代码）；检测器按 assembly.build_detector
组装、不追加连续提醒。引擎取 rows4 的 L30 与 L180F30 行（补齐上限跟随 30 秒短窗口），按 grid7 同一套门槛与条件重算。
比较（提醒所在秒、方向、窗口）。用法：verify7.py C|D 进程数 [合约-月份 ...]（不给时取样本外该方案提醒最多的 40 个合约-月份）
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from common import load_klines, month_start, previous_month
from engine import EXTRA_WARM, load_trades
from grid7 import LONG, PERIODS, alerts_for, hits
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, load, row_count
from spike_grid import FOLDERS

from price_alert.assembly import build_detector
from price_alert.common.models import Candle, PriceTick
from price_alert.config import load_config
from price_alert.detectors.spike.config import InstantConfig

ROOT = Path(__file__).resolve().parents[2]
SHORT = Params(1.0, 1.0, 1, 15, 30.0, 0)
LONG_ANY = Params(LONG.a, LONG.b, LONG.c, LONG.trades, LONG.cooldown, 0)
# 方案 -> （引擎的条件，长窗口是否启用，配置里的附加条件）
SCHEMES = {
    "C": (((300, 0.0), None), True, {"prior_seconds": 300, "min_prior_move_percent": 0, "min_trade_burst_ratio": None}),
    "D": (((900, 0.0), 1.5), False, {"prior_seconds": 900, "min_prior_move_percent": 0, "min_trade_burst_ratio": 1.5}),
}


def instant_config(name: str) -> InstantConfig:
    _, long_on, extra = SCHEMES[name]
    data = load_config(ROOT / "config" / "default").instant.model_dump()
    data["short_window"] |= {"lookback_seconds": 30, "min_change_percent": 1.0, "trigger_atr_multiple": 1.0,
                             "confirmation_seconds": 1} | extra
    data["long_window"] |= {"enabled": long_on} | extra
    return InstantConfig.model_validate(data)


def detector_alerts(name: str, key: str) -> set[tuple[int, int, str]]:
    symbol, month = key.rsplit("-", 1)
    start = month_start(month)
    prev = load_klines(symbol, previous_month(month))
    prev = prev[prev[:, 0] >= start - EXTRA_WARM * 60] if len(prev) else prev
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    detector = build_detector(instant_config(name), streak=False)
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    out = set()
    for ts, px, sz in zip(t, price, size, strict=True):
        for alert in detector.add_tick(PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))):
            out.add((int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1, alert.window))
    return out


def engine_alerts(name: str, key: str) -> set[tuple[int, int, str]]:
    gate, long_on, _ = SCHEMES[name]
    month = key.rsplit("-", 1)[1]
    out = set()
    parts = [("L30", SHORT, "short")] + ([("L180F30", LONG_ANY, "long")] if long_on else [])
    for variant, p, window in parts:
        path = FOLDERS[variant] / f"{key}.npz"
        if not path.exists() or not row_count(path, variant):
            continue
        rows = load(variant, [month], COLUMNS + EXTRA_COLUMNS, [path])
        # 对拍不限合约池
        rows.cols["pool"] = np.full(len(rows), 0xFF, np.uint8)
        sim = Simulator(rows)
        for i in hits(rows, p, sim.eligible(p), gate):
            out.add((int(np.floor(rows["t"][i])), int(rows["dir"][i]), window))
    return out


def compare(args: tuple[str, str]) -> tuple[str, int, int, int, list]:
    name, key = args
    a, b = detector_alerts(name, key), engine_alerts(name, key)
    return key, len(a), len(b), len(a & b), sorted(a ^ b)[:5]


if __name__ == "__main__":
    name, workers, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    if not keys:
        tab = alerts_for("L30", PERIODS["oos"], Params(1.0, 1.0, 1, 15, 30.0, 1), SCHEMES[name][0])
        keys = [k for k, _ in Counter(tab["symbol"]).most_common(40)]
    total = Counter()
    with Pool(workers) as pool:
        for key, na, nb, both, diff in pool.imap_unordered(compare, [(name, k) for k in keys]):
            total.update(det=na, eng=nb, both=both)
            print(f"{key}：检测器 {na}，引擎 {nb}，一致 {both}", *(f"\n   差异 {datetime.fromtimestamp(x[0], UTC)} {x[1:]}" for x in diff), flush=True)
    print(f"合计：检测器 {total['det']}，引擎 {total['eng']}，一致 {total['both']}，"
          f"一致率 {total['both'] / max(total['det'] + total['eng'] - total['both'], 1):.2%}")
