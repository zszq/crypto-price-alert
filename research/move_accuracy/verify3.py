"""第三轮对拍：带反向极值距离的反弹回落检测器与研究引擎（L10F10 行 + opp3 的定义）逐条比较，不限合约池。

用法：verify3.py 确认秒数 距离上限 合约-月份 ...（如 verify3.py 2 10 LAB_USDT-202605）
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
from common import load_klines, month_start, previous_month
from engine import load_trades
from opp3 import group
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, files_for, load, scan

from price_alert.common.models import Candle, PriceTick
from price_alert.detectors.reversal.config import ReversalConfig
from price_alert.detectors.reversal.detector import ReversalDetector


def detector_alerts(symbol: str, month: str, config: ReversalConfig) -> set[tuple[int, int]]:
    start = month_start(month)
    rule = config.to_rule()
    # 月中才上线的合约没有之前的 K 线，检测器从空白开始，靠实时成交攒满 24 小时
    months = (previous_month(previous_month(month)), previous_month(month))
    prev = np.concatenate([np.empty((0, 6)), *(x for x in map(lambda m: load_klines(symbol, m), months) if len(x))])
    prev = prev[(prev[:, 0] >= start - rule.warmup_candles * 60) & (prev[:, 0] < start)]
    candles = [Candle(datetime.fromtimestamp(r[0], UTC), r[5], r[3], r[4], r[2]) for r in prev]
    detector = ReversalDetector(rule)
    detector.add_symbol(symbol, candles, 1e9)
    t, price, size = load_trades(symbol, month)
    out = set()
    for ts, px, sz in zip(t, price, size, strict=True):
        tick = PriceTick(symbol, float(px), abs(float(sz)), datetime.fromtimestamp(float(ts), UTC))
        for alert in detector.add_tick(tick):
            out.add((int(alert.timestamp.timestamp()), 1 if alert.direction == "surge" else -1))
    return out


def engine_alerts(symbol: str, month: str, config: ReversalConfig) -> set[tuple[int, int]]:
    files = [f for f in files_for([month], "L10F10") if f.stem == f"{symbol}-{month}"]
    if not files:
        return set()
    rows = load("L10F10", [month], COLUMNS + EXTRA_COLUMNS, files)
    p = Params(config.min_change_percent, config.trigger_atr_multiple, config.confirmation_seconds, config.min_window_trades, config.cooldown_seconds)
    idx = Simulator(rows).eligible(p)
    ok = (rows["pre"][idx] <= -config.min_prior_move_percent) & (rows["burst"][idx] >= config.min_trade_burst_ratio)
    if config.max_extreme_distance_percent is not None:
        sel = idx[ok]
        _, feats = group((f"{symbol}-{month}", sel, rows["sec"][sel], rows["t"][sel], rows["dir"][sel]))
        dist = np.full(len(idx), np.inf)
        dist[ok] = np.where(np.isnan(feats["opp_1440"]), np.inf, feats["opp_1440"])
        ok &= dist <= config.max_extreme_distance_percent
    hit = idx[scan(rows.group[idx], rows["sec"][idx], rows["t"][idx], rows["real"][idx], rows["lchg"][idx], rows["latrm"][idx], ok, p.a, p.b, p.c, p.cooldown)]
    return {(int(np.floor(rows["t"][i])), int(rows["dir"][i])) for i in hit}


if __name__ == "__main__":
    config = ReversalConfig(confirmation_seconds=int(sys.argv[1]), max_extreme_distance_percent=float(sys.argv[2]))
    total_a = total_b = same = 0
    for key in sys.argv[3:]:
        symbol, month = key.rsplit("-", 1)
        a, b = detector_alerts(symbol, month, config), engine_alerts(symbol, month, config)
        total_a, total_b, same = total_a + len(a), total_b + len(b), same + len(a & b)
        print(f"{key}：检测器 {len(a)} 条，引擎 {len(b)} 条，一致 {len(a & b)}", flush=True)
        for x in sorted(a ^ b)[:5]:
            print("   差异", datetime.fromtimestamp(x[0], UTC), x[1], "仅检测器" if x in a else "仅引擎")
    print(f"合计：检测器 {total_a}，引擎 {total_b}，一致 {same}")
