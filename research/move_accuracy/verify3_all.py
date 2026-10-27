"""第三轮全量对拍：样本外方案 B 有提醒的全部合约-月份（不限合约池），检测器与引擎逐条比较。用法：verify3_all.py r3|dev [进程数]"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from common import DATA
from grid3 import combos
from verify3 import detector_alerts, engine_alerts

from price_alert.detectors.reversal.config import ReversalConfig

CONFIG = ReversalConfig(confirmation_seconds=2, max_extreme_distance_percent=10.0)


def one(key: str) -> tuple[str, int, int, int]:
    symbol, month = key.rsplit("-", 1)
    a, b = detector_alerts(symbol, month, CONFIG), engine_alerts(symbol, month, CONFIG)
    return key, len(a), len(b), len(a & b)


if __name__ == "__main__":
    name = sys.argv[1]
    g = np.load(DATA / f"grid3_{name}.npz")
    sel = json.load(open(DATA / "selection3.json", encoding="utf-8"))
    k = combos().index(tuple(sel["B"]["combo"]))
    a = g["alerts"]
    keys = sorted(set(g["symbols"][np.unique(a[a[:, 0] == k, 1].astype(int))]))
    totals = np.zeros(3, int)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        for key, na, nb, same in pool.imap_unordered(one, keys):
            totals += (na, nb, same)
            if not (na == nb == same):
                print(f"  差异 {key}：检测器 {na}，引擎 {nb}，一致 {same}", flush=True)
    print(f"{name} {len(keys)} 个合约-月份：检测器 {totals[0]}，引擎 {totals[1]}，一致 {totals[2]}")
