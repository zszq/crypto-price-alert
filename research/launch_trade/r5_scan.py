"""第五轮：提醒特征分档，五年 10 个半年段里每档每笔收益为正的段数（看哪些特征在熊市、牛市都同向）。

用法：r5_scan.py <数据集>；交易规则为 r5_studies.LONG、所有波次、不限仓（看单笔质量，不受限仓先到先买干扰）。
"""

import sys

import numpy as np
from r5_studies import HALVES, HOLDOUT_START, LONG, Book

sys.stdout.reconfigure(encoding="utf-8")
book = Book(sys.argv[1])
d = book.data
tr = book.trades(LONG)
f = dict(book.features)
f.update(
    {
        "涨幅%": d.meta["change"],
        "提醒前1h": d.meta["ret_1h"],
        "提醒前24h": d.meta["ret_24h"],
        "距30天低": d.meta["start_price"] / d.meta["low_30d"] - 1,
        "60分振幅": d.meta["range_60"],
        "基准区间": d.meta["base_high"] / d.meta["base_low"] - 1,
        "波次": d.wave.astype(float),
        "北京时": (((d.closed_at // 3600) + 8) % 24).astype(float),
    }
)
base = (d.direction == 1) & ~np.isnan(tr.net) & (d.closed_at >= HOLDOUT_START)
edges = [HOLDOUT_START, *[h for h in HALVES if h > HOLDOUT_START], d.end + 1]
seg = np.digitize(d.closed_at, edges) - 1
print(
    f"全部 {base.sum()} 笔，每笔 {tr.net[base].mean() * 100:+.2f}%；各段每笔：",
    " ".join(f"{tr.net[base & (seg == k)].mean() * 100:+.1f}" for k in range(len(edges) - 1)),
)
for name, x in f.items():
    ok = base & ~np.isnan(x)
    qs = np.unique(np.nanpercentile(x[ok], [20, 40, 60, 80]))
    bins = np.digitize(x, qs)
    parts = []
    for b in range(len(qs) + 1):
        m = ok & (bins == b)
        per = [tr.net[m & (seg == k)] for k in range(len(edges) - 1)]
        pos = sum(1 for y in per if len(y) >= 20 and y.mean() > 0)
        cnt = sum(1 for y in per if len(y) >= 20)
        lo = -np.inf if b == 0 else qs[b - 1]
        parts.append(f"[≥{lo:.3g}] {tr.net[m].mean() * 100:+.2f}% {pos}/{cnt}")
    print(f"{name}: " + "  ".join(parts))
