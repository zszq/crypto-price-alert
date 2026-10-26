"""描述性分析（只用开发期，不参与选参）：某组参数的提醒按各特征分档后的方向保持率。

用法：explore.py 变体 涨跌幅 ATR 确认 笔数 冷却
"""

from __future__ import annotations

import sys

import numpy as np
from common import DEV_MONTHS
from sim import Params, alert_rows

variant = sys.argv[1]
a, b, c, trades, cd = (float(x) for x in sys.argv[2:7])
ar = alert_rows(variant, DEV_MONTHS, Params(a, b, int(c), int(trades), cd, 1))
ret = ar["ret"]
valid = ~np.isnan(ret).any(axis=1)
pos = np.nan_to_num(ret, nan=-1) > 0
h5 = pos[:, :5].all(1)
print(
    f"{variant} {a}% {b}ATR 确认{c:g} 笔数{trades:g} 冷却{cd:g}：{len(h5)} 条，"
    f"hold1 {pos[valid, 0].mean():.1%} hold5 {h5[valid].mean():.1%} 5 分钟顺势中位 {np.median(ret[valid, 4]):+.3f}%"
)
for d in (1, -1):
    s = valid & (ar["dir"] == d)
    print(f"  {'急涨' if d > 0 else '急跌'} {s.sum()} 条 hold5 {h5[s].mean():.1%}")
for name in ("chg", "atrm", "trades", "imb", "vr", "retr", "pre", "brk", "eff", "atrpct"):
    x = ar[name].astype(np.float64)
    ok = valid & np.isfinite(x)
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 6)))
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        s = ok & (x >= lo) & (x <= hi)
        cells.append(f"[{lo:.3g},{hi:.3g}] {h5[s].mean():.1%}({s.sum()})")
    print(f"  {name:<7}", "  ".join(cells))
