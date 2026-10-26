"""第二轮探索（4~9 月都已看过，只用来找方向；结论要在 2025-10~2026-03 新数据上验证）。

收集一组宽松参数的提醒（不加前置走势），对前置走势阈值、涨跌幅等做二维分档，看方向保持率能到多高。
用法：explore2.py 变体 涨跌幅 ATR 确认 笔数 冷却 合约池档
"""

from __future__ import annotations

import sys

import numpy as np
from common import MONTHS
from sim import Params, alert_rows

variant = sys.argv[1]
a, b, c, trades, cd, pool = (float(x) for x in sys.argv[2:8])
ar = alert_rows(variant, MONTHS, Params(a, b, int(c), int(trades), cd, int(pool)))
ret = ar["ret"]
valid = ~np.isnan(ret).any(axis=1)
pos = np.nan_to_num(ret, nan=-1) > 0
h5 = pos[:, :5].all(1)
days = 183
print(
    f"{variant} {a}% {b}ATR 确认{c:g} 笔数{trades:g} 冷却{cd:g} 池{pool:g}："
    f"{valid.sum()} 条（每天 {len(h5) / days:.1f}），hold5 {h5[valid].mean():.1%}"
)
pre = ar["pre"]
chg = ar["chg"]
for lo in (None, 0, -1, -2, -3, -5, -8):
    s = valid if lo is None else valid & (pre <= lo)
    row = [f"前置 ≤ {lo}%" if lo is not None else "前置不限"]
    for cmin in (0, 3, 5, 8):
        t = s & (chg >= max(cmin, a))
        row.append(f"涨跌≥{cmin}: {h5[t].mean():.1%}({t.sum() / days:.1f}/天)" if t.sum() else f"涨跌≥{cmin}: -")
    print("  ", "  ".join(row))
