"""第二轮：在宽松提醒上做可解释规则的网格（前置走势深度 × 成交放大 × 涨跌幅 × 主动买卖），看 4~7 月与 8~9 月。"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from common import DATA

name = sys.argv[1]
d = np.load(DATA / f"feat_{name}.npz", allow_pickle=True)
ret = d["ret"]
valid = ~np.isnan(ret).any(axis=1)
h5 = (np.nan_to_num(ret, nan=-1) > 0)[:, :5].all(1)
m = d["month"]
dev, hold = valid & (m <= 3), valid & (m >= 4)
res = []
pos5 = ret[:, 4] > 0
for pre_col, pre_max, act_col, act_min, cmin, imb_max, ext_max in itertools.product(
    ("ret_300", "ret_900", "pre"),
    (0, -2, -4, -6, -8),
    ("qratio_60", "nratio_60"),
    (8, 12, 16, 24, 32),
    (1.0, 1.5, 2.0),
    (None, 0.0),
    (None, -5, -10, -20),
):
    s = (d[pre_col] <= pre_max) & (d[act_col] >= act_min) & (d["chg"] >= cmin)
    if imb_max is not None:
        s &= d["imb_300"] <= imb_max
    if ext_max is not None:
        s &= d["ext6h"] <= ext_max
    a, b = s & dev, s & hold
    if a.sum() < int(sys.argv[2]) if len(sys.argv) > 2 else a.sum() < 200:
        continue
    res.append(
        (
            h5[a].mean(),
            a.sum() / 122,
            h5[b].mean() if b.sum() else np.nan,
            b.sum() / 61,
            f"{pre_col}≤{pre_max} {act_col}≥{act_min} 涨跌≥{cmin} imb_300≤{imb_max} ext6h≤{ext_max} "
            f"| 8~9月 5分钟后顺势 {pos5[b].mean():.0%}",
        )
    )
res.sort(key=lambda r: -r[0])
print("按 4~7 月 hold5 排序（4~7 月至少 200 条）：")
for r in res[:25]:
    print(f"  4~7月 {r[0]:.1%}（{r[1]:.1f}/天）  8~9月 {r[2]:.1%}（{r[3]:.1f}/天）  {r[4]}")
