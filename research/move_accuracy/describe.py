"""留出期的直观指标（只描述，不参与选参与采用）：选中参数、只用 10 秒窗口与现行参数。"""

from __future__ import annotations

import numpy as np
from common import HOLD_MONTHS
from select_params import CURRENT
from sim import Params, alert_rows

GATE = (("pre_neg", 0.0),)
CONFIGS = {
    "现行": [(v, p, ()) for v, p in CURRENT],
    "选中（10 秒 + 20 秒）": [
        ("L10", Params(2.0, 1.0, 2, 60, 30, 2), GATE),
        ("L20F10", Params(2.0, 1.0, 3, 20, 30, 2), GATE),
    ],
    "只用 10 秒窗口": [("L10", Params(2.0, 1.0, 2, 60, 30, 2), GATE)],
}

for name, windows in CONFIGS.items():
    parts = [alert_rows(v, HOLD_MONTHS, p, g) for v, p, g in windows]
    ret = np.concatenate([x["ret"] for x in parts])
    ret = ret[~np.isnan(ret).any(axis=1)]
    print(f"{name}（{len(ret)} 条）")
    for k in (1, 3, 5, 10):
        r = ret[:, k - 1]
        print(
            f"  {k:>2} 分钟后：顺势 {np.mean(r > 0):.1%}、逆势 {np.mean(r < 0):.1%}，"
            f"顺势 > 0.1% {np.mean(r > 0.1):.1%}，"
            f"中位 {np.median(r):+.3f}%，均值 {np.mean(np.clip(r, -20, 20)):+.3f}%（截断 ±20%）"
        )
    print(f"  反方向 5 分钟保持（每个整分钟点都在入场价另一侧）：{(ret[:, :5] < 0).all(axis=1).mean():.1%}")
