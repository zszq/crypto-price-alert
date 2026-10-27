"""第三轮（PLAN.md 第三轮）：收集宽松反弹回落候选秒的全部列，写 cand3_<名称>.npz。

宽松候选 = 10 秒窗口、涨跌幅 ≥ 1%、1 ATR、确认 1、5 笔、500 万池、前置走势 ≤ −2%、成交暴增 ≥ 6、不冷却时会提醒的每一秒，
是候选族里任何一组的提醒行的超集：确认 2 秒的连续性另由 grid3.py 在全部行上算，这里只需要可以提醒的那一秒。
用法：cand3.py dev|r3|r4（dev = 2025-10 ~ 2026-09 选参期，r3 = 2025-04 ~ 09 样本外，r4 = 2024-10 ~ 2025-03 补充检验）
"""

from __future__ import annotations

import sys

import numpy as np
from common import DATA, FRESH_MONTHS, MONTHS, ROUND3_MONTHS, ROUND4_MONTHS
from sim import COLUMNS, EXTRA_COLUMNS, Params, Simulator, chunked_files, load

WIDE = Params(1.0, 1.0, 1, 5, 0, 0)
GATES = (("pre_neg", -2.0), ("burst", 6.0))
KEEP = COLUMNS + EXTRA_COLUMNS
PERIODS = {"dev": MONTHS + FRESH_MONTHS, "r3": ROUND3_MONTHS, "r4": ROUND4_MONTHS}

if __name__ == "__main__":
    name = sys.argv[1]
    months = PERIODS[name]
    parts = []
    for files in chunked_files("L10F10", months):
        rows = load("L10F10", months, KEEP, files)
        hit = Simulator(rows).alerts(WIDE, GATES)
        part = {c: rows[c][hit] for c in KEEP}
        part["month"] = rows.month[hit]
        part["symbol"] = np.array([rows.symbols[g] for g in rows.group[hit]])
        parts.append(part)
    out = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    np.savez(DATA / f"cand3_{name}.npz", **out)
    print(name, "候选", len(out["t"]))
