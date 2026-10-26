"""第二轮选中参数的相邻参数（任一项挪一格）：hold5 与每天条数。用法：neighbors2.py explore|fresh"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import DATA, FRESH_MONTHS, MONTHS
from sim import COLUMNS, EXTRA_COLUMNS, Params, alert_rows, month_days

months = MONTHS if sys.argv[1] == "explore" else FRESH_MONTHS
days = sum(month_days(m) for m in months)
sel = json.load(open(DATA / "selection2.json", encoding="utf-8"))
p0 = Params(**sel["params"])
g0 = tuple(tuple(g) for g in sel["gates"])


def run(label: str, p: Params, gates: tuple) -> None:
    ar = alert_rows("L10F10", months, p, gates, COLUMNS + EXTRA_COLUMNS)
    ret = ar["ret"]
    valid = ~np.isnan(ret).any(axis=1)
    h5 = (np.nan_to_num(ret, nan=-1) > 0)[:, :5].all(1) & valid
    print(
        f"  {label:<18} hold5 {h5.sum() / max(valid.sum(), 1):.1%}  每天 {len(h5) / days:.2f}  "
        f"5 分钟后顺势 {np.mean(ret[valid, 4] > 0):.0%}"
    )


def with_gate(name: str, value: float | None) -> tuple:
    gates = tuple(g for g in g0 if g[0] != name)
    return gates if value is None else gates + ((name, value),)


print(f"{sys.argv[1]}：{p0} {g0}")
run("选中", p0, g0)
for field, values in (("a", (1.0, 2.0)), ("c", (2,)), ("trades", (20,)), ("cooldown", (300,)), ("pool", (0,))):
    for v in values:
        run(f"{field} {v}", Params(**(vars(p0) | {field: v})), g0)
for v in (-4.0, -8.0, None):
    run(f"前置 ≤ {v}", p0, with_gate("pre_neg", v))
for v in (12.0, 24.0, None):
    run(f"成交暴增 ≥ {v}", p0, with_gate("burst", v))
