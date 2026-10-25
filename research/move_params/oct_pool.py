"""用 10 月 REST 成交本身算滚动 24 小时成交额，得出 10-03~10-11 的合约池。

与 pool.py 同一口径：> 800 万进、≤ 640 万出，每 10 分钟刷新。
"""

import json

import numpy as np
import sim
from record import load_trades

contracts = json.load(open(sim.DATA.parent / "bt" / "contracts.json"))
out = {}
for symbol in json.load(open(sim.DATA / "oct_symbols.json")):
    mult = float(contracts.get(symbol, {}).get("quanto_multiplier") or 0)
    trades = load_trades(symbol, "oct")
    if mult <= 0 or not len(trades):
        print("跳过", symbol, mult, len(trades))
        continue
    marks = sim.OCT_START + 600 * np.arange(9 * 144)
    quote = np.abs(trades[:, 3]) * mult * trades[:, 2]
    cs = np.concatenate([[0], np.cumsum(quote)])
    hi = np.searchsorted(trades[:, 0], marks, side="left")
    lo = np.searchsorted(trades[:, 0], marks - 86400, side="left")
    vol = cs[hi] - cs[lo]
    member, inside = [], False
    for v in vol:
        inside = v > 6.4e6 if inside else v > 8e6
        member.append(int(inside))
    out[symbol] = {"mult": mult, "frac": float(np.mean(member)), "member": member}
json.dump(out, open(sim.DATA / "oct_pool.json", "w"))
print(len(out), "个合约；平均同时在池", round(sum(v["frac"] for v in out.values()), 1))
