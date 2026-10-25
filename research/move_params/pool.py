"""按 1 分钟 K 线估算每个合约滚动 24 小时成交额，得出某月的秒级异动合约池（> 800 万进、≤ 640 万出，每 10 分钟刷新）。

用法：pool.py [2026MM]，默认九月（输出 pool.json、volume.json），其他月份输出 pool_<月>.json 与 symbols_<月>.txt。
"""

import glob
import gzip
import json
import os
import sys

import numpy as np
import sim

HIST = sim.DATA.parent / "hist"
ENTER, EXIT = 8e6, 8e6 * 0.8
month = sys.argv[1] if len(sys.argv) > 1 else "202609"
ds = sim.dataset(sim.month_variant(month))
start, end = ds["start"], ds["start"] + ds["days"] * 86400
prev = sim.previous_month(month)
contracts = json.load(open(sim.DATA.parent / "bt" / "contracts.json"))
out = {}
vols = {}
for path in sorted(glob.glob(str(HIST / f"*-{month}.csv.gz"))):
    sym = os.path.basename(path)[: -len(f"-{month}.csv.gz")]
    mult = float(contracts.get(sym, {}).get("quanto_multiplier") or 0)
    if mult <= 0:
        continue
    rows = []
    for p in (str(HIST / f"{sym}-{prev}.csv.gz"), path):
        if os.path.exists(p):
            with gzip.open(p, "rt") as f:
                rows += [line.split(",") for line in f if line.strip()]
    if not rows:
        continue
    a = np.array(rows, dtype=float)
    t = a[:, 0].astype(np.int64)
    q = np.abs(a[:, 1]) * mult * a[:, 2]
    minute = (t - (start - 86400)) // 60
    n = (end - start + 86400) // 60
    ok = (minute >= 0) & (minute < n)
    per = np.zeros(n)
    np.add.at(per, minute[ok], q[ok])
    cs = np.concatenate([[0], np.cumsum(per)])
    # 每 10 分钟刷新一次：取刷新时刻之前 24 小时成交额
    marks = np.arange(1440, n + 1, 10)
    vol = cs[marks] - cs[marks - 1440]
    member = np.zeros(len(marks), bool)
    inside = False
    for i, v in enumerate(vol):
        inside = v > EXIT if inside else v > ENTER
        member[i] = inside
    if member.any():
        out[sym] = {"mult": mult, "frac": float(member.mean()), "member": member.astype(int).tolist()}
        vols[sym] = vol.round().tolist()
if month == "202609":
    json.dump(out, open(sim.DATA / "pool.json", "w"))
    json.dump(vols, open(sim.DATA / "volume.json", "w"))
else:
    json.dump(out, open(sim.DATA / ds["pool"], "w"))
    # 只留下载站当月有逐笔成交文件的合约
    sizes = json.load(open(sim.DATA / f"trade_sizes_{month}.json"))
    names = sorted(s for s in out if s in sizes)
    (sim.DATA / ds["symbols"]).write_text("".join(f"{s}\n" for s in names))
    print("下载站有文件的", len(names), "个，约", round(sum(sizes[s] for s in names) / 1e9, 2), "GB")
print(len(out), "个合约曾进池；平均同时在池", round(sum(v["frac"] for v in out.values()), 1))
