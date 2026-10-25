"""把 grid.py 的短、长窗口结果两两组合（长窗口也可以关闭），汇总每组的数量、质量与大行情覆盖。

输出 results/combos_<variant>.pkl：每个组合在开发期、留出期的各项指标，选参见 select_params.py。
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import sim

SEGMENT_GAP = 600
DUP_GAP = 300
LONG_COOLDOWN = 180  # 现行长窗口冷却，新提醒最晚出现在旧提醒之后这么久
FIELDS = ("good", "fake", "cont", "rev", "held")


def summarize(cols, period_mask):
    out = {"n": int(period_mask.sum())}
    for k in FIELDS:
        out[k] = float(cols[k][period_mask].sum())
    out["r5"] = float(cols["r5"][period_mask].sum())
    key = cols["sym"][period_mask] * 1e10 + cols["t"][period_mask]
    out["key"] = np.sort(key)
    out["up"] = np.sort(key[cols["d"][period_mask] > 0])
    out["down"] = np.sort(key[cols["d"][period_mask] < 0])
    return out


def duplicates(a, b):
    """同一合约同方向 DUP_GAP 秒内已有提醒（任一窗口）的条数。"""
    total = 0
    for side in ("up", "down"):
        keys = np.sort(np.concatenate([a[side], b[side]]))
        total += int((np.diff(keys) <= DUP_GAP).sum())
    return total


def segment_spans(keys):
    """把提醒按 SEGMENT_GAP 切成行情段，返回每段的（首条、末条）键。"""
    if len(keys) == 0:
        return np.zeros((0, 2))
    cut = np.flatnonzero(np.diff(keys) > SEGMENT_GAP)
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(keys) - 1]])
    return np.stack([keys[starts], keys[ends]], axis=1)


def kept(spans, keys):
    """现行参数的行情段里，新参数仍至少报一条的段数（新提醒只会出现在旧提醒及其冷却之内）。"""
    lo = np.searchsorted(keys, spans[:, 0], side="left")
    hi = np.searchsorted(keys, spans[:, 1] + LONG_COOLDOWN, side="right")
    return int((hi > lo).sum())


def segments(keys):
    if len(keys) == 0:
        return 0
    return int(1 + (np.diff(keys) > SEGMENT_GAP).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s30_l180")
    ns = parser.parse_args()
    data = pickle.load(open(sim.DATA / "results" / f"grid_{ns.variant}.pkl", "rb"))
    rows = build(data)
    with open(sim.DATA / "results" / f"combos_{ns.variant}.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(len(rows), "组合")


def build(data, dev_end=sim.DEV_END, keys=None):
    """grid.py 的结果两两组合；keys 给出时只算这些（短, 长）组合。dev_end 之前算开发期，之后算留出期。"""
    ctx = setup(data, dev_end)
    return [row(ctx, key) for key in keys or ctx["pairs"]]


def setup(data, dev_end=sim.DEV_END):
    """各窗口按开发/留出期汇总一次，供 row 反复组合（组合很多时由多个进程共用）。"""
    ev = data["events"]
    ev_period = np.where(ev[:, 1] < dev_end, 0, 1)
    shorts = [(w, r) for (name, w), r in data["results"].items() if name == "short"]
    longs = [(w, r) for (name, w), r in data["results"].items() if name == "long"] + [(None, None)]
    empty = {"sym": np.zeros(0), "t": np.zeros(0), **{k: np.zeros(0) for k in (*FIELDS, "r5", "d", "chg")}}
    prepared = {}
    for side, items in (("short", shorts), ("long", longs)):
        for w, r in items:
            cols, delay = r if r is not None else (empty, np.full(len(ev), np.inf))
            prepared[(side, w)] = prepare(cols, delay, dev_end)
    base_s = prepared[("short", sim.CURRENT.short)][0]
    base_l = prepared[("long", sim.CURRENT.long)][0]
    base_spans = [segment_spans(np.sort(np.concatenate([base_s[p]["key"], base_l[p]["key"]]))) for p in (0, 1)]
    pairs = [(sw, lw) for sw, _ in shorts for lw, _ in longs]
    return {"prepared": prepared, "base_spans": base_spans, "ev": ev, "ev_period": ev_period, "pairs": pairs}


def row(ctx, key):
    sw, lw = key
    p = ctx["prepared"]
    return combo_row(sw, lw, p[("short", sw)], p[("long", lw)], ctx["base_spans"], ctx["ev"], ctx["ev_period"])


def prepare(cols, delay, dev_end=sim.DEV_END):
    dev = cols["t"] < dev_end
    return [summarize(cols, dev), summarize(cols, ~dev)], delay


def combo_row(sw, lw, short_prepared, long_prepared, base_spans, ev, ev_period):
    (s_stats, s_delay) = short_prepared
    (l_stats, l_delay) = long_prepared
    delay = np.minimum(s_delay, l_delay)
    row = {"short": sw, "long": lw}
    for p, name in ((0, "dev"), (1, "hold")):
        a, b = s_stats[p], l_stats[p]
        n = a["n"] + b["n"]
        row[f"{name}_n"] = n
        row[f"{name}_short"] = a["n"]
        for k in (*FIELDS, "r5"):
            row[f"{name}_{k}"] = (a[k] + b[k]) / max(n, 1)
        keys = np.sort(np.concatenate([a["key"], b["key"]]))
        row[f"{name}_seg"] = segments(keys)
        row[f"{name}_segkeep"] = kept(base_spans[p], keys) / max(len(base_spans[p]), 1)
        row[f"{name}_dup"] = duplicates(a, b) / max(n, 1)
        for level in (3.0, 5.0):
            mask = (ev_period == p) & (ev[:, 4] == level)
            caught = np.isfinite(delay[mask])
            tag = f"{name}_big{int(level)}"
            row[tag] = caught.mean() if mask.any() else np.nan
            row[tag + "_delay"] = float(np.median(delay[mask][caught])) if caught.any() else np.nan
    return row


if __name__ == "__main__":
    main()
