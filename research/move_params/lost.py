"""新参数丢掉的现行行情段：段内现行提醒的最大涨跌幅、之后 15 分钟最大顺势幅度，与保留下来的段对比。"""

import pickle

import combine
import numpy as np
import sim


def report(res, pick=sim.ROUND1):
    def cat(*keys):
        cols = [res[k][0] for k in keys]
        return {f: np.concatenate([c[f] for c in cols]) for f in ("sym", "t", "chg", "d", "held", "fake")}

    base = cat(("short", sim.CURRENT.short), ("long", sim.CURRENT.long))
    new = cat(("short", pick.short), ("long", pick.long))
    key_b = base["sym"] * 1e10 + base["t"]
    order = np.argsort(key_b)
    key_b = key_b[order]
    chg_b = np.abs(base["chg"][order])
    key_n = np.sort(new["sym"] * 1e10 + new["t"])
    spans = combine.segment_spans(key_b)
    cut = np.concatenate([[0], np.flatnonzero(np.diff(key_b) > combine.SEGMENT_GAP) + 1, [len(key_b)]])
    lo = np.searchsorted(key_n, spans[:, 0], side="left")
    hi = np.searchsorted(key_n, spans[:, 1] + combine.LONG_COOLDOWN, side="right")
    kept = hi > lo
    seg_max = np.array([chg_b[cut[i] : cut[i + 1]].max() for i in range(len(spans))])
    seg_n = np.diff(cut)
    for label, m in (("保留的段", kept), ("丢掉的段", ~kept)):
        print(
            f"{label} {m.sum():5d} 段  段内最大涨跌幅中位 {np.median(seg_max[m]):.2f}%  "
            f"≥3% 的占 {np.mean(seg_max[m] >= 3):.1%}  "
            f"段内现行提醒数中位 {np.median(seg_n[m]):.0f}  只有 1 条的占 {np.mean(seg_n[m] == 1):.0%}"
        )


if __name__ == "__main__":
    report(pickle.load(open(sim.DATA / "results" / "grid_s30_l180.pkl", "rb"))["results"])
