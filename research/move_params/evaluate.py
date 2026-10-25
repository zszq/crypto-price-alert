"""按参数组汇总：每天提醒数、独立行情段数、提醒质量、大行情覆盖率与首报延迟，开发期与留出期分开。"""

from __future__ import annotations

import json
import math
from functools import cache

import numpy as np
import sim

SEGMENT_GAP = 600  # 同一合约两条提醒间隔超过 10 分钟才算新的一段行情
BIG_WINDOW = 5  # 大行情：5 分钟收盘价变动


def symbols(variant: str = "s30_l180"):
    path = sim.DATA / sim.dataset(variant)["symbols"]
    names = json.load(open(path)) if path.suffix == ".json" else [line.strip() for line in open(path) if line.strip()]
    return [s for s in names if (sim.DATA / "rec" / variant / f"{s}.npz").exists()]


def period_of(t: float) -> str:
    return "dev" if t < sim.DEV_END else "hold"


@cache
def big_moves(symbol: str, threshold: float, variant: str = "s30_l180") -> list[tuple[float, float, int]]:
    """5 分钟收盘价变动 ≥ threshold% 的行情段（开始、结束、方向），只算当时在合约池（800 万）里的。"""
    data = sim.load(symbol, variant)
    sec, last = data["sec"], data["last"]
    ds = sim.dataset(variant)
    if len(sec) == 0:
        return []
    minutes = np.arange(ds["start"] // 60, ds["start"] // 60 + ds["days"] * 1440)
    idx = np.searchsorted(sec, (minutes + 1) * 60, side="left") - 1
    valid = idx >= 0
    close = np.where(valid, last[np.clip(idx, 0, None)], np.nan)
    ret = (close[BIG_WINDOW:] / close[:-BIG_WINDOW] - 1) * 100
    events = []
    for direction in (1, -1):
        hit = np.flatnonzero(direction * ret >= threshold)
        if not len(hit):
            continue
        # 相邻（重叠）的 5 分钟窗口合并成一段
        start = prev = hit[0]
        for h in list(hit[1:]) + [None]:
            if h is not None and h - prev <= BIG_WINDOW:
                prev = h
                continue
            t0 = (minutes[start] + 1) * 60  # 起点那根收盘
            t1 = (minutes[prev + BIG_WINDOW] + 1) * 60
            if sim.in_pool(symbol, t0, variant=variant):
                events.append((float(t0), float(t1), direction))
            if h is not None:
                start = prev = h
    return events


def evaluate(params: sim.Params, variant: str = "s30_l180") -> dict:
    stats = {
        p: {
            "n": 0,
            "short": 0,
            "long": 0,
            "segments": 0,
            "good": 0,
            "fake": 0,
            "held": 0,
            "r5": [],
            "mfe": [],
            "big3": [0, 0, []],
            "big5": [0, 0, []],
        }
        for p in ("dev", "hold")
    }
    for symbol in symbols(variant):
        alerts = sim.alerts_for(symbol, params, variant)
        res = sim.outcomes(symbol, alerts, variant)
        last_t = -math.inf
        for o in res:
            s = stats[period_of(o["t"])]
            s["n"] += 1
            s[o["win"]] += 1
            s["segments"] += o["t"] - last_t > SEGMENT_GAP
            last_t = o["t"]
            s["good"] += o["good"]
            s["fake"] += o["fake"]
            s["held"] += o["held"]
            s["r5"].append(o["r5"])
            s["mfe"].append(o["mfe"])
        times = np.array([a[0] for a in alerts])
        dirs = np.array([a[2] for a in alerts])
        for key, threshold in (("big3", 3.0), ("big5", 5.0)):
            for t0, t1, d in big_moves(symbol, threshold, variant):
                s = stats[period_of(t0)][key]
                s[0] += 1
                # 行情段从第一个达标 5 分钟窗口的起点算起，到最后一个窗口结束后 1 分钟内报到都算
                hit = np.flatnonzero((times >= t0) & (times <= t1 + 60) & (dirs == d))
                if len(hit):
                    s[1] += 1
                    s[2].append(times[hit[0]] - t0)
    out = {}
    for p, s in stats.items():
        days = 20 if p == "dev" else 10
        n = max(s["n"], 1)
        out[p] = {
            "per_day": s["n"] / days,
            "short": s["short"] / days,
            "long": s["long"] / days,
            "seg_day": s["segments"] / days,
            "good": s["good"] / n,
            "fake": s["fake"] / n,
            "held": s["held"] / n,
            "r5_med": float(np.median(s["r5"])) if s["r5"] else 0.0,
            "mfe_med": float(np.median(s["mfe"])) if s["mfe"] else 0.0,
            "big3": s["big3"][1] / max(s["big3"][0], 1),
            "big3_n": s["big3"][0],
            "big3_delay": float(np.median(s["big3"][2])) if s["big3"][2] else math.nan,
            "big5": s["big5"][1] / max(s["big5"][0], 1),
            "big5_n": s["big5"][0],
            "big5_delay": float(np.median(s["big5"][2])) if s["big5"][2] else math.nan,
        }
    return out


def fmt(r: dict) -> str:
    return (
        f"{r['per_day']:6.1f}/天(短{r['short']:5.1f} 长{r['long']:5.1f}) 段{r['seg_day']:5.1f} "
        f"好{r['good']:.1%} 假{r['fake']:.1%} 持{r['held']:.1%} r5{r['r5_med']:+.2f} mfe{r['mfe_med']:.2f} "
        f"大3%{r['big3']:.1%}({r['big3_n']}) 延{r['big3_delay']:+.0f}s "
        f"大5%{r['big5']:.1%}({r['big5_n']}) 延{r['big5_delay']:+.0f}s"
    )


if __name__ == "__main__":
    r = evaluate(sim.CURRENT)
    print("现行 dev ", fmt(r["dev"]))
    print("现行 hold", fmt(r["hold"]))
