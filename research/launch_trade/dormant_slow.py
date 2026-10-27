"""冷门慢启动：平时几乎不动的冷门币，量价在一两个小时里缓慢拉升，口径与判据见 dormant_slow.md（跑数前写定）。

冷门、冷门启动行情、「准」与 dormant_solo / dormant_early 相同（直接沿用其代码）；本脚本抽 30~120 分钟长窗口的候选分钟，
并逐分钟算出现行冷门启动（dormant_best.md 的 B3，即 default.yaml）的提醒，用来衡量慢启动在现行之外多报早了多少。

用法（仓库根目录，$P 见 README.md）：
  & $P research/launch_trade/dormant_slow.py extract    # 候选分钟、特征、B3 提醒、冷门启动行情
  & $P research/launch_trade/dormant_slow.py explore    # 开发期探索（设计候选族用，只看 2025 年之前）
  & $P research/launch_trade/dormant_slow.py verify     # 网格判定与逐分钟直接实现对拍
  & $P research/launch_trade/dormant_slow.py search     # 全部候选逐周计数（全期，选参只用开发期）
  & $P research/launch_trade/dormant_slow.py select     # 开发期选参、PBO
  & $P research/launch_trade/dormant_slow.py evaluate   # 样本外检验
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from dormant_calm import _features
from dormant_early import _leg_position
from dormant_solo import PRE, _dormant_legs
from launch_alerts import (
    BOOTSTRAP,
    CSCV_BLOCKS,
    DEV_WEEKS,
    FORWARD,
    GAINS,
    HOLDS,
    TARGETS,
    WEEK,
    _bootstrap_weights,
    _validity,
)
from launch_grid import EPOCH, _multiplier, _rolling_extreme
from load_data import load_contracts
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "dormant_slow"
GATE = 1e6  # 冷门：窗口之前 24 小时成交额上限，与现行相同
CALM = 240  # 涨前波动的统计长度，与现行相同
WINDOWS = (30, 60, 90, 120)
# 抽取下限，比候选族最松的一档还松。
MIN_QUOTE = 5e3
MIN_CHANGE = 4.0
MAX_JUMP = 5.0
COOLDOWN = 120  # 慢启动的冷却（分钟），不报第 N 波；不在候选族里

# 现行冷门启动（B3，config/default.yaml 的 dormant 段）。
B3_W = 15
B3_CHANGE = 8.0
B3_QUOTE = 2e4
B3_ACTIVE = 8
B3_LAST = 1e3
B3_LATE = 0.4
B3_JUMP = 5.0
B3_BODY = 0.6
B3_VOL = 0.5
B3_MULTIPLE = 10.0
B3_COOLDOWN = 60
B3_STEP = 3.0

FEATURES = ("change", "quote", "active", "pre", "jump", "vol", "er", "ratio", "late", "retrace", "body")


# ---------------------------------------------------------------- 抽取


@njit(cache=True)
def _b3_alerts(open_, high, low, close, quote, cum_quote, cum_active, cum_uncovered, cum_tr, jump_max):
    """逐分钟按现行冷门启动（B3）判定并走冷却与第 N 波，返回提醒的 K 线下标；与 DormantDetector 同一口径。"""
    n = len(close)
    window = B3_W
    half = (window + 1) // 2
    out = []
    last_t = -1
    last_p = 0.0
    for t in range(window + PRE, n):
        s = t - window
        first = s + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:
            continue
        if cum_quote[s + 1] - cum_quote[first] > GATE:
            continue
        q = cum_quote[t + 1] - cum_quote[s + 1]
        if q < B3_QUOTE:
            continue
        change = (close[t] / close[s] - 1) * 100
        if change < B3_CHANGE or jump_max[t] > B3_JUMP:
            continue
        if cum_active[t + 1] - cum_active[s + 1] < B3_ACTIVE or quote[t] < B3_LAST:
            continue
        if (cum_quote[t + 1] - cum_quote[t + 1 - half]) / q < B3_LATE:
            continue
        vol = (cum_tr[s + 1] - cum_tr[s + 1 - CALM]) / CALM
        if vol > B3_VOL or (vol > 0 and change / vol < B3_MULTIPLE):
            continue
        body = 0.0
        span = 0.0
        for j in range(s + 1, t + 1):
            body += abs(close[j] - open_[j])
            span += high[j] - low[j]
        if span <= 0 or body / span < B3_BODY:
            continue
        if last_t >= 0 and t - last_t < B3_COOLDOWN and close[t] < last_p * (1 + B3_STEP / 100):
            continue
        last_t = t
        last_p = close[t]
        out.append(t)
    return np.array(out, np.int64)


@njit(cache=True)
def _extract(open_, high, low, close, cum_quote, cum_active, cum_uncovered, cum_tr, cum_move, cum_body, cum_span,
             jump_max, win_high, window):
    n = len(close)
    keep = np.zeros(n, np.bool_)
    for t in range(window + PRE, n):
        s = t - window
        first = s + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:  # 含窗口前 24 小时第一根的前收
            continue
        if cum_quote[s + 1] - cum_quote[first] > GATE:
            continue
        if cum_quote[t + 1] - cum_quote[s + 1] < MIN_QUOTE:
            continue
        if (close[t] / close[s] - 1) * 100 < MIN_CHANGE or jump_max[t] > MAX_JUMP:
            continue
        keep[t] = True
    index = np.flatnonzero(keep)
    m = len(index)
    f = np.empty((m, 11))
    half = (window + 1) // 2
    for k in range(m):
        t = index[k]
        s = t - window
        change = (close[t] / close[s] - 1) * 100
        q = cum_quote[t + 1] - cum_quote[s + 1]
        pre = cum_quote[s + 1] - cum_quote[s + 1 - PRE]
        vol = (cum_tr[s + 1] - cum_tr[s + 1 - CALM]) / CALM
        path = cum_move[t + 1] - cum_move[s + 1]
        hi = win_high[t]
        span = cum_span[t + 1] - cum_span[s + 1]
        f[k, 0] = change
        f[k, 1] = q
        f[k, 2] = cum_active[t + 1] - cum_active[s + 1]
        f[k, 3] = pre
        f[k, 4] = jump_max[t]
        f[k, 5] = vol
        f[k, 6] = abs(close[t] - close[s]) / path if path > 0 else 0.0  # 效率：净位移 ÷ 逐分钟路程
        f[k, 7] = q / (pre / PRE * window) if pre > 0 else np.inf  # 窗口成交额相对之前 24 小时同长度均值
        f[k, 8] = (cum_quote[t + 1] - cum_quote[t + 1 - half]) / q
        f[k, 9] = (hi - close[t]) / (hi - close[s]) if hi > close[s] else 0.0  # 从窗口最高价回吐的比例
        f[k, 10] = (cum_body[t + 1] - cum_body[s + 1]) / span if span > 0 else 0.0
    return index, f


def _cumsum(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def _extract_symbol(args: tuple[int, str, float]) -> tuple[int, dict]:
    sym_id, symbol, multiplier = args
    out: dict = {}
    prepared = _features(symbol, multiplier)
    if prepared is None:
        return sym_id, out
    series, cum_quote, cum_active, cum_uncovered, cum_tr, minute_change = prepared
    open_, high, low, close = series.open, series.high, series.low, series.close
    offset = (series.t0 - EPOCH) // 60
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    covered = (cum_uncovered[1:] - cum_uncovered[:-1]) == 0
    ok = covered & np.concatenate([[False], covered[:-1]])
    move = np.where(ok, np.abs(np.diff(close, prepend=close[0])), 0.0)
    cum_move = _cumsum(np.nan_to_num(move))
    cum_body = _cumsum(np.nan_to_num(np.abs(close - open_)))
    cum_span = _cumsum(np.nan_to_num(high - low))
    legs = {gain: _dormant_legs(close, cum_quote, cum_uncovered, gain) for gain in GAINS}
    for gain, (starts, _) in legs.items():
        weeks = ((starts + offset) * 60 // WEEK).astype(np.int64)
        out[f"events{int(gain * 100)}"] = weeks[weeks >= 0]

    def tag_legs(index: np.ndarray) -> dict:
        tagged = {}
        for gain in GAINS:
            starts, ends = legs[gain]
            leg, frac = _leg_position(close, index, starts, ends)
            g = int(gain * 100)
            week = np.full(len(index), -1, np.int64)
            hit = leg >= 0
            week[hit] = (starts[leg[hit]] + offset) * 60 // WEEK
            tagged[f"leg{g}"] = leg
            tagged[f"frac{g}"] = frac
            tagged[f"leg_week{g}"] = week
        return tagged

    # 现行冷门启动（B3）的提醒，以及每段行情里 B3 的第一次提醒（时刻、已走完的比例）。
    b3 = _b3_alerts(open_, high, low, close, series.quote, cum_quote, cum_active, cum_uncovered, cum_tr,
                    _rolling_extreme(minute_change, B3_W, True))
    b3_tags = tag_legs(b3)
    first_b3 = {}
    for gain in GAINS:
        g = int(gain * 100)
        n_legs = len(legs[gain][0])
        first_t = np.full(n_legs, -1, np.int64)
        first_f = np.full(n_legs, np.nan)
        for k in range(len(b3)):
            leg = b3_tags[f"leg{g}"][k]
            if leg >= 0 and first_t[leg] < 0:
                first_t[leg] = b3[k] + offset
                first_f[leg] = b3_tags[f"frac{g}"][k]
        first_b3[g] = (first_t, first_f)
    keep = b3 + offset >= 0
    ones = np.ones(len(b3), np.int8)
    out["b3"] = {"t": (b3 + offset)[keep], "price": close[b3][keep],
                 "valid": _validity(high, low, close, cum_uncovered, b3, ones, targets, holds, FORWARD)[keep],
                 **{k: v[keep] for k, v in b3_tags.items()}}
    for window in WINDOWS:
        index, f = _extract(open_, high, low, close, cum_quote, cum_active, cum_uncovered, cum_tr, cum_move, cum_body,
                            cum_span, _rolling_extreme(minute_change, window, True),
                            _rolling_extreme(high, window, True), window)
        recent = index + offset >= 0  # 2021-09 之前不计（开发期从那时算起）
        index, f = index[recent], f[recent]
        ones = np.ones(len(index), np.int8)
        record = {"t": (index + offset).astype(np.int64), "price": close[index], "f": f.astype(np.float32),
                  "valid": _validity(high, low, close, cum_uncovered, index, ones, targets, holds, FORWARD),
                  **tag_legs(index)}
        for gain in GAINS:
            g = int(gain * 100)
            leg = record[f"leg{g}"]
            first_t, first_f = first_b3[g]
            hit = leg >= 0
            bt = np.full(len(index), -1, np.int64)
            bf = np.full(len(index), np.nan)
            bt[hit] = first_t[leg[hit]]
            bf[hit] = first_f[leg[hit]]
            record[f"b3t{g}"] = bt
            record[f"b3f{g}"] = bf
        out[window] = record
    return sym_id, out


def cmd_extract() -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = [(i, s, _multiplier(contracts[s])) for i, s in enumerate(symbols) if _multiplier(contracts[s])]
    parts: dict = {w: [] for w in (*WINDOWS, "b3")}
    events: dict = {g: [] for g in GAINS}
    started = time.time()
    with ProcessPoolExecutor(max_workers=11) as executor:
        for done, (sym_id, result) in enumerate(executor.map(_extract_symbol, jobs, chunksize=2), 1):
            if result:
                for key in parts:
                    fields = result[key]
                    fields["sym"] = np.full(len(fields["t"]), sym_id, np.int32)
                    parts[key].append(fields)
                for gain in GAINS:
                    events[gain].append(result[f"events{int(gain * 100)}"])
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    np.savez(OUT / "events.npz", **{f"up_{int(g * 100)}": np.concatenate(v) for g, v in events.items()})
    for key, chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        name = "b3" if key == "b3" else f"cand_W{key}"
        np.savez(OUT / f"{name}.npz", **merged)
        print(f"{name}: {len(merged['t'])} 条", flush=True)


def _attach(rec: dict) -> dict:
    edges = np.flatnonzero(np.diff(rec["sym"])) + 1
    rec["seg"] = np.concatenate([[0], edges, [len(rec["sym"])]]).astype(np.int64)
    rec["week"] = ((rec["t"] * 60 + 60) // WEEK).astype(np.int64)
    valid = rec["valid"].astype(np.float32)
    valid[valid < 0] = np.nan
    rec["precision"] = valid.mean(1).astype(np.float64)
    for g in (10, 20):
        rec[f"frac{g}"] = rec[f"frac{g}"].astype(np.float64)
    return rec


def load(window: int) -> dict:
    with np.load(OUT / f"cand_W{window}.npz") as data:
        rec = {name: data[name] for name in data.files}
    for i, name in enumerate(FEATURES):
        rec[name] = rec["f"][:, i].astype(np.float64)
    rec.pop("f")
    return _attach(rec)


def load_b3() -> dict:
    with np.load(OUT / "b3.npz") as data:
        return _attach({name: data[name] for name in data.files})


def _data_weeks() -> int:
    with np.load(OUT / f"cand_W{WINDOWS[0]}.npz") as data:
        last = int(data["t"].max())
    return (last * 60 + 60) // WEEK + 1


def event_counts(first_week: int, n_weeks: int) -> np.ndarray:
    with np.load(OUT / "events.npz") as data:
        rows = []
        for gain in GAINS:
            weeks = data[f"up_{int(gain * 100)}"] - first_week
            rows.append(np.bincount(weeks[(weeks >= 0) & (weeks < n_weeks)], minlength=n_weeks))
    return np.array(rows, np.float64)


# ---------------------------------------------------------------- 计数与指标

COUNTS = ("alerts", "scored", "correct", "credit10", "credit20", "caught10", "caught20",
          "dcredit10", "dcredit20", "dcaught10", "dcaught20")


@njit(cache=True)
def _b3_weekly(sym, week, precision, leg10, leg20, frac10, frac20, lw10, lw20, n_weeks, out):
    """B3 已经走过冷却，这里只按周计数；每段行情只算第一次提醒。"""
    last_s = -1
    l10 = -1
    l20 = -1
    for i in range(len(sym)):
        if sym[i] != last_s:
            last_s = sym[i]
            l10 = -1
            l20 = -1
        w = week[i]
        if w < n_weeks:
            out[0, w] += 1
            if not np.isnan(precision[i]):
                out[1, w] += 1
                out[2, w] += precision[i]
        if leg10[i] >= 0 and leg10[i] != l10:
            l10 = leg10[i]
            if 0 <= lw10[i] < n_weeks:
                out[3, lw10[i]] += 1 - frac10[i]
                out[5, lw10[i]] += frac10[i] <= 0.5
        if leg20[i] >= 0 and leg20[i] != l20:
            l20 = leg20[i]
            if 0 <= lw20[i] < n_weeks:
                out[4, lw20[i]] += 1 - frac20[i]
                out[6, lw20[i]] += frac20[i] <= 0.5


def b3_weekly(n_weeks: int) -> np.ndarray:
    rec = load_b3()
    out = np.zeros((len(COUNTS), n_weeks))
    _b3_weekly(rec["sym"], rec["week"], rec["precision"], rec["leg10"], rec["leg20"], rec["frac10"], rec["frac20"],
               rec["leg_week10"], rec["leg_week20"], n_weeks, out)
    return out


@njit(cache=True)
def _passes(i, p, change, quote, active, jump, er, ratio, late, vol, retrace):
    return not (change[i] < p[0] or quote[i] < p[1] or active[i] < p[2] or jump[i] > p[3] or er[i] < p[4]
                or ratio[i] < p[5] or late[i] < p[6] or vol[i] > p[7] or retrace[i] > p[8])


@njit(cache=True)
def _leg_delta(t, frac, b3t, b3f):
    """慢启动在这一段的第一次提醒（t, frac）加进现行之后，这一段早报分与前半段报到的增量。"""
    if b3t < 0:
        return 1 - frac, 1.0 if frac <= 0.5 else 0.0
    if t < b3t:
        return b3f - frac, (1.0 if frac <= 0.5 else 0.0) - (1.0 if b3f <= 0.5 else 0.0)
    return 0.0, 0.0


@njit(parallel=True, cache=True)
def _grid(seg, t, week, change, quote, active, jump, er, ratio, late, vol, retrace, precision, leg10, leg20,
          frac10, frac20, lw10, lw20, b3t10, b3t20, b3f10, b3f20, params, n_weeks, out):
    n_params = params.shape[0]
    block = 32
    for b in prange((n_params + block - 1) // block):
        c0 = b * block
        c1 = min(n_params, c0 + block)
        last_t = np.empty(block, np.int64)
        last_l10 = np.empty(block, np.int64)
        last_l20 = np.empty(block, np.int64)
        for s in range(len(seg) - 1):
            for k in range(block):
                last_t[k] = -1
                last_l10[k] = -1
                last_l20[k] = -1
            for i in range(seg[s], seg[s + 1]):
                w = week[i]
                if w >= n_weeks:
                    break
                for c in range(c0, c1):
                    if not _passes(i, params[c], change, quote, active, jump, er, ratio, late, vol, retrace):
                        continue
                    k = c - c0
                    if last_t[k] >= 0 and t[i] - last_t[k] < COOLDOWN:
                        continue
                    last_t[k] = t[i]
                    out[c, 0, w] += 1
                    if not np.isnan(precision[i]):
                        out[c, 1, w] += 1
                        out[c, 2, w] += precision[i]
                    if leg10[i] >= 0 and leg10[i] != last_l10[k] and lw10[i] >= 0:
                        last_l10[k] = leg10[i]
                        out[c, 3, lw10[i]] += 1 - frac10[i]
                        out[c, 5, lw10[i]] += frac10[i] <= 0.5
                        d, dc = _leg_delta(t[i], frac10[i], b3t10[i], b3f10[i])
                        out[c, 7, lw10[i]] += d
                        out[c, 9, lw10[i]] += dc
                    if leg20[i] >= 0 and leg20[i] != last_l20[k] and lw20[i] >= 0:
                        last_l20[k] = leg20[i]
                        out[c, 4, lw20[i]] += 1 - frac20[i]
                        out[c, 6, lw20[i]] += frac20[i] <= 0.5
                        d, dc = _leg_delta(t[i], frac20[i], b3t20[i], b3f20[i])
                        out[c, 8, lw20[i]] += d
                        out[c, 10, lw20[i]] += dc


def _grid_args(rec: dict) -> tuple:
    return (rec["seg"], rec["t"], rec["week"], rec["change"], rec["quote"], rec["active"], rec["jump"], rec["er"],
            rec["ratio"], rec["late"], rec["vol"], rec["retrace"], rec["precision"], rec["leg10"], rec["leg20"],
            rec["frac10"], rec["frac20"], rec["leg_week10"], rec["leg_week20"], rec["b3t10"], rec["b3t20"],
            rec["b3f10"], rec["b3f20"])


def run_weekly(rec: dict, params: np.ndarray, n_weeks: int) -> np.ndarray:
    out = np.zeros((len(params), len(COUNTS), n_weeks), np.float32)
    _grid(*_grid_args(rec), np.ascontiguousarray(params, np.float64), n_weeks, out)
    return out


def scores(totals: np.ndarray, base: np.ndarray, events: np.ndarray, days) -> dict:
    """totals 为慢启动各计数（COUNTS 顺序，在时间上求和，可带前导维度），base 为 B3 的同样计数。"""
    def unpack(x):
        x = np.asarray(x, np.float64)
        return {name: x[..., i] for i, name in enumerate(COUNTS)}

    s, b = unpack(totals), unpack(base)
    e10, e20 = events[..., 0], events[..., 1]

    def metrics(c):
        scored = c["scored"]
        precision = np.divide(c["correct"], scored, out=np.zeros(np.shape(scored)), where=scored > 0)
        per_day = c["alerts"] / days
        return {"precision": precision, "per_day": per_day, "fpd": (1 - precision) * per_day,
                "re": 0.5 * (c["credit10"] / e10 + c["credit20"] / e20),
                "recall": 0.5 * (c["caught10"] / e10 + c["caught20"] / e20)}

    out = metrics(s)
    ref = metrics(b)
    out["dre"] = 0.5 * (s["dcredit10"] / e10 + s["dcredit20"] / e20)
    out["drecall"] = 0.5 * (s["dcaught10"] / e10 + s["dcaught20"] / e20)
    out["gain"] = out["dre"] / ref["re"]  # 早报分相对现行提高的比例
    out.update({f"b3_{k}": v for k, v in ref.items()})
    return out


PARAM_NAMES = ("change", "quote", "active", "jump", "er", "ratio", "late", "vol", "retrace")
# 不设限时的取值：下限类取 -inf，上限类取 inf。
OPEN = {"change": -np.inf, "quote": -np.inf, "active": -np.inf, "jump": np.inf, "er": -np.inf, "ratio": -np.inf,
        "late": -np.inf, "vol": np.inf, "retrace": np.inf}


def _need(window: int, share: float) -> float:
    return float(max(1, math.ceil(window * share - 1e-9)))


def row_of(config: dict) -> np.ndarray:
    """config 的 active 是占窗口的比例，换成分钟数。"""
    values = {**OPEN, **{k: v for k, v in config.items() if k != "W"}}
    if np.isfinite(values["active"]):
        values["active"] = _need(config["W"], values["active"])
    return np.array([values[name] for name in PARAM_NAMES], np.float64)


@njit(cache=True)
def _alert_mask(seg, t, ok):
    alert = np.zeros(len(t), np.bool_)
    for s in range(len(seg) - 1):
        last = -1
        for i in range(seg[s], seg[s + 1]):
            if ok[i] and (last < 0 or t[i] - last >= COOLDOWN):
                last = t[i]
                alert[i] = True
    return alert


def ok_of(rec: dict, config: dict) -> np.ndarray:
    p = row_of(config)
    ok = np.ones(len(rec["t"]), np.bool_)
    for name, v in zip(PARAM_NAMES, p, strict=True):
        if name in ("jump", "vol", "retrace"):
            ok &= rec[name] <= v
        else:
            ok &= rec[name] >= v
    return ok


def alerts_for(rec: dict, config: dict) -> np.ndarray:
    return _alert_mask(rec["seg"], rec["t"], ok_of(rec, config))


def cmd_explore() -> None:
    """开发期探索：B3 的开发期指标（与 dormant_best.md 对拍），各窗口宽松方案的指标，以及提醒按特征分档的准确率。"""
    days = DEV_WEEKS * 7
    events = event_counts(0, DEV_WEEKS).sum(1)
    base = b3_weekly(DEV_WEEKS).sum(1)
    b = scores(np.zeros(len(COUNTS)), base, events, days)
    print(f"B3 开发期：准确率 {b['b3_precision']:.3f}、每天 {b['b3_per_day']:.2f}、假启动 {b['b3_fpd']:.2f}、"
          f"早报分 {b['b3_re']:.4f}、前半段召回 {b['b3_recall']:.3f}；行情 {events}")
    for window in WINDOWS:
        rec = load(window)
        dev = rec["week"] < DEV_WEEKS
        print(f"\nW{window}：开发期候选 {int(dev.sum())} 条")
        for change, quote in ((4, 5e3), (6, 1e4), (8, 1e4), (8, 2e4), (12, 2e4)):
            config = {"W": window, "change": change, "quote": quote}
            counts = run_weekly(rec, row_of(config)[None], DEV_WEEKS)[0].sum(1)
            s = scores(counts, base, events, days)
            print(f"  涨幅 ≥ {change}% 成交额 ≥ {quote:.0f}：准确率 {s['precision']:.3f} 每天 {s['per_day']:.2f} "
                  f"早报分 {s['re']:.4f} 相对现行多 {s['gain']:+.1%} 前半段召回 +{s['drecall']:.4f}")
        alert = alerts_for(rec, {"W": window, "change": 6, "quote": 1e4}) & dev
        sub = {k: v[alert] for k, v in rec.items() if k != "seg"}
        has = ~np.isnan(sub["precision"])
        overall = np.nanmean(sub["precision"])
        print(f"  分档（涨幅 ≥ 6%、成交额 ≥ 1 万的 {int(alert.sum())} 条提醒，整体 {overall:.3f}）：")
        for name in ("change", "quote", "active", "jump", "vol", "er", "ratio", "late", "retrace", "body", "pre"):
            x = sub[name][has]
            y = sub["precision"][has]
            finite = np.isfinite(x)
            edges = np.unique(np.quantile(x[finite], [0.2, 0.4, 0.6, 0.8]))
            bins = np.digitize(np.where(finite, x, np.inf), edges)
            cells = [f"{y[bins == k].mean():.3f}({int((bins == k).sum())})" for k in range(len(edges) + 1)]
            print(f"    {name:8s} 切点 {', '.join(f'{e:.3g}' for e in edges)} → {' '.join(cells)}")


def _reference(series, cum_uncovered, minute_change, begin: int, end: int, config: dict) -> set[int]:
    """逐分钟直接按 dormant_slow.md 的口径计算的提醒时刻（K 线下标），不经过抽取与网格。"""
    window = config["W"]
    p = dict(zip(PARAM_NAMES, row_of(config), strict=True))
    high, low, close, quote = series.high, series.low, series.close, series.quote
    covered = (cum_uncovered[1:] - cum_uncovered[:-1]) == 0
    last = None
    out = set()
    for t in range(begin, end):
        s = t - window
        first = s + 1 - PRE
        if first < 1 or cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:
            continue
        pre = float(quote[first : s + 1].sum())
        q = float(quote[s + 1 : t + 1].sum())
        change = (close[t] / close[s] - 1) * 100
        jump = float(minute_change[s + 1 : t + 1].max())
        if pre > GATE or q < MIN_QUOTE or change < MIN_CHANGE or jump > MAX_JUMP:
            continue
        ranges = [(max(high[k], close[k - 1]) - min(low[k], close[k - 1])) / close[k - 1] * 100
                  for k in range(s - CALM + 1, s + 1)]
        vol = sum(ranges) / CALM
        path = sum(abs(close[k] - close[k - 1]) for k in range(s + 1, t + 1) if covered[k] and covered[k - 1])
        er = abs(close[t] - close[s]) / path if path > 0 else 0.0
        ratio = q / (pre / PRE * window) if pre > 0 else math.inf
        half = (window + 1) // 2
        late = float(quote[t + 1 - half : t + 1].sum()) / q
        hi = float(high[s + 1 : t + 1].max())
        retrace = (hi - close[t]) / (hi - close[s]) if hi > close[s] else 0.0
        active = int(np.sum(quote[s + 1 : t + 1] > 0))
        if (change < p["change"] or q < p["quote"] or active < p["active"] or jump > p["jump"] or er < p["er"]
                or ratio < p["ratio"] or late < p["late"] or vol > p["vol"] or retrace > p["retrace"]):
            continue
        if last is not None and t - last < COOLDOWN:
            continue
        last = t
        out.add(t)
    return out


def cmd_verify() -> None:
    contracts = load_contracts()
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    cases = [
        {"W": 60, "change": 6.0, "quote": 1e4},
        {"W": 30, "change": 4.0, "quote": 5e3, "active": 0.8, "er": 0.3, "late": 0.5, "vol": 0.3},
        {"W": 90, "change": 8.0, "quote": 2e4, "jump": 3.0, "ratio": 5.0, "retrace": 0.3},
        {"W": 120, "change": 12.0, "quote": 1e4, "active": 0.5, "er": 0.2, "vol": 0.5},
    ]
    span_minutes = 30 * 1440
    mismatches = 0
    total = 0
    for config in cases:
        rec_all = load(config["W"])
        busiest = np.argsort(-np.bincount(rec_all["sym"][alerts_for(rec_all, config)], minlength=len(symbols)))[:3]
        for sym_id in busiest:
            symbol = symbols[sym_id]
            series, _, _, cum_uncovered, _, minute_change = _features(symbol, _multiplier(contracts[symbol]))
            offset = (series.t0 - EPOCH) // 60
            keep = rec_all["sym"] == sym_id
            sub = {k: v[keep] for k, v in rec_all.items() if k != "seg"}
            sub["seg"] = np.array([0, int(keep.sum())], np.int64)
            got_all = sub["t"][alerts_for(sub, config)] - offset
            begin = max(int(np.median(got_all)) - span_minutes // 2, PRE + config["W"] + 1, -offset)
            # 网格从合约第一条候选开始走冷却：区间开头若还在网格上一条提醒的冷却里，
            # 往后挪到冷却结束，两边起点状态才相同。
            before = got_all[got_all < begin]
            if len(before) and before[-1] + COOLDOWN > begin:
                begin = int(before[-1]) + COOLDOWN
            end = min(begin + span_minutes, len(series))
            expected = _reference(series, cum_uncovered, minute_change, begin, end, config)
            got = {int(x) for x in got_all if begin <= x < end}
            diff = expected ^ got
            mismatches += len(diff)
            total += len(expected)
            print(f"W{config['W']} {symbol}: 直接实现 {len(expected)} 条，网格 {len(got)} 条，不一致 {len(diff)}",
                  flush=True)
    print(f"共 {total} 条，" + ("对拍通过" if mismatches == 0 else f"对拍不一致 {mismatches} 条"))


# ---------------------------------------------------------------- 候选族、选参与检验（dormant_slow.md 第二节）

GRID = {  # dormant_slow.md 第二节第 3 小节，探索之后写定
    "change": (6.0, 8.0, 10.0, 12.0, 15.0),
    "quote": (5e3, 1e4, 2e4, 5e4),
    "active": (0.5, 0.8),  # 占窗口的比例
    "jump": (2.0, 3.0, 5.0),
    "er": (-np.inf, 0.3, 0.5),
    "ratio": (-np.inf, 3.0, 10.0),
    "late": (-np.inf, 0.5),
    "vol": (0.2, 0.5, np.inf),
    "retrace": (0.3, np.inf),
}
P_SLACK = 0.01  # 准确率允许比现行低的幅度（约一个标准误）
NONINFERIOR = 0.02  # 样本外非劣检验的界值
GAIN_MIN = 0.10  # 样本外早报分至少比现行多一成
GAIN_FLOOR = 0.05  # 自助法下早报分增量低于现行 5% 的概率要 < 0.05


def configs(window: int) -> list[dict]:
    return [dict(zip(("W", *GRID), (window, *combo), strict=True)) for combo in itertools.product(*GRID.values())]


def cmd_search() -> None:
    n_weeks = _data_weeks()
    for window in WINDOWS:
        started = time.time()
        cfgs = configs(window)
        rec = load(window)
        params = np.array([row_of(c) for c in cfgs])
        counts = run_weekly(rec, params, n_weeks)
        np.savez(OUT / f"grid_W{window}.npz", configs=np.array([[c[k] for k in ("W", *GRID)] for c in cfgs]),
                 counts=counts)
        print(f"W{window}: {len(cfgs)} 组、{len(rec['t'])} 条候选 {time.time() - started:.0f}s", flush=True)


def _load_grid() -> tuple[np.ndarray, np.ndarray]:
    rows, counts = [], []
    for window in WINDOWS:
        with np.load(OUT / f"grid_W{window}.npz") as data:
            rows.append(data["configs"])
            counts.append(data["counts"])
    return np.concatenate(rows), np.concatenate(counts)


def _describe(row: np.ndarray) -> dict:
    return {k: (int(v) if k == "W" else float(v)) for k, v in zip(("W", *GRID), row, strict=True)}


def eligible_of(s: dict) -> np.ndarray:
    return (s["precision"] >= s["b3_precision"] - P_SLACK) & (s["per_day"] <= s["b3_per_day"])


def _pick(s: dict) -> int:
    return int(np.argmax(np.where(eligible_of(s), s["gain"], -np.inf)))


def _row(s: dict, i: int) -> dict:
    return {k: float(np.asarray(v)[i]) if np.ndim(v) else float(v) for k, v in s.items()}


def cmd_select() -> None:
    rows, counts = _load_grid()
    dev = counts[:, :, :DEV_WEEKS].astype(np.float64)
    base = b3_weekly(DEV_WEEKS)
    events = event_counts(0, DEV_WEEKS)
    s = scores(dev.sum(2), base.sum(1), events.sum(1), DEV_WEEKS * 7)
    ok = eligible_of(s)
    best = _pick(s)
    summary: dict = {"trials": len(rows), "eligible": int(ok.sum()), "chosen": _describe(rows[best]),
                     "chosen_dev": _row(s, best)}
    edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
    block = np.add.reduceat(dev, edges[:-1], axis=2)
    base_block = np.add.reduceat(base, edges[:-1], axis=1)
    ev_block = np.add.reduceat(events, edges[:-1], axis=1)
    lengths = np.diff(edges) * 7
    logits = []
    picks = []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        mask = np.zeros(CSCV_BLOCKS, bool)
        mask[list(combo)] = True
        part = {side: scores(block[:, :, m].sum(2), base_block[:, m].sum(1), ev_block[:, m].sum(1),
                             float(lengths[m].sum())) for side, m in (("is", mask), ("oos", ~mask))}
        pick = _pick(part["is"])
        picks.append(pick)
        oos = part["oos"]
        perf = np.where(eligible_of(oos), oos["gain"], -np.inf)  # 不满足限制的组排最后，与选法口径一致
        below = np.sum(perf < perf[pick]) + 0.5 * (np.sum(perf == perf[pick]) - 1)
        omega = (below + 1) / (len(perf) + 1)
        logits.append(math.log(omega / (1 - omega)))
    summary["pbo"] = float(np.mean(np.array(logits) <= 0))
    picked = np.bincount(np.array(picks), minlength=len(rows))
    summary["cscv_top_picks"] = [{**_describe(rows[i]), "share": float(picked[i] / len(picks))}
                                 for i in np.argsort(-picked)[:5] if picked[i]]
    order = np.argsort(-np.where(ok, s["gain"], -np.inf))
    summary["top10"] = [{**_describe(rows[i]), **_row(s, int(i))} for i in order[:10] if ok[i]]
    summary["best_by_window"] = {}
    for window in WINDOWS:
        inside = ok & (rows[:, 0] == window)
        if inside.any():
            i = int(np.flatnonzero(inside)[np.argmax(s["gain"][inside])])
            summary["best_by_window"][window] = {**_describe(rows[i]), **_row(s, i)}
    print(json.dumps({k: v for k, v in summary.items() if k != "top10"}, ensure_ascii=False, indent=1), flush=True)
    (OUT / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_evaluate() -> None:
    chosen = json.loads((OUT / "select.json").read_text(encoding="utf-8"))["chosen"]
    total = _data_weeks()
    n_weeks = total - DEV_WEEKS
    days = n_weeks * 7
    rec = load(chosen["W"])
    weekly = run_weekly(rec, row_of(chosen)[None], total)[0][:, DEV_WEEKS:].astype(np.float64)
    base = b3_weekly(total)[:, DEV_WEEKS:]
    events = event_counts(DEV_WEEKS, n_weeks)
    s = scores(weekly.sum(1), base.sum(1), events.sum(1), days)
    alert = alerts_for(rec, chosen) & (rec["week"] >= DEV_WEEKS)
    valid = rec["valid"][alert]
    valid = valid[valid[:, 0] >= 0]
    b3 = load_b3()
    b3_valid = b3["valid"][(b3["week"] >= DEV_WEEKS) & (b3["valid"][:, 0] >= 0)]
    by_setting = [float(v) for v in valid.mean(0)]
    b3_setting = [float(v) for v in b3_valid.mean(0)]
    weights = _bootstrap_weights(n_weeks, BOOTSTRAP, seed=2610).astype(np.float64)
    boot = scores(np.einsum("bw,cw->bc", weights, weekly), np.einsum("bw,cw->bc", weights, base),
                  np.einsum("bw,cw->bc", weights, events), weights.sum(1) * 7)
    p_inferior = float(np.mean(boot["precision"] - boot["b3_precision"] < -NONINFERIOR))
    p_gain = float(np.mean(boot["gain"] < GAIN_FLOOR))
    result = {"chosen": chosen, "oos_weeks": n_weeks, "events": events.sum(1).tolist(),
              "oos": {k: float(v) for k, v in s.items()}, "by_setting": by_setting, "b3_by_setting": b3_setting,
              "bootstrap_p_inferior": p_inferior, "bootstrap_p_gain_below_floor": p_gain}
    result["checks"] = {
        "precision": bool(s["precision"] >= s["b3_precision"] - P_SLACK),
        "settings_ok": int(sum(a >= b - P_SLACK for a, b in zip(by_setting, b3_setting, strict=True))),
        "noninferior": p_inferior < 0.05,
        "gain": bool(s["gain"] >= GAIN_MIN and p_gain < 0.05),
        "per_day": bool(0.5 <= s["per_day"] <= s["b3_per_day"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = {"extract": cmd_extract, "explore": cmd_explore, "verify": cmd_verify, "search": cmd_search,
                "select": cmd_select, "evaluate": cmd_evaluate}
    parser.add_argument("command", choices=tuple(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
