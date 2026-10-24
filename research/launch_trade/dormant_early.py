"""冷门启动：提醒更早、假启动更少，口径与判据见 dormant_early.md（跑数前写定）。

冷门、冷门启动行情、「准」与 dormant_solo / dormant_calm 相同（直接沿用其代码）；本脚本多抽几项形态特征，
并记下每个候选分钟落在哪一段冷门启动行情、此时已走完这一段涨幅的多少，用来衡量提醒早不早。

用法（仓库根目录，$P 见 README.md；先跑过 dormant_solo.py extract）：
  & $P research/launch_trade/dormant_early.py extract    # 候选分钟与特征
  & $P research/launch_trade/dormant_early.py verify     # 网格判定与逐分钟直接实现对拍
  & $P research/launch_trade/dormant_early.py search     # 开发期全量网格，逐周计数
  & $P research/launch_trade/dormant_early.py select     # 基准、开发期选参、PBO
  & $P research/launch_trade/dormant_early.py select3    # 第三轮：假启动更少优先
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
from dormant_solo import PRE, _dormant_legs
from launch_alerts import (
    CSCV_BLOCKS,
    DEV_WEEKS,
    FORWARD,
    GAINS,
    HOLDS,
    TARGETS,
    WEEK,
    _validity,
)
from launch_grid import EPOCH, _multiplier, _rolling_extreme
from load_data import load_contracts
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "dormant_early"
GATE = 1e6  # 冷门：窗口之前 24 小时成交额上限，与现行相同，不在候选族里
CALM = 240  # 涨前波动、涨前区间的统计长度，与现行相同
COOLDOWN = 60
STEP = 3.0  # 冷却内再报「第 N 波」的步长，与现行相同
WINDOWS = (3, 5, 10, 15)
MIN_QUOTE = 3e3  # 抽取的下限，比候选族最松的一档还松
MIN_CHANGE = 3.0


# ---------------------------------------------------------------- 抽取


@njit(cache=True)
def _extract(open_, high, low, close, quote, cum_quote, cum_active, cum_uncovered, cum_tr, pre_high, day_high,
             high3d, window):
    n = len(close)
    keep = np.zeros(n, np.bool_)
    for t in range(window + PRE, n):
        first = t - window + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:  # 含窗口前 24 小时第一根的前收
            continue
        if cum_quote[t - window + 1] - cum_quote[first] > GATE:
            continue
        if cum_quote[t + 1] - cum_quote[t - window + 1] < MIN_QUOTE:
            continue
        if (close[t] / close[t - window] - 1) * 100 < MIN_CHANGE:
            continue
        keep[t] = True
    index = np.flatnonzero(keep)
    m = len(index)
    f = np.empty((m, 21))
    for k in range(m):
        t = index[k]
        s = t - window  # 窗口开始前一根
        change = (close[t] / close[s] - 1) * 100
        q = cum_quote[t + 1] - cum_quote[s + 1]
        vol = (cum_tr[s + 1] - cum_tr[s + 1 - CALM]) / CALM
        jump = -np.inf
        up = 0
        body = 0.0
        span = 0.0
        hi = close[s]
        for j in range(s + 1, t + 1):
            c = (close[j] / close[j - 1] - 1) * 100
            jump = max(jump, c)
            up += close[j] > close[j - 1]
            body += abs(close[j] - open_[j])
            span += high[j] - low[j]
            hi = max(hi, high[j])
        half = (window + 1) // 2
        late = cum_quote[t + 1] - cum_quote[t + 1 - half]
        f[k, 0] = change
        f[k, 1] = q
        f[k, 2] = cum_active[t + 1] - cum_active[s + 1]
        f[k, 3] = cum_quote[s + 1] - cum_quote[s + 1 - PRE]  # 之前 24 小时成交额
        f[k, 4] = jump
        f[k, 5] = body / span if span > 0 else 0.0
        f[k, 6] = vol
        f[k, 7] = change / vol if vol > 0 else np.inf
        f[k, 8] = up
        f[k, 9] = (hi - close[t]) / (hi - close[s]) if hi > close[s] else 0.0  # 从窗口最高价的回吐比例
        f[k, 10] = (close[t] / pre_high[s] - 1) * 100  # 收盘超过涨前 240 分钟最高价的幅度
        f[k, 11] = late / q if q > 0 else 0.0  # 后半窗口成交额占比
        f[k, 12] = quote[t]  # 最后一分钟成交额
        # 经验最小报价单位：之前 24 小时相邻收盘价的最小正差值，相对价格（%）。跳一档就是 1% 的币噪声大。
        tick = np.inf
        for j in range(s - PRE + 2, s + 1):
            d = abs(close[j] - close[j - 1])
            if d > 0:
                tick = min(tick, d)
        f[k, 13] = tick / close[s] * 100 if np.isfinite(tick) else np.nan
        f[k, 14] = (close[t] / close[t - 1] - 1) * 100  # 最后一分钟涨幅
        h = window // 2
        f[k, 15] = ((close[t] / close[t - h] - 1) - (close[t - h] / close[s] - 1)) * 100 if h > 0 else 0.0  # 加速
        f[k, 16] = (close[s] / close[s - PRE + 1] - 1) * 100  # 涨前 24 小时涨跌
        f[k, 17] = (close[t] / day_high[s] - 1) * 100  # 收盘超过涨前 24 小时最高价的幅度
        f[k, 18] = (close[s] / high3d[s] - 1) * 100  # 涨前距 3 天最高价
        week_quote = cum_quote[s + 1] - cum_quote[max(0, s + 1 - 7 * PRE)]
        f[k, 19] = f[k, 3] / (week_quote / 7) if week_quote > 0 else np.nan  # 之前 24 小时成交额 ÷ 7 天日均
        f[k, 20] = 0.0  # 上市天数，抽取后按合约创建时间填
    return index, f


FEATURES = ("change", "quote", "active", "pre", "jump", "body", "vol", "multiple", "up", "retrace", "breakout",
            "late", "last_quote", "tick", "ret_last", "accel", "chg24", "brk24", "dd3d", "quote7", "age")


@njit(cache=True)
def _leg_position(close, index, starts, ends):
    """候选落在哪一段冷门启动行情（开区间起点、闭区间终点）及此时已走完的对数涨幅比例；不在任何一段为 -1。"""
    m = len(index)
    leg = np.full(m, -1, np.int64)
    frac = np.full(m, np.nan)
    for k in range(m):
        t = index[k]
        lo = 0
        hi = len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j < 0 or t > ends[j]:
            continue
        leg[k] = j
        frac[k] = math.log(close[t] / close[starts[j]]) / math.log(close[ends[j]] / close[starts[j]])
    return leg, frac


def _extract_symbol(args: tuple[int, str, float, int]) -> tuple[int, dict]:
    sym_id, symbol, multiplier, created = args
    out: dict = {}
    prepared = _features(symbol, multiplier)
    if prepared is None:
        return sym_id, out
    series, cum_quote, cum_active, cum_uncovered, cum_tr, _ = prepared
    open_, high, low, close = series.open, series.high, series.low, series.close
    offset = (series.t0 - EPOCH) // 60
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    legs = {gain: _dormant_legs(close, cum_quote, cum_uncovered, gain) for gain in GAINS}
    for gain, (starts, _) in legs.items():
        out[f"events{int(gain * 100)}"] = ((starts + offset) * 60 // WEEK).astype(np.int64)
    pre_high = _rolling_extreme(high, CALM, True)
    day_high = _rolling_extreme(high, PRE, True)
    high3d = _rolling_extreme(np.nan_to_num(high, nan=0.0), 3 * PRE, True)
    for window in WINDOWS:
        index, f = _extract(open_, high, low, close, series.quote, cum_quote, cum_active, cum_uncovered, cum_tr,
                            pre_high, day_high, high3d, window)
        f[:, 20] = (series.t0 + index * 60 + 60 - created) / 86400
        ones = np.ones(len(index), np.int8)
        record = {"t": (index + offset).astype(np.int64), "price": close[index], "f": f.astype(np.float32),
                  "valid": _validity(high, low, close, cum_uncovered, index, ones, targets, holds, FORWARD)}
        for gain in GAINS:
            starts, ends = legs[gain]
            leg, frac = _leg_position(close, index, starts, ends)
            tag = int(gain * 100)
            week = np.full(len(index), -1, np.int64)
            hit = leg >= 0
            week[hit] = (starts[leg[hit]] + offset) * 60 // WEEK
            record[f"leg{tag}"] = leg
            record[f"frac{tag}"] = frac.astype(np.float32)
            record[f"leg_week{tag}"] = week
        out[window] = record
    return sym_id, out


def cmd_extract() -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = [(i, s, _multiplier(contracts[s]), int(contracts[s].get("create_time") or 0))
            for i, s in enumerate(symbols) if _multiplier(contracts[s])]
    parts: dict = {w: [] for w in WINDOWS}
    events: dict = {g: [] for g in GAINS}
    started = time.time()
    with ProcessPoolExecutor(max_workers=14) as executor:
        for done, (sym_id, result) in enumerate(executor.map(_extract_symbol, jobs, chunksize=2), 1):
            if result:
                for window in WINDOWS:
                    fields = result[window]
                    fields["sym"] = np.full(len(fields["t"]), sym_id, np.int32)
                    parts[window].append(fields)
                for gain in GAINS:
                    events[gain].append(result[f"events{int(gain * 100)}"])
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    # 冷门启动行情与 dormant_solo 同一定义，按本次加载的数据重算（数据可能比 dormant_solo 抽取时新）。
    np.savez(OUT / "events.npz", **{f"up_{int(g * 100)}": np.concatenate(v) for g, v in events.items()})
    for window, chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        np.savez(OUT / f"cand_W{window}.npz", **merged)
        print(f"W{window}: {len(merged['t'])} 条候选", flush=True)


def load(window: int) -> dict:
    with np.load(OUT / f"cand_W{window}.npz") as data:
        rec = {name: data[name] for name in data.files}
    for i, name in enumerate(FEATURES):
        rec[name] = rec["f"][:, i].astype(np.float64)
    edges = np.flatnonzero(np.diff(rec["sym"])) + 1
    rec["seg"] = np.concatenate([[0], edges, [len(rec["sym"])]]).astype(np.int64)
    rec["week"] = ((rec["t"] * 60 + 60) // WEEK).astype(np.int64)
    valid = rec["valid"].astype(np.float32)
    valid[valid < 0] = np.nan
    rec["precision"] = valid.mean(1)
    return rec


# ---------------------------------------------------------------- 候选族与判定

CHANGES = (4.0, 5.0, 6.0, 7.0, 8.0)
QUOTES = (5e3, 1e4, 2e4)
ACTIVE_SHARES = (0.5, 0.8, 1.0)
LAST_QUOTES = (0.0, 1e3, 3e3)
UP_SHARES = (0.0, 0.6)
BREAKOUTS = (0.0, 1.0)
LATES = (0.0, 0.4)
PARAM_NAMES = ("change", "quote", "active", "last_quote", "up", "breakout", "late")
COUNTS = ("alerts", "scored", "correct", "credit10", "credit20", "caught10", "caught20")
BASELINE = {"W": 10, "change": 8.0, "quote": 5e3, "active": 5.0, "last_quote": 0.0, "up": 0.0, "breakout": 0.0,
            "late": 0.0}


def _need(window: int, share: float) -> int:
    return max(1, math.ceil(window * share - 1e-9)) if share > 0 else 0


def param_grid(window: int) -> np.ndarray:
    rows = [(c, q, _need(window, a), lq, _need(window, u), b, la)
            for c, q, a, lq, u, b, la in itertools.product(
                CHANGES, QUOTES, ACTIVE_SHARES, LAST_QUOTES, UP_SHARES, BREAKOUTS, LATES)]
    return np.array(rows, dtype=np.float64)


def shape_mask(rec: dict) -> np.ndarray:
    """用户指定、不在候选族里的形态条件（现行默认值），以及族里最松的涨幅与成交额。"""
    return ((rec["jump"] <= 5.0) & (rec["body"] >= 0.6) & (rec["multiple"] >= 10.0) & (rec["vol"] <= 0.5)
            & (rec["change"] >= CHANGES[0]) & (rec["quote"] >= QUOTES[0]))


def load_shaped(window: int) -> dict:
    rec = load(window)
    keep = shape_mask(rec)
    out = {k: v[keep] for k, v in rec.items() if k not in ("seg", "f")}
    edges = np.flatnonzero(np.diff(out["sym"])) + 1
    out["seg"] = np.concatenate([[0], edges, [len(out["sym"])]]).astype(np.int64)
    return out


@njit(cache=True)
def _passes(i, p, change, quote, active, last_quote, up, breakout, late):
    return not (change[i] < p[0] or quote[i] < p[1] or active[i] < p[2] or last_quote[i] < p[3] or up[i] < p[4]
                or (p[5] > 0 and breakout[i] <= 0) or late[i] < p[6])


@njit(parallel=True, cache=True)
def _grid(seg, t, week, price, change, quote, active, last_quote, up, breakout, late, precision, leg10, leg20,
          frac10, frac20, leg_week10, leg_week20, params, n_weeks, alerts, scored, correct, credit10, credit20,
          caught10, caught20):
    n_params = params.shape[0]
    block = 32
    for b in prange((n_params + block - 1) // block):
        c0 = b * block
        c1 = min(n_params, c0 + block)
        last_t = np.empty(block, np.int64)
        last_p = np.empty(block)
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
                    p = params[c]
                    if not _passes(i, p, change, quote, active, last_quote, up, breakout, late):
                        continue
                    k = c - c0
                    # 冷却内只有比上次提醒价再涨 STEP 才报下一波，与 DormantDetector 相同。
                    if last_t[k] >= 0 and t[i] - last_t[k] < COOLDOWN and price[i] < last_p[k] * (1 + STEP / 100):
                        continue
                    last_t[k] = t[i]
                    last_p[k] = price[i]
                    alerts[c, w] += 1
                    if not np.isnan(precision[i]):
                        scored[c, w] += 1
                        correct[c, w] += precision[i]
                    if leg10[i] >= 0 and leg10[i] != last_l10[k]:
                        last_l10[k] = leg10[i]
                        credit10[c, leg_week10[i]] += 1 - frac10[i]
                        caught10[c, leg_week10[i]] += frac10[i] <= 0.5
                    if leg20[i] >= 0 and leg20[i] != last_l20[k]:
                        last_l20[k] = leg20[i]
                        credit20[c, leg_week20[i]] += 1 - frac20[i]
                        caught20[c, leg_week20[i]] += frac20[i] <= 0.5


def _grid_args(rec: dict) -> tuple:
    return (rec["seg"], rec["t"], rec["week"], rec["price"], rec["change"], rec["quote"], rec["active"],
            rec["last_quote"], rec["up"], rec["breakout"], rec["late"], rec["precision"], rec["leg10"], rec["leg20"],
            rec["frac10"].astype(np.float64), rec["frac20"].astype(np.float64), rec["leg_week10"], rec["leg_week20"])


def run_weekly(rec: dict, params: np.ndarray, n_weeks: int) -> dict:
    """各组逐周计数：提醒、计分、准（4 种设置平均之和）按提醒所在周，早报分与前半段报到按行情起点所在周。"""
    shape = (len(params), n_weeks)
    out = {name: np.zeros(shape, np.int32 if name in ("alerts", "scored", "caught10", "caught20") else np.float64)
           for name in COUNTS}
    _grid(*_grid_args(rec), params, n_weeks, *(out[name] for name in COUNTS))
    return out


def cmd_search() -> None:
    for window in WINDOWS:
        started = time.time()
        params = param_grid(window)
        rec = load_shaped(window)
        counts = run_weekly(rec, params, DEV_WEEKS)
        np.savez(OUT / f"dev_W{window}.npz", params=params, **counts)
        print(f"W{window}: {len(rec['t'])} 条记录 {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------- 指标、选参与检验


def event_counts(first_week: int, n_weeks: int) -> np.ndarray:
    with np.load(OUT / "events.npz") as data:
        rows = []
        for gain in GAINS:
            weeks = data[f"up_{int(gain * 100)}"] - first_week
            rows.append(np.bincount(weeks[(weeks >= 0) & (weeks < n_weeks)], minlength=n_weeks))
    return np.array(rows, np.float64)


def scores(totals: dict, events: np.ndarray, days) -> dict:
    """totals 为各计数在时间上求和后的数组（可带前导维度），events 为两种行情的段数。"""
    scored = np.asarray(totals["scored"], np.float64)
    precision = np.divide(totals["correct"], scored, out=np.zeros(scored.shape), where=scored > 0)
    per_day = np.asarray(totals["alerts"], np.float64) / days
    return {"precision": precision, "per_day": per_day, "fpd": (1 - precision) * per_day,
            "re": 0.5 * (totals["credit10"] / events[0] + totals["credit20"] / events[1]),
            "recall": 0.5 * (totals["caught10"] / events[0] + totals["caught20"] / events[1])}


def objective(s: dict, base: dict) -> np.ndarray:
    return 2 * (1 - s["fpd"] / base["fpd"]) + (s["re"] / base["re"] - 1)


def eligible_of(s: dict, base: dict) -> np.ndarray:
    return (s["precision"] >= base["precision"]) & (s["fpd"] <= base["fpd"]) & (s["re"] >= base["re"])


def _load_dev() -> dict:
    parts = []
    for window in WINDOWS:
        with np.load(OUT / f"dev_W{window}.npz") as data:
            part = {name: data[name] for name in data.files}
        part["window"] = np.full(len(part["params"]), window)
        parts.append(part)
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def _describe(dev: dict, i: int) -> dict:
    return {"W": int(dev["window"][i]), **dict(zip(PARAM_NAMES, (float(v) for v in dev["params"][i]), strict=True))}


def _baseline_index(dev: dict) -> int:
    row = np.array([BASELINE[name] for name in PARAM_NAMES])
    hit = np.flatnonzero((dev["window"] == BASELINE["W"]) & np.all(dev["params"] == row, axis=1))
    assert len(hit) == 1, hit
    return int(hit[0])


def _pick(s: dict, base_index: int) -> int:
    base = {k: v[base_index] for k, v in s.items()}
    j = np.where(eligible_of(s, base), objective(s, base), -np.inf)
    return int(np.argmax(j))


def _row_dict(s: dict, i: int) -> dict:
    return {k: float(v[i]) for k, v in s.items()}


def cmd_select() -> None:
    dev = _load_dev()
    events = event_counts(0, DEV_WEEKS)
    days = DEV_WEEKS * 7
    b = _baseline_index(dev)
    s = scores({name: dev[name].sum(1, dtype=np.float64) for name in COUNTS}, events.sum(1), days)
    base = {k: v[b] for k, v in s.items()}
    ok = eligible_of(s, base)
    j = objective(s, base)
    best = _pick(s, b)
    summary: dict = {"trials": len(j), "eligible": int(ok.sum()), "baseline_dev": _row_dict(s, b),
                     "chosen": _describe(dev, best), "chosen_dev": {**_row_dict(s, best), "J": float(j[best])}}
    # CSCV：IS 内同样以 IS 的基准判断三项不变差、取 J 最高；OOS 按 J（以 OOS 的基准）排名。
    edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
    block = {name: np.add.reduceat(dev[name].astype(np.float64), edges[:-1], axis=1) for name in COUNTS}
    ev_block = np.add.reduceat(events, edges[:-1], axis=1)
    lengths = np.diff(edges) * 7
    logits = []
    picks = []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        mask = np.zeros(CSCV_BLOCKS, bool)
        mask[list(combo)] = True
        part = {}
        for side, m in (("is", mask), ("oos", ~mask)):
            part[side] = scores({name: block[name][:, m].sum(1) for name in COUNTS}, ev_block[:, m].sum(1),
                                float(lengths[m].sum()))
        pick = _pick(part["is"], b)
        picks.append(pick)
        oos_base = {k: v[b] for k, v in part["oos"].items()}
        oos_j = objective(part["oos"], oos_base)
        below = np.sum(oos_j < oos_j[pick]) + 0.5 * (np.sum(oos_j == oos_j[pick]) - 1)
        omega = (below + 1) / (len(oos_j) + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    summary["pbo"] = float(np.mean(logits <= 0))
    summary["cscv_baseline_picked"] = float(np.mean(np.array(picks) == b))
    order = np.argsort(-np.where(ok, j, -np.inf))
    summary["top20"] = [{**_describe(dev, int(i)), **_row_dict(s, int(i)), "J": float(j[i])}
                        for i in order[:20] if ok[i]]
    # 每个窗口各自的最优，只供解读。
    summary["best_by_window"] = {}
    for window in WINDOWS:
        inside = ok & (dev["window"] == window)
        if inside.any():
            i = int(np.flatnonzero(inside)[np.argmax(j[inside])])
            summary["best_by_window"][window] = {**_describe(dev, i), **_row_dict(s, i), "J": float(j[i])}
    print(json.dumps({k: v for k, v in summary.items() if k != "top20"}, ensure_ascii=False, indent=1), flush=True)
    (OUT / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _data_weeks() -> int:
    last = 0
    for window in WINDOWS:
        with np.load(OUT / f"cand_W{window}.npz") as data:
            last = max(last, int(data["t"].max()))
    return (last * 60 + 60) // WEEK + 1


@njit(cache=True)
def _alert_mask(seg, t, price, change, quote, active, last_quote, up, breakout, late, p):
    alert = np.zeros(len(t), np.bool_)
    for s in range(len(seg) - 1):
        last_t = -1
        last_p = 0.0
        for i in range(seg[s], seg[s + 1]):
            if not _passes(i, p, change, quote, active, last_quote, up, breakout, late):
                continue
            if last_t >= 0 and t[i] - last_t < COOLDOWN and price[i] < last_p * (1 + STEP / 100):
                continue
            last_t = t[i]
            last_p = price[i]
            alert[i] = True
    return alert


def alerts_for(rec: dict, config: dict) -> np.ndarray:
    row = np.array([config[name] for name in PARAM_NAMES], dtype=np.float64)
    return _alert_mask(rec["seg"], rec["t"], rec["price"], rec["change"], rec["quote"], rec["active"],
                       rec["last_quote"], rec["up"], rec["breakout"], rec["late"], row)


# ---------------------------------------------------------------- 第三轮：假启动更少优先、允许稍晚

RE_FLOOR = 0.9  # 早报分最多比基准低一成


def eligible3(s: dict, base: dict) -> np.ndarray:
    return (s["re"] >= RE_FLOOR * base["re"]) & (s["fpd"] <= base["fpd"])


def _pick3(s: dict, base_index: int) -> int:
    base = {k: v[base_index] for k, v in s.items()}
    return int(np.argmax(np.where(eligible3(s, base), s["precision"], -np.inf)))


def cmd_select3() -> None:
    dev = _load_dev()
    events = event_counts(0, DEV_WEEKS)
    b = _baseline_index(dev)
    s = scores({name: dev[name].sum(1, dtype=np.float64) for name in COUNTS}, events.sum(1), DEV_WEEKS * 7)
    best = _pick3(s, b)
    edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
    block = {name: np.add.reduceat(dev[name].astype(np.float64), edges[:-1], axis=1) for name in COUNTS}
    ev_block = np.add.reduceat(events, edges[:-1], axis=1)
    lengths = np.diff(edges) * 7
    logits = []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        mask = np.zeros(CSCV_BLOCKS, bool)
        mask[list(combo)] = True
        part = {side: scores({name: block[name][:, m].sum(1) for name in COUNTS}, ev_block[:, m].sum(1),
                             float(lengths[m].sum())) for side, m in (("is", mask), ("oos", ~mask))}
        pick = _pick3(part["is"], b)
        oos = part["oos"]
        # 不满足两项限制的组排最后，与选法口径一致。
        perf = np.where(eligible3(oos, {k: v[b] for k, v in oos.items()}), oos["precision"], -np.inf)
        below = np.sum(perf < perf[pick]) + 0.5 * (np.sum(perf == perf[pick]) - 1)
        omega = (below + 1) / (len(perf) + 1)
        logits.append(math.log(omega / (1 - omega)))
    summary = {"eligible": int(eligible3(s, {k: v[b] for k, v in s.items()}).sum()), "baseline_dev": _row_dict(s, b),
               "chosen": _describe(dev, best), "chosen_dev": _row_dict(s, best),
               "pbo": float(np.mean(np.array(logits) <= 0))}
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    (OUT / "select3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 对拍


def _reference(series, cum_uncovered, begin: int, end: int, config: dict) -> set[int]:
    """逐分钟直接按 dormant_early.md 第二节计算的提醒时刻（K 线下标），不经过抽取与网格。"""
    window = config["W"]
    open_, high, low, close, quote = series.open, series.high, series.low, series.close, series.quote
    last = None
    last_price = 0.0
    out = set()
    for t in range(begin, end):
        first = t - window + 1 - PRE
        if first < 1 or cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:
            continue
        s = t - window
        win = range(s + 1, t + 1)
        pre_q = float(quote[first : s + 1].sum())
        win_q = float(quote[s + 1 : t + 1].sum())
        change = (close[t] / close[s] - 1) * 100
        ranges = [(max(high[k], close[k - 1]) - min(low[k], close[k - 1])) / close[k - 1] * 100
                  for k in range(s - CALM + 1, s + 1)]
        vol = sum(ranges) / CALM
        multiple = change / vol if vol > 0 else math.inf
        jump = max((close[k] / close[k - 1] - 1) * 100 for k in win)
        span = sum(high[k] - low[k] for k in win)
        body = sum(abs(close[k] - open_[k]) for k in win) / span if span > 0 else 0.0
        up = sum(close[k] > close[k - 1] for k in win)
        half = (window + 1) // 2
        late = float(quote[t + 1 - half : t + 1].sum()) / win_q if win_q > 0 else 0.0
        breakout = close[t] > high[s - CALM + 1 : s + 1].max()
        if (pre_q > GATE or win_q < config["quote"] or change < config["change"]
                or int(np.sum(quote[s + 1 : t + 1] > 0)) < config["active"] or vol > 0.5 or multiple < 10
                or jump > 5 or body < 0.6 or quote[t] < config["last_quote"] or up < config["up"]
                or (config["breakout"] > 0 and not breakout) or late < config["late"]):
            continue
        if last is not None and t - last < COOLDOWN and close[t] < last_price * (1 + STEP / 100):
            continue
        last = t
        last_price = close[t]
        out.add(t)
    return out


def cmd_verify() -> None:
    contracts = load_contracts()
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    cases = [
        BASELINE,
        dict(W=5, change=5.0, quote=1e4, active=4.0, last_quote=1e3, up=3.0, breakout=1.0, late=0.4),
        dict(W=15, change=6.0, quote=5e3, active=12.0, last_quote=3e3, up=0.0, breakout=0.0, late=0.4),
        dict(W=3, change=4.0, quote=2e4, active=2.0, last_quote=0.0, up=2.0, breakout=1.0, late=0.0),
    ]
    span_minutes = 30 * 1440
    mismatches = 0
    total = 0
    for config in cases:
        rec_all = load_shaped(config["W"])
        busiest = np.argsort(-np.bincount(rec_all["sym"], minlength=len(symbols)))[:4]
        for sym_id in busiest:
            symbol = symbols[sym_id]
            series, _, _, cum_uncovered, _, _ = _features(symbol, _multiplier(contracts[symbol]))
            offset = (series.t0 - EPOCH) // 60
            mine = rec_all["t"][rec_all["sym"] == sym_id] - offset
            begin = max(int(np.median(mine)) - span_minutes // 2, PRE + config["W"] + 1)
            end = min(begin + span_minutes, len(series))
            expected = _reference(series, cum_uncovered, begin, end, config)
            keep = (rec_all["sym"] == sym_id) & (rec_all["t"] - offset >= begin) & (rec_all["t"] - offset < end)
            sub = {k: v[keep] for k, v in rec_all.items() if k != "seg"}
            sub["seg"] = np.array([0, int(keep.sum())], np.int64)
            got = {int(t - offset) for t in sub["t"][alerts_for(sub, config)]}
            diff = expected ^ got
            mismatches += len(diff)
            total += len(expected)
            print(f"W{config['W']} {symbol}: 直接实现 {len(expected)} 条，网格 {len(got)} 条，不一致 {len(diff)}",
                  flush=True)
    print(f"共 {total} 条，" + ("对拍通过" if mismatches == 0 else f"对拍不一致 {mismatches} 条"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "verify", "search", "select", "select3"))
    {"extract": cmd_extract, "verify": cmd_verify, "search": cmd_search, "select": cmd_select,
     "select3": cmd_select3}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
