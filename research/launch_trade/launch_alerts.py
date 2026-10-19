"""第二轮：按提醒本身的好坏（准确率 × 大行情前半段召回率的 F1）搜索放量上涨 / 下跌参数，口径见 launch_grid.md 第三节。

候选族、判定逻辑与第一轮相同（复用 launch_grid 的特征抽取与冷却、再推进实现，已与 LaunchDetector 对拍）。

用法（仓库根目录）：
  & $P research/launch_trade/launch_alerts.py extract    # 候选分钟 + 准不准 + 落在哪段大行情的前半段
  & $P research/launch_trade/launch_alerts.py search     # 开发期全量网格，逐周计数
  & $P research/launch_trade/launch_alerts.py select     # 开发期选参、Reality Check、PBO
  & $P research/launch_trade/launch_alerts.py evaluate   # 选中方案与现行配置的样本外
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
from launch_grid import (
    BASELINES,
    CHANGES,
    CURRENT,
    EPOCH,
    QUOTES,
    RATIOS,
    SIGNS,
    WINDOWS,
    _extract,
    _last_jump,
    _multiplier,
    _params_row,
    _rolling_extreme,
    alerts_of,
    param_grid,
)
from load_data import load_contracts, load_series
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "launch_alerts"
WEEK = 7 * 86400
DEV_WEEKS = (1735689600 - EPOCH) // WEEK  # 2021-09-01 ~ 2025-01-01 正好 174 周
TARGETS = (0.03, 0.05)
HOLDS = (60, 240)
FORWARD = max(HOLDS)
GAINS = (0.10, 0.20)
MAX_LEG_MINUTES = 1440
MIN_LEG_QUOTE = 1e6
CSCV_BLOCKS = 16
BOOTSTRAP = 2000
BLOCK_WEEKS = 4
# F-beta 的 beta：第二轮 1（F1），第三轮 0.5（更看重准确率），由命令行 --beta 设定；变量名里的 f1 都指 F-beta。
BETA = 1.0


# ---------------------------------------------------------------- 大行情与准不准


@njit(cache=True)
def _qualifies(close, cum_quote, start, end, trend, gain):
    move = close[end] / close[start] - 1
    big = move >= gain if trend == 1 else -move >= gain
    return big and end - start <= MAX_LEG_MINUTES and cum_quote[end + 1] - cum_quote[start + 1] >= MIN_LEG_QUOTE


@njit(cache=True)
def _legs(close, cum_quote, cum_uncovered, gain):
    """之字线（反转门槛 gain/2）切出的、合格的大行情段：返回起点、终点下标与方向。"""
    n = len(close)
    rev = gain / 2
    starts = []
    ends = []
    signs = []
    a = 0
    while a < n:
        while a < n and cum_uncovered[a + 1] - cum_uncovered[a] == 1:
            a += 1
        b = a
        while b < n and cum_uncovered[b + 1] - cum_uncovered[b] == 0:
            b += 1
        # [a, b) 是一段连续有数据的分钟，之字线不跨越缺口。
        trend = 0
        hi = a
        lo = a
        start = a
        ext = a
        for t in range(a + 1, b):
            c = close[t]
            if trend == 0:
                if c > close[hi]:
                    hi = t
                if c < close[lo]:
                    lo = t
                if c >= close[lo] * (1 + rev):
                    trend, start, ext = 1, lo, t
                elif c <= close[hi] * (1 - rev):
                    trend, start, ext = -1, hi, t
            elif trend == 1:
                if c > close[ext]:
                    ext = t
                elif c <= close[ext] * (1 - rev):
                    if _qualifies(close, cum_quote, start, ext, 1, gain):
                        starts.append(start)
                        ends.append(ext)
                        signs.append(1)
                    trend, start, ext = -1, ext, t
            else:
                if c < close[ext]:
                    ext = t
                elif c >= close[ext] * (1 + rev):
                    if _qualifies(close, cum_quote, start, ext, -1, gain):
                        starts.append(start)
                        ends.append(ext)
                        signs.append(-1)
                    trend, start, ext = 1, ext, t
        # 数据到头（或遇到缺口）时还没确认结束的最后一段，按已走出的部分计。
        if trend != 0 and _qualifies(close, cum_quote, start, ext, trend, gain):
            starts.append(start)
            ends.append(ext)
            signs.append(trend)
        a = b
    return np.array(starts, np.int64), np.array(ends, np.int64), np.array(signs, np.int8)


@njit(cache=True)
def _validity(high, low, close, cum_uncovered, index, sign, targets, holds, forward):
    """每条候选在各（幅度 X, 时长 H）设置下准不准：1 准，0 不准，-1 之后没有完整数据不计。"""
    m = len(index)
    out = np.full((m, len(targets) * len(holds)), -1, np.int8)
    n = len(close)
    for k in range(m):
        t = index[k]
        if t + forward >= n or cum_uncovered[t + forward + 1] - cum_uncovered[t] != 0:
            continue
        p = close[t]
        s = sign[k]
        for xi in range(len(targets)):
            x = targets[xi]
            hit_minute = forward + 1
            good = False
            for j in range(1, forward + 1):
                if s > 0:
                    reach = high[t + j] >= p * (1 + x)
                    stop = low[t + j] <= p * (1 - x)
                else:
                    reach = low[t + j] <= p * (1 - x)
                    stop = high[t + j] >= p * (1 + x)
                if reach or stop:
                    hit_minute = j
                    # 同一分钟两边都碰到，分不清先后，按不准算。
                    good = reach and not stop
                    break
            for hi_ in range(len(holds)):
                out[k, xi * len(holds) + hi_] = 1 if (good and hit_minute <= holds[hi_]) else 0
    return out


@njit(cache=True)
def _early_event(close, index, sign, starts, ends, signs):
    """候选落在哪一段同方向大行情的前半段：返回该段在本合约中的序号，不在则 -1。"""
    m = len(index)
    out = np.full(m, -1, np.int64)
    for k in range(m):
        t = index[k]
        s = sign[k]
        # 同方向的段互不重叠且按起点有序，二分找起点 < t 的最后一段。
        lo = 0
        hi = len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        while j >= 0 and signs[j] != s:
            j -= 1
        if j < 0 or t > ends[j]:
            continue
        whole = s * math.log(close[ends[j]] / close[starts[j]])
        done = s * math.log(close[t] / close[starts[j]])
        if done <= 0.5 * whole:
            out[k] = j
    return out


def _extract_symbol(args: tuple[int, str, float]) -> tuple[int, dict, dict]:
    sym_id, symbol, multiplier = args
    series = load_series(symbol, multiplier)
    result: dict = {}
    events: dict = {}
    if series is None or len(series) < 2000:
        return sym_id, result, events
    covered = series.covered.copy()
    covered[: _last_jump(series.close)] = False
    cum_quote = np.concatenate([[0.0], np.cumsum(series.quote)])
    cum_uncovered = np.concatenate([[0], np.cumsum(~covered)]).astype(np.int64)
    offset = (series.t0 - EPOCH) // 60
    legs = {}
    for gain in GAINS:
        starts, ends, signs = _legs(series.close, cum_quote, cum_uncovered, gain)
        legs[gain] = (starts, ends, signs)
        for direction, s in SIGNS.items():
            events[(direction, gain)] = ((starts[signs == s] + offset) * 60 // WEEK).astype(np.int64)
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    win_cache = {w: (_rolling_extreme(series.high, w, True), _rolling_extreme(series.low, w, False)) for w in WINDOWS}
    for baseline in BASELINES:
        base_high = _rolling_extreme(series.high, baseline, True)
        base_low = _rolling_extreme(series.low, baseline, False)
        for window in WINDOWS:
            win_high, win_low = win_cache[window]
            fields = _extract(series.high, series.low, series.close, cum_quote, cum_uncovered, window, baseline,
                              base_high, base_low, win_high, win_low, RATIOS[0], QUOTES[0], CHANGES[0],
                              np.array([0], np.int64), np.array([1], np.int64), 1, 1.0, 0.0)
            index, sign = fields[0], fields[1]
            record = {
                "t": (index + offset).astype(np.int64),
                "price": series.close[index],
                "sign": sign,
                "change": fields[2],
                "ratio": fields[3],
                "quote": fields[4],
                "breakout": fields[5],
                "retrace": fields[6],
                "volume": fields[7],
                "valid": _validity(series.high, series.low, series.close, cum_uncovered, index, sign, targets, holds,
                                   FORWARD),
            }
            for gain in GAINS:
                starts, ends, signs = legs[gain]
                event = _early_event(series.close, index, sign, starts, ends, signs)
                tag = int(gain * 100)
                record[f"event{tag}"] = event
                week = np.full(len(index), -1, np.int64)
                hit = event >= 0
                week[hit] = (starts[event[hit]] + offset) * 60 // WEEK
                record[f"event_week{tag}"] = week
            result[(window, baseline)] = record
    return sym_id, result, events


def cmd_extract() -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = [(i, s, _multiplier(contracts[s])) for i, s in enumerate(symbols) if _multiplier(contracts[s])]
    parts: dict[tuple[int, int], list] = {(w, b): [] for w in WINDOWS for b in BASELINES}
    events: dict = {(d, g): [] for d in SIGNS for g in GAINS}
    started = time.time()
    with ProcessPoolExecutor(max_workers=14) as executor:
        for done, (sym_id, result, found) in enumerate(executor.map(_extract_symbol, jobs, chunksize=2), 1):
            for key, fields in result.items():
                fields["sym"] = np.full(len(fields["t"]), sym_id, np.int32)
                parts[key].append(fields)
            for key, weeks in found.items():
                events[key].append(weeks)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "events.npz", **{f"{d}_{int(g * 100)}": np.concatenate(v) for (d, g), v in events.items()})
    for (window, baseline), chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        np.savez(OUT / f"cand_W{window}_B{baseline}.npz", **merged)
        print(f"W{window} B{baseline}: {len(merged['t'])} 条候选", flush=True)
    for (d, g), v in events.items():
        print(f"{d} G{g:.0%}: {sum(len(x) for x in v)} 段大行情")


# ---------------------------------------------------------------- 网格（逐周计数）


def _load(window: int, baseline: int, direction: str) -> dict:
    with np.load(OUT / f"cand_W{window}_B{baseline}.npz") as data:
        mask = data["sign"] == SIGNS[direction]
        rec = {name: data[name][mask] for name in data.files}
    edges = np.flatnonzero(np.diff(rec["sym"])) + 1
    rec["seg"] = np.concatenate([[0], edges, [len(rec["sym"])]]).astype(np.int64)
    rec["week"] = ((rec["t"] * 60 + 60) // WEEK).astype(np.int64)
    valid = rec["valid"].astype(np.float32)
    valid[valid < 0] = np.nan
    rec["precision"] = valid.mean(1)  # 4 种设置的平均；没有完整后续数据为 NaN
    return rec


def event_counts(direction: str, first_week: int, n_weeks: int) -> np.ndarray:
    """逐周的大行情段数，形状 (G 种数, n_weeks)。"""
    with np.load(OUT / "events.npz") as data:
        rows = []
        for gain in GAINS:
            weeks = data[f"{direction}_{int(gain * 100)}"] - first_week
            weeks = weeks[(weeks >= 0) & (weeks < n_weeks)]
            rows.append(np.bincount(weeks, minlength=n_weeks))
    return np.array(rows, np.float64)


@njit(parallel=True, cache=True)
def _grid(seg, t, week, price, change, ratio, quote, breakout, retrace, volume, precision, event10, event20,
          event_week10, event_week20, params, baseline, sign, n_weeks, scored, correct, caught10, caught20, alerts):
    n_params = params.shape[0]
    block = 32
    for b in prange((n_params + block - 1) // block):
        c0 = b * block
        c1 = min(n_params, c0 + block)
        last_t = np.empty(block, np.int64)
        last_p = np.empty(block)
        last_e10 = np.empty(block, np.int64)
        last_e20 = np.empty(block, np.int64)
        for s in range(len(seg) - 1):
            for k in range(block):
                last_t[k] = -1
                last_e10[k] = -1
                last_e20[k] = -1
            for i in range(seg[s], seg[s + 1]):
                w = week[i]
                if w >= n_weeks:
                    break
                for c in range(c0, c1):
                    p = params[c]
                    if (ratio[i] < p[0] or quote[i] < p[1] or change[i] < p[2] or (p[4] > 0 and not breakout[i])
                            or volume[i] <= p[5] or retrace[i] > p[8]):
                        continue
                    k = c - c0
                    if last_t[k] >= 0:
                        elapsed = t[i] - last_t[k]
                        if elapsed < p[6]:
                            continue
                        if elapsed < baseline and sign * (price[i] / last_p[k] - 1) * 100 < p[7]:
                            continue
                    last_t[k] = t[i]
                    last_p[k] = price[i]
                    if change[i] > p[3]:
                        continue
                    alerts[c, w] += 1
                    if not np.isnan(precision[i]):
                        scored[c, w] += 1
                        correct[c, w] += precision[i]
                    # 同一段大行情只算报到一次；同合约同方向的段按时间先后编号，记录也按时间排序。
                    if event10[i] >= 0 and event10[i] != last_e10[k]:
                        last_e10[k] = event10[i]
                        caught10[c, event_week10[i]] += 1
                    if event20[i] >= 0 and event20[i] != last_e20[k]:
                        last_e20[k] = event20[i]
                        caught20[c, event_week20[i]] += 1


def cmd_search() -> None:
    params = param_grid()
    for direction in SIGNS:
        for window in WINDOWS:
            for baseline in BASELINES:
                started = time.time()
                rec = _load(window, baseline, direction)
                shape = (len(params), DEV_WEEKS)
                arrays = {name: np.zeros(shape, np.float32 if name == "correct" else np.int32)
                          for name in ("scored", "correct", "caught10", "caught20", "alerts")}
                _grid(rec["seg"], rec["t"], rec["week"], rec["price"], rec["change"], rec["ratio"], rec["quote"],
                      rec["breakout"], rec["retrace"], rec["volume"], rec["precision"], rec["event10"],
                      rec["event20"], rec["event_week10"], rec["event_week20"], params, baseline, SIGNS[direction],
                      DEV_WEEKS, arrays["scored"], arrays["correct"], arrays["caught10"], arrays["caught20"],
                      arrays["alerts"])
                np.savez(OUT / f"dev_{direction}_W{window}_B{baseline}.npz", params=params, **arrays)
                print(f"{direction} W{window} B{baseline}: {len(rec['t'])} 条记录 {time.time() - started:.0f}s",
                      flush=True)


# ---------------------------------------------------------------- F1、Reality Check、PBO


def f1_score(scored, correct, caught10, caught20, events10, events20):
    """各量可以是任意形状，按最后一维已经汇总好的总数计算；返回 (F1, 准确率, 召回率)。"""
    precision = np.divide(correct, scored, out=np.zeros(np.shape(correct)), where=np.asarray(scored) > 0)
    recall = 0.5 * (caught10 / events10 + caught20 / events20)
    total = BETA**2 * precision + recall
    f1 = np.divide((1 + BETA**2) * precision * recall, total, out=np.zeros(np.shape(total)), where=total > 0)
    return f1, precision, recall


def _load_dev(direction: str) -> dict:
    parts = []
    for window in WINDOWS:
        for baseline in BASELINES:
            with np.load(OUT / f"dev_{direction}_W{window}_B{baseline}.npz") as data:
                part = {name: data[name] for name in data.files}
            part["window"] = np.full(len(part["params"]), window)
            part["baseline"] = np.full(len(part["params"]), baseline)
            parts.append(part)
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def weekly_of(config: dict, direction: str, first_week: int, n_weeks: int) -> dict:
    """单组参数逐周的计数，另附逐条提醒的明细（各设置准不准、波次、提前量），供样本外分项统计。"""
    rec = _load(config["W"], config["B"], direction)
    alert, wave = alerts_of(rec, _params_row(config), config["B"], direction)
    week = rec["week"] - first_week
    inside = alert & (week >= 0) & (week < n_weeks)
    valid = rec["valid"][inside]
    scored_mask = valid[:, 0] >= 0
    out = {
        "alerts": np.bincount(week[inside], minlength=n_weeks).astype(np.float64),
        "scored": np.bincount(week[inside][scored_mask], minlength=n_weeks).astype(np.float64),
        "correct": np.bincount(week[inside][scored_mask], weights=rec["precision"][inside][scored_mask],
                               minlength=n_weeks),
        "wave": wave[inside],
        "valid": valid,
    }
    for tag in (10, 20):
        caught = np.zeros(n_weeks)
        seen = set()
        for i in np.flatnonzero(inside):
            event = rec[f"event{tag}"][i]
            if event < 0:
                continue
            key = (int(rec["sym"][i]), int(event))
            if key in seen:
                continue
            seen.add(key)
            w = rec[f"event_week{tag}"][i] - first_week
            if 0 <= w < n_weeks:
                caught[w] += 1
        out[f"caught{tag}"] = caught
    return out


def _bootstrap_weights(n_weeks: int, draws: int, seed: int) -> np.ndarray:
    """循环块自助法：每次抽若干个长 BLOCK_WEEKS 周的块拼满 n_weeks 周，返回各周被抽中的次数。"""
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_weeks / BLOCK_WEEKS)
    weights = np.zeros((draws, n_weeks), np.float32)
    offsets = np.arange(BLOCK_WEEKS)
    for k in range(draws):
        picks = ((rng.integers(0, n_weeks, n_blocks)[:, None] + offsets) % n_weeks).ravel()[:n_weeks]
        weights[k] = np.bincount(picks, minlength=n_weeks)
    return weights


@njit(cache=True)
def _f1_scalar(scored, correct, caught10, caught20, events10, events20, beta):
    pr = correct / scored if scored > 0 else 0.0
    rc = 0.5 * (caught10 / events10 + caught20 / events20)
    total = beta * beta * pr + rc
    return (1 + beta * beta) * pr * rc / total if total > 0 else 0.0


@njit(parallel=True, cache=True)
def _cscv(scored, correct, caught10, caught20, events10, events20, masks, beta):
    n_split = masks.shape[0]
    n = scored.shape[0]
    logits = np.empty(n_split)
    for j in prange(n_split):
        is_f1 = np.empty(n)
        oos_f1 = np.empty(n)
        e10_in = 0.0
        e20_in = 0.0
        e10_out = 0.0
        e20_out = 0.0
        for b in range(masks.shape[1]):
            if masks[j, b]:
                e10_in += events10[b]
                e20_in += events20[b]
            else:
                e10_out += events10[b]
                e20_out += events20[b]
        best = 0
        for c in range(n):
            v = np.zeros(8)
            for b in range(masks.shape[1]):
                o = 0 if masks[j, b] else 4
                v[o] += scored[c, b]
                v[o + 1] += correct[c, b]
                v[o + 2] += caught10[c, b]
                v[o + 3] += caught20[c, b]
            # 不用元组循环写数组：numba 里那样写不回原数组。
            is_f1[c] = _f1_scalar(v[0], v[1], v[2], v[3], e10_in, e20_in, beta)
            oos_f1[c] = _f1_scalar(v[4], v[5], v[6], v[7], e10_out, e20_out, beta)
            if is_f1[c] > is_f1[best]:
                best = c
        below = 0.0
        for c in range(n):
            if oos_f1[c] < oos_f1[best]:
                below += 1.0
            elif oos_f1[c] == oos_f1[best] and c != best:
                below += 0.5
        omega = (below + 1.0) / (n + 1.0)
        logits[j] = math.log(omega / (1.0 - omega))
    return logits


def _describe(dev: dict, i: int) -> dict:
    names = ("ratio", "quote", "change", "max_change", "breakout", "volume_24h", "cooldown", "step", "retrace")
    return {"W": int(dev["window"][i]), "B": int(dev["baseline"][i]),
            **dict(zip(names, (float(v) for v in dev["params"][i]), strict=True))}


def cmd_select() -> None:
    summary = {}
    edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
    masks = np.array([[b in combo for b in range(CSCV_BLOCKS)]
                      for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2)])
    for direction in SIGNS:
        dev = _load_dev(direction)
        events = event_counts(direction, 0, DEV_WEEKS)
        totals = {name: dev[name].sum(1, dtype=np.float64) for name in ("scored", "correct", "caught10", "caught20")}
        f1, precision, recall = f1_score(totals["scored"], totals["correct"], totals["caught10"], totals["caught20"],
                                         events[0].sum(), events[1].sum())
        best = int(np.argmax(f1))
        config = _describe(dev, best)
        # 网格与单组评估是两套实现，选中方案用单组评估再算一遍，两者必须一致。
        check = weekly_of(config, direction, 0, DEV_WEEKS)
        check_f1 = f1_score(check["scored"].sum(), check["correct"].sum(), check["caught10"].sum(),
                            check["caught20"].sum(), events[0].sum(), events[1].sum())[0]
        assert abs(float(check_f1) - float(f1[best])) < 1e-6, (check_f1, f1[best])
        bench = weekly_of(CURRENT[direction], direction, 0, DEV_WEEKS)
        bench_f1, bench_p, bench_r = f1_score(bench["scored"].sum(), bench["correct"].sum(), bench["caught10"].sum(),
                                              bench["caught20"].sum(), events[0].sum(), events[1].sum())

        # Reality Check：自助样本里「全族最好的一组比基准好多少」减去原样本的差，看原样本的优势有多少是运气。
        started = time.time()
        observed = f1 - bench_f1
        stat = float(observed.max())
        exceed = 0
        weights_all = _bootstrap_weights(DEV_WEEKS, BOOTSTRAP, seed=20261019 + SIGNS[direction])
        stacked = {name: dev[name].astype(np.float32) for name in ("scored", "correct", "caught10", "caught20")}
        for chunk in np.array_split(np.arange(BOOTSTRAP), 20):
            wts = weights_all[chunk]
            sums = {name: wts @ arr.T for name, arr in stacked.items()}  # (draws, 候选数)
            ev10 = wts @ events[0].astype(np.float32)
            ev20 = wts @ events[1].astype(np.float32)
            f1_star, _, _ = f1_score(sums["scored"], sums["correct"], sums["caught10"], sums["caught20"],
                                     ev10[:, None], ev20[:, None])
            b_star, _, _ = f1_score(wts @ bench["scored"].astype(np.float32), wts @ bench["correct"].astype(np.float32),
                                    wts @ bench["caught10"].astype(np.float32),
                                    wts @ bench["caught20"].astype(np.float32), ev10, ev20)
            centered = (f1_star - b_star[:, None]) - observed[None, :]
            exceed += int(np.sum(centered.max(1) >= stat))
        reality_p = exceed / BOOTSTRAP
        rc_seconds = time.time() - started

        block = {name: np.add.reduceat(dev[name].astype(np.float64), edges[:-1], axis=1)
                 for name in ("scored", "correct", "caught10", "caught20")}
        logits = _cscv(block["scored"], block["correct"], block["caught10"], block["caught20"],
                       np.add.reduceat(events[0], edges[:-1]), np.add.reduceat(events[1], edges[:-1]), masks, BETA)
        pbo = float(np.mean(logits <= 0))
        order = np.argsort(-f1)
        summary[direction] = {
            "chosen": config,
            "dev_f1": float(f1[best]),
            "dev_precision": float(precision[best]),
            "dev_recall": float(recall[best]),
            "dev_alerts_per_day": float(dev["alerts"][best].sum() / (DEV_WEEKS * 7)),
            "current_dev_f1": float(bench_f1),
            "current_dev_precision": float(bench_p),
            "current_dev_recall": float(bench_r),
            "current_dev_alerts_per_day": float(bench["alerts"].sum() / (DEV_WEEKS * 7)),
            "current_dev_rank": int(np.sum(f1 > bench_f1)) + 1,
            "events_dev": [float(events[0].sum()), float(events[1].sum())],
            "reality_check_p": reality_p,
            "pbo": pbo,
            "trials": len(f1),
            "rc_seconds": rc_seconds,
            "top30": [
                {**_describe(dev, int(i)), "f1": float(f1[i]), "precision": float(precision[i]),
                 "recall": float(recall[i]), "alerts_per_day": float(dev["alerts"][i].sum() / (DEV_WEEKS * 7))}
                for i in order[:30]
            ],
        }
        print(json.dumps({k: v for k, v in summary[direction].items() if k != "top30"}, ensure_ascii=False, indent=1),
              flush=True)
    (OUT / _named("select")).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _report(data: dict, events: np.ndarray, n_weeks: int) -> dict:
    f1, p, r = f1_score(data["scored"].sum(), data["correct"].sum(), data["caught10"].sum(), data["caught20"].sum(),
                        events[0].sum(), events[1].sum())
    valid = data["valid"][data["valid"][:, 0] >= 0]
    per_setting = valid.mean(0)  # 顺序：(3%, 60)、(3%, 240)、(5%, 60)、(5%, 240)
    recalls = [data["caught10"].sum() / events[0].sum(), data["caught20"].sum() / events[1].sum()]
    b2 = BETA**2
    pairs = [(1 + b2) * ps * rg / (b2 * ps + rg) if b2 * ps + rg > 0 else 0.0 for ps in per_setting for rg in recalls]
    return {
        "f1": float(f1), "precision": float(p), "recall": float(r),
        "precision_by_setting": [float(v) for v in per_setting],
        "recall_by_gain": [float(v) for v in recalls],
        "f1_by_pair": [float(v) for v in pairs],
        "alerts_per_day": float(data["alerts"].sum() / (n_weeks * 7)),
        "wave1_per_day": float(np.sum(data["wave"] == 1) / (n_weeks * 7)),
        "events": [float(events[0].sum()), float(events[1].sum())],
    }


def _data_weeks() -> int:
    """数据覆盖到的周数（含最后不满一周的那周）。按数据算而不是按运行当天算：数据不更新时晚些重跑，
    样本外的周数与每天提醒数也不变。"""
    last = 0
    for path in OUT.glob("cand_W*_B*.npz"):
        with np.load(path) as data:
            last = max(last, int(data["t"].max()))
    return (last * 60 + 60) // WEEK + 1


def cmd_evaluate() -> None:
    chosen = json.loads((OUT / _named("select")).read_text(encoding="utf-8"))
    n_weeks = _data_weeks() - DEV_WEEKS
    result = {}
    for direction in SIGNS:
        events = event_counts(direction, DEV_WEEKS, n_weeks)
        runs = {name: weekly_of(config, direction, DEV_WEEKS, n_weeks)
                for name, config in (("chosen", chosen[direction]["chosen"]), ("current", CURRENT[direction]))}
        weights = _bootstrap_weights(n_weeks, BOOTSTRAP, seed=1019 + SIGNS[direction]).astype(np.float64)
        ev10, ev20 = weights @ events[0], weights @ events[1]
        star = {name: f1_score(weights @ run["scored"], weights @ run["correct"], weights @ run["caught10"],
                               weights @ run["caught20"], ev10, ev20)[0] for name, run in runs.items()}
        reports = {name: _report(run, events, n_weeks) for name, run in runs.items()}
        result[direction] = {
            "chosen": {"config": chosen[direction]["chosen"], "oos": reports["chosen"]},
            "current": {"config": CURRENT[direction], "oos": reports["current"]},
            "oos_weeks": n_weeks,
            # 单侧：自助样本里选中方案不比现行配置好的比例。
            "bootstrap_p": float(np.mean(star["chosen"] - star["current"] <= 0)),
            "pairs_better": int(sum(a > b for a, b in zip(reports["chosen"]["f1_by_pair"],
                                                          reports["current"]["f1_by_pair"], strict=True))),
        }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / _named("evaluate")).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def _named(stem: str) -> str:
    return f"{stem}.json" if BETA == 1.0 else f"{stem}_b{BETA:g}.json"


def _baseline_symbol(args: tuple[str, float]) -> np.ndarray:
    """随机时刻（每隔 61 分钟一个）在两个方向、4 种设置下准的次数与计数，按开发期 / 样本外分开。"""
    symbol, multiplier = args
    out = np.zeros((2, 2, 5))  # (期间, 方向, 4 种设置的准次数 + 计数)
    series = load_series(symbol, multiplier)
    if series is None or len(series) < 2000:
        return out
    covered = series.covered.copy()
    covered[: _last_jump(series.close)] = False
    cum_quote = np.concatenate([[0.0], np.cumsum(series.quote)])
    cum_uncovered = np.concatenate([[0], np.cumsum(~covered)]).astype(np.int64)
    index = np.arange(1440, len(series), 61)
    volume = cum_quote[index + 1] - cum_quote[index - 1439]
    index = index[(volume > 1e6) & covered[index]]
    period = ((series.t0 + index * 60 + 60) >= EPOCH + DEV_WEEKS * WEEK).astype(np.int64)
    for d, s in enumerate((1, -1)):
        valid = _validity(series.high, series.low, series.close, cum_uncovered, index,
                          np.full(len(index), s, np.int8), np.array(TARGETS), np.array(HOLDS, np.int64), FORWARD)
        ok = valid[:, 0] >= 0
        for p in (0, 1):
            sel = ok & (period == p)
            out[p, d, :4] = valid[sel].sum(0)
            out[p, d, 4] = sel.sum()
    return out


def cmd_baseline() -> None:
    contracts = load_contracts()
    jobs = [(s, _multiplier(contracts[s])) for s in sorted(contracts) if _multiplier(contracts[s])]
    total = np.zeros((2, 2, 5))
    with ProcessPoolExecutor(max_workers=14) as executor:
        for part in executor.map(_baseline_symbol, jobs, chunksize=4):
            total += part
    result = {}
    for p, period in enumerate(("dev", "oos")):
        for d, direction in enumerate(SIGNS):
            by_setting = total[p, d, :4] / total[p, d, 4]
            result[f"{period}_{direction}"] = {"precision": float(by_setting.mean()),
                                               "by_setting": [float(v) for v in by_setting],
                                               "samples": int(total[p, d, 4])}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_selftest() -> None:
    """用手造的价格检查之字线与准不准的判定。"""
    close = np.array([100, 101, 99, 100, 106, 112, 111, 105, 104, 98, 90, 95, 100, 105], np.float64)
    quote = np.full(len(close), 1e6)
    cum_quote = np.concatenate([[0.0], np.cumsum(quote)])
    cum_uncovered = np.zeros(len(close) + 1, np.int64)
    starts, ends, signs = _legs(close, cum_quote, cum_uncovered, 0.10)
    # 99 → 112 涨 13.1%（112 回落到 105 跌 6.25% ≥ 5% 结束），112 → 90 跌 19.6%，90 → 105 涨 16.7%（数据到头）。
    assert list(zip(starts, ends, signs, strict=True)) == [(2, 5, 1), (5, 10, -1), (10, 13, 1)], (starts, ends, signs)
    event = _early_event(close, np.array([4, 5, 9, 11]), np.array([1, 1, -1, 1], np.int8), starts, ends, signs)
    # 106 离 99 的对数涨幅 0.068 > 整段 0.123 的一半；112 是终点；98 已过半（0.134 > 0.109）；95 在前半段。
    assert list(event) == [-1, -1, -1, 2], event
    high = close + 0.5
    low = close - 0.5
    valid = _validity(np.concatenate([high, np.full(300, 200.0)]), np.concatenate([low, np.full(300, 199.0)]),
                      np.concatenate([close, np.full(300, 199.5)]), np.zeros(len(close) + 301, np.int64),
                      np.array([0]), np.array([1], np.int8), np.array([0.03, 0.05]), np.array([60, 240]), 240)
    # 100 之后先到 101.5、99.5（未触及 ±3%），第 4 分钟最高 106.5 ≥ 103 且最低 105.5 > 97：3% 准，5% 也准。
    assert list(valid[0]) == [1, 1, 1, 1], valid
    print("自测通过")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "extract", "search", "select", "evaluate", "baseline"))
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()
    global BETA
    BETA = args.beta
    command = args.command
    {"selftest": cmd_selftest, "extract": cmd_extract, "search": cmd_search, "select": cmd_select,
     "evaluate": cmd_evaluate, "baseline": cmd_baseline}[command]()


if __name__ == "__main__":
    main()
