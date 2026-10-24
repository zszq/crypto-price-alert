"""单独的冷门启动检测器：只看冷门币自己的启动行情与提醒，口径与判据见 dormant_solo.md（跑数前写定）。

准不准沿用 launch_alerts；之字线换成冷门启动的口径（起点前 24 小时冷门、段内成交额 ≥ 10 万）。

用法（仓库根目录，$P 见 README.md）：
  & $P research/launch_trade/dormant_solo.py extract    # 候选分钟、冷门启动行情、冷门随机时刻
  & $P research/launch_trade/dormant_solo.py verify     # 网格判定与逐分钟直接实现对拍
  & $P research/launch_trade/dormant_solo.py search     # 开发期全量网格，逐周计数
  & $P research/launch_trade/dormant_solo.py select     # 开发期选参、PBO
  & $P research/launch_trade/dormant_solo.py evaluate   # 选中方案的样本外与 IRYS 例子
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
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
    _early_event,
    _validity,
)
from launch_grid import EPOCH, _last_jump, _multiplier, _rolling_extreme
from load_data import load_contracts, load_series
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "dormant_solo"
BETA = 0.5
PRE = 1440  # 「之前 24 小时」的分钟数
DORMANT = 1e6  # 冷门：之前 24 小时成交额上限（目标行情的定义，也是闸门 D 的最大候选）
MIN_LEG_QUOTE = 1e5
MAX_LEG_MINUTES = 1440
RANDOM_STEP = 61

WINDOWS = (3, 5, 10, 15)
QUOTES = (3e3, 5e3, 1e4, 2e4, 5e4)
CHANGES = (3.0, 5.0, 8.0, 12.0)
MAXES = (20.0, math.inf)
RATIOS = (0.0, 5.0, 20.0, 100.0)
BREAKOUTS = (0.0, 1.0)
ACTIVE_SHARES = (0.0, 0.5, 1.0)  # 活跃分钟数下限 = max(1, ⌈W × 比例⌉)
RETRACES = (math.inf, 0.5)
RANGES = (math.inf, 0.10)
GATES = (3e5, 1e6)
COOLDOWNS = (60.0, 360.0)
PARAM_NAMES = ("quote", "change", "max_change", "ratio", "breakout", "active", "retrace", "range", "gate", "cooldown")
COUNTS = ("scored", "correct", "caught10", "caught20", "alerts")


def _active_need(window: int, share: float) -> int:
    return max(1, math.ceil(window * share - 1e-9))


def param_grid(window: int) -> np.ndarray:
    rows = [(q, c, m, r, b, _active_need(window, a), p, z, d, cd)
            for q, c, m, r, b, a, p, z, d, cd in itertools.product(
                QUOTES, CHANGES, MAXES, RATIOS, BREAKOUTS, ACTIVE_SHARES, RETRACES, RANGES, GATES, COOLDOWNS)]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------- 冷门启动行情


@njit(cache=True)
def _dormant_legs(close, cum_quote, cum_uncovered, gain):
    """之字线（反转门槛 gain/2）切出的冷门启动上涨段：返回起点、终点下标。"""
    n = len(close)
    rev = gain / 2
    starts = []
    ends = []
    a = 0
    while a < n:
        while a < n and cum_uncovered[a + 1] - cum_uncovered[a] == 1:
            a += 1
        b = a
        while b < n and cum_uncovered[b + 1] - cum_uncovered[b] == 0:
            b += 1
        trend = 0
        hi = a
        lo = a
        start = a
        ext = a
        for t in range(a + 1, b + 1):
            # t == b 时收尾：数据到头或遇到缺口，还没确认结束的最后一段按已走出的部分计。
            ended = t == b
            if not ended:
                c = close[t]
                # 价格相同时取最后一次：冷门币常在同一价位横盘很久，取最早一次会把横盘算进上涨段，
                # 段长超过 24 小时被排除。
                if trend == 0:
                    if c >= close[hi]:
                        hi = t
                    if c <= close[lo]:
                        lo = t
                    if c >= close[lo] * (1 + rev):
                        trend, start, ext = 1, lo, t
                    elif c <= close[hi] * (1 - rev):
                        trend, start, ext = -1, hi, t
                    continue
                if trend == 1:
                    if c >= close[ext]:
                        ext = t
                        continue
                    if c > close[ext] * (1 - rev):
                        continue
                else:
                    if c <= close[ext]:
                        ext = t
                        continue
                    if c < close[ext] * (1 + rev):
                        continue
            if trend == 1:
                ok = (close[ext] / close[start] - 1 >= gain and ext - start <= MAX_LEG_MINUTES
                      and cum_quote[ext + 1] - cum_quote[start + 1] >= MIN_LEG_QUOTE and start >= PRE - 1
                      and cum_uncovered[start + 1] - cum_uncovered[start + 1 - PRE] == 0
                      and cum_quote[start + 1] - cum_quote[start + 1 - PRE] <= DORMANT)
                if ok:
                    starts.append(start)
                    ends.append(ext)
            if ended:
                break
            if trend == 1:
                trend, start, ext = -1, ext, t
            elif trend == -1:
                trend, start, ext = 1, ext, t
        a = b
    return np.array(starts, np.int64), np.array(ends, np.int64)


# ---------------------------------------------------------------- 抽取


@njit(cache=True)
def _extract(high, close, cum_quote, cum_active, cum_uncovered, window, pre_high, pre_low, win_high, min_quote,
             min_change, max_gate):
    n = len(close)
    keep = np.zeros(n, np.bool_)
    for t in range(window + PRE - 1, n):
        first = t - window + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first] != 0:
            continue
        if cum_quote[t - window + 1] - cum_quote[first] > max_gate:
            continue
        if cum_quote[t + 1] - cum_quote[t - window + 1] < min_quote:
            continue
        if (close[t] / close[t - window] - 1) * 100 < min_change:
            continue
        keep[t] = True
    index = np.flatnonzero(keep)
    m = len(index)
    change = np.empty(m)
    ratio = np.empty(m)
    quote = np.empty(m)
    breakout = np.empty(m, np.bool_)
    active = np.empty(m, np.int16)
    retrace = np.empty(m)
    span = np.empty(m)
    pre = np.empty(m)
    for k in range(m):
        t = index[k]
        first = t - window + 1 - PRE
        start = close[t - window]
        change[k] = (close[t] / start - 1) * 100
        q = cum_quote[t + 1] - cum_quote[t - window + 1]
        quote[k] = q
        p = cum_quote[t - window + 1] - cum_quote[first]
        pre[k] = p
        ratio[k] = q / (p / PRE * window) if p > 0 else np.inf
        breakout[k] = close[t] > pre_high[t - window]
        active[k] = cum_active[t + 1] - cum_active[t - window + 1]
        retrace[k] = (win_high[t] - close[t]) / (win_high[t] - start)
        span[k] = pre_high[t - window] / pre_low[t - window] - 1
    return index, change, ratio, quote, breakout, active, retrace, span, pre


def _week(index: np.ndarray, offset: int) -> np.ndarray:
    return ((index + offset) * 60 + 60) // WEEK  # 提醒时刻 = K 线开盘 + 1 分钟


def _prepare(symbol: str, multiplier: float):
    series = load_series(symbol, multiplier)
    if series is None or len(series) < 2000:
        return None
    covered = series.covered.copy()
    covered[: _last_jump(series.close)] = False
    cum_quote = np.concatenate([[0.0], np.cumsum(series.quote)])
    cum_active = np.concatenate([[0], np.cumsum(series.quote > 0)]).astype(np.int64)
    cum_uncovered = np.concatenate([[0], np.cumsum(~covered)]).astype(np.int64)
    return series, cum_quote, cum_active, cum_uncovered


def _extract_symbol(args: tuple[int, str, float]) -> tuple[int, dict]:
    sym_id, symbol, multiplier = args
    prepared = _prepare(symbol, multiplier)
    out: dict = {}
    if prepared is None:
        return sym_id, out
    series, cum_quote, cum_active, cum_uncovered = prepared
    high, low, close = series.high, series.low, series.close
    offset = (series.t0 - EPOCH) // 60
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    legs = {}
    for gain in GAINS:
        starts, ends = _dormant_legs(close, cum_quote, cum_uncovered, gain)
        legs[gain] = (starts, ends, np.ones(len(starts), np.int8))
        out[f"events{int(gain * 100)}"] = ((starts + offset) * 60 // WEEK).astype(np.int64)

    # 冷门随机时刻：每隔 RANDOM_STEP 分钟一个，之前 24 小时有数据且冷门，按上涨提醒算准不准。
    index = np.arange(PRE, len(series), RANDOM_STEP)
    index = index[(cum_uncovered[index + 1] - cum_uncovered[index + 1 - PRE] == 0)
                  & (cum_quote[index + 1] - cum_quote[index + 1 - PRE] <= DORMANT)]
    valid = _validity(high, low, close, cum_uncovered, index, np.ones(len(index), np.int8), targets, holds, FORWARD)
    ok = valid[:, 0] >= 0
    oos = _week(index, offset) >= DEV_WEEKS
    out["random"] = np.array([np.concatenate([valid[ok & (oos == p)].sum(0), [np.sum(ok & (oos == p))]])
                              for p in (False, True)], np.float64)

    pre_high = _rolling_extreme(high, PRE, True)
    pre_low = _rolling_extreme(low, PRE, False)
    for window in WINDOWS:
        win_high = _rolling_extreme(high, window, True)
        index, change, ratio, quote, breakout, active, retrace, span, pre = _extract(
            high, close, cum_quote, cum_active, cum_uncovered, window, pre_high, pre_low, win_high,
            QUOTES[0], CHANGES[0], max(GATES))
        ones = np.ones(len(index), np.int8)
        record = {"t": (index + offset).astype(np.int64), "price": close[index], "change": change, "ratio": ratio,
                  "quote": quote, "breakout": breakout, "active": active, "retrace": retrace, "range": span,
                  "pre": pre, "valid": _validity(high, low, close, cum_uncovered, index, ones, targets, holds, FORWARD)}
        for gain in GAINS:
            starts, ends, signs = legs[gain]
            event = _early_event(close, index, ones, starts, ends, signs)
            tag = int(gain * 100)
            week = np.full(len(index), -1, np.int64)
            hit = event >= 0
            week[hit] = (starts[event[hit]] + offset) * 60 // WEEK
            record[f"event{tag}"] = event
            record[f"event_week{tag}"] = week
        out[window] = record
    return sym_id, out


def cmd_extract() -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = [(i, s, _multiplier(contracts[s])) for i, s in enumerate(symbols) if _multiplier(contracts[s])]
    parts: dict = {w: [] for w in WINDOWS}
    events: dict = {g: [] for g in GAINS}
    random = np.zeros((2, 5))
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
                random += result["random"]
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    np.savez(OUT / "events.npz", **{f"up_{int(g * 100)}": np.concatenate(v) for g, v in events.items()})
    np.save(OUT / "random.npy", random)
    for window, chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        np.savez(OUT / f"cand_W{window}.npz", **merged)
        print(f"W{window}: {len(merged['t'])} 条候选", flush=True)
    for gain, v in events.items():
        weeks = np.concatenate(v)
        print(f"冷门启动 G{gain:.0%}: 开发期 {np.sum(weeks < DEV_WEEKS)} 段，样本外 {np.sum(weeks >= DEV_WEEKS)} 段")
    for p, name in enumerate(("开发期", "样本外")):
        print(f"冷门随机时刻 {name}：准确率 {random[p, :4].mean() / random[p, 4]:.3f}（{random[p, 4]:.0f} 个）")


# ---------------------------------------------------------------- 判定与网格


def _load(window: int) -> dict:
    with np.load(OUT / f"cand_W{window}.npz") as data:
        rec = {name: data[name] for name in data.files}
    edges = np.flatnonzero(np.diff(rec["sym"])) + 1
    rec["seg"] = np.concatenate([[0], edges, [len(rec["sym"])]]).astype(np.int64)
    rec["week"] = ((rec["t"] * 60 + 60) // WEEK).astype(np.int64)
    valid = rec["valid"].astype(np.float32)
    valid[valid < 0] = np.nan
    rec["precision"] = valid.mean(1)
    return rec


@njit(cache=True)
def _passes(i, p, quote, change, ratio, breakout, active, retrace, span, pre):
    return not (quote[i] < p[0] or change[i] < p[1] or ratio[i] < p[3] or (p[4] > 0 and not breakout[i])
                or active[i] < p[5] or retrace[i] > p[6] or span[i] > p[7] or pre[i] > p[8])


@njit(parallel=True, cache=True)
def _grid(seg, t, week, quote, change, ratio, breakout, active, retrace, span, pre, precision, event10, event20,
          event_week10, event_week20, params, n_weeks, scored, correct, caught10, caught20, alerts):
    n_params = params.shape[0]
    block = 32
    for b in prange((n_params + block - 1) // block):
        c0 = b * block
        c1 = min(n_params, c0 + block)
        last_t = np.empty(block, np.int64)
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
                    if not _passes(i, p, quote, change, ratio, breakout, active, retrace, span, pre):
                        continue
                    k = c - c0
                    if last_t[k] >= 0 and t[i] - last_t[k] < p[9]:
                        continue
                    last_t[k] = t[i]
                    # 超上限：记入冷却但不发出（防止插针后在高位补报）。
                    if change[i] > p[2]:
                        continue
                    alerts[c, w] += 1
                    if not np.isnan(precision[i]):
                        scored[c, w] += 1
                        correct[c, w] += precision[i]
                    if event10[i] >= 0 and event10[i] != last_e10[k]:
                        last_e10[k] = event10[i]
                        caught10[c, event_week10[i]] += 1
                    if event20[i] >= 0 and event20[i] != last_e20[k]:
                        last_e20[k] = event20[i]
                        caught20[c, event_week20[i]] += 1


@njit(cache=True)
def _alert_mask(seg, t, quote, change, ratio, breakout, active, retrace, span, pre, p):
    n = len(t)
    alert = np.zeros(n, np.bool_)
    for s in range(len(seg) - 1):
        last_t = -1
        for i in range(seg[s], seg[s + 1]):
            if not _passes(i, p, quote, change, ratio, breakout, active, retrace, span, pre):
                continue
            if last_t >= 0 and t[i] - last_t < p[9]:
                continue
            last_t = t[i]
            alert[i] = change[i] <= p[2]
    return alert


def _row(config: dict) -> np.ndarray:
    return np.array([config[name] for name in PARAM_NAMES], dtype=np.float64)


def alerts_for(rec: dict, config: dict) -> np.ndarray:
    return _alert_mask(rec["seg"], rec["t"], rec["quote"], rec["change"], rec["ratio"], rec["breakout"],
                       rec["active"], rec["retrace"], rec["range"], rec["pre"], _row(config))


def cmd_search() -> None:
    for window in WINDOWS:
        started = time.time()
        params = param_grid(window)
        rec = _load(window)
        shape = (len(params), DEV_WEEKS)
        arrays = {name: np.zeros(shape, np.float32 if name == "correct" else np.int32) for name in COUNTS}
        _grid(rec["seg"], rec["t"], rec["week"], rec["quote"], rec["change"], rec["ratio"], rec["breakout"],
              rec["active"], rec["retrace"], rec["range"], rec["pre"], rec["precision"], rec["event10"],
              rec["event20"], rec["event_week10"], rec["event_week20"], params, DEV_WEEKS, arrays["scored"],
              arrays["correct"], arrays["caught10"], arrays["caught20"], arrays["alerts"])
        np.savez(OUT / f"dev_W{window}.npz", params=params, **arrays)
        print(f"W{window}: {len(rec['t'])} 条记录 {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------- 选参与检验


def f_score(scored, correct, caught10, caught20, events10, events20):
    precision = np.divide(correct, scored, out=np.zeros(np.shape(correct)), where=np.asarray(scored) > 0)
    recall = 0.5 * (np.asarray(caught10) / events10 + np.asarray(caught20) / events20)
    total = BETA**2 * precision + recall
    f = np.divide((1 + BETA**2) * precision * recall, total, out=np.zeros(np.shape(total)), where=total > 0)
    return f, precision, recall


def event_counts(first_week: int, n_weeks: int) -> np.ndarray:
    with np.load(OUT / "events.npz") as data:
        rows = []
        for gain in GAINS:
            weeks = data[f"up_{int(gain * 100)}"] - first_week
            weeks = weeks[(weeks >= 0) & (weeks < n_weeks)]
            rows.append(np.bincount(weeks, minlength=n_weeks))
    return np.array(rows, np.float64)


def _bars() -> dict:
    """准确的标准：现行放量上涨在同一份数据上的准确率。

    原由已删除的 dormant_launch 研究算出；那份脚本只为给出这几个数，删掉后固定在这里，判据仍可复现。
    """
    return {"dev": 0.39785473920358944, "oos": 0.437489154954017,
            "oos_by_setting": [0.44990456359534964, 0.4853027936838452, 0.3683498178032275, 0.4463994447336457]}


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


@njit(cache=True)
def _f_scalar(scored, correct, caught10, caught20, events10, events20, beta):
    pr = correct / scored if scored > 0 else 0.0
    rc = 0.5 * (caught10 / events10 + caught20 / events20)
    total = beta * beta * pr + rc
    return (1 + beta * beta) * pr * rc / total if total > 0 else 0.0


@njit(parallel=True, cache=True)
def _cscv(scored, correct, caught10, caught20, events10, events20, masks, beta, min_precision):
    n_split = masks.shape[0]
    n = scored.shape[0]
    logits = np.full(n_split, np.nan)
    for j in prange(n_split):
        is_f = np.empty(n)
        oos_f = np.empty(n)
        eligible = np.zeros(n, np.bool_)
        e_in = np.zeros(2)
        e_out = np.zeros(2)
        for b in range(masks.shape[1]):
            if masks[j, b]:
                e_in[0] += events10[b]
                e_in[1] += events20[b]
            else:
                e_out[0] += events10[b]
                e_out[1] += events20[b]
        best = -1
        for c in range(n):
            v = np.zeros(8)
            for b in range(masks.shape[1]):
                o = 0 if masks[j, b] else 4
                v[o] += scored[c, b]
                v[o + 1] += correct[c, b]
                v[o + 2] += caught10[c, b]
                v[o + 3] += caught20[c, b]
            is_f[c] = _f_scalar(v[0], v[1], v[2], v[3], e_in[0], e_in[1], beta)
            oos_f[c] = _f_scalar(v[4], v[5], v[6], v[7], e_out[0], e_out[1], beta)
            eligible[c] = v[0] > 0 and v[1] / v[0] >= min_precision
            if eligible[c] and (best < 0 or is_f[c] > is_f[best]):
                best = c
        if best < 0:
            continue
        below = 0.0
        count = 0
        for c in range(n):
            count += 1
            if oos_f[c] < oos_f[best]:
                below += 1.0
            elif oos_f[c] == oos_f[best] and c != best:
                below += 0.5
        omega = (below + 1.0) / (count + 1.0)
        logits[j] = math.log(omega / (1.0 - omega))
    return logits


def cmd_select() -> None:
    bars = _bars()
    dev = _load_dev()
    events = event_counts(0, DEV_WEEKS)
    totals = {name: dev[name].sum(1, dtype=np.float64) for name in COUNTS}
    f, precision, recall = f_score(totals["scored"], totals["correct"], totals["caught10"], totals["caught20"],
                                   events[0].sum(), events[1].sum())
    eligible = precision >= bars["dev"]
    days = DEV_WEEKS * 7
    summary: dict = {"bar_dev_precision": bars["dev"], "trials": len(f), "eligible": int(eligible.sum()),
                     "events_dev": [float(events[0].sum()), float(events[1].sum())],
                     "max_dev_precision": float(precision.max()), "max_dev_recall": float(recall.max())}
    if eligible.any():
        best = int(np.flatnonzero(eligible)[np.argmax(f[eligible])])
        config = _describe(dev, best)
        check = weekly_of(config, 0, DEV_WEEKS)
        check_f = f_score(check["scored"].sum(), check["correct"].sum(), check["caught10"].sum(),
                          check["caught20"].sum(), events[0].sum(), events[1].sum())[0]
        assert abs(float(check_f) - float(f[best])) < 1e-6, (check_f, f[best])
        edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
        masks = np.array([[b in combo for b in range(CSCV_BLOCKS)]
                          for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2)])
        block = {name: np.add.reduceat(dev[name].astype(np.float64), edges[:-1], axis=1) for name in COUNTS[:4]}
        logits = _cscv(block["scored"], block["correct"], block["caught10"], block["caught20"],
                       np.add.reduceat(events[0], edges[:-1]), np.add.reduceat(events[1], edges[:-1]), masks, BETA,
                       bars["dev"])
        valid_logits = logits[~np.isnan(logits)]
        summary.update({
            "chosen": config, "dev_f": float(f[best]), "dev_precision": float(precision[best]),
            "dev_recall": float(recall[best]), "dev_alerts_per_day": float(totals["alerts"][best] / days),
            "pbo": float(np.mean(valid_logits <= 0)), "pbo_splits": int(len(valid_logits)),
        })
    order = np.argsort(-np.where(eligible, f, -1.0))
    summary["top30"] = [{**_describe(dev, int(i)), "f": float(f[i]), "precision": float(precision[i]),
                         "recall": float(recall[i]), "alerts_per_day": float(totals["alerts"][i] / days)}
                        for i in order[:30] if eligible[i]]
    # 准确率最高的几组（不论召回），只供解读：看这个族里准确率能到多高、代价是什么。
    busy = totals["scored"] >= DEV_WEEKS  # 平均每周至少 1 条可计分的提醒
    by_precision = np.argsort(-np.where(busy, precision, -1.0))
    summary["most_precise"] = [{**_describe(dev, int(i)), "f": float(f[i]), "precision": float(precision[i]),
                                "recall": float(recall[i]), "alerts_per_day": float(totals["alerts"][i] / days)}
                               for i in by_precision[:10]]
    print(json.dumps({k: v for k, v in summary.items() if k not in ("top30", "most_precise")}, ensure_ascii=False,
                     indent=1), flush=True)
    (OUT / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def weekly_of(config: dict, first_week: int, n_weeks: int) -> dict:
    rec = _load(config["W"])
    alert = alerts_for(rec, config)
    week = rec["week"] - first_week
    inside = alert & (week >= 0) & (week < n_weeks)
    valid = rec["valid"][inside]
    ok = valid[:, 0] >= 0
    out = {
        "alerts": np.bincount(week[inside], minlength=n_weeks).astype(np.float64),
        "scored": np.bincount(week[inside][ok], minlength=n_weeks).astype(np.float64),
        "correct": np.bincount(week[inside][ok], weights=rec["precision"][inside][ok], minlength=n_weeks),
        "valid": valid,
        "rec": rec,
        "alert": alert,
    }
    for tag in (10, 20):
        caught = np.zeros(n_weeks)
        seen = set()
        for i in np.flatnonzero(inside):
            event = rec[f"event{tag}"][i]
            if event < 0 or (int(rec["sym"][i]), int(event)) in seen:
                continue
            seen.add((int(rec["sym"][i]), int(event)))
            w = rec[f"event_week{tag}"][i] - first_week
            if 0 <= w < n_weeks:
                caught[w] += 1
        out[f"caught{tag}"] = caught
    return out


def _data_weeks() -> int:
    last = 0
    for path in OUT.glob("cand_W*.npz"):
        with np.load(path) as data:
            last = max(last, int(data["t"].max()))
    return (last * 60 + 60) // WEEK + 1


def _example(run: dict, symbol: str, start: str, end: str) -> list[str]:
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    rec, alert = run["rec"], run["alert"]
    lo = (int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp()) - EPOCH) // 60
    hi = (int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp()) - EPOCH) // 60
    keep = np.flatnonzero(alert & (rec["sym"] == symbols.index(symbol)) & (rec["t"] >= lo) & (rec["t"] < hi))
    return [f"{datetime.fromtimestamp(EPOCH + int(rec['t'][i]) * 60 + 60 + 8 * 3600, UTC):%m-%d %H:%M} 北京 "
            f"{rec['price'][i]:.5g} 涨 {rec['change'][i]:.2f}% 成交额 {rec['quote'][i]:.0f} "
            f"之前24h {rec['pre'][i]:.0f}" for i in keep]


def cmd_evaluate() -> None:
    bars = _bars()
    selected = json.loads((OUT / "select.json").read_text(encoding="utf-8"))
    if "chosen" not in selected:
        print("开发期没有满足准确率下限的候选，没有方案")
        return
    config = selected["chosen"]
    n_weeks = _data_weeks() - DEV_WEEKS
    events = event_counts(DEV_WEEKS, n_weeks)
    run = weekly_of(config, DEV_WEEKS, n_weeks)
    f, p, r = f_score(run["scored"].sum(), run["correct"].sum(), run["caught10"].sum(), run["caught20"].sum(),
                      events[0].sum(), events[1].sum())
    weights = _bootstrap_weights(n_weeks, BOOTSTRAP, seed=2210).astype(np.float64)
    p_star = np.divide(weights @ run["correct"], weights @ run["scored"], out=np.zeros(BOOTSTRAP),
                       where=weights @ run["scored"] > 0)
    valid = run["valid"][run["valid"][:, 0] >= 0]
    by_setting = valid.mean(0) if len(valid) else np.zeros(4)
    random = np.load(OUT / "random.npy")
    result = {
        "config": config,
        "oos_weeks": n_weeks,
        "events": [float(events[0].sum()), float(events[1].sum())],
        "f": float(f), "precision": float(p), "recall": float(r),
        "recall_by_gain": [float(run["caught10"].sum() / events[0].sum()),
                           float(run["caught20"].sum() / events[1].sum())],
        "precision_by_setting": [float(v) for v in by_setting],
        "bar_oos_precision": bars["oos"], "bar_oos_by_setting": bars["oos_by_setting"],
        "bootstrap_p_precision_gt_bar": float(np.mean(p_star <= bars["oos"])),
        "settings_not_below_bar": int(sum(a >= b for a, b in zip(by_setting, bars["oos_by_setting"], strict=True))),
        "alerts_per_day": float(run["alerts"].sum() / (n_weeks * 7)),
        "scored_alerts": int(len(valid)),
        "random_precision": {"dev": float(random[0, :4].mean() / random[0, 4]),
                             "oos": float(random[1, :4].mean() / random[1, 4])},
        "irys_20261021": _example(run, "IRYS_USDT", "2026-10-21T06:00", "2026-10-21T10:00"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 对拍


def _reference(series, cum_uncovered, begin: int, end: int, config: dict) -> set[int]:
    """逐分钟直接按 dormant_solo.md 第 3 节计算的提醒时刻（K 线下标），不经过抽取与网格，用来核对两者。"""
    window = config["W"]
    need = _active_need(window, config["active_share"])
    last = None
    out = set()
    for t in range(begin, end):
        first = t - window + 1 - PRE
        if first < 0 or cum_uncovered[t + 1] - cum_uncovered[first] != 0:
            continue
        pre_q = float(series.quote[first : t - window + 1].sum())
        win_q = float(series.quote[t - window + 1 : t + 1].sum())
        start = series.close[t - window]
        change = (series.close[t] / start - 1) * 100
        pre_high = series.high[first : t - window + 1].max()
        pre_low = series.low[first : t - window + 1].min()
        win_high = series.high[t - window + 1 : t + 1].max()
        ratio = win_q / (pre_q / PRE * window) if pre_q > 0 else math.inf
        if (pre_q > config["gate"] or win_q < config["quote"] or change < config["change"] or ratio < config["ratio"]
                or (config["breakout"] and not series.close[t] > pre_high)
                or int(np.sum(series.quote[t - window + 1 : t + 1] > 0)) < need
                or (win_high - series.close[t]) / (win_high - start) > config["retrace"]
                or pre_high / pre_low - 1 > config["range"]):
            continue
        if last is not None and t - last < config["cooldown"]:
            continue
        last = t
        if change <= config["max_change"]:
            out.add(t)
    return out


def cmd_verify() -> None:
    contracts = load_contracts()
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    cases = [
        dict(W=5, quote=5e3, change=3.0, max_change=20.0, ratio=5.0, breakout=0.0, active_share=0.5,
             retrace=math.inf, range=math.inf, gate=1e6, cooldown=60.0),
        dict(W=3, quote=3e3, change=5.0, max_change=math.inf, ratio=0.0, breakout=1.0, active_share=1.0,
             retrace=0.5, range=0.10, gate=3e5, cooldown=360.0),
        dict(W=15, quote=1e4, change=3.0, max_change=20.0, ratio=20.0, breakout=0.0, active_share=0.0,
             retrace=math.inf, range=math.inf, gate=1e6, cooldown=60.0),
    ]
    span_minutes = 30 * 1440
    mismatches = 0
    total = 0
    for config in cases:
        config = {**config, "active": float(_active_need(config["W"], config["active_share"]))}
        rec_all = _load(config["W"])
        # 候选分钟最多的 4 个合约（冷门币），各取其候选最密的 30 天。
        busiest = np.argsort(-np.bincount(rec_all["sym"], minlength=len(symbols)))[:4]
        for sym_id in busiest:
            symbol = symbols[sym_id]
            prepared = _prepare(symbol, _multiplier(contracts[symbol]))
            series, _, _, cum_uncovered = prepared
            offset = (series.t0 - EPOCH) // 60
            mine = rec_all["t"][rec_all["sym"] == sym_id] - offset
            begin = int(np.median(mine)) - span_minutes // 2
            begin = max(begin, PRE + config["W"])
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


def cmd_selftest() -> None:
    """手造价格检查冷门启动行情的切分。"""
    flat = np.full(PRE, 1.0)
    close = np.concatenate([flat, [1.0, 1.05, 1.12, 1.10, 1.25, 1.18, 1.10]])
    quote = np.concatenate([np.full(PRE, 100.0), np.full(7, 5e4)])
    cum_quote = np.concatenate([[0.0], np.cumsum(quote)])
    cum_uncovered = np.zeros(len(close) + 1, np.int64)
    starts, ends = _dormant_legs(close, cum_quote, cum_uncovered, 0.20)
    # 横盘取最后一个 1.0（下标 PRE）为起点；1.0 → 1.25 涨 25%，回落到 1.10 跌 12% ≥ 10% 结束；
    # 段内成交额 4 × 5 万 = 20 万；起点及之前共 24 小时成交额 1439 × 100 + 5 万 ≈ 19.4 万。
    assert list(starts) == [PRE] and list(ends) == [PRE + 4], (starts, ends)
    loud = quote.copy()
    loud[:PRE] = 1e4  # 之前 24 小时 1440 万，不冷门
    cum_loud = np.concatenate([[0.0], np.cumsum(loud)])
    assert len(_dormant_legs(close, cum_loud, cum_uncovered, 0.20)[0]) == 0
    print("自测通过")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "extract", "verify", "search", "select", "evaluate"))
    command = parser.parse_args().command
    {"selftest": cmd_selftest, "extract": cmd_extract, "verify": cmd_verify, "search": cmd_search,
     "select": cmd_select, "evaluate": cmd_evaluate}[command]()


if __name__ == "__main__":
    main()
