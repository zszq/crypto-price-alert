"""冷门启动：涨前平稳、涨幅是涨前波动的很多倍，口径与判据见 dormant_calm.md（跑数前写定）。

冷门启动行情、冷门随机时刻直接用 dormant_solo 的抽取结果（定义完全相同），本脚本只换检测器的候选族。

用法（仓库根目录，$P 见 README.md；先跑过 dormant_solo.py extract）：
  & $P research/launch_trade/dormant_calm.py extract    # 候选分钟与涨前波动、倍数、单分钟涨幅
  & $P research/launch_trade/dormant_calm.py verify     # 网格判定与逐分钟直接实现对拍
  & $P research/launch_trade/dormant_calm.py search     # 开发期全量网格，逐周计数
  & $P research/launch_trade/dormant_calm.py select     # 基准、开发期选参、PBO
  & $P research/launch_trade/dormant_calm.py evaluate   # 选中方案与基准的样本外、例子
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
from dormant_solo import (
    BETA,
    PRE,
    _active_need,
    _bars,
    _cscv,
    _dormant_legs,
    _prepare,
    event_counts,
    f_score,
)
from dormant_solo import OUT as SOLO
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
from launch_grid import EPOCH, _multiplier, _rolling_extreme
from load_data import load_contracts
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "dormant_calm"
GATE = 1e6
COOLDOWN = 60.0

WINDOWS = (10, 15, 20, 30)
QUOTES = (5e3, 1e4, 2e4, 5e4)
CHANGES = (5.0, 8.0, 12.0)
ACTIVE_SHARES = (0.0, 0.5, 1.0)
SPANS = (60, 240)  # 涨前波动的统计长度 N
CALMS = (math.inf, 0.5, 0.2)
MULTIPLES = (5.0, 10.0, 20.0, 50.0)
JUMPS = (3.0, 5.0, 8.0)
# span 存的是 SPANS 的下标，numba 里按下标取对应的特征列。
PARAM_NAMES = ("quote", "change", "active", "span", "calm", "multiple", "jump", "cooldown")
COUNTS = ("scored", "correct", "caught10", "caught20", "alerts")
# 现行冷门启动默认参数（dormant_solo 选中方案），不带本轮三项：倍数 0、上限无穷即不限。
BASELINE = {"W": 10, "quote": 2e4, "change": 8.0, "active": 5.0, "span": 0.0, "calm": math.inf, "multiple": 0.0,
            "jump": math.inf, "cooldown": COOLDOWN}


def param_grid(window: int) -> np.ndarray:
    rows = [(q, c, _active_need(window, a), s, f, k, j, COOLDOWN)
            for q, c, a, s, f, k, j in itertools.product(
                QUOTES, CHANGES, ACTIVE_SHARES, range(len(SPANS)), CALMS, MULTIPLES, JUMPS)]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------- 抽取


def _minute_moves(high, low, close, covered):
    """每分钟真实波幅（%）与收盘涨幅（%）；缺数据的分钟及其后一分钟记 0，免得 NaN 污染累计和与滚动最大值。"""
    prev = np.concatenate([[np.nan], close[:-1]])
    true_range = (np.maximum(high, prev) - np.minimum(low, prev)) / prev * 100
    change = (close / prev - 1) * 100
    ok = covered & np.concatenate([[False], covered[:-1]])
    true_range = np.where(ok, np.nan_to_num(true_range), 0.0)
    change = np.where(ok, np.nan_to_num(change), 0.0)
    return true_range, change


@njit(cache=True)
def _extract(close, cum_quote, cum_active, cum_uncovered, cum_tr, jump_max, window, min_quote, min_change, gate,
             spans):
    n = len(close)
    keep = np.zeros(n, np.bool_)
    for t in range(window + PRE, n):
        first = t - window + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:  # 含窗口前 24 小时第一根的前收
            continue
        if cum_quote[t - window + 1] - cum_quote[first] > gate:
            continue
        if cum_quote[t + 1] - cum_quote[t - window + 1] < min_quote:
            continue
        if (close[t] / close[t - window] - 1) * 100 < min_change:
            continue
        keep[t] = True
    index = np.flatnonzero(keep)
    m = len(index)
    change = np.empty(m)
    quote = np.empty(m)
    active = np.empty(m, np.int16)
    pre = np.empty(m)
    jump = np.empty(m)
    vol = np.empty((m, len(spans)))
    multiple = np.empty((m, len(spans)))
    for k in range(m):
        t = index[k]
        s = t - window  # 窗口开始前一根
        change[k] = (close[t] / close[s] - 1) * 100
        quote[k] = cum_quote[t + 1] - cum_quote[t - window + 1]
        active[k] = cum_active[t + 1] - cum_active[t - window + 1]
        pre[k] = cum_quote[t - window + 1] - cum_quote[t - window + 1 - PRE]
        jump[k] = jump_max[t]
        for j in range(len(spans)):
            v = (cum_tr[s + 1] - cum_tr[s + 1 - spans[j]]) / spans[j]
            vol[k, j] = v
            multiple[k, j] = change[k] / v if v > 0 else np.inf
    return index, change, quote, active, pre, jump, vol, multiple


def _week(index: np.ndarray, offset: int) -> np.ndarray:
    return ((index + offset) * 60 + 60) // WEEK  # 提醒时刻 = K 线开盘 + 1 分钟


def _features(symbol: str, multiplier: float):
    prepared = _prepare(symbol, multiplier)
    if prepared is None:
        return None
    series, cum_quote, cum_active, cum_uncovered = prepared
    covered = (cum_uncovered[1:] - cum_uncovered[:-1]) == 0
    true_range, minute_change = _minute_moves(series.high, series.low, series.close, covered)
    cum_tr = np.concatenate([[0.0], np.cumsum(true_range)])
    return series, cum_quote, cum_active, cum_uncovered, cum_tr, minute_change


def _extract_symbol(args: tuple[int, str, float]) -> tuple[int, dict]:
    sym_id, symbol, multiplier = args
    out: dict = {}
    prepared = _features(symbol, multiplier)
    if prepared is None:
        return sym_id, out
    series, cum_quote, cum_active, cum_uncovered, cum_tr, minute_change = prepared
    high, low, close = series.high, series.low, series.close
    offset = (series.t0 - EPOCH) // 60
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    # 冷门启动行情与 dormant_solo 相同，这里重算一遍只为拿到本合约各段的下标。
    legs = {gain: _dormant_legs(close, cum_quote, cum_uncovered, gain) for gain in GAINS}
    spans = np.array(SPANS, np.int64)
    for window in WINDOWS:
        jump_max = _rolling_extreme(minute_change, window, True)
        index, change, quote, active, pre, jump, vol, multiple = _extract(
            close, cum_quote, cum_active, cum_uncovered, cum_tr, jump_max, window, QUOTES[0], CHANGES[0], GATE, spans)
        ones = np.ones(len(index), np.int8)
        record = {"t": (index + offset).astype(np.int64), "price": close[index], "change": change, "quote": quote,
                  "active": active, "pre": pre, "jump": jump, "vol": vol, "multiple": multiple,
                  "valid": _validity(high, low, close, cum_uncovered, index, ones, targets, holds, FORWARD)}
        for gain in GAINS:
            starts, ends = legs[gain]
            event = _early_event(close, index, ones, starts, ends, np.ones(len(starts), np.int8))
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
    started = time.time()
    with ProcessPoolExecutor(max_workers=14) as executor:
        for done, (sym_id, result) in enumerate(executor.map(_extract_symbol, jobs, chunksize=2), 1):
            if result:
                for window in WINDOWS:
                    fields = result[window]
                    fields["sym"] = np.full(len(fields["t"]), sym_id, np.int32)
                    parts[window].append(fields)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    for window, chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        np.savez(OUT / f"cand_W{window}.npz", **merged)
        print(f"W{window}: {len(merged['t'])} 条候选", flush=True)


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
def _passes(i, p, quote, change, active, vol, multiple, jump):
    s = int(p[3])
    return not (quote[i] < p[0] or change[i] < p[1] or active[i] < p[2] or vol[i, s] > p[4]
                or multiple[i, s] < p[5] or jump[i] > p[6])


@njit(parallel=True, cache=True)
def _grid(seg, t, week, quote, change, active, vol, multiple, jump, precision, event10, event20, event_week10,
          event_week20, params, n_weeks, scored, correct, caught10, caught20, alerts):
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
                    if not _passes(i, p, quote, change, active, vol, multiple, jump):
                        continue
                    k = c - c0
                    if last_t[k] >= 0 and t[i] - last_t[k] < p[7]:
                        continue
                    last_t[k] = t[i]
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
def _alert_mask(seg, t, quote, change, active, vol, multiple, jump, p):
    alert = np.zeros(len(t), np.bool_)
    for s in range(len(seg) - 1):
        last_t = -1
        for i in range(seg[s], seg[s + 1]):
            if not _passes(i, p, quote, change, active, vol, multiple, jump):
                continue
            if last_t >= 0 and t[i] - last_t < p[7]:
                continue
            last_t = t[i]
            alert[i] = True
    return alert


def alerts_for(rec: dict, config: dict) -> np.ndarray:
    row = np.array([config[name] for name in PARAM_NAMES], dtype=np.float64)
    return _alert_mask(rec["seg"], rec["t"], rec["quote"], rec["change"], rec["active"], rec["vol"],
                       rec["multiple"], rec["jump"], row)


def cmd_search() -> None:
    for window in WINDOWS:
        started = time.time()
        params = param_grid(window)
        rec = _load(window)
        shape = (len(params), DEV_WEEKS)
        arrays = {name: np.zeros(shape, np.float32 if name == "correct" else np.int32) for name in COUNTS}
        _grid(rec["seg"], rec["t"], rec["week"], rec["quote"], rec["change"], rec["active"], rec["vol"],
              rec["multiple"], rec["jump"], rec["precision"], rec["event10"], rec["event20"], rec["event_week10"],
              rec["event_week20"], params, DEV_WEEKS, arrays["scored"], arrays["correct"], arrays["caught10"],
              arrays["caught20"], arrays["alerts"])
        np.savez(OUT / f"dev_W{window}.npz", params=params, **arrays)
        print(f"W{window}: {len(rec['t'])} 条记录 {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------- 选参与检验


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


def _summary(run: dict, events: np.ndarray, days: float) -> dict:
    f, p, r = f_score(run["scored"].sum(), run["correct"].sum(), run["caught10"].sum(), run["caught20"].sum(),
                      events[0].sum(), events[1].sum())
    valid = run["valid"][run["valid"][:, 0] >= 0]
    return {"f": float(f), "precision": float(p), "recall": float(r),
            "precision_by_setting": [float(v) for v in (valid.mean(0) if len(valid) else np.zeros(4))],
            "alerts_per_day": float(run["alerts"].sum() / days), "scored_alerts": int(len(valid))}


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


def cmd_select() -> None:
    dev = _load_dev()
    events = event_counts(0, DEV_WEEKS)
    days = DEV_WEEKS * 7
    base = _summary(weekly_of(BASELINE, 0, DEV_WEEKS), events, days)
    bar = base["precision"]
    totals = {name: dev[name].sum(1, dtype=np.float64) for name in COUNTS}
    f, precision, recall = f_score(totals["scored"], totals["correct"], totals["caught10"], totals["caught20"],
                                   events[0].sum(), events[1].sum())
    eligible = precision >= bar
    summary: dict = {"baseline_dev": base, "bar_dev_precision": bar, "trials": len(f), "eligible": int(eligible.sum()),
                     "max_dev_precision": float(precision.max())}
    if eligible.any():
        best = int(np.flatnonzero(eligible)[np.argmax(f[eligible])])
        config = _describe(dev, best)
        check = _summary(weekly_of(config, 0, DEV_WEEKS), events, days)
        assert abs(check["f"] - float(f[best])) < 1e-6, (check["f"], f[best])
        edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
        masks = np.array([[b in combo for b in range(CSCV_BLOCKS)]
                          for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2)])
        block = {name: np.add.reduceat(dev[name].astype(np.float64), edges[:-1], axis=1) for name in COUNTS[:4]}
        logits = _cscv(block["scored"], block["correct"], block["caught10"], block["caught20"],
                       np.add.reduceat(events[0], edges[:-1]), np.add.reduceat(events[1], edges[:-1]), masks, BETA,
                       bar)
        valid_logits = logits[~np.isnan(logits)]
        summary.update({"chosen": config, "chosen_dev": check, "pbo": float(np.mean(valid_logits <= 0)),
                        "pbo_splits": int(len(valid_logits))})
    order = np.argsort(-np.where(eligible, f, -1.0))
    summary["top30"] = [{**_describe(dev, int(i)), "f": float(f[i]), "precision": float(precision[i]),
                         "recall": float(recall[i]), "alerts_per_day": float(totals["alerts"][i] / days)}
                        for i in order[:30] if eligible[i]]
    # 准确率最高的几组（每周至少 1 条可计分提醒），只供解读。
    busy = totals["scored"] >= DEV_WEEKS
    by_precision = np.argsort(-np.where(busy, precision, -1.0))
    summary["most_precise"] = [{**_describe(dev, int(i)), "f": float(f[i]), "precision": float(precision[i]),
                                "recall": float(recall[i]), "alerts_per_day": float(totals["alerts"][i] / days)}
                               for i in by_precision[:10]]
    print(json.dumps({k: v for k, v in summary.items() if k not in ("top30", "most_precise")}, ensure_ascii=False,
                     indent=1), flush=True)
    (OUT / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _data_weeks() -> int:
    last = 0
    for path in OUT.glob("cand_W*.npz"):
        with np.load(path) as data:
            last = max(last, int(data["t"].max()))
    return (last * 60 + 60) // WEEK + 1


def _example(run: dict, symbol: str, start: str, end: str) -> list[str]:
    """区间内的提醒（start、end 为 UTC）。"""
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    rec, alert = run["rec"], run["alert"]
    lo = (int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp()) - EPOCH) // 60
    hi = (int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp()) - EPOCH) // 60
    keep = np.flatnonzero(alert & (rec["sym"] == symbols.index(symbol)) & (rec["t"] >= lo) & (rec["t"] < hi))
    return [f"{datetime.fromtimestamp(EPOCH + int(rec['t'][i]) * 60 + 60 + 8 * 3600, UTC):%m-%d %H:%M} 北京 "
            f"{rec['price'][i]:.5g} 涨 {rec['change'][i]:.2f}% 成交额 {rec['quote'][i]:.0f}" for i in keep]


def cmd_evaluate() -> None:
    selected = json.loads((OUT / "select.json").read_text(encoding="utf-8"))
    if "chosen" not in selected:
        print("开发期没有满足准确率下限的候选，没有方案")
        return
    config = selected["chosen"]
    n_weeks = _data_weeks() - DEV_WEEKS
    days = n_weeks * 7
    events = event_counts(DEV_WEEKS, n_weeks)
    base_run = weekly_of(BASELINE, DEV_WEEKS, n_weeks)
    base = _summary(base_run, events, days)
    run = weekly_of(config, DEV_WEEKS, n_weeks)
    chosen = _summary(run, events, days)
    weights = _bootstrap_weights(n_weeks, BOOTSTRAP, seed=2210).astype(np.float64)
    p_star = np.divide(weights @ run["correct"], weights @ run["scored"], out=np.zeros(BOOTSTRAP),
                       where=weights @ run["scored"] > 0)
    examples = {}
    for name, span in {"IRYS_USDT": ("2026-10-21T05:30", "2026-10-21T10:00"),
                       "ORBIO_USDT": ("2026-10-20T16:00", "2026-10-22T05:00")}.items():
        examples[name] = {"chosen": _example(run, name, *span), "baseline": _example(base_run, name, *span)}
    result = {
        "config": config, "oos_weeks": n_weeks, "events": [float(events[0].sum()), float(events[1].sum())],
        "chosen": chosen, "baseline": base,
        "bootstrap_p_precision_gt_baseline": float(np.mean(p_star <= base["precision"])),
        "settings_not_below_baseline": int(sum(a >= b for a, b in zip(
            chosen["precision_by_setting"], base["precision_by_setting"], strict=True))),
        "launch_rise_oos_precision": _bars()["oos"],
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 对拍


def _reference(series, cum_uncovered, begin: int, end: int, config: dict) -> set[int]:
    """逐分钟直接按 dormant_calm.md 第一节计算的提醒时刻（K 线下标），不经过抽取与网格。"""
    window = config["W"]
    span = SPANS[int(config["span"])]
    high, low, close, quote = series.high, series.low, series.close, series.quote
    last = None
    out = set()
    for t in range(begin, end):
        first = t - window + 1 - PRE
        if first < 1 or cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:
            continue
        s = t - window
        pre_q = float(quote[first : s + 1].sum())
        win_q = float(quote[s + 1 : t + 1].sum())
        change = (close[t] / close[s] - 1) * 100
        ranges = [(max(high[k], close[k - 1]) - min(low[k], close[k - 1])) / close[k - 1] * 100
                  for k in range(s - span + 1, s + 1)]
        vol = sum(ranges) / span
        multiple = change / vol if vol > 0 else math.inf
        jump = max((close[k] / close[k - 1] - 1) * 100 for k in range(s + 1, t + 1))
        if (pre_q > GATE or win_q < config["quote"] or change < config["change"]
                or int(np.sum(quote[s + 1 : t + 1] > 0)) < config["active"] or vol > config["calm"]
                or multiple < config["multiple"] or jump > config["jump"]):
            continue
        if last is not None and t - last < config["cooldown"]:
            continue
        last = t
        out.add(t)
    return out


def cmd_verify() -> None:
    contracts = load_contracts()
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    cases = [
        dict(W=10, quote=5e3, change=5.0, active=5.0, span=0.0, calm=0.5, multiple=10.0, jump=5.0),
        dict(W=20, quote=1e4, change=5.0, active=1.0, span=1.0, calm=math.inf, multiple=5.0, jump=8.0),
        dict(W=30, quote=5e3, change=8.0, active=15.0, span=1.0, calm=0.2, multiple=20.0, jump=3.0),
        BASELINE,
    ]
    span_minutes = 30 * 1440
    mismatches = 0
    total = 0
    for config in cases:
        config = {"cooldown": COOLDOWN, **config}
        rec_all = _load(config["W"])
        busiest = np.argsort(-np.bincount(rec_all["sym"], minlength=len(symbols)))[:4]
        for sym_id in busiest:
            symbol = symbols[sym_id]
            series, _, _, cum_uncovered = _prepare(symbol, _multiplier(contracts[symbol]))
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
    assert (SOLO / "events.npz").exists(), "先跑 dormant_solo.py extract"
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "verify", "search", "select", "evaluate"))
    command = parser.parse_args().command
    {"extract": cmd_extract, "verify": cmd_verify, "search": cmd_search, "select": cmd_select,
     "evaluate": cmd_evaluate}[command]()


if __name__ == "__main__":
    main()
