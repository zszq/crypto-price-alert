"""放量上涨 / 放量下跌提醒参数全量搜索，口径与判据见 launch_grid.md（跑数前写定）。

用法（仓库根目录，$P 见 README.md）：
  & $P research/launch_trade/launch_grid.py extract    # 逐合约抽出各 (窗口, 基准期) 下的候选分钟
  & $P research/launch_trade/launch_grid.py verify     # 与真实 LaunchDetector 对拍提醒时刻
  & $P research/launch_trade/launch_grid.py search     # 开发期全量网格
  & $P research/launch_trade/launch_grid.py select     # 开发期选参、Deflated Sharpe、PBO
  & $P research/launch_trade/launch_grid.py evaluate   # 选中方案与现行配置的样本外

只有「量比、窗口成交额、涨跌幅都达到候选族最宽门槛」的分钟会影响判定与冷却状态（其余分钟在任何候选下都判定失败、
不改状态），所以先把这些分钟抽成稀疏记录，网格只在记录上跑冷却与再推进，不必对每组参数重扫全部分钟。
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
from statistics import NormalDist

import numpy as np
from load_data import load_contracts, load_series
from numba import njit, prange
from paths import RESULTS

OUT = RESULTS / "launch_grid"
EPOCH = 1630454400  # 2021-09-01 00:00 UTC，下载站最早的月份
DEV_END = 1735689600  # 2025-01-01 00:00 UTC：之前为开发期
DAY = 86400

WINDOWS = (3, 5, 10, 15)
BASELINES = (180, 360, 720, 1440)
RATIOS = (2.0, 3.0, 5.0, 8.0)
QUOTES = (5e4, 1e5, 3e5, 1e6)
CHANGES = (1.5, 2.0, 3.0, 5.0)
MAXES = (8.0, 15.0, math.inf)
BREAKOUTS = (0.0, 1.0)
VOLUMES = (0.0, 1e6, 5e6)
COOLDOWNS = (15.0, 60.0)
STEPS = (3.0, 8.0)
RETRACES = (math.inf, 0.5)
PARAM_NAMES = ("ratio", "quote", "change", "max_change", "breakout", "volume_24h", "cooldown", "step", "retrace")

DELAYS = (0, 1, 3)
HORIZONS = (10, 30, 60, 120, 240)
FORWARD = max(DELAYS) + max(HORIZONS)
CLIP = 0.3
COST = 0.002
MIN_ALERTS_PER_DAY = 2.0
CSCV_BLOCKS = 16
JUMP_RATIO = 3.0

# 现行 default.yaml 的取值（对照，不参与挑选）；网格外的冷却 10 分钟由单组评估直接计算。
CURRENT = {
    "surge": dict(W=5, B=360, ratio=3.0, quote=1e5, change=3.0, max_change=15.0, breakout=0.0,
                  volume_24h=5e5, cooldown=10.0, step=5.0, retrace=math.inf),
    "drop": dict(W=5, B=360, ratio=4.0, quote=1e5, change=2.0, max_change=10.0, breakout=0.0,
                 volume_24h=5e5, cooldown=10.0, step=5.0, retrace=math.inf),
}
SIGNS = {"surge": 1, "drop": -1}


def _multiplier(item: dict) -> float | None:
    try:
        value = float(item.get("quanto_multiplier"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# ---------------------------------------------------------------- 抽取候选分钟


@njit(cache=True)
def _rolling_extreme(values, width, is_max):
    """out[j] = values[j-width+1 .. j] 的最大（最小）值，单调队列 O(n)。"""
    n = len(values)
    out = np.empty(n)
    queue = np.empty(n, np.int64)
    head = 0
    tail = 0
    for j in range(n):
        v = values[j]
        while tail > head and ((values[queue[tail - 1]] <= v) if is_max else (values[queue[tail - 1]] >= v)):
            tail -= 1
        queue[tail] = j
        tail += 1
        if queue[head] <= j - width:
            head += 1
        out[j] = values[queue[head]]
    return out


@njit(cache=True)
def _extract(high, low, close, cum_quote, cum_uncovered, window, baseline, base_high, base_low, win_high, win_low,
             min_ratio, min_quote, min_change, delays, horizons, forward, clip, cost):
    n = len(close)
    keep = np.zeros(n, np.bool_)
    for t in range(window + baseline - 1, n):
        first = t - window - baseline + 1
        # 基准期 + 窗口必须全部有数据，与实盘预热拿到完整历史后才判定一致。
        if cum_uncovered[t + 1] - cum_uncovered[first] != 0:
            continue
        quote = cum_quote[t + 1] - cum_quote[t - window + 1]
        if quote < min_quote:
            continue
        change = (close[t] / close[t - window] - 1) * 100
        if abs(change) < min_change:
            continue
        base = (cum_quote[t - window + 1] - cum_quote[first]) / baseline * window
        if base > 0 and quote / base < min_ratio:
            continue
        keep[t] = True
    index = np.flatnonzero(keep)
    m = len(index)
    sign = np.empty(m, np.int8)
    change_abs = np.empty(m)
    ratio = np.empty(m)
    quote_out = np.empty(m)
    breakout = np.empty(m, np.bool_)
    retrace = np.empty(m)
    volume = np.empty(m)
    score = np.full(m, np.nan)
    by_delay = np.full((m, len(delays)), np.nan)
    for k in range(m):
        t = index[k]
        first = t - window - baseline + 1
        start = close[t - window]
        end = close[t]
        change = (end / start - 1) * 100
        s = 1 if change >= 0 else -1
        sign[k] = s
        change_abs[k] = abs(change)
        quote = cum_quote[t + 1] - cum_quote[t - window + 1]
        quote_out[k] = quote
        base = (cum_quote[t - window + 1] - cum_quote[first]) / baseline * window
        ratio[k] = quote / base if base > 0 else np.inf
        if s > 0:
            breakout[k] = end > base_high[t - window]
            extreme = win_high[t]
            retrace[k] = (extreme - end) / (extreme - start)
        else:
            breakout[k] = end < base_low[t - window]
            extreme = win_low[t]
            retrace[k] = (end - extreme) / (start - extreme)
        volume[k] = cum_quote[t + 1] - cum_quote[max(0, t - 1439)]
        if t + forward < n and cum_uncovered[t + forward + 1] - cum_uncovered[t] == 0:
            total = 0.0
            for di in range(len(delays)):
                entry = close[t + delays[di]]
                acc = 0.0
                for h in horizons:
                    r = s * (close[t + delays[di] + h] / entry - 1)
                    r = min(max(r, -clip), clip)
                    acc += r - cost
                by_delay[k, di] = acc / len(horizons)
                total += by_delay[k, di]
            score[k] = total / len(delays)
    return index, sign, change_abs, ratio, quote_out, breakout, retrace, volume, score, by_delay


def _last_jump(close: np.ndarray) -> int:
    """最后一次价格持久跳变的位置（与 check_jumps.py 同一判据），没有返回 0。"""
    step = np.abs(np.diff(np.log(close)))
    last = 0
    for k in np.flatnonzero(step > math.log(JUMP_RATIO)) + 1:
        if k < 1440 or k + 1440 > len(close):
            continue
        before = float(np.median(close[k - 1440 : k]))
        after = float(np.median(close[k : k + 1440]))
        if max(before, after) / min(before, after) > JUMP_RATIO:
            last = int(k)
    return last


def _extract_symbol(args: tuple[int, str, float]) -> tuple[int, dict]:
    sym_id, symbol, multiplier = args
    series = load_series(symbol, multiplier)
    result: dict = {}
    if series is None or len(series) < 2000:
        return sym_id, result
    covered = series.covered.copy()
    # 改面值之前的成交额按当前面值估算会错，整段不用。
    covered[: _last_jump(series.close)] = False
    cum_quote = np.concatenate([[0.0], np.cumsum(series.quote)])
    cum_uncovered = np.concatenate([[0], np.cumsum(~covered)]).astype(np.int64)
    offset = (series.t0 - EPOCH) // 60
    delays = np.array(DELAYS, np.int64)
    horizons = np.array(HORIZONS, np.int64)
    win_cache = {w: (_rolling_extreme(series.high, w, True), _rolling_extreme(series.low, w, False)) for w in WINDOWS}
    for baseline in BASELINES:
        base_high = _rolling_extreme(series.high, baseline, True)
        base_low = _rolling_extreme(series.low, baseline, False)
        for window in WINDOWS:
            win_high, win_low = win_cache[window]
            fields = _extract(series.high, series.low, series.close, cum_quote, cum_uncovered, window, baseline,
                              base_high, base_low, win_high, win_low, RATIOS[0], QUOTES[0], CHANGES[0],
                              delays, horizons, FORWARD, CLIP, COST)
            index = fields[0]
            result[(window, baseline)] = {
                "t": (index + offset).astype(np.int64),
                "price": series.close[index],
                "sign": fields[1],
                "change": fields[2],
                "ratio": fields[3],
                "quote": fields[4],
                "breakout": fields[5],
                "retrace": fields[6],
                "volume": fields[7],
                "score": fields[8].astype(np.float32),
                "by_delay": fields[9].astype(np.float32),
            }
    return sym_id, result


def cmd_extract() -> None:
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = [(i, s, _multiplier(contracts[s])) for i, s in enumerate(symbols) if _multiplier(contracts[s])]
    parts: dict[tuple[int, int], list] = {(w, b): [] for w in WINDOWS for b in BASELINES}
    started = time.time()
    with ProcessPoolExecutor(max_workers=14) as executor:
        for done, (sym_id, result) in enumerate(executor.map(_extract_symbol, jobs, chunksize=2), 1):
            for key, fields in result.items():
                fields["sym"] = np.full(len(fields["t"]), sym_id, np.int32)
                parts[key].append(fields)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    for (window, baseline), chunks in parts.items():
        merged = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        np.savez(OUT / f"cand_W{window}_B{baseline}.npz", **merged)
        print(f"W{window} B{baseline}: {len(merged['t'])} 条候选", flush=True)


# ---------------------------------------------------------------- 冷却、再推进与网格


def _load(window: int, baseline: int, direction: str) -> dict:
    with np.load(OUT / f"cand_W{window}_B{baseline}.npz") as data:
        mask = data["sign"] == SIGNS[direction]
        rec = {name: data[name][mask] for name in data.files}
    # 记录已按合约、时间排序（逐合约抽取后拼接）；seg 是各合约的起止下标。
    edges = np.flatnonzero(np.diff(rec["sym"])) + 1
    rec["seg"] = np.concatenate([[0], edges, [len(rec["sym"])]]).astype(np.int64)
    rec["day"] = ((rec["t"] * 60 + 60) // DAY).astype(np.int64)  # 提醒时刻 = K 线开盘 + 1 分钟
    return rec


@njit(parallel=True, cache=True)
def _grid(seg, t, day, price, change, ratio, quote, breakout, retrace, volume, score, params, baseline, sign,
          n_days, sum_score, n_scored, n_alerts):
    n_params = params.shape[0]
    block = 32
    for b in prange((n_params + block - 1) // block):
        c0 = b * block
        c1 = min(n_params, c0 + block)
        last_t = np.empty(block, np.int64)
        last_p = np.empty(block)
        for s in range(len(seg) - 1):
            for k in range(block):
                last_t[k] = -1
            for i in range(seg[s], seg[s + 1]):
                d = day[i]
                if d >= n_days:
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
                    # 只因超上限失败：按提醒记入冷却与再推进，但不发出（与 LaunchDetector._record_overshoot 一致）。
                    if change[i] > p[3]:
                        continue
                    n_alerts[c, d] += 1
                    if not np.isnan(score[i]):
                        sum_score[c, d] += score[i]
                        n_scored[c, d] += 1


def _run(rec: dict, params: np.ndarray, baseline: int, direction: str, first_day: int, n_days: int):
    """返回每组参数 [first_day, first_day + n_days) 内逐日的得分和、计分提醒数、提醒数。"""
    shape = (len(params), n_days)
    sums, scored, alerts = np.zeros(shape), np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    _grid(rec["seg"], rec["t"], rec["day"] - first_day, rec["price"], rec["change"], rec["ratio"], rec["quote"],
          rec["breakout"], rec["retrace"], rec["volume"], rec["score"], params, baseline, SIGNS[direction],
          n_days, sums, scored, alerts)
    return sums, scored, alerts


@njit(cache=True)
def _alert_mask(seg, t, price, change, ratio, quote, breakout, retrace, volume, p, baseline, sign):
    """单组参数逐条记录的判定：1 提醒，0 不提醒；wave 为波次。"""
    n = len(t)
    alert = np.zeros(n, np.bool_)
    wave = np.zeros(n, np.int32)
    for s in range(len(seg) - 1):
        last_t = -1
        last_p = 0.0
        last_wave = 0
        for i in range(seg[s], seg[s + 1]):
            if (ratio[i] < p[0] or quote[i] < p[1] or change[i] < p[2] or (p[4] > 0 and not breakout[i])
                    or volume[i] <= p[5] or retrace[i] > p[8]):
                continue
            current = 1
            if last_t >= 0:
                elapsed = t[i] - last_t
                if elapsed < p[6]:
                    continue
                if elapsed < baseline:
                    if sign * (price[i] / last_p - 1) * 100 < p[7]:
                        continue
                    current = last_wave + 1
            last_t = t[i]
            last_p = price[i]
            if change[i] > p[3]:
                last_wave = current - 1
                continue
            last_wave = current
            alert[i] = True
            wave[i] = current
    return alert, wave


def alerts_of(rec: dict, params: np.ndarray, baseline: int, direction: str):
    return _alert_mask(rec["seg"], rec["t"], rec["price"], rec["change"], rec["ratio"], rec["quote"],
                       rec["breakout"], rec["retrace"], rec["volume"], params, baseline, SIGNS[direction])


def param_grid() -> np.ndarray:
    rows = [row for row in itertools.product(RATIOS, QUOTES, CHANGES, MAXES, BREAKOUTS, VOLUMES, COOLDOWNS, STEPS,
                                             RETRACES) if row[3] > row[2]]
    return np.array(rows, dtype=np.float64)


DEV_DAYS = (DEV_END - EPOCH) // DAY


def _block_edges(n_days: int) -> np.ndarray:
    return np.linspace(0, n_days, CSCV_BLOCKS + 1).round().astype(np.int64)


def cmd_search() -> None:
    params = param_grid()
    edges = _block_edges(DEV_DAYS)
    for direction in ("surge", "drop"):
        for window in WINDOWS:
            for baseline in BASELINES:
                started = time.time()
                rec = _load(window, baseline, direction)
                sums, scored, alerts = _run(rec, params, baseline, direction, 0, DEV_DAYS)
                daily = np.divide(sums, scored, out=np.zeros_like(sums), where=scored > 0)
                mean = daily.mean(1)
                std = daily.std(1, ddof=1)
                total_scored = scored.sum(1)
                np.savez(
                    OUT / f"dev_{direction}_W{window}_B{baseline}.npz",
                    params=params,
                    sharpe=np.divide(mean, std, out=np.zeros_like(mean), where=std > 0),
                    alerts_per_day=alerts.sum(1) / DEV_DAYS,
                    mean_score=np.divide(sums.sum(1), total_scored, out=np.zeros_like(mean), where=total_scored > 0),
                    block_sum=np.add.reduceat(daily, edges[:-1], axis=1),
                    block_sq=np.add.reduceat(daily**2, edges[:-1], axis=1),
                    block_alerts=np.add.reduceat(alerts, edges[:-1], axis=1),
                    block_len=np.diff(edges),
                )
                print(f"{direction} W{window} B{baseline}: {len(rec['t'])} 条记录 {time.time() - started:.0f}s",
                      flush=True)


# ---------------------------------------------------------------- 选参与过拟合检验


def _load_dev(direction: str) -> dict:
    parts = []
    for window in WINDOWS:
        for baseline in BASELINES:
            with np.load(OUT / f"dev_{direction}_W{window}_B{baseline}.npz") as data:
                part = {name: data[name] for name in data.files}
            n = len(part["sharpe"])
            part["window"] = np.full(n, window)
            part["baseline"] = np.full(n, baseline)
            parts.append(part)
    merged = {name: np.concatenate([p[name] for p in parts]) for name in parts[0] if name != "block_len"}
    merged["block_len"] = parts[0]["block_len"]
    return merged


@njit(parallel=True, cache=True)
def _cscv(block_sum, block_sq, block_alerts, block_len, masks, min_rate):
    n_split = masks.shape[0]
    n = block_sum.shape[0]
    logits = np.full(n_split, np.nan)
    for j in prange(n_split):
        is_mask = masks[j]
        best = -1
        best_sr = -np.inf
        is_sr = np.empty(n)
        oos_sr = np.empty(n)
        eligible = np.zeros(n, np.bool_)
        for c in range(n):
            s_in = 0.0
            q_in = 0.0
            a_in = 0.0
            l_in = 0
            s_out = 0.0
            q_out = 0.0
            l_out = 0
            for b in range(block_sum.shape[1]):
                if is_mask[b]:
                    s_in += block_sum[c, b]
                    q_in += block_sq[c, b]
                    a_in += block_alerts[c, b]
                    l_in += block_len[b]
                else:
                    s_out += block_sum[c, b]
                    q_out += block_sq[c, b]
                    l_out += block_len[b]
            m_in = s_in / l_in
            v_in = (q_in - l_in * m_in * m_in) / (l_in - 1)
            m_out = s_out / l_out
            v_out = (q_out - l_out * m_out * m_out) / (l_out - 1)
            is_sr[c] = m_in / math.sqrt(v_in) if v_in > 0 else 0.0
            oos_sr[c] = m_out / math.sqrt(v_out) if v_out > 0 else 0.0
            eligible[c] = a_in / l_in >= min_rate
            if eligible[c] and is_sr[c] > best_sr:
                best_sr = is_sr[c]
                best = c
        if best < 0:
            continue
        below = 0.0
        count = 0
        for c in range(n):
            if eligible[c]:
                count += 1
                if oos_sr[c] < oos_sr[best]:
                    below += 1.0
                elif oos_sr[c] == oos_sr[best] and c != best:
                    below += 0.5
        omega = (below + 1.0) / (count + 1.0)
        logits[j] = math.log(omega / (1.0 - omega))
    return logits


def _moments(x: np.ndarray) -> tuple[float, float, float]:
    """日夏普（不年化）、偏度、峰度（非超额）。"""
    std = x.std(ddof=1)
    sr = x.mean() / std if std > 0 else 0.0
    z = (x - x.mean()) / x.std()
    return float(sr), float((z**3).mean()), float((z**4).mean())


def psr(x: np.ndarray, benchmark: float = 0.0) -> float:
    sr, skew, kurt = _moments(x)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return NormalDist().cdf((sr - benchmark) * math.sqrt(len(x) - 1) / denom)


def expected_max_sharpe(variance: float, trials: int) -> float:
    gamma = 0.5772156649
    inv = NormalDist().inv_cdf
    return math.sqrt(variance) * ((1 - gamma) * inv(1 - 1 / trials) + gamma * inv(1 - 1 / (trials * math.e)))


def _describe(window: int, baseline: int, row: np.ndarray) -> dict:
    values = dict(zip(PARAM_NAMES, (float(v) for v in row), strict=True))
    return {"W": window, "B": baseline, **values}


def _params_row(config: dict) -> np.ndarray:
    return np.array([config[name] for name in PARAM_NAMES], dtype=np.float64)


def daily_series(config: dict, direction: str, first_day: int, n_days: int, delay: int | None = None):
    """单组参数逐日收益 x_d、逐日提醒数与提醒记录；delay 给定时只用该入场延迟的得分。"""
    rec = _load(config["W"], config["B"], direction)
    alert, wave = alerts_of(rec, _params_row(config), config["B"], direction)
    score = rec["score"] if delay is None else rec["by_delay"][:, DELAYS.index(delay)]
    day = rec["day"] - first_day
    inside = alert & (day >= 0) & (day < n_days)
    valid = inside & ~np.isnan(score)
    sums = np.bincount(day[valid], weights=score[valid], minlength=n_days)
    counts = np.bincount(day[valid], minlength=n_days)
    daily = np.divide(sums, counts, out=np.zeros(n_days), where=counts > 0)
    alerts = np.bincount(day[inside], minlength=n_days)
    return daily, alerts, {"wave": wave[inside], "score": score[inside], "valid": valid[inside]}


def cmd_select() -> None:
    summary = {}
    masks = np.array([[b in combo for b in range(CSCV_BLOCKS)]
                      for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2)])
    for direction in ("surge", "drop"):
        dev = _load_dev(direction)
        eligible = dev["alerts_per_day"] >= MIN_ALERTS_PER_DAY
        trials = len(dev["sharpe"])
        candidates = np.flatnonzero(eligible)
        best = candidates[np.argmax(dev["sharpe"][candidates])]
        config = _describe(int(dev["window"][best]), int(dev["baseline"][best]), dev["params"][best])
        x, _, _ = daily_series(config, direction, 0, DEV_DAYS)
        sr, skew, kurt = _moments(x)
        variance = float(np.var(dev["sharpe"][candidates], ddof=1))
        sr0 = expected_max_sharpe(variance, trials)
        dsr = psr(x, sr0)
        started = time.time()
        logits = _cscv(dev["block_sum"], dev["block_sq"], dev["block_alerts"].astype(np.float64),
                       dev["block_len"], masks, MIN_ALERTS_PER_DAY)
        pbo = float(np.mean(logits[~np.isnan(logits)] <= 0))
        # 现行配置在开发期的位置（仅供参考）：冷却 10 不在网格里，按单组评估算夏普后与网格比较。
        current_x, current_alerts, _ = daily_series(CURRENT[direction], direction, 0, DEV_DAYS)
        current_sr = _moments(current_x)[0]
        order = np.argsort(-np.where(eligible, dev["sharpe"], -np.inf))
        top = [
            {
                **_describe(int(dev["window"][i]), int(dev["baseline"][i]), dev["params"][i]),
                "sharpe_ann": float(dev["sharpe"][i] * math.sqrt(365)),
                "alerts_per_day": float(dev["alerts_per_day"][i]),
                "mean_score": float(dev["mean_score"][i]),
            }
            for i in order[:30]
        ]
        summary[direction] = {
            "chosen": config,
            "dev_sharpe_ann": sr * math.sqrt(365),
            "dev_sharpe_check": float(dev["sharpe"][best] * math.sqrt(365)),
            "dev_alerts_per_day": float(dev["alerts_per_day"][best]),
            "dev_mean_score": float(dev["mean_score"][best]),
            "skew": skew,
            "kurtosis": kurt,
            "trials": trials,
            "eligible": int(len(candidates)),
            "sharpe_var_daily": variance,
            "expected_max_sharpe_ann": sr0 * math.sqrt(365),
            "dsr": dsr,
            "pbo": pbo,
            "pbo_splits": int(np.sum(~np.isnan(logits))),
            "current_dev_sharpe_ann": current_sr * math.sqrt(365),
            "current_dev_alerts_per_day": float(current_alerts.mean()),
            "current_dev_rank": int(np.sum(dev["sharpe"][candidates] > current_sr)) + 1,
            "top30": top,
            "cscv_seconds": time.time() - started,
        }
        print(json.dumps({k: v for k, v in summary[direction].items() if k != "top30"}, ensure_ascii=False, indent=1),
              flush=True)
    (OUT / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _period_report(config: dict, direction: str, first_day: int, n_days: int) -> dict:
    x, alerts, info = daily_series(config, direction, first_day, n_days)
    sr = _moments(x)[0]
    report = {
        "sharpe_ann": sr * math.sqrt(365),
        "psr_gt0": psr(x),
        "alerts_per_day": float(alerts.mean()),
        "wave1_per_day": float(np.sum(info["wave"] == 1) / n_days),
        "mean_score_pct": float(np.nanmean(info["score"][info["valid"]]) * 100),
        "hit_rate": float(np.mean(info["score"][info["valid"]] > 0)),
        "mean_daily_pct": float(x.mean() * 100),
    }
    for delay in DELAYS:
        xd, _, info_d = daily_series(config, direction, first_day, n_days, delay)
        report[f"delay{delay}_mean_score_pct"] = float(np.mean(info_d["score"][info_d["valid"]]) * 100)
        report[f"delay{delay}_sharpe_ann"] = _moments(xd)[0] * math.sqrt(365)
    return report


def _data_days() -> int:
    """数据覆盖到的天数（含最后不满一天的那天）。按数据算而不是按运行当天算：数据不更新时晚些重跑，
    样本外的天数与每天提醒数也不变（与 launch_alerts._data_weeks 同理）。"""
    last = 0
    for path in OUT.glob("cand_W*_B*.npz"):
        with np.load(path) as data:
            last = max(last, int(data["t"].max()))
    return (last * 60 + 60) // DAY + 1


def cmd_evaluate() -> None:
    chosen = json.loads((OUT / "select.json").read_text(encoding="utf-8"))
    oos_days = _data_days() - DEV_DAYS
    result = {}
    for direction in ("surge", "drop"):
        result[direction] = {}
        for name, config in (("chosen", chosen[direction]["chosen"]), ("current", CURRENT[direction])):
            result[direction][name] = {
                "config": config,
                "dev": _period_report(config, direction, 0, DEV_DAYS),
                "oos": _period_report(config, direction, DEV_DAYS, oos_days),
            }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 对拍


def cmd_verify() -> None:

    from price_alert.launch.detector import LaunchDetector
    from price_alert.launch.rule import LaunchRule
    from price_alert.models import Candle

    contracts = load_contracts()
    symbols = json.loads((OUT / "symbols.json").read_text(encoding="utf-8"))
    cases = [
        (5, 360, dict(ratio=3.0, quote=1e5, change=3.0, max_change=15.0, breakout=0.0, cooldown=10.0, step=5.0)),
        (3, 180, dict(ratio=2.0, quote=5e4, change=1.5, max_change=8.0, breakout=1.0, cooldown=15.0, step=3.0)),
        (15, 720, dict(ratio=5.0, quote=3e5, change=5.0, max_change=math.inf, breakout=0.0, cooldown=60.0, step=8.0)),
    ]
    picks = ["PEPE_USDT", "WIF_USDT", "DOGE_USDT", "ORDI_USDT", "TRB_USDT"]
    span_start = datetime(2024, 3, 1, tzinfo=UTC)
    span_minutes = 30 * 1440
    mismatches = 0
    for window, baseline, values in cases:
        config = {"W": window, "B": baseline, **values, "volume_24h": 0.0, "retrace": math.inf}
        for symbol in picks:
            series = load_series(symbol, _multiplier(contracts[symbol]))
            begin = int((span_start.timestamp() - series.t0) // 60)
            end = begin + span_minutes
            assert series.covered[begin:end].all(), symbol
            # 两个方向各一个检测器，参数相同。
            detectors = [
                LaunchDetector(LaunchRule(
                    direction=direction, window_minutes=window, baseline_minutes=baseline,
                    min_volume_ratio=values["ratio"], min_window_quote=values["quote"],
                    min_change_percent=values["change"], require_breakout=bool(values["breakout"]),
                    cooldown_minutes=int(values["cooldown"]), realert_step_percent=values["step"],
                    max_change_percent=None if math.isinf(values["max_change"]) else values["max_change"],
                ))
                for direction in ("surge", "drop")
            ]
            for detector in detectors:
                detector.add_symbol(symbol, [], 0.0, quanto_multiplier=1.0)
            expected = set()
            for i in range(begin, end):
                stamp = datetime.fromtimestamp(series.t0 + i * 60, UTC)
                candle = Candle(stamp, float(series.open[i]), float(series.high[i]), float(series.low[i]),
                                float(series.close[i]), float(series.quote[i]))
                for detector in detectors:
                    for alert in detector.add_candle(symbol, candle):
                        expected.add((int(alert.timestamp.timestamp()), alert.metrics.direction, alert.wave))
            got = set()
            first_t = (series.t0 - EPOCH) // 60 + begin
            for direction in ("surge", "drop"):
                rec = _load(window, baseline, direction)
                sym_id = symbols.index(symbol)
                in_span = (rec["t"] >= first_t + window + baseline - 1) & (rec["t"] < first_t + span_minutes)
                keep = (rec["sym"] == sym_id) & in_span
                sub = {k: v[keep] for k, v in rec.items() if k != "seg"}
                sub["seg"] = np.array([0, int(keep.sum())], np.int64)
                alert, wave = alerts_of(sub, _params_row(config), baseline, direction)
                for t, w in zip(sub["t"][alert], wave[alert], strict=True):
                    got.add((int(EPOCH + t * 60 + 60), direction, int(w)))
            diff = expected ^ got
            mismatches += len(diff)
            print(f"W{window} B{baseline} {symbol}: 检测器 {len(expected)} 条，脚本 {len(got)} 条，不一致 {len(diff)}"
                  + (f" {sorted(diff)[:4]}" if diff else ""), flush=True)
    print("对拍通过" if mismatches == 0 else f"对拍不一致 {mismatches} 条")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "verify", "search", "select", "evaluate"))
    command = parser.parse_args().command
    {"extract": cmd_extract, "verify": cmd_verify, "search": cmd_search, "select": cmd_select,
     "evaluate": cmd_evaluate}[command]()


if __name__ == "__main__":
    main()
