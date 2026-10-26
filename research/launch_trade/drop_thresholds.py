"""放量下跌门槛（launch.drop）搜索：有没有比提议组更早、延续性也不差的参数。判据见同名 .md，跑数前写定。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/drop_thresholds.py verify SOL_USDT WIF_USDT ORDI_USDT  # 与真检测器逐条对照
    & $P research/launch_trade/drop_thresholds.py run       # 120 组候选，写 results/drop_thresholds/
    & $P research/launch_trade/drop_thresholds.py report
"""

from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
from load_data import load_contracts, load_series
from numba import njit
from paths import RESULTS

OUT = RESULTS / "drop_thresholds"
# 两个方向共用的字段取当前 config/default.yaml 的 launch 段，只改 drop 开放的三项。
WINDOW, BASELINE, MIN_QUOTE, COOLDOWN, STEP, POOL = 5, 360, 100_000.0, 10, 5.0, 500_000.0
CURRENT = (3.0, 2.5, 8.0)
PROPOSED = (4.0, 2.0, 10.0)
CANDIDATES = [
    (ratio, low, high)
    for ratio in (2.0, 2.5, 3.0, 4.0, 5.0)
    for low in (1.5, 2.0, 2.5, 3.0)
    for high in (6.0, 8.0, 10.0, 12.0, 15.0, math.inf)
    if high > low
]
LEVELS = np.log1p(np.array([0.03, 0.05, 0.10]))
HORIZON = 1440
MATCH_SECONDS = 3600
# 列：收盘时刻、波次、提醒价、±3/5/10 延续、合约下标。
T, WAVE, PRICE, C0, SYM = 0, 1, 2, 3, 6


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


DEV = (ts("2024-10-10"), 10**12)
OOS = (ts("2021-10-10"), ts("2024-09-30"))
SEGMENT_STARTS = [ts(f"{y}-{m:02d}-10") for y in range(2021, 2027) for m in (4, 10)]
SEGMENT_STARTS = [s for s in SEGMENT_STARTS if ts("2021-10-10") <= s <= ts("2026-04-10")]


@njit(cache=True)
def detect(close, high, low, quote, covered, ratio_min, change_min, change_max, levels):
    """逐分钟复刻 LaunchDetector 的下跌判定（add_candle 口径，不复核空档过期）。"""
    n = close.size
    hist = BASELINE + WINDOW
    cq = np.zeros(n + 1)
    cq[1:] = np.cumsum(quote)
    bad = np.zeros(n + 1, dtype=np.int64)
    bad[1:] = np.cumsum(~covered)
    out = []
    last_i, last_price, last_wave = -(10**9), 0.0, 0
    for i in range(hist - 1, n):
        lo = i - hist + 1
        if bad[i + 1] - bad[lo] > 0:
            continue
        if cq[i + 1] - cq[max(0, i - 1439)] <= POOL:
            continue
        start, end = close[i - WINDOW], close[i]
        change = (end / start - 1) * 100
        if change >= 0:
            continue
        window_quote = cq[i + 1] - cq[i + 1 - WINDOW]
        base = (cq[i + 1 - WINDOW] - cq[lo]) / BASELINE * WINDOW
        if window_quote < MIN_QUOTE or (base > 0 and window_quote / base < ratio_min) or -change < change_min:
            continue
        # _repeat_outcome：冷却先于基准期重置，再看是否比上次再推进。
        if i - last_i < COOLDOWN:
            continue
        if i - last_i >= BASELINE:
            wave = 1
        elif (end / last_price - 1) * -100 < STEP:
            continue
        else:
            wave = last_wave + 1
        if -change > change_max:
            # 插针：静默记入冷却与再推进的基准，波次不加。
            last_i, last_price, last_wave = i, end, wave - 1
            continue
        last_i, last_price, last_wave = i, end, wave
        row = np.full(3 + levels.size, np.nan)
        row[0], row[1], row[2] = i, wave, end
        for k in range(levels.size):
            down, up = end * np.exp(-levels[k]), end * np.exp(levels[k])
            for j in range(i + 1, min(n, i + 1 + HORIZON)):
                if not covered[j]:
                    break
                hit_down, hit_up = low[j] <= down, high[j] >= up
                if hit_down or hit_up:
                    row[3 + k] = 0.5 if (hit_down and hit_up) else (1.0 if hit_down else 0.0)
                    break
        out.append(row)
    return out


def _items():
    items = []
    for index, (name, info) in enumerate(sorted(load_contracts().items())):
        try:
            multiplier = float(info.get("quanto_multiplier") or 0)
        except ValueError:
            multiplier = 0.0
        items.append((index, name, multiplier if multiplier > 0 else None))
    return items


def _run_symbol(item):
    index, name, multiplier = item
    series = load_series(name, multiplier)
    if series is None or len(series) < BASELINE + WINDOW:
        return []
    result = []
    for ratio, low, high in CANDIDATES:
        rows = detect(series.close, series.high, series.low, series.quote, series.covered, ratio, low, high, LEVELS)
        arr = np.array(rows) if rows else np.empty((0, 6))
        arr[:, T] = series.t0 + (arr[:, T] + 1) * 60
        result.append(np.hstack([arr, np.full((len(arr), 1), index, dtype=float)]))
    return result


def run() -> None:
    parts: list[list[np.ndarray]] = [[] for _ in CANDIDATES]
    with ProcessPoolExecutor(14) as pool:
        for result in pool.map(_run_symbol, _items(), chunksize=2):
            for k, arr in enumerate(result):
                if len(arr):
                    parts[k].append(arr)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "alerts.npz",
        **{f"c{k}": np.vstack(p) if p else np.empty((0, 7)) for k, p in enumerate(parts)},
        candidates=np.array(CANDIDATES),
    )


def verify(symbols: list[str]) -> None:
    """与真 LaunchDetector 逐条对照 2025-10~11 两个月（这段没有数据缺口、合约池始终满足）。"""
    from price_alert.detectors.launch.detector import LaunchDetector
    from price_alert.detectors.launch.rule import LaunchRule
    from price_alert.common.models import Candle

    contracts = load_contracts()
    for name in symbols:
        series = load_series(name, float(contracts[name]["quanto_multiplier"]))
        lo = (ts("2025-10-01") - series.t0) // 60
        hi = lo + 61 * 1440
        assert series.covered[lo:hi].all()
        sl = slice(lo, hi)
        for ratio, low, high in (CURRENT, PROPOSED, (2.0, 1.5, 6.0), (5.0, 3.0, math.inf)):
            mine = [
                (int(r[0]), int(r[1]))
                for r in detect(
                    series.close[sl], series.high[sl], series.low[sl], series.quote[sl],
                    np.ones(hi - lo, dtype=bool), ratio, low, high, LEVELS,
                )
            ]
            # 放量下跌是单独的检测器，只用下跌这一套门槛。
            rule = LaunchRule(
                direction="drop", window_minutes=WINDOW, baseline_minutes=BASELINE, min_volume_ratio=ratio,
                min_window_quote=MIN_QUOTE, min_change_percent=low, require_breakout=False, cooldown_minutes=COOLDOWN,
                realert_step_percent=STEP, max_change_percent=None if math.isinf(high) else high,
            )
            detector = LaunchDetector(rule)
            detector.add_symbol(name, [], 1e9, quanto_multiplier=1)
            real = []
            for k in range(lo, hi):
                moment = datetime.fromtimestamp(series.t0 + k * 60, UTC)
                candle = Candle(moment, series.open[k], series.high[k], series.low[k], series.close[k], series.quote[k])
                for alert in detector.add_candle(name, candle):
                    if alert.metrics.direction == "drop":
                        real.append((k - lo, alert.wave))
            print(name, (ratio, low, high), len(mine), "一致" if mine == real else f"不一致（真检测器 {len(real)} 条）")


# ---------- 统计 ----------


def _first(arr):
    first = arr[arr[:, WAVE] == 1]
    return first[np.lexsort((first[:, T], first[:, SYM]))]


def match_scores(base_first, cand_first):
    """当前组每条第 1 波的得分：候选同合约 ±60 分钟内最近的第 1 波更早 +1、同时 0、更晚或没有 -1。"""
    scores = np.full(len(base_first), -1.0)
    b_sym, c_sym = base_first[:, SYM], cand_first[:, SYM]
    for sym in np.unique(b_sym):
        bi = np.flatnonzero(b_sym == sym)
        ci = np.flatnonzero(c_sym == sym)
        if ci.size == 0:
            continue
        bt, ct = base_first[bi, T], cand_first[ci, T]
        pos = np.searchsorted(ct, bt)
        left = ct[np.clip(pos - 1, 0, ct.size - 1)]
        right = ct[np.clip(pos, 0, ct.size - 1)]
        gap = np.where(np.abs(left - bt) <= np.abs(right - bt), left - bt, right - bt)
        ok = np.abs(gap) <= MATCH_SECONDS
        scores[bi[ok]] = np.where(gap[ok] < 0, 1.0, np.where(gap[ok] == 0, 0.0, -1.0))
    return scores


class Aggregates:
    """按时间块累计各候选的计数，任意块组合的指标都由块的和算出（CSCV 要用）。"""

    def __init__(self, alerts, base_first, scores, edges):
        n_blocks = len(edges) - 1
        k = len(alerts)
        self.cont_sum = np.zeros((k, n_blocks, 3))
        self.cont_cnt = np.zeros((k, n_blocks, 3))
        self.alerts = np.zeros((k, n_blocks))
        self.score = np.zeros((k, n_blocks))
        base_block = np.searchsorted(edges, base_first[:, T], side="right") - 1
        valid = (base_block >= 0) & (base_block < n_blocks)
        self.base_first = np.bincount(base_block[valid], minlength=n_blocks).astype(float)
        for c, arr in enumerate(alerts):
            block = np.searchsorted(edges, arr[:, T], side="right") - 1
            keep = (block >= 0) & (block < n_blocks)
            b = block[keep]
            self.alerts[c] = np.bincount(b, minlength=n_blocks)
            for level in range(3):
                values = arr[keep, C0 + level]
                has = ~np.isnan(values)
                self.cont_sum[c, :, level] = np.bincount(b[has], weights=values[has], minlength=n_blocks)
                self.cont_cnt[c, :, level] = np.bincount(b[has], minlength=n_blocks)
            self.score[c] = np.bincount(base_block[valid], weights=scores[c][valid], minlength=n_blocks)

    def metrics(self, mask, base_index):
        """返回各候选的 E、ΔC（百分点）、R。mask 选中的块合在一起算。"""
        cont = (self.cont_sum[:, mask].sum(1) / self.cont_cnt[:, mask].sum(1)).mean(1) * 100
        e = self.score[:, mask].sum(1) / self.base_first[mask].sum()
        r = self.alerts[:, mask].sum(1) / self.alerts[base_index, mask].sum()
        return e, cont - cont[base_index], r


def report() -> None:
    stored = np.load(OUT / "alerts.npz")
    candidates = [tuple(row) for row in stored["candidates"]]
    assert candidates == CANDIDATES
    alerts = [stored[f"c{k}"] for k in range(len(CANDIDATES))]
    base_i, prop_i = CANDIDATES.index(CURRENT), CANDIDATES.index(PROPOSED)
    t_end = max(arr[:, T].max() for arr in alerts)
    base_first = _first(alerts[base_i])
    scores = [match_scores(base_first, _first(arr)) for arr in alerts]
    label = lambda c: f"量比{c[0]:g} {c[1]:g}~{'不限' if math.isinf(c[2]) else f'{c[2]:g}'}%"  # noqa: E731
    print(f"数据截至 {datetime.fromtimestamp(t_end, UTC):%Y-%m-%d %H:%M} UTC，候选 {len(CANDIDATES)} 组\n")

    # ---- 开发期选择 ----
    dev_days = np.arange(DEV[0] // 86400, t_end // 86400 + 1)
    dev = Aggregates(alerts, base_first, scores, (dev_days * 86400).astype(float).tolist() + [t_end + 1])
    all_days = np.ones(len(dev_days), dtype=bool)
    e, dc, r = dev.metrics(all_days, base_i)
    busiest = np.argsort(dev.alerts[base_i])[-30:]
    trimmed = all_days.copy()
    trimmed[busiest] = False
    _, dc30, _ = dev.metrics(trimmed, base_i)
    feasible = (r <= 1.5) & (dc >= 0) & (dc30 >= 0)
    order = np.argsort(-np.where(feasible, e, -np.inf))
    print("开发期（E 越大越早；ΔC 对当前组，百分点；去 30 天为去掉当前组提醒最多的 30 天）")
    print("| 候选 | E | ΔC | ΔC 去 30 天 | R | 可行 |")
    print("|---|---|---|---|---|---|")
    shown = list(order[:15]) + [i for i in (prop_i, base_i) if i not in order[:15]]
    for i in shown:
        mark = "是" if feasible[i] else ""
        print(f"| {label(CANDIDATES[i])} | {e[i]:+.3f} | {dc[i]:+.2f} | {dc30[i]:+.2f} | {r[i]:.2f} | {mark} |")
    print(f"\n可行候选 {feasible.sum()} 组")
    if not feasible.any():
        print("没有可行候选 → 用提议组")
        return
    w = int(order[0])
    print(f"W = {label(CANDIDATES[w])}")
    # 目标是比当前组更早：可行候选里最早的若不早于当前组（含选回当前组本身），等同于没有可行候选。
    if e[w] <= 0:
        print("没有比当前组更早的可行候选 → 用提议组")
        return
    if w == prop_i:
        print("W 就是提议组 → 用提议组")
        return

    # ---- 检验 1：PBO（开发期 16 块 CSCV）----
    edges = np.linspace(DEV[0], t_end + 1, 17)
    blocks = Aggregates(alerts, base_first, scores, edges.tolist())
    below, logits = 0, []
    combos = list(itertools.combinations(range(16), 8))
    for combo in combos:
        is_mask = np.zeros(16, dtype=bool)
        is_mask[list(combo)] = True
        e_is, dc_is, r_is = blocks.metrics(is_mask, base_i)
        ok = (r_is <= 1.5) & (dc_is >= 0)
        if not ok.any():
            below += 1
            continue
        chosen = int(np.argmax(np.where(ok, e_is, -np.inf)))
        e_oos, dc_oos, _ = blocks.metrics(~is_mask, base_i)
        score = e_oos - (dc_oos < 0)
        rank = (score < score[chosen]).sum() + 0.5 * ((score == score[chosen]).sum() - 1) + 1
        omega = rank / (len(CANDIDATES) + 1)
        logits.append(math.log(omega / (1 - omega)))
        below += omega <= 0.5
    pbo = below / len(combos)
    print(f"\n检验 1 PBO = {pbo:.3f}（< 0.5 通过）{'通过' if pbo < 0.5 else '不通过'}")

    # ---- 检验 2：样本外 ----
    oos_days = np.arange(OOS[0] // 86400, OOS[1] // 86400)
    oos = Aggregates(alerts, base_first, scores, (oos_days * 86400).astype(float).tolist() + [OOS[1]])
    mask = np.ones(len(oos_days), dtype=bool)
    e_o, dc_o, r_o = oos.metrics(mask, base_i)
    rng = np.random.default_rng(20261017)
    diff_day = oos.score[w] - oos.score[prop_i]
    n_day = oos.base_first
    boot = np.empty(2000)
    for b in range(boot.size):
        pick = rng.integers(0, len(oos_days), len(oos_days))
        boot[b] = diff_day[pick].sum() / n_day[pick].sum()
    lower = np.quantile(boot, 0.05)
    pass2 = (e_o[w] > e_o[prop_i]) and lower > 0 and dc_o[w] >= 0 and dc_o[w] >= dc_o[prop_i] - 0.3
    print(
        f"检验 2 样本外：E W {e_o[w]:+.3f} / 提议 {e_o[prop_i]:+.3f}，差的单侧 95% 下界 {lower:+.4f}；"
        f"ΔC W {dc_o[w]:+.2f} / 提议 {dc_o[prop_i]:+.2f}；R W {r_o[w]:.2f} / 提议 {r_o[prop_i]:.2f}"
        f" → {'通过' if pass2 else '不通过'}"
    )

    # ---- 检验 3：半年段 ----
    seg_edges = SEGMENT_STARTS + [t_end + 1]
    seg = Aggregates(alerts, base_first, scores, seg_edges)
    e_wins = c_ok = 0
    cells = []
    for s in range(len(SEGMENT_STARTS)):
        m = np.zeros(len(SEGMENT_STARTS), dtype=bool)
        m[s] = True
        e_s, dc_s, _ = seg.metrics(m, base_i)
        e_wins += e_s[w] > e_s[prop_i]
        c_ok += dc_s[w] >= 0
        cells.append(f"E {e_s[w]:+.2f}/{e_s[prop_i]:+.2f} ΔC {dc_s[w]:+.1f}/{dc_s[prop_i]:+.1f}")
    pass3 = e_wins >= 7 and c_ok >= 5
    print(f"检验 3 半年段：E 胜提议 {e_wins}/10，ΔC ≥ 0 {c_ok}/10 → {'通过' if pass3 else '不通过'}")
    for start, cell in zip(SEGMENT_STARTS, cells, strict=True):
        print(f"  {datetime.fromtimestamp(start, UTC):%y-%m}  W/提议  {cell}")
    final = pbo < 0.5 and pass2 and pass3
    print(f"\n结论：{'W 全部通过，采用 W' if final else 'W 未全部通过 → 用提议组'}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "verify":
        verify(sys.argv[2:])
    elif command == "run":
        run()
    else:
        report()
