"""放量上涨参数搜索：怎么搭配能报得更早、之后延续性更高。判据见同名 .md，跑数前写定。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/surge_params.py verify SOL_USDT WIF_USDT ORDI_USDT  # 与真检测器逐条对照
    & $P research/launch_trade/surge_params.py scan       # 逐分钟特征，写 results/surge_params/rows.npz
    & $P research/launch_trade/surge_params.py evaluate   # 7776 组候选按天聚合，写 aggregates.npz
    & $P research/launch_trade/surge_params.py report

候选只在阈值上不同，提醒一定落在「最宽松的那组也满足」的分钟里，所以先把这些分钟的特征与延续结果扫出来（scan），
每个候选再在这些行上筛选、跑冷却与波次的状态机（evaluate），不必对每个候选重扫全部分钟。
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

OUT = RESULTS / "surge_params"
COOLDOWN, STEP, POOL = 10, 5.0, 500_000.0
WINDOWS = (2, 3, 5)
BASELINES = (120, 360, 1440)
QUOTES = (50_000.0, 100_000.0, 200_000.0)
BREAKOUTS = (False, True)
RATIOS = (2.0, 3.0, 5.0)
CHANGES = (2.0, 2.5, 3.0)
MAXES = (15.0, math.inf)
SWITCHES = (False, True)
SHARED = list(itertools.product(WINDOWS, BASELINES, QUOTES, BREAKOUTS))
# 候选：(W, B, Q, 突破, 量比, 涨幅下限, 涨幅上限, 收在高位, 末分钟不缩量, 日线顺势)
CANDIDATES = [
    shared + rest
    for shared in SHARED
    for rest in itertools.product(RATIOS, CHANGES, MAXES, SWITCHES, SWITCHES, SWITCHES)
]
CURRENT = (5, 360, 100_000.0, False, 3.0, 3.0, 15.0, False, False, False)
DROP = (4.0, 2.0, 10.0)
# 扫描时最宽松的门槛：上涨取候选族的最小值，下跌取固定门槛（窗口成交额取族内最小）。
LOOSE_QUOTE, LOOSE_SURGE_RATIO, LOOSE_SURGE_CHANGE = min(QUOTES), min(RATIOS), min(CHANGES)
LEVELS = np.log1p(np.array([0.03, 0.05, 0.10]))
HORIZON = 1440
MATCH_SECONDS = 3600
# 行的列（float64）：
COLS = ("sym", "i", "t", "dir", "change", "ratio", "quote", "broke", "pos", "nofade", "daytrend", "price",
        "c3", "c5", "c10", "runup")
C = {name: k for k, name in enumerate(COLS)}
# 每天聚合的列：提醒数、三档延续的和与计数、E 得分之和。
N_ALERTS, CSUM, CCNT, SCORE = 0, 1, 4, 7


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


DAY0 = ts("2021-10-10") // 86400
DEV = ts("2024-10-10")
OOS = (ts("2021-10-10"), ts("2024-09-30"))
SEGMENT_STARTS = [ts(f"{y}-{m:02d}-10") for y in range(2021, 2027) for m in (4, 10)]
SEGMENT_STARTS = [s for s in SEGMENT_STARTS if ts("2021-10-10") <= s <= ts("2026-04-10")]


def label(c) -> str:
    w, b, q, bo, r, lo, hi, f1, f2, f3 = c
    extra = "".join(name for on, name in ((f1, "+高位"), (f2, "+不缩量"), (f3, "+顺势")) if on)
    top = "不限" if math.isinf(hi) else f"{hi:g}"
    return f"W{w} B{b} Q{q / 1e4:g}万 {'突破' if bo else '不突破'} 量比{r:g} {lo:g}~{top}%{extra}"


# ---------- 扫描 ----------


@njit(cache=True)
def scan(close, high, low, quote, covered, window, baseline, loose_quote, surge_ratio, surge_change, drop_ratio,
         drop_change):
    """返回满足最宽松门槛的分钟：下标、方向与各项特征（add_candle 口径，判定第 i 根收盘后的窗口）。"""
    n = close.size
    hist = baseline + window
    cq = np.zeros(n + 1)
    cq[1:] = np.cumsum(quote)
    bad = np.zeros(n + 1, dtype=np.int64)
    bad[1:] = np.cumsum(~covered)
    out = []
    for i in range(hist - 1, n):
        lo = i - hist + 1
        if bad[i + 1] - bad[lo] > 0:
            continue
        if cq[i + 1] - cq[max(0, i - 1439)] <= POOL:
            continue
        window_quote = cq[i + 1] - cq[i + 1 - window]
        if window_quote < loose_quote:
            continue
        start, end = close[i - window], close[i]
        change = (end / start - 1) * 100
        base = (cq[i + 1 - window] - cq[lo]) / baseline * window
        ratio = window_quote / base if base > 0 else np.inf
        if change >= surge_change and ratio >= surge_ratio:
            direction = 1.0
            broke = 1.0 if end > high[lo : i + 1 - window].max() else 0.0
        elif change <= -drop_change and ratio >= drop_ratio:
            direction = -1.0
            broke = 1.0 if end < low[lo : i + 1 - window].min() else 0.0
        else:
            continue
        span = high[i] - low[i]
        pos = 1.0 if span <= 0 else (end - low[i]) / span
        nofade = 1.0 if quote[i] >= window_quote / window else 0.0
        k = i - window - 1440
        daytrend = 1.0 if k >= 0 and covered[k] and start >= close[k] else 0.0
        out.append((float(i), direction, change, ratio, window_quote, broke, pos, nofade, daytrend, end))
    return out


@njit(cache=True)
def continuation(high, low, close, covered, index, levels):
    """每个下标：从收盘价起先涨 X% 记 1、先跌 X% 记 0、同一分钟都碰到 0.5，未决或遇到无数据为 nan。
    另返回提醒价相对前 60 分钟最低价的涨幅（%）。"""
    n = close.size
    res = np.full((index.size, levels.size + 1), np.nan)
    for r in range(index.size):
        i = index[r]
        price = close[i]
        res[r, levels.size] = (price / low[max(0, i - 59) : i + 1].min() - 1) * 100
        pending = levels.size
        for j in range(i + 1, min(n, i + 1 + HORIZON)):
            if not covered[j]:
                break
            for k in range(levels.size):
                if not np.isnan(res[r, k]):
                    continue
                up, down = price * np.exp(levels[k]), price * np.exp(-levels[k])
                hit_up, hit_down = high[j] >= up, low[j] <= down
                if hit_up or hit_down:
                    res[r, k] = 0.5 if (hit_up and hit_down) else (1.0 if hit_up else 0.0)
                    pending -= 1
            if pending == 0:
                break
    return res


def _items():
    items = []
    for index, (name, info) in enumerate(sorted(load_contracts().items())):
        try:
            multiplier = float(info.get("quanto_multiplier") or 0)
        except ValueError:
            multiplier = 0.0
        items.append((index, name, multiplier if multiplier > 0 else None))
    return items


def _scan_series(series, sym: int) -> dict[tuple[int, int], np.ndarray]:
    parts = {}
    for window, baseline in itertools.product(WINDOWS, BASELINES):
        if len(series) < baseline + window:
            continue
        rows = scan(series.close, series.high, series.low, series.quote, series.covered, window, baseline,
                    LOOSE_QUOTE, LOOSE_SURGE_RATIO, LOOSE_SURGE_CHANGE, DROP[0], DROP[1])
        if rows:
            parts[(window, baseline)] = np.array(rows)
    if not parts:
        return {}
    index = np.unique(np.concatenate([arr[:, 0] for arr in parts.values()])).astype(np.int64)
    cont = continuation(series.high, series.low, series.close, series.covered, index, LEVELS)
    result = {}
    for key, arr in parts.items():
        at = np.searchsorted(index, arr[:, 0].astype(np.int64))
        full = np.empty((len(arr), len(COLS)))
        full[:, C["sym"]] = sym
        full[:, C["i"]] = arr[:, 0]
        full[:, C["t"]] = series.t0 + (arr[:, 0] + 1) * 60
        full[:, C["dir"] : C["price"] + 1] = arr[:, 1:]
        full[:, C["c3"] :] = cont[at]
        result[key] = full
    return result


def _run_symbol(item):
    index, name, multiplier = item
    series = load_series(name, multiplier)
    if series is None:
        return {}
    return _scan_series(series, index)


def run_scan() -> None:
    parts: dict[tuple[int, int], list[np.ndarray]] = {}
    with ProcessPoolExecutor(14) as pool:
        for result in pool.map(_run_symbol, _items(), chunksize=2):
            for key, arr in result.items():
                parts.setdefault(key, []).append(arr)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "rows.npz", **{f"w{w}_b{b}": np.vstack(v) for (w, b), v in parts.items()})
    for (w, b), v in sorted(parts.items()):
        print(f"W{w} B{b}: {sum(len(a) for a in v)} 行")


# ---------- 候选 ----------


@njit(cache=True)
def machine(sym, idx, price, change, sign, change_max, baseline):
    """冷却、波次与插针记录（_repeat_outcome / _record_overshoot）。行已按合约、时间排序，且除涨跌幅上限外全部满足。
    返回每行的波次，0 表示不提醒。"""
    n = sym.size
    wave_out = np.zeros(n, dtype=np.int64)
    current = -1.0
    last_i, last_price, last_wave = -(10**9), 0.0, 0
    for r in range(n):
        if sym[r] != current:
            current = sym[r]
            last_i, last_price, last_wave = -(10**9), 0.0, 0
        i = idx[r]
        if i - last_i < COOLDOWN:
            continue
        if i - last_i >= baseline:
            wave = 1
        elif (price[r] / last_price - 1) * 100 * sign < STEP:
            continue
        else:
            wave = last_wave + 1
        if abs(change[r]) > change_max:
            last_i, last_price, last_wave = i, price[r], wave - 1
            continue
        last_i, last_price, last_wave = i, price[r], wave
        wave_out[r] = wave
    return wave_out


@njit(cache=True)
def match(base_sym, base_t, cand_sym, cand_t):
    """当前组每条第 1 波：候选同合约 ±60 分钟内时间最近的提醒（同距取更早）更早 +1、同一分钟 0、更晚或没有 -1。"""
    scores = np.full(base_t.size, -1.0)
    j = 0
    m_total = cand_t.size
    for k in range(base_t.size):
        s, t = base_sym[k], base_t[k]
        while j < m_total and (cand_sym[j] < s or (cand_sym[j] == s and cand_t[j] < t - MATCH_SECONDS)):
            j += 1
        best = 1e12
        m = j
        while m < m_total and cand_sym[m] == s and cand_t[m] <= t + MATCH_SECONDS:
            gap = cand_t[m] - t
            if abs(gap) < abs(best):
                best = gap
            m += 1
        if best != 1e12:
            scores[k] = 1.0 if best < 0 else (0.0 if best == 0 else -1.0)
    return scores


class Rows:
    def __init__(self):
        stored = np.load(OUT / "rows.npz")
        self.groups: dict[tuple[int, int, int], np.ndarray] = {}
        for w, b in itertools.product(WINDOWS, BASELINES):
            arr = stored[f"w{w}_b{b}"]
            for sign in (1, -1):
                part = arr[arr[:, C["dir"]] == sign]
                self.groups[(w, b, sign)] = part[np.lexsort((part[:, C["i"]], part[:, C["sym"]]))]
        self.t_end = int(max(a[:, C["t"]].max() for a in self.groups.values()))
        self.n_days = self.t_end // 86400 - DAY0 + 1

    def surge_alerts(self, c) -> np.ndarray:
        w, b, q, bo, ratio, lo, hi, f1, f2, f3 = c
        g = self.groups[(w, b, 1)]
        mask = (g[:, C["quote"]] >= q) & (g[:, C["ratio"]] >= ratio) & (g[:, C["change"]] >= lo)
        if bo:
            mask &= g[:, C["broke"]] > 0
        if f1:
            mask &= g[:, C["pos"]] >= 0.5
        if f2:
            mask &= g[:, C["nofade"]] > 0
        if f3:
            mask &= g[:, C["daytrend"]] > 0
        return self._run(g[mask], 1.0, hi, b)

    def drop_alerts(self, shared) -> np.ndarray:
        w, b, q, bo = shared
        g = self.groups[(w, b, -1)]
        mask = (g[:, C["quote"]] >= q) & (g[:, C["ratio"]] >= DROP[0]) & (-g[:, C["change"]] >= DROP[1])
        if bo:
            mask &= g[:, C["broke"]] > 0
        alerts = self._run(g[mask], -1.0, DROP[2], b)
        # 下跌的延续是先跌 X%：把上涨口径的结果翻过来（0.5 与 nan 不变）。
        alerts[:, C["c3"] : C["c10"] + 1] = 1 - alerts[:, C["c3"] : C["c10"] + 1]
        return alerts

    @staticmethod
    def _run(g, sign, change_max, baseline):
        waves = machine(g[:, C["sym"]], g[:, C["i"]].astype(np.int64), g[:, C["price"]], g[:, C["change"]], sign,
                        change_max, baseline)
        keep = waves > 0
        out = g[keep].copy()
        out[:, C["dir"]] = waves[keep]  # 提醒里 dir 列改存波次
        return out

    def aggregate(self, alerts, base_first) -> np.ndarray:
        """按天（UTC）聚合：提醒数、三档延续和与计数、当前组第 1 波的 E 得分之和。"""
        agg = np.zeros((self.n_days, 8), dtype=np.float64)
        day = alerts[:, C["t"]].astype(np.int64) // 86400 - DAY0
        ok = day >= 0
        d = day[ok]
        agg[:, N_ALERTS] = np.bincount(d, minlength=self.n_days)
        for k in range(3):
            v = alerts[ok, C["c3"] + k]
            has = ~np.isnan(v)
            agg[:, CSUM + k] = np.bincount(d[has], weights=v[has], minlength=self.n_days)
            agg[:, CCNT + k] = np.bincount(d[has], minlength=self.n_days)
        scores = match(base_first[:, C["sym"]], base_first[:, C["t"]], alerts[:, C["sym"]], alerts[:, C["t"]])
        bday = base_first[:, C["t"]].astype(np.int64) // 86400 - DAY0
        bok = bday >= 0
        agg[:, SCORE] = np.bincount(bday[bok], weights=scores[bok], minlength=self.n_days)
        return agg


def first_waves(alerts):
    return alerts[alerts[:, C["dir"]] == 1]


def base_counts(rows, first) -> np.ndarray:
    day = first[:, C["t"]].astype(np.int64) // 86400 - DAY0
    return np.bincount(day[day >= 0], minlength=rows.n_days).astype(float)


def evaluate() -> None:
    rows = Rows()
    base = rows.surge_alerts(CURRENT)
    base_first = first_waves(base)
    surge = np.zeros((len(CANDIDATES), rows.n_days, 8), dtype=np.float32)
    for k, c in enumerate(CANDIDATES):
        surge[k] = rows.aggregate(rows.surge_alerts(c), base_first)
        if k % 500 == 0:
            print(f"{k}/{len(CANDIDATES)}", flush=True)
    base_drop = rows.drop_alerts(CURRENT[:4])
    base_drop_first = first_waves(base_drop)
    drop = np.zeros((len(SHARED), rows.n_days, 8), dtype=np.float32)
    for k, s in enumerate(SHARED):
        drop[k] = rows.aggregate(rows.drop_alerts(s), base_drop_first)
    np.savez(
        OUT / "aggregates.npz",
        surge=surge,
        drop=drop,
        base_first=base_counts(rows, base_first),
        base_drop_first=base_counts(rows, base_drop_first),
        t_end=rows.t_end,
    )


# ---------- 判据 ----------


class Metrics:
    """由按天聚合求任意日子集合上的 E、ΔC（百分点）、R、S；base 为当前组的下标。"""

    def __init__(self, agg, base_first, base):
        self.agg = agg.astype(np.float64)
        self.base_first = base_first
        self.base = base

    def on(self, days):
        a = self.agg[:, days].sum(1)
        cont = (a[:, CSUM : CSUM + 3] / a[:, CCNT : CCNT + 3]).mean(1) * 100
        e = a[:, SCORE] / self.base_first[days].sum()
        dc = cont - cont[self.base]
        r = a[:, N_ALERTS] / a[self.base, N_ALERTS]
        return e, dc, r, e + 0.1 * dc


def block_sums(agg, base_first, edges_days):
    """把按天聚合合并成块（CSCV 用），返回 (块聚合, 块的基准第 1 波数)。"""
    k, _, f = agg.shape
    out = np.zeros((k, len(edges_days) - 1, f))
    bf = np.zeros(len(edges_days) - 1)
    for j in range(len(edges_days) - 1):
        out[:, j] = agg[:, edges_days[j] : edges_days[j + 1]].sum(1)
        bf[j] = base_first[edges_days[j] : edges_days[j + 1]].sum()
    return out, bf


def report() -> None:
    stored = np.load(OUT / "aggregates.npz")
    surge, drop = stored["surge"], stored["drop"]
    t_end = int(stored["t_end"])
    n_days = surge.shape[1]
    base_i = CANDIDATES.index(CURRENT)
    shared_base = SHARED.index(CURRENT[:4])
    shared_of = np.array([SHARED.index(c[:4]) for c in CANDIDATES])
    changed = shared_of != shared_base
    up = Metrics(surge, stored["base_first"], base_i)
    down = Metrics(drop, stored["base_drop_first"], shared_base)
    day_t = (np.arange(n_days) + DAY0) * 86400
    dev = day_t >= DEV
    oos = (day_t >= OOS[0]) & (day_t < OOS[1])
    print(f"数据截至 {datetime.fromtimestamp(t_end, UTC):%Y-%m-%d %H:%M} UTC，候选 {len(CANDIDATES)} 组")

    def guard(days):
        e_d, dc_d, _, _ = down.on(days)
        return ~changed | ((e_d[shared_of] >= -0.05) & (dc_d[shared_of] >= -0.3)), e_d[shared_of], dc_d[shared_of]

    # ---- 开发期选择 ----
    e, dc, r, s = up.on(dev)
    dev_days = np.flatnonzero(dev)
    busiest = dev_days[np.argsort(surge[base_i, dev_days, N_ALERTS])[-30:]]
    trimmed = dev.copy()
    trimmed[busiest] = False
    _, dc30, _, _ = up.on(trimmed)
    g_ok, e_d, dc_d = guard(dev)
    feasible = (r <= 1.5) & (e >= 0) & (dc >= 0) & (dc30 >= 0) & g_ok
    order = np.argsort(-np.where(feasible, s, -np.inf))
    a = up.agg[base_i][dev].sum(0)
    base_c = (a[CSUM : CSUM + 3] / a[CCNT : CCNT + 3]).mean() * 100
    print(f"\n开发期当前组：上涨提醒 {a[N_ALERTS] / dev.sum():.1f} 条/天，第 1 波 {up.base_first[dev].sum():.0f} 条，"
          f"C = {base_c:.2f}%（3/5/10 档 {', '.join(f'{x * 100:.1f}' for x in a[CSUM:CSUM + 3] / a[CCNT:CCNT + 3])}）")
    print(f"可行候选 {feasible.sum()} 组")
    print("| 候选 | S | E | ΔC | ΔC 去 30 天 | R | E↓ | ΔC↓ |")
    print("|---|---|---|---|---|---|---|---|")

    def row(i):
        return (f"| {label(CANDIDATES[i])} | {s[i]:+.3f} | {e[i]:+.3f} | {dc[i]:+.2f} | {dc30[i]:+.2f} | {r[i]:.2f} "
                f"| {e_d[i]:+.3f} | {dc_d[i]:+.2f} |")

    shown = list(order[: min(20, max(1, feasible.sum()))])
    for i in shown + ([base_i] if base_i not in shown else []):
        print(row(i))

    # 参考：帕累托前沿（只看 E、ΔC，R ≤ 1.5）
    cand = np.flatnonzero(r <= 1.5)
    front = [
        i for i in cand
        if not np.any((e[cand] >= e[i]) & (dc[cand] >= dc[i]) & ((e[cand] > e[i]) | (dc[cand] > dc[i])))
    ]
    print(f"\n参考：开发期 E-ΔC 帕累托前沿（R ≤ 1.5），{len(front)} 组")
    for i in sorted(front, key=lambda i: e[i]):
        print(row(i))

    if not feasible.any():
        print("\n没有可行候选 → 维持当前组")
        return
    w = int(order[0])
    print(f"\nW = {label(CANDIDATES[w])}")
    if w == base_i or s[w] <= 0:
        print("W 不比当前组好 → 维持当前组")
        return

    # ---- 检验 1：PBO ----
    edges = np.linspace(dev_days[0], dev_days[-1] + 1, 17).astype(int)
    bu, bfu = block_sums(up.agg, up.base_first, edges)
    bd, bfd = block_sums(down.agg, down.base_first, edges)
    blocks_up = Metrics(bu, bfu, base_i)
    blocks_down = Metrics(bd, bfd, shared_base)
    below = 0
    combos = list(itertools.combinations(range(16), 8))
    for combo in combos:
        m = np.zeros(16, dtype=bool)
        m[list(combo)] = True
        e_is, dc_is, r_is, s_is = blocks_up.on(m)
        ed_is, dcd_is, _, _ = blocks_down.on(m)
        g = ~changed | ((ed_is[shared_of] >= -0.05) & (dcd_is[shared_of] >= -0.3))
        ok = (r_is <= 1.5) & (e_is >= 0) & (dc_is >= 0) & g
        if not ok.any():
            below += 1
            continue
        chosen = int(np.argmax(np.where(ok, s_is, -np.inf)))
        e_o, dc_o, _, s_o = blocks_up.on(~m)
        score = s_o - ((e_o < 0) | (dc_o < 0))
        rank = (score < score[chosen]).sum() + 0.5 * ((score == score[chosen]).sum() - 1) + 1
        below += rank / (len(CANDIDATES) + 1) <= 0.5
    pbo = below / len(combos)
    pass1 = pbo < 0.5
    print(f"\n检验 1 PBO = {pbo:.3f}（< 0.5 通过）{'通过' if pass1 else '不通过'}")

    # ---- 检验 2：样本外 ----
    e_o, dc_o, r_o, s_o = up.on(oos)
    g_o, ed_o, dcd_o = guard(oos)
    rng = np.random.default_rng(20261019)
    oos_days = np.flatnonzero(oos)
    pair = up.agg[[w, base_i]][:, oos_days]
    bf = up.base_first[oos_days]
    boot = np.empty(2000)
    for b in range(boot.size):
        pick = rng.integers(0, oos_days.size, oos_days.size)
        a = pair[:, pick].sum(1)
        cont = (a[:, CSUM : CSUM + 3] / a[:, CCNT : CCNT + 3]).mean(1) * 100
        boot[b] = a[0, SCORE] / bf[pick].sum() + 0.1 * (cont[0] - cont[1])
    lower = np.quantile(boot, 0.05)
    pass2 = lower > 0 and e_o[w] >= 0 and dc_o[w] >= 0 and r_o[w] <= 1.5 and bool(g_o[w])
    print(f"检验 2 样本外：S {s_o[w]:+.3f}（单侧 95% 下界 {lower:+.4f}），E {e_o[w]:+.3f}，"
          f"ΔC {dc_o[w]:+.2f}，R {r_o[w]:.2f}，E↓ {ed_o[w]:+.3f}，ΔC↓ {dcd_o[w]:+.2f}"
          f" → {'通过' if pass2 else '不通过'}")

    # ---- 检验 3：半年段 ----
    seg_days = [s_ // 86400 - DAY0 for s_ in SEGMENT_STARTS] + [n_days]
    s_wins = c_ok = 0
    for j in range(len(SEGMENT_STARTS)):
        m = np.zeros(n_days, dtype=bool)
        m[seg_days[j] : seg_days[j + 1]] = True
        e_s, dc_s, r_s, s_s = up.on(m)
        s_wins += s_s[w] > 0
        c_ok += dc_s[w] >= 0
        print(f"  {datetime.fromtimestamp(SEGMENT_STARTS[j], UTC):%y-%m}  S {s_s[w]:+.3f}  E {e_s[w]:+.3f}  "
              f"ΔC {dc_s[w]:+.2f}  R {r_s[w]:.2f}")
    pass3 = s_wins >= 7 and c_ok >= 5
    print(f"检验 3 半年段：S > 0 {s_wins}/10，ΔC ≥ 0 {c_ok}/10 → {'通过' if pass3 else '不通过'}")
    final = pass1 and pass2 and pass3
    print(f"\n结论：{'W 全部通过，推荐 W' if final else 'W 未全部通过 → 维持当前组'}")


def sensitivity() -> None:
    """参考：以当前组为中心每次只改一项，开发期与样本外的 E、ΔC、R（看完判据结论后补的说明，不参与选择）。"""
    stored = np.load(OUT / "aggregates.npz")
    surge = stored["surge"]
    base_i = CANDIDATES.index(CURRENT)
    up = Metrics(surge, stored["base_first"], base_i)
    day_t = (np.arange(surge.shape[1]) + DAY0) * 86400
    dev = up.on(day_t >= DEV)
    oos = up.on((day_t >= OOS[0]) & (day_t < OOS[1]))
    print("| 改动 | 开发期 E | 开发期 ΔC | 开发期 R | 样本外 E | 样本外 ΔC | 样本外 R |")
    print("|---|---|---|---|---|---|---|")
    grid = (WINDOWS, BASELINES, QUOTES, BREAKOUTS, RATIOS, CHANGES, MAXES, SWITCHES, SWITCHES, SWITCHES)
    for k, values in enumerate(grid):
        for v in values:
            if v == CURRENT[k]:
                continue
            c = CURRENT[:k] + (v,) + CURRENT[k + 1 :]
            i = CANDIDATES.index(c)
            print(f"| {label(c)} | {dev[0][i]:+.3f} | {dev[1][i]:+.2f} | {dev[2][i]:.2f} "
                  f"| {oos[0][i]:+.3f} | {oos[1][i]:+.2f} | {oos[2][i]:.2f} |")


def runup(which: list[str]) -> None:
    """参考：提醒价相对前 60 分钟最低价的涨幅中位数（开发期 / 样本外）。"""
    rows = Rows()
    for text in which:
        c = CANDIDATES[int(text)] if text.isdigit() else CURRENT
        alerts = rows.surge_alerts(c)
        t = alerts[:, C["t"]]
        for name, m in (("开发期", t >= DEV), ("样本外", (t >= OOS[0]) & (t < OOS[1]))):
            first = m & (alerts[:, C["dir"]] == 1)
            print(f"{label(c)} {name}：全部 {np.median(alerts[m, C['runup']]):.2f}%，第 1 波 "
                  f"{np.median(alerts[first, C['runup']]):.2f}%")


def verify(symbols: list[str]) -> None:
    """与真 LaunchDetector 逐条对照 2025-10~11 两个月（检测器现有参数能表达的候选）。"""
    from price_alert.launch.detector import LaunchDetector
    from price_alert.launch.rule import LaunchRule, Thresholds
    from price_alert.models import Candle

    contracts = load_contracts()
    checks = [c for c in CANDIDATES if not any(c[7:])]
    rng = np.random.default_rng(0)
    picks = [CURRENT] + [checks[k] for k in rng.choice(len(checks), 5, replace=False)]
    for name in symbols:
        series = load_series(name, float(contracts[name]["quanto_multiplier"]))
        lo = (ts("2025-10-01") - series.t0) // 60
        hi = lo + 61 * 1440
        assert series.covered[lo:hi].all()
        sl = slice(lo, hi)

        class Part:
            pass

        part = Part()
        part.t0 = series.t0 + lo * 60
        for field in ("open", "high", "low", "close", "quote"):
            setattr(part, field, getattr(series, field)[sl])
        part.covered = np.ones(hi - lo, dtype=bool)
        scanned = {}
        for window, baseline in itertools.product(WINDOWS, BASELINES):
            got = scan(part.close, part.high, part.low, part.quote, part.covered, window, baseline,
                       LOOSE_QUOTE, LOOSE_SURGE_RATIO, LOOSE_SURGE_CHANGE, DROP[0], DROP[1])
            arr = np.zeros((len(got), len(COLS)))
            if got:
                g = np.array(got)
                arr[:, C["i"]] = g[:, 0]
                arr[:, C["dir"] : C["price"] + 1] = g[:, 1:]
            scanned[(window, baseline)] = arr
        for c in picks:
            w, b, q, bo, ratio, low_, high_, *_ = c
            fake = Rows.__new__(Rows)
            fake.groups = {}
            for sign in (1, -1):
                arr = scanned[(w, b)]
                fake.groups[(w, b, sign)] = arr[arr[:, C["dir"]] == sign]
            mine = sorted(
                [(int(x[C["i"]]), "surge", int(x[C["dir"]])) for x in fake.surge_alerts(c)]
                + [(int(x[C["i"]]), "drop", int(x[C["dir"]])) for x in fake.drop_alerts(c[:4])]
            )
            rule = LaunchRule(
                window_minutes=w, baseline_minutes=b, min_volume_ratio=ratio, min_window_quote=q,
                min_change_percent=low_, require_breakout=bo, alert_drops=True, cooldown_minutes=COOLDOWN,
                realert_step_percent=STEP, max_change_percent=None if math.isinf(high_) else high_,
                drop=Thresholds(*DROP),
            )
            detector = LaunchDetector(rule)
            detector.add_symbol(name, [], 1e9, quanto_multiplier=1)
            real = []
            for k in range(hi - lo):
                moment = datetime.fromtimestamp(part.t0 + k * 60, UTC)
                candle = Candle(moment, part.open[k], part.high[k], part.low[k], part.close[k], part.quote[k])
                for alert in detector.add_candle(name, candle):
                    real.append((k, alert.metrics.direction, alert.wave))
            real.sort()
            status = "一致" if mine == real else f"不一致（真检测器 {len(real)} 条）"
            print(name, label(c), len(mine), status)
            if mine != real:
                only_mine, only_real = sorted(set(mine) - set(real))[:5], sorted(set(real) - set(mine))[:5]
                print("  只在脚本：", only_mine, " 只在检测器：", only_real)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command == "verify":
        verify(sys.argv[2:])
    elif command == "scan":
        run_scan()
    elif command == "evaluate":
        evaluate()
    elif command == "sensitivity":
        sensitivity()
    elif command == "runup":
        runup(sys.argv[2:])
    else:
        report()
