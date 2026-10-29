"""第五轮（PLAN.md 第五轮）：短时异动（方案 A）提醒后的出场规则，在逐笔成交上算 1170 组出场的每笔收益。

用法：
  exit5.py alerts       取选参期与样本外的提醒明细，写 exit5_alerts.npz
  exit5.py grid 进程数   逐笔成交上算每条提醒在各组出场下的收益（不扣成本），写 exit5_grid.npz
  exit5.py select       只用选参期选规则、算 PBO，写 exit5_selection.json
  exit5.py oos          样本外按写定判据检验（只运行一次）
  exit5.py trends       止损宽度 × 持有时间的每笔均值表（另报告，不作判据）
"""

from __future__ import annotations

import itertools
import json
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA, month_start
from engine import load_trades
from sim import month_days
from spike_grid import GATES, LONG, PERIODS, SHORT_PARAMS, VARIANTS, alerts_for

ALERTS = DATA / "exit5_alerts.npz"
GRID = DATA / "exit5_grid.npz"
SELECTION = DATA / "exit5_selection.json"

# 方案 A：spike4_selection.json 的 combo (10, 0, 37, "gated")
SCHEME_GATE = GATES[37]
SCHEME_SHORT = SHORT_PARAMS[0]
assert SCHEME_GATE == ((300, -2.0), None, 4.0) and (SCHEME_SHORT.a, SCHEME_SHORT.b, SCHEME_SHORT.c) == (1.0, 1.0, 1)

STOPS = (1.0, 1.5, 2.0, 3.0, 5.0, None)
TRAILS = (None,) + tuple(itertools.product((1.0, 2.0, 3.0), (0.5, 1.0, 1.5, 2.0)))
TAKES = (None, 2.0, 4.0)
HOLDS = (1, 2, 3, 5, 10)
COMBOS = list(itertools.product(STOPS, TRAILS, TAKES, HOLDS))
BASELINE = COMBOS.index((None, None, None, 5))
USER_RULE = COMBOS.index((1.0, (1.0, 1.0), None, 5))
DELAY = 5.0
COST, COST_HIGH = 0.1, 0.2
MODES = ("delayed", "immediate", "random")
YEARS = (PERIODS["sel"][:12], PERIODS["sel"][12:])  # 2024-10~2025-09、2025-10~2026-09（PERIODS 按时间顺序）


def collect_alerts() -> None:
    out = {}
    for period, months in PERIODS.items():
        short, long = VARIANTS[10]
        parts = []
        for window, variant, params in (("short", short, SCHEME_SHORT), ("long", long, LONG)):
            a = alerts_for(variant, months, params, SCHEME_GATE)
            a["window"] = np.full(len(a["t"]), window)
            parts.append(a)
        merged = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "dir", "month", "symbol", "window")}
        order = np.argsort(merged["t"], kind="stable")
        for k, v in merged.items():
            out[f"{period}_{k}"] = v[order]
        print(period, len(order), "条，每天", round(len(order) / sum(month_days(m) for m in months), 1))
    np.savez(ALERTS, **{k: (v.astype(str) if v.dtype == object else v) for k, v in out.items()})


def path_returns(t: np.ndarray, p: np.ndarray, k0: int, sign: int) -> tuple[np.ndarray, np.ndarray] | None:
    """入场那一笔之后（含）到 10 分钟期限后第一笔的逐笔收益（%）；数据不够覆盖期限返回 None。"""
    deadline = t[k0] + 60 * max(HOLDS)
    k1 = np.searchsorted(t, deadline, side="left")
    if k1 >= len(t):
        return None
    seg_t = t[k0 : k1 + 1] - t[k0]
    seg_r = sign * (p[k0 : k1 + 1] / p[k0] - 1) * 100.0
    return seg_t, seg_r


def first(mask: np.ndarray) -> int:
    i = int(np.argmax(mask))
    return i if mask[i] else len(mask)


def combo_returns(seg_t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """每组出场的收益：各类出场事件的最早下标先各算一次，组合只取最小值，不必对每组逐笔重走一遍。"""
    n = len(r)
    best = np.maximum.accumulate(r)
    hold_idx = {h: min(int(np.searchsorted(seg_t, 60 * h, side="left")), n - 1) for h in HOLDS}
    take_idx = {tp: first(r >= tp) for tp in TAKES if tp is not None}
    stop_idx = {s: first(r <= -s) for s in STOPS if s is not None}
    trail_idx, armed_idx = {}, {}
    for tr in TRAILS:
        if tr is None:
            continue
        act, dist = tr
        armed = first(best >= act)
        armed_idx[tr] = armed
        if armed >= n:
            trail_idx[tr] = n
            continue
        # 从最有利价格回撤 dist：(1 + r) / (1 + best) - 1 ≤ −dist（收益按 % 计）
        drop = (1 + r[armed:] / 100) / (1 + best[armed:] / 100) - 1 <= -dist / 100
        trail_idx[tr] = armed + first(drop)
    out = np.empty(len(COMBOS), np.float32)
    for c, (s, tr, tp, h) in enumerate(COMBOS):
        k = hold_idx[h]
        if tp is not None:
            k = min(k, take_idx[tp])
        armed = n
        if tr is not None:
            k = min(k, trail_idx[tr])
            armed = armed_idx[tr]
        # 移动止损启用后取代止损：只认启用之前碰到的止损
        if s is not None and stop_idx[s] < armed:
            k = min(k, stop_idx[s])
        out[c] = r[k]
    return out


def stem_job(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray]:
    stem, times, signs = args
    symbol, month = stem.rsplit("-", 1)
    t, p, _ = load_trades(symbol, month)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    res = np.full((len(MODES), len(times), len(COMBOS)), np.nan, np.float32)
    for i, (ta, sign) in enumerate(zip(times, signs, strict=True)):
        entries = (
            (0, int(np.searchsorted(t, ta + DELAY, side="left")), int(sign)),
            (1, int(np.searchsorted(t, ta, side="right")), int(sign)),
        )
        # 随机对照：同合约同月份随机时刻、随机方向
        lo, hi = max(t[0], month_start(month)), t[-1] - 60 * max(HOLDS) - 60
        rt = rng.uniform(lo, hi) if hi > lo else t[0]
        entries += ((2, int(np.searchsorted(t, rt, side="left")), int(rng.choice((-1, 1)))),)
        for mode, k0, sg in entries:
            if k0 >= len(t):
                continue
            seg = path_returns(t, p, k0, sg)
            if seg is not None:
                res[mode, i] = combo_returns(*seg)
    return stem, res


def run_grid(workers: int) -> None:
    a = np.load(ALERTS)
    jobs, index = [], []
    for period in PERIODS:
        groups = defaultdict(list)
        for i, stem in enumerate(a[f"{period}_symbol"]):
            groups[stem].append(i)
        for stem, idx in groups.items():
            idx = np.array(idx)
            jobs.append((stem, a[f"{period}_t"][idx], a[f"{period}_dir"][idx]))
            index.append((period, idx))
    out = {period: np.full((len(MODES), len(a[f"{period}_t"]), len(COMBOS)), np.nan, np.float32) for period in PERIODS}
    where = {stem: (period, idx) for (stem, *_), (period, idx) in zip(jobs, index, strict=True)}
    # 大文件先跑，免得最后只剩一个进程在读大合约
    jobs.sort(key=lambda j: -len(j[1]))
    with Pool(workers) as pool:
        for n, (stem, res) in enumerate(pool.imap_unordered(stem_job, jobs), 1):
            period, idx = where[stem]
            out[period][:, idx] = res
            if n % 50 == 0 or n == len(jobs):
                print(n, "/", len(jobs), flush=True)
    np.savez(GRID, **out)


def load_period(period: str) -> dict:
    a = np.load(ALERTS)
    g = np.load(GRID)[period]
    # 三种入场任一缺数据（月末）就整条剔除，各口径比较同一批提醒
    valid = ~np.isnan(g).any(axis=(0, 2))
    first_day = month_start(PERIODS[period][0])
    return {
        "ret": g[:, valid],
        "t": a[f"{period}_t"][valid],
        "dir": a[f"{period}_dir"][valid],
        "month": a[f"{period}_month"][valid],
        "window": a[f"{period}_window"][valid],
        "symbol": a[f"{period}_symbol"][valid],
        "day": ((a[f"{period}_t"][valid] - first_day) // 86400).astype(np.int64),
        "dropped": int((~valid).sum()),
    }


def describe(c: int) -> str:
    s, tr, tp, h = COMBOS[c]
    return (
        f"止损 {'不设' if s is None else f'{s:g}%'}、移动止损 {'不设' if tr is None else f'浮盈 {tr[0]:g}% 启用 / 回撤 {tr[1]:g}%'}、"
        f"止盈 {'不设' if tp is None else f'{tp:g}%'}、持有 {h} 分钟"
    )


def rules_used(c: int) -> int:
    s, tr, tp, _ = COMBOS[c]
    return sum(x is not None for x in (s, tr, tp))


def stats(r: np.ndarray, day: np.ndarray, cost: float = COST) -> str:
    net = r - cost
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    daily = np.bincount(day, weights=net)
    curve = np.cumsum(daily)
    dd = (np.maximum.accumulate(np.r_[0, curve]) - np.r_[0, curve]).max()
    return (
        f"{len(net)} 笔，每笔均值 {net.mean():+.3f}%，中位 {np.median(net):+.3f}%，胜率 {(net > 0).mean():.1%}，"
        f"盈亏比 {pf:.2f}，最差 {net.min():+.2f}%，合计 ${net.sum():+,.0f}，按天累计最大回撤 ${dd:,.0f}"
    )


def select() -> None:
    d = load_period("sel")
    r = d["ret"][0] - COST  # 主口径：延迟 5 秒入场，扣 0.1%
    months = d["month"]
    sel_idx = [ALL_MONTHS.index(m) for m in PERIODS["sel"]]
    year_mask = [np.isin(months, [ALL_MONTHS.index(m) for m in y]) for y in YEARS]
    mean = r.mean(0)
    year_mean = np.array([r[m].mean(0) for m in year_mask])
    ok = (year_mean > 0).all(0)
    print(f"选参期有效提醒 {len(r)} 条（剔除 {d['dropped']} 条），两个年度都 > 0 的组合 {ok.sum()} / {len(COMBOS)}")
    for name, c in (("基准", BASELINE), ("用户规则", USER_RULE)):
        print(f"  {name}（{describe(c)}）：{stats(d['ret'][0][:, c], d['day'])}；两年度 {year_mean[0, c]:+.3f}% / {year_mean[1, c]:+.3f}%")
    top = np.argsort(-mean)[:15]
    print("  选参期每笔均值前 15：")
    for c in top:
        print(f"    {mean[c]:+.3f}%（{year_mean[0, c]:+.3f} / {year_mean[1, c]:+.3f}）{'' if ok[c] else '✗年度'} {describe(c)}")
    # CSCV：两个月一块，12 块对半分
    blocks = np.array([sel_idx.index(m) // 2 for m in months])
    bsum = np.array([r[blocks == b].sum(0) for b in range(12)])
    bcnt = np.array([(blocks == b).sum() for b in range(12)])
    logits = []
    for ins in itertools.combinations(range(12), 6):
        ins = list(ins)
        outs = [b for b in range(12) if b not in ins]
        is_mean = bsum[ins].sum(0) / bcnt[ins].sum()
        oos_mean = bsum[outs].sum(0) / bcnt[outs].sum()
        best = int(np.argmax(is_mean))
        rank = (oos_mean < oos_mean[best]).sum() + 0.5 * ((oos_mean == oos_mean[best]).sum() - 1) + 1
        w = rank / (len(COMBOS) + 1)
        logits.append(np.log(w / (1 - w)))
    pbo = float((np.array(logits) <= 0).mean())
    print(f"  PBO {pbo:.2f}")
    if not ok.any():
        print("没有两个年度都 > 0 的组合，按写定规则不推荐任何出场规则。")
        json.dump({"combo": None, "pbo": pbo}, open(SELECTION, "w"))
        return
    cand = np.flatnonzero(ok)
    best = cand[np.argmax(mean[cand])]
    tied = [c for c in cand if mean[c] >= mean[best] - 0.02]
    chosen = min(tied, key=lambda c: (rules_used(c), COMBOS[c][3], -mean[c]))
    print(f"选中：{describe(chosen)}（同等 {len(tied)} 组，最高 {describe(best)} {mean[best]:+.3f}%）")
    print(f"  选参期：{stats(d['ret'][0][:, chosen], d['day'])}")
    json.dump({"combo": int(chosen), "pbo": pbo, "sel_mean": float(mean[chosen])}, open(SELECTION, "w"))


def boot(x: np.ndarray, day: np.ndarray, n: int = 5000) -> tuple[float, float]:
    nd = int(day.max()) + 1
    s = np.bincount(day, weights=x, minlength=nd)
    c = np.bincount(day, minlength=nd)
    pick = np.random.default_rng(0).integers(0, nd, (n, nd))
    return tuple(np.percentile(s[pick].sum(1) / c[pick].sum(1), [2.5, 97.5]))


def neighbors(c: int) -> list[tuple[str, int]]:
    s, tr, tp, h = COMBOS[c]
    out = []
    for vals, pos, name in ((STOPS, 0, "止损"), (TRAILS, 1, "移动止损"), (TAKES, 2, "止盈"), (HOLDS, 3, "持有")):
        cur = list(COMBOS[c])
        k = vals.index(cur[pos])
        for j in (k - 1, k + 1):
            if 0 <= j < len(vals):
                new = list(cur)
                new[pos] = vals[j]
                out.append((f"{name} → {vals[j]}", COMBOS.index(tuple(new))))
    return out


def oos() -> None:
    selection = json.load(open(SELECTION))
    d = load_period("oos")
    s = load_period("sel")
    print(f"样本外有效提醒 {len(d['t'])} 条（剔除 {d['dropped']} 条）")
    for name, c in (("基准", BASELINE), ("用户规则", USER_RULE)):
        print(f"{name}（{describe(c)}）\n  样本外 {stats(d['ret'][0][:, c], d['day'])}"
              f"\n  随机对照 {stats(d['ret'][2][:, c], d['day'])}\n  立即入场 {stats(d['ret'][1][:, c], d['day'])}")
    c = selection["combo"]
    if c is None:
        print("选参期没有选出规则，不做判据。")
        return
    r = d["ret"][0][:, c]
    net = r - COST
    lo, hi = boot(net, d["day"])
    base = d["ret"][0][:, BASELINE].mean() - COST
    ok = [net.mean() > 0 and lo > 0, (r - COST_HIGH).mean() > 0, net.mean() >= base, selection["pbo"] < 0.5]
    print(f"\n选中：{describe(c)}\n  样本外 {stats(r, d['day'])}\n  扣 0.2%：{stats(r, d['day'], COST_HIGH)}")
    print(f"  立即入场 {stats(d['ret'][1][:, c], d['day'])}\n  随机对照 {stats(d['ret'][2][:, c], d['day'])}")
    print(
        f"  ① 每笔均值 {net.mean():+.3f}% [{lo:+.3f}, {hi:+.3f}] {'✓' if ok[0] else '✗'}\n"
        f"  ② 扣 0.2% 均值 {(r - COST_HIGH).mean():+.3f}% {'✓' if ok[1] else '✗'}\n"
        f"  ③ 基准 {base:+.3f}% {'✓' if ok[2] else '✗'}\n"
        f"  ④ PBO {selection['pbo']:.2f} {'✓' if ok[3] else '✗'}\n"
        f"  结论：{'全部通过，推荐' if all(ok) else '未通过，不推荐'}"
    )
    print("  逐月：" + "  ".join(
        f"{ALL_MONTHS[m][4:]}月 {net[d['month'] == m].mean():+.3f}%（{(d['month'] == m).sum()}）" for m in sorted(set(d["month"]))))
    up = d["dir"] > 0
    print(f"  急涨 {net[up].mean():+.3f}%（{up.sum()}）急跌 {net[~up].mean():+.3f}%（{(~up).sum()}）")
    for w in ("short", "long"):
        m = d["window"] == w
        print(f"  {w} 窗口 {net[m].mean():+.3f}%（{m.sum()}）")
    print("  相邻参数（选参期 / 样本外每笔均值，扣 0.1%）：")
    for label, k in neighbors(c):
        print(f"    {label}：{s['ret'][0][:, k].mean() - COST:+.3f}% / {d['ret'][0][:, k].mean() - COST:+.3f}%")


def trends() -> None:
    """止损宽度、持有时间单独变化（其他项不设）时两个时期的每笔均值，另报告不作判据。"""
    for period in PERIODS:
        d = load_period(period)
        r = d["ret"][0] - COST
        print(period)
        for h in HOLDS:
            cells = [f"{'不设' if s is None else s:>4}: {r[:, COMBOS.index((s, None, None, h))].mean():+.3f}" for s in STOPS]
            print(f"  持有 {h:2d} 分钟 | 止损 " + "  ".join(cells))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "alerts":
        collect_alerts()
    elif cmd == "grid":
        run_grid(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == "select":
        select()
    elif cmd == "oos":
        oos()
    elif cmd == "trends":
        trends()
