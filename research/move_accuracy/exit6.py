"""第六轮（PLAN.md 第六轮）：只做急涨、只用短窗口，在第五轮同样的 1170 组出场上选规则并在新下载的 2023-10 ~ 2024-03 检验。

出场计算、候选族、成本与选参规则都沿用 exit5.py；不同的只有提醒（方案 A 短窗口的急涨）与随机对照（随机时刻做多）。
用法：
  exit6.py alerts 时期 ...   取提醒明细（sel、ref、oos），写 exit6_alerts_<时期>.npz
  exit6.py grid 时期 进程数   算每条提醒在各组出场下的收益，写 exit6_grid_<时期>.npy
  exit6.py select           只用选参期选规则、算 PBO，写 exit6_selection.json
  exit6.py oos              2023-10 ~ 2024-03 按写定判据检验（只运行一次），并列出 2024-04 ~ 09 作参考
"""

from __future__ import annotations

import itertools
import json
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA, EXIT_OOS_MONTHS, SPIKE_OOS_MONTHS, month_start
from engine import load_trades
from exit5 import (
    BASELINE,
    COMBOS,
    COST,
    COST_HIGH,
    DELAY,
    HOLDS,
    MODES,
    SCHEME_GATE,
    SCHEME_SHORT,
    USER_RULE,
    YEARS,
    boot,
    combo_returns,
    describe,
    neighbors,
    path_returns,
    rules_used,
    stats,
)
from sim import month_days
from spike_grid import PERIODS as SPIKE_PERIODS
from spike_grid import VARIANTS, alerts_for

PERIODS = {"sel": SPIKE_PERIODS["sel"], "ref": SPIKE_OOS_MONTHS, "oos": EXIT_OOS_MONTHS}
SELECTION = DATA / "exit6_selection.json"


def alerts_path(period: str):
    return DATA / f"exit6_alerts_{period}.npz"


def grid_path(period: str):
    return DATA / f"exit6_grid_{period}.npy"


def collect_alerts(period: str) -> None:
    months = PERIODS[period]
    # 急跌提醒在检测器里同样占用短窗口冷却，所以先按完整检测器取提醒，再只留急涨
    a = alerts_for(VARIANTS[10][0], months, SCHEME_SHORT, SCHEME_GATE)
    up = a["dir"] > 0
    order = np.argsort(a["t"][up], kind="stable")
    out = {k: a[k][up][order] for k in ("t", "dir", "month", "symbol")}
    out["symbol"] = out["symbol"].astype(str)
    np.savez(alerts_path(period), **out)
    print(period, f"急涨 {up.sum()} 条（急跌 {(~up).sum()} 条不开仓），每天 {up.sum() / sum(month_days(m) for m in months):.1f}")


def stem_job(args: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    stem, times = args
    symbol, month = stem.rsplit("-", 1)
    t, p, _ = load_trades(symbol, month)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    res = np.full((len(MODES), len(times), len(COMBOS)), np.nan, np.float32)
    for i, ta in enumerate(times):
        lo, hi = max(t[0], month_start(month)), t[-1] - 60 * max(HOLDS) - 60
        rt = rng.uniform(lo, hi) if hi > lo else t[0]
        # 随机对照与信号同方向（做多），区分信号本身与这段时间合约整体上涨
        entries = (
            (0, int(np.searchsorted(t, ta + DELAY, side="left"))),
            (1, int(np.searchsorted(t, ta, side="right"))),
            (2, int(np.searchsorted(t, rt, side="left"))),
        )
        for mode, k0 in entries:
            if k0 >= len(t):
                continue
            seg = path_returns(t, p, k0, 1)
            if seg is not None:
                res[mode, i] = combo_returns(*seg)
    return stem, res


def run_grid(period: str, workers: int) -> None:
    a = np.load(alerts_path(period))
    groups = defaultdict(list)
    for i, stem in enumerate(a["symbol"]):
        groups[stem].append(i)
    jobs = sorted(((stem, a["t"][idx]) for stem, idx in groups.items()), key=lambda j: -len(j[1]))
    out = np.full((len(MODES), len(a["t"]), len(COMBOS)), np.nan, np.float32)
    with Pool(workers) as pool:
        for n, (stem, res) in enumerate(pool.imap_unordered(stem_job, jobs), 1):
            out[:, groups[stem]] = res
            if n % 50 == 0 or n == len(jobs):
                print(n, "/", len(jobs), flush=True)
    np.save(grid_path(period), out)


def load_period(period: str) -> dict:
    a = np.load(alerts_path(period))
    g = np.load(grid_path(period))
    valid = ~np.isnan(g).any(axis=(0, 2))
    first_day = month_start(PERIODS[period][0])
    return {
        "ret": g[:, valid],
        "t": a["t"][valid],
        "month": a["month"][valid],
        "symbol": a["symbol"][valid],
        "day": ((a["t"][valid] - first_day) // 86400).astype(np.int64),
        "dropped": int((~valid).sum()),
    }


def select() -> None:
    d = load_period("sel")
    r = d["ret"][0] - COST
    months = d["month"]
    sel_idx = [ALL_MONTHS.index(m) for m in PERIODS["sel"]]
    year_mask = [np.isin(months, [ALL_MONTHS.index(m) for m in y]) for y in YEARS]
    mean = r.mean(0)
    year_mean = np.array([r[m].mean(0) for m in year_mask])
    ok = (year_mean > 0).all(0)
    print(f"选参期有效提醒 {len(r)} 条（剔除 {d['dropped']} 条），两个年度都 > 0 的组合 {ok.sum()} / {len(COMBOS)}")
    for name, c in (("基准", BASELINE), ("用户规则", USER_RULE)):
        print(f"  {name}（{describe(c)}）：{stats(d['ret'][0][:, c], d['day'])}；两年度 {year_mean[0, c]:+.3f}% / {year_mean[1, c]:+.3f}%")
        print(f"    随机做多 {stats(d['ret'][2][:, c], d['day'])}")
    print("  选参期每笔均值前 15：")
    for c in np.argsort(-mean)[:15]:
        print(f"    {mean[c]:+.3f}%（{year_mean[0, c]:+.3f} / {year_mean[1, c]:+.3f}）{'' if ok[c] else '✗年度'} {describe(c)}")
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


def report(d: dict, c: int, title: str) -> None:
    r = d["ret"][0][:, c]
    print(f"  {title} {stats(r, d['day'])}；亏损超过 20% {int((r <= -20).sum())} 笔")
    print(f"    立即入场 {stats(d['ret'][1][:, c], d['day'])}\n    随机做多 {stats(d['ret'][2][:, c], d['day'])}")


def oos() -> None:
    selection = json.load(open(SELECTION))
    d = load_period("oos")
    ref = load_period("ref")
    sel = load_period("sel")
    print(f"2023-10 ~ 2024-03 有效提醒 {len(d['t'])} 条（剔除 {d['dropped']} 条），每天 {len(d['t']) / sum(month_days(m) for m in PERIODS['oos']):.1f}")
    for name, c in (("基准", BASELINE), ("用户规则", USER_RULE)):
        print(f"{name}（{describe(c)}）")
        report(d, c, "2023-10~2024-03")
        report(ref, c, "参考 2024-04~09")
    c = selection["combo"]
    if c is None:
        print("选参期没有选出规则，不做判据。")
        return
    r = d["ret"][0][:, c]
    net = r - COST
    lo, hi = boot(net, d["day"])
    base = d["ret"][0][:, BASELINE].mean() - COST
    ok = [net.mean() > 0 and lo > 0, (r - COST_HIGH).mean() > 0, net.mean() >= base, selection["pbo"] < 0.5]
    print(f"\n选中：{describe(c)}")
    report(d, c, "2023-10~2024-03")
    print(f"    扣 0.2%：{stats(r, d['day'], COST_HIGH)}")
    report(ref, c, "参考 2024-04~09")
    print(
        f"  ① 每笔均值 {net.mean():+.3f}% [{lo:+.3f}, {hi:+.3f}] {'✓' if ok[0] else '✗'}\n"
        f"  ② 扣 0.2% 均值 {(r - COST_HIGH).mean():+.3f}% {'✓' if ok[1] else '✗'}\n"
        f"  ③ 基准 {base:+.3f}% {'✓' if ok[2] else '✗'}\n"
        f"  ④ PBO {selection['pbo']:.2f} {'✓' if ok[3] else '✗'}\n"
        f"  结论：{'全部通过，推荐' if all(ok) else '未通过，不推荐'}"
    )
    print("  逐月：" + "  ".join(
        f"{ALL_MONTHS[m][2:]} {net[d['month'] == m].mean():+.3f}%（{(d['month'] == m).sum()}）" for m in sorted(set(d["month"]))))
    print("  相邻参数（选参期 / 2023-10~2024-03 / 参考 2024-04~09，每笔均值扣 0.1%）：")
    for label, k in neighbors(c):
        print(f"    {label}：{sel['ret'][0][:, k].mean() - COST:+.3f}% / {d['ret'][0][:, k].mean() - COST:+.3f}% / "
              f"{ref['ret'][0][:, k].mean() - COST:+.3f}%")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "alerts":
        for period in sys.argv[2:]:
            collect_alerts(period)
    elif cmd == "grid":
        run_grid(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif cmd == "select":
        select()
    elif cmd == "oos":
        oos()
