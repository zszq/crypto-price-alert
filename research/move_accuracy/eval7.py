"""第七轮样本外验证（PLAN.md 第七轮，2023-10 ~ 2024-03，只运行一次）：读 select7.json，对方案 C、D 与现行逐条判据。

依赖 grid7.py oos / oos4 的汇总（相邻参数、前沿、只加一项的参考）与 alerts_for 的明细（自助法、与短时异动的重叠）。
"""

from __future__ import annotations

import json

import numpy as np
from common import ALL_MONTHS, DATA, month_start
from engine4 import ROWS4
from grid7 import BURST, GATES, LONG, PERIODS, PRE, SHORT_PARAMS, A, B, C, alerts_for
from select7 import CURRENT, PLAIN, describe, references, table
from sim import Params, month_days
from spike_grid import VARIANTS
from spike_select import H3, H5, H5D, S5, N, V, rate

OOS = PERIODS["oos"]
SEL = PERIODS["sel"]
DAYS = sum(month_days(m) for m in OOS)
FIRST = month_start(OOS[0])
# 短时异动（第四轮方案 A，config/default/spike.yaml）：用来算重叠，看新方案是不是把短时异动再报一遍
SPIKE_A = (Params(1.0, 1.0, 1, 15, 30.0, 1), ((300, -2.0), 4.0))


def detail(combo: tuple) -> dict[str, np.ndarray]:
    length, i, g, long = combo
    short, long_variant = VARIANTS[length]
    parts = [alerts_for(short, OOS, SHORT_PARAMS[i], GATES[g])]
    if long != "off":
        parts.append(alerts_for(long_variant, OOS, LONG, GATES[g] if long == "gated" else GATES[PLAIN]))
    return finish(parts)


def finish(parts: list[dict]) -> dict[str, np.ndarray]:
    out = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    out["day"] = ((out["t"] - FIRST) // 86400).astype(np.int64)
    return out


def boot(x: dict, y: dict, col: int, n: int = 5000) -> tuple[float, float, float]:
    """按天块重抽样，x 与 y 的指标率之差与 95% 区间（与第四轮 spike_eval.boot 相同，天数换成本轮样本外）。"""
    nd = int(np.ceil(DAYS))

    def per_day(tab):
        return (np.bincount(tab["day"], weights=tab["lab"][:, col], minlength=nd),
                np.bincount(tab["day"], weights=tab["lab"][:, V], minlength=nd))

    xs, xv = per_day(x)
    ys, yv = per_day(y)
    pick = np.random.default_rng(0).integers(0, nd, (n, nd))
    diff = xs[pick].sum(1) / xv[pick].sum(1) - ys[pick].sum(1) / yv[pick].sum(1)
    return xs.sum() / xv.sum() - ys.sum() / yv.sum(), *np.percentile(diff, [2.5, 97.5])


def summary(tab: dict) -> str:
    lab = tab["lab"]
    v = max(lab[:, V].sum(), 1)
    ret = tab["ret"][lab[:, V] == 1]
    up = tab["dir"] > 0
    months = " ".join(
        f"{ALL_MONTHS[m][2:]} {lab[tab['month'] == m, H5].sum() / max(lab[tab['month'] == m, V].sum(), 1):.0%}"
        for m in sorted(set(tab["month"]))
    )
    return (
        f"每天 {len(lab) / DAYS:.1f} hold5 {lab[:, H5].sum() / v:.1%} side5 {lab[:, S5].sum() / v:.1%} "
        f"hold3 {lab[:, H3].sum() / v:.1%} 延迟5秒 {lab[:, H5D].sum() / v:.1%} "
        f"5分钟收益中位 {np.median(ret[:, 4]) if len(ret) else np.nan:+.2f}% "
        f"急涨 {lab[up, H5].sum() / max(lab[up, V].sum(), 1):.1%} 急跌 {lab[~up, H5].sum() / max(lab[~up, V].sum(), 1):.1%}"
        f"（{months}）"
    )


def random_baseline() -> str:
    rets = []
    for path in sorted(ROWS4.glob("*.npz")):
        if path.stem.rsplit("-", 1)[1] in OOS and ".tmp" not in path.name:
            with np.load(path) as d:
                if "rand_ret" in d:
                    rets.append(d["rand_ret"][(d["rand_pool"] >> 1) & 1 == 1])
    r = np.concatenate(rets)
    r = r[~np.isnan(r).any(1)]
    pos = r > 0
    return f"随机：hold5 {pos[:, :5].all(1).mean():.1%} side5 {pos[:, 4].mean():.1%} hold3 {pos[:, :3].all(1).mean():.1%}（{len(r)} 个）"


def overlap(tab: dict, spike: dict, seconds: float = 10.0) -> float:
    """同合约同方向、±seconds 秒内有短时异动提醒的比例。"""
    hit = np.zeros(len(tab["t"]), bool)
    for sym in np.unique(tab["symbol"]):
        mine = np.flatnonzero(tab["symbol"] == sym)
        other = np.flatnonzero(spike["symbol"] == sym)
        for d in (-1, 1):
            o = np.sort(spike["t"][other[spike["dir"][other] == d]])
            m = mine[tab["dir"][mine] == d]
            if not len(o) or not len(m):
                continue
            k = np.clip(np.searchsorted(o, tab["t"][m]), 1, len(o) - 1) if len(o) > 1 else np.zeros(len(m), int)
            near = np.minimum(np.abs(o[k] - tab["t"][m]), np.abs(o[np.maximum(k - 1, 0)] - tab["t"][m]))
            hit[m] = near <= seconds
    return hit.mean()


def neighbors(combo: tuple, tab, data_oos, data_sel) -> None:
    """任一项挪一格，选参期 / 样本外 hold5、side5 与每天条数。"""
    length, i, g, long = combo
    p = SHORT_PARAMS[i]
    pre, r = GATES[g]
    found = []
    for vals, idx, name in ((A, 0, "涨跌幅"), (B, 1, "ATR"), (C, 2, "确认")):
        cur = [p.a, p.b, p.c]
        k = vals.index(cur[idx])
        for j in (k - 1, k + 1):
            if 0 <= j < len(vals):
                new = list(cur)
                new[idx] = vals[j]
                ii = next(n for n, q in enumerate(SHORT_PARAMS) if [q.a, q.b, q.c] == new)
                found.append((f"{name} {vals[j]:g}", (length, ii, g, long)))
    found.append((f"窗口 {40 - length} 秒", (40 - length, i, g, long)))
    for vals, idx, name in ((PRE, 0, "前置"), (BURST, 1, "暴增")):
        k = vals.index((pre, r)[idx])
        for j in sorted({k - 1, k + 1, 0}):
            if 0 <= j < len(vals) and j != k:
                new = [pre, r]
                new[idx] = vals[j]
                found.append((f"{name} {'不限' if vals[j] is None else vals[j]}", (length, i, GATES.index(tuple(new)), long)))
    for lo in ("off", "plain", "gated"):
        if lo != long:
            found.append((f"长窗口 {lo}", (length, i, g, lo)))
    days_sel = sum(month_days(m) for m in SEL)
    for label, c in found:
        k = tab.index(c)
        print(f"    {label:16s} 选参期 hold5 {rate(data_sel, H5)[k]:.1%} side5 {rate(data_sel, S5)[k]:.1%}（{data_sel[k, :, N].sum() / days_sel:.1f}）"
              f" / 样本外 hold5 {rate(data_oos, H5)[k]:.1%} side5 {rate(data_oos, S5)[k]:.1%}（{data_oos[k, :, N].sum() / DAYS:.1f}）")


def frontier(tab, data, days, base: int) -> None:
    per_day = data[:, :, N].sum(1) / days
    h5, s5 = rate(data, H5), rate(data, S5)
    for share in (1.0, 0.75, 0.5, 0.25, 0.1):
        cand = np.flatnonzero(per_day >= per_day[base] * share)
        k = cand[np.argmax(h5[cand])]
        print(f"  ≥ 现行 × {share:<4g}: hold5 {h5[k]:.1%} side5 {s5[k]:.1%} 每天 {per_day[k]:.1f}  {describe(tab[k])}")


if __name__ == "__main__":
    selection = json.load(open(DATA / "select7.json", encoding="utf-8"))
    tab, data_oos = table(np.load(DATA / "grid7_oos.npz"), [ALL_MONTHS.index(m) for m in OOS])
    _, data_sel = table(np.load(DATA / "grid7_sel.npz"))
    print(random_baseline())
    current = detail(CURRENT)
    print("现行：" + summary(current))
    p, gate = SPIKE_A
    spike = finish([alerts_for("L10F10", OOS, p, gate), alerts_for("L180F10", OOS, LONG, gate)])
    print("短时异动（方案 A）：" + summary(spike))
    print(f"现行与短时异动重叠 {overlap(current, spike):.1%}")
    for name, info in selection.items():
        combo = (int(info["combo"][0]), int(info["combo"][1]), int(info["combo"][2]), info["combo"][3])
        tab_ = detail(combo)
        print(f"\n方案 {name}：{describe(combo)}\n  样本外 {summary(tab_)}")
        print(f"  选参期 每天 {info['per_day']:.1f} hold5 {info['hold5']:.1%} side5 {info['side5']:.1%}")
        h5, s5, h3, hd = (boot(tab_, current, col) for col in (H5, S5, H3, H5D))
        ok = [h5[0] > 0 and h5[1] > 0, s5[0] > 0 and s5[1] > 0, h3[0] >= 0 and hd[0] >= 0, info["pbo"] < 0.5]
        print(
            f"  ① hold5 差 {h5[0]:+.1%} [{h5[1]:+.1%}, {h5[2]:+.1%}] {'✓' if ok[0] else '✗'}\n"
            f"  ② side5 差 {s5[0]:+.1%} [{s5[1]:+.1%}, {s5[2]:+.1%}] {'✓' if ok[1] else '✗'}\n"
            f"  ③ hold3 差 {h3[0]:+.1%}、延迟 5 秒 hold5 差 {hd[0]:+.1%} {'✓' if ok[2] else '✗'}\n"
            f"  ④ PBO {info['pbo']:.2f} {'✓' if ok[3] else '✗'}\n  结论：{'全部通过，推荐' if all(ok) else '未通过，不推荐'}"
        )
        print(f"  与短时异动重叠 {overlap(tab_, spike):.1%}")
        print("  相邻参数：")
        neighbors(combo, tab, data_oos, data_sel)
    print("\n样本外：只改一项的现行参数")
    references(tab, data_oos, DAYS)
    print("\n样本外前沿（不要求优于现行，每天至少 → 最高 hold5）：")
    frontier(tab, data_oos, DAYS, tab.index(CURRENT))
    _, data_oos4 = table(np.load(DATA / "grid7_oos4.npz"), [ALL_MONTHS.index(m) for m in PERIODS["oos4"]])
    days4 = sum(month_days(m) for m in PERIODS["oos4"])
    print("\n参考（2024-04 ~ 09，第四轮样本外，已不干净）：")
    for name, info in [("现行", {"combo": CURRENT})] + list(selection.items()):
        c = (int(info["combo"][0]), int(info["combo"][1]), int(info["combo"][2]), info["combo"][3])
        k = tab.index(c)
        print(f"  {name}: 每天 {data_oos4[k, :, N].sum() / days4:.1f} hold5 {rate(data_oos4, H5)[k]:.1%} side5 {rate(data_oos4, S5)[k]:.1%}")
