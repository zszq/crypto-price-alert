"""查看组合结果：现行参数、单项改动的边际效果。"""

from __future__ import annotations

import argparse
import dataclasses
import pickle

import sim

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="s30_l180")
parser.add_argument("--period", default="dev")
ns = parser.parse_args()
rows = pickle.load(open(sim.DATA / "results" / f"combos_{ns.variant}.pkl", "rb"))
index = {(r["short"], r["long"]): r for r in rows}
days = 20 if ns.period == "dev" else 10
P = ns.period


def line(label, r, base=None):
    n = r[f"{P}_n"] / days
    text = (
        f"{label:<24} {n:6.1f}/天 短{r[f'{P}_short'] / days:5.1f} "
        f"段保留{r[f'{P}_segkeep']:.1%} 重复{r[f'{P}_dup']:.1%} "
        f"持{r[f'{P}_held']:.1%} 假{r[f'{P}_fake']:.1%} 顺{r[f'{P}_cont']:.1%}/逆{r[f'{P}_rev']:.1%} "
        f"大3%{r[f'{P}_big3']:.1%} 延{r[f'{P}_big3_delay']:4.0f}s "
        f"大5%{r[f'{P}_big5']:.1%} 延{r[f'{P}_big5_delay']:4.0f}s"
    )
    if base is not None:
        text += f"  数量{n / (base[f'{P}_n'] / days) - 1:+.0%}"
    print(text)


base = index[(sim.CURRENT.short, sim.CURRENT.long)]
line("现行", base)
s0, l0 = sim.CURRENT.short, sim.CURRENT.long
fields = {"change": "涨跌幅", "atr": "ATR倍数", "trades": "笔数", "confirm": "确认", "cooldown": "冷却"}
for side in ("short", "long"):
    w0 = s0 if side == "short" else l0
    for field, label in fields.items():
        values = sorted(
            {
                getattr(k[0] if side == "short" else k[1], field)
                for k in index
                if (k[0] if side == "short" else k[1]) is not None
            }
        )
        for v in values:
            if v == getattr(w0, field):
                continue
            w = dataclasses.replace(w0, **{field: v})
            key = (w, l0) if side == "short" else (s0, w)
            if key in index:
                line(f"{'短' if side == 'short' else '长'}{label}={v}", index[key], base)
line("关闭长窗口", index[(s0, None)], base)
