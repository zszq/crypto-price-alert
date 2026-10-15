"""第五轮：市场状态分档。每个状态变量按五分位分档，看做多 / 做空每笔收益在各段是否一致。

用法：r5_regime.py <数据集> <起点 YYYY-MM-DD> <分段点 YYYY-MM-DD,...>
"""

import sys

import numpy as np
from r5_lib import Market, Rule, alert_density, beijing, load, simulate

sys.stdout.reconfigure(encoding="utf-8")
LONG = Rule(confirm=0.02)
SHORT = Rule(delay=3, confirm=0.0, stop_gap=0.05, trail=0.15, trail_atr=np.nan)


def features(data, market: Market) -> dict[str, np.ndarray]:
    t = data.closed_at
    btc = market.at(market.btc, t)
    up24 = alert_density(data, 86400, 1)
    dn24 = alert_density(data, 86400, -1)
    return {
        "宽度30天": market.at(market.breadth[30], t),
        "宽度7天": market.at(market.breadth[7], t),
        "BTC/MA20日": btc / market.at(market.btc_ma[20], t) - 1,
        "BTC/MA50日": btc / market.at(market.btc_ma[50], t) - 1,
        "中位涨跌24h": market.at(market.median_ret[24], t),
        "中位涨跌7天": market.at(market.median_ret[168], t),
        "中位涨跌30天": market.at(market.median_ret[720], t),
        "全市场放量24h/30天": market.at(market.volume_surge, t),
        "24h拉升提醒数": up24,
        "24h下跌提醒数": dn24,
        "24h拉升占比": up24 / np.maximum(up24 + dn24, 1),
        "币自身7天涨跌": data.meta["ret_7d"],
        "币自身24h涨跌": data.meta["ret_24h"],
        "距30天高点": data.meta["start_price"] / data.meta["high_30d"] - 1,
        "距30天低点": data.meta["start_price"] / data.meta["low_30d"] - 1,
        "量比": data.meta["ratio"],
        "窗口成交额": data.meta["window_quote"],
        "24h成交额": data.meta["volume_24h"],
        "已突破": data.meta["broke_out"],
    }


def main() -> None:
    dataset = sys.argv[1]
    start = beijing(*map(int, sys.argv[2].split("-")))
    cuts = [beijing(*map(int, s.split("-"))) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
    data = load(dataset)
    market = Market()
    feats = features(data, market)
    edges = [start, *cuts, data.end + 1]
    for side, rule, label in ((1, LONG, "做多（X 确认2%）"), (-1, SHORT, "做空（等3分钟 确认0）")):
        tr = simulate(data, rule)
        for waves in ("首波", "全部"):
            base = (data.direction == side) & ~np.isnan(tr.net) & (data.closed_at >= start)
            if waves == "首波":
                base &= data.wave == 1
            print(f"==== {label} {waves}：{base.sum()} 笔，每笔 {np.nanmean(tr.net[base]) * 100:+.2f}%")
            for name, x in feats.items():
                ok = base & ~np.isnan(x)
                qs = np.unique(np.nanpercentile(x[ok], [20, 40, 60, 80]))
                bins = np.digitize(x, qs)
                parts = []
                for b in range(len(qs) + 1):
                    m = ok & (bins == b)
                    seg = []
                    for a, z in zip(edges[:-1], edges[1:], strict=True):
                        y = tr.net[m & (data.closed_at >= a) & (data.closed_at < z)]
                        seg.append(f"{y.mean() * 100:+.1f}" if len(y) > 20 else "  · ")
                    lo = -np.inf if b == 0 else qs[b - 1]
                    parts.append(f"[≥{lo:.3g}] {np.mean(tr.net[m]) * 100:+.2f}%({'/'.join(seg)})")
                print(f"  {name}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
