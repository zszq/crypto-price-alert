"""第二版研究数据：每条提醒保存提醒前 PRE 分钟、提醒后 POST 分钟的完整开高低收与成交额，以及提醒时的价位。

成交量类指标（等待期、持仓期的成交额变化）需要提醒后的成交额，压力/支撑类指标需要提醒前的价格，第一版只有提醒后的价格。
输出 data/backtest/results/r2/*.npy（不压缩，加载快）：
- bars: (N, 5, PRE+POST) float64，开高低收、计价成交额；第 PRE-1 列是提醒所在分钟（窗口最后一根），没有数据为 NaN；
- meta: (N, len(META)) float64；symbol: (N,) 合约名。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_trade_backtest import REPO, RESULTS, PoolRule, detect  # noqa: E402
from load_data import load_contracts, load_series  # noqa: E402

from price_alert.config import load_config  # noqa: E402

PRE = 1440
POST = 720
META = [
    "closed_at",
    "wave",
    "change",
    "ratio",
    "window_quote",
    "volume_24h",
    "breakout",  # 基准期最高价：放量突破的这条压力位，突破后理论上变成支撑
    "baseline_low",
    "start_price",
    "baseline_quote_per_min",
    "high_24h",
    "low_24h",
    "high_7d",
    "high_30d",
    "low_7d",
    "btc_1h",
    "btc_4h",
    "btc_24h",
]


def _btc_return(btc, closed_at: int, back: int) -> float:
    """提醒前 back 分钟 BTC 的涨跌，用来看大盘环境。"""
    j = (closed_at - btc.t0) // 60 - 1
    return float(btc.close[j] / btc.close[j - back] - 1) if 0 <= j - back and j < len(btc) else np.nan


def run(args):
    symbol, multiplier, rule, pool = args
    series = load_series(symbol, multiplier)
    if series is None:
        return [], [], []
    btc_contracts = load_contracts()
    btc = load_series("BTC_USDT", float(btc_contracts["BTC_USDT"]["quanto_multiplier"]))
    n = len(series)
    o, h, low, c, q = series.open, series.high, series.low, series.close, series.quote
    bars_list, meta_list, names = [], [], []
    for alert in detect(series, rule, pool):
        i = (alert.closed_at - series.t0) // 60 - 1
        w = rule.window_minutes
        base = slice(i - rule.history_minutes + 1, i - w + 1)
        lo_i = i - PRE + 1
        bars = np.full((5, PRE + POST), np.nan)
        src_lo, src_hi = max(0, lo_i), min(n, i + 1 + POST)
        cov = series.covered[src_lo:src_hi]
        dst = slice(src_lo - lo_i, src_hi - lo_i)
        for row, values in enumerate((o, h, low, c, q)):
            chunk = values[src_lo:src_hi].astype(np.float64).copy()
            chunk[~cov] = np.nan
            bars[row, dst] = chunk
        # 提醒之后遇到缺口就截断：之后即使有数据也不能当作连续持仓。
        post = bars[3, PRE:]
        gap = np.isnan(post)
        if gap.any():
            bars[:, PRE + int(np.argmax(gap)) :] = np.nan

        pre_end = i - w + 1  # 窗口开始之前
        meta_list.append(
            [
                alert.closed_at,
                alert.wave,
                alert.change_percent,
                alert.window_quote / (q[base].sum() / rule.baseline_minutes * w) if q[base].sum() > 0 else 1e6,
                alert.window_quote,
                alert.volume_24h,
                float(h[base].max()),
                float(low[base].min()),
                float(c[i - w]),
                float(q[base].mean()),
                float(h[max(0, i - 1440) : pre_end].max()),
                float(low[max(0, i - 1440) : pre_end].min()),
                float(h[max(0, i - 10080) : pre_end].max()),
                float(h[max(0, i - 43200) : pre_end].max()),
                float(low[max(0, i - 10080) : pre_end].min()),
                _btc_return(btc, alert.closed_at, 60),
                _btc_return(btc, alert.closed_at, 240),
                _btc_return(btc, alert.closed_at, 1440),
            ]
        )
        bars_list.append(bars)
        names.append(symbol)
    return bars_list, meta_list, names


def main() -> None:
    config = load_config(REPO / "config" / "default.yaml")
    rule = config.launch.to_rule()
    pool = PoolRule(
        config.launch.min_volume_24h_quote,
        config.launch.max_volume_24h_quote,
        config.gate.universe_exit_volume_ratio,
        config.gate.universe_refresh_seconds,
    )
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    jobs = [(symbol, multiplier(symbol), rule, pool) for symbol in sorted(contracts)]
    bars, meta, names = [], [], []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for b, m, s in executor.map(run, jobs, chunksize=4):
            bars.extend(b)
            meta.extend(m)
            names.extend(s)
    out = RESULTS / "r2"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "bars.npy", np.stack(bars))
    np.save(out / "meta.npy", np.array(meta))
    np.save(out / "symbol.npy", np.array(names))
    (out / "meta_names.txt").write_text("\n".join(META), encoding="utf-8")
    print(f"导出 {len(meta)} 条提醒，bars {np.stack(bars).shape}")


if __name__ == "__main__":
    main()
