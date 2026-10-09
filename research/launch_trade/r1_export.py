"""导出放量启动提醒的特征与之后的价格路径，供快速试验交易规则（不必每次重跑检测）。

输出 data/backtest/results/research_alerts.npz：
- 每条提醒（全部波次）一行特征：波次、窗口涨幅、量比、成交额、提醒前各时段涨跌、BTC 同期涨跌等；
- path：提醒所在分钟收盘后 PATH_MINUTES 分钟的开高低收，按提醒价（收盘价）归一化，没有数据的分钟为 NaN。

用法：data/backtest/.venv/Scripts/python.exe research/launch_trade/r1_export.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_trade_backtest import REPO, RESULTS, PoolRule, detect  # noqa: E402
from load_data import Series, load_contracts, load_series  # noqa: E402

from price_alert.config import load_config  # noqa: E402

PATH_MINUTES = 360
FEATURES = [
    "closed_at",
    "wave",
    "change",
    "ratio",
    "window_quote",
    "volume_24h",
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "ret_3d",
    "vs_7d_high",
    "upper_wick",
    "last_share",
    "baseline_range",
    "btc_1h",
    "btc_4h",
    "btc_24h",
    "listed_days",
]
_btc: Series | None = None


def _btc_series() -> Series:
    global _btc
    if _btc is None:
        contracts = load_contracts()
        _btc = load_series("BTC_USDT", float(contracts["BTC_USDT"]["quanto_multiplier"]))
    return _btc


def _ret(c: np.ndarray, i: int, back: int) -> float:
    return float(c[i] / c[i - back] - 1) if i - back >= 0 else np.nan


def _btc_ret(at: int, back: int) -> float:
    btc = _btc_series()
    i = (at - btc.t0) // 60 - 1
    if i - back < 0 or i >= len(btc):
        return np.nan
    return float(btc.close[i] / btc.close[i - back] - 1)


def run(args: tuple[str, float | None, object, PoolRule]) -> tuple[list[list[float]], list[np.ndarray], list[str]]:
    symbol, multiplier, rule, pool = args
    series = load_series(symbol, multiplier)
    if series is None:
        return [], [], []
    rows, paths, names = [], [], []
    o, h, low, c, q = series.open, series.high, series.low, series.close, series.quote
    for alert in detect(series, rule, pool):
        i = (alert.closed_at - series.t0) // 60 - 1
        window = rule.window_minutes
        base = slice(i - rule.history_minutes + 1, i - window + 1)
        baseline_quote = q[base].sum() / rule.baseline_minutes * window
        ratio = alert.window_quote / baseline_quote if baseline_quote > 0 else 1e6
        high_7d = h[max(0, i - 10080) : i - window + 1].max()
        span = h[i] - low[i]
        rows.append(
            [
                alert.closed_at,
                alert.wave,
                alert.change_percent,
                ratio,
                alert.window_quote,
                alert.volume_24h,
                _ret(c, i, 60),
                _ret(c, i, 360),
                _ret(c, i, 1440),
                _ret(c, i, 4320),
                float(c[i] / high_7d - 1),
                float((h[i] - c[i]) / span) if span > 0 else 0.0,
                float(q[i] / alert.window_quote) if alert.window_quote > 0 else 0.0,
                float(h[base].max() / low[base].min() - 1),
                _btc_ret(alert.closed_at, 60),
                _btc_ret(alert.closed_at, 240),
                _btc_ret(alert.closed_at, 1440),
                (i + 1) / 1440,
            ]
        )
        path = np.full((4, PATH_MINUTES), np.nan, dtype=np.float32)
        end = min(len(series), i + 1 + PATH_MINUTES)
        seg = slice(i + 1, end)
        ok = series.covered[seg]
        # 遇到缺口就截断：之后的分钟即使有数据也不能当作连续持仓。
        stop = int(np.argmin(ok)) if not ok.all() else len(ok)
        for row, values in enumerate((o, h, low, c)):
            path[row, :stop] = values[seg][:stop] / c[i]
        paths.append(path)
        names.append(symbol)
    return rows, paths, names


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
    rows, paths, names = [], [], []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for r, p, n in executor.map(run, jobs, chunksize=4):
            rows.extend(r)
            paths.extend(p)
            names.extend(n)
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(
        RESULTS / "research_alerts.npz",
        features=np.array(rows, dtype=np.float64),
        feature_names=np.array(FEATURES),
        path=np.stack(paths),
        symbol=np.array(names),
    )
    print(f"导出 {len(rows)} 条提醒")


if __name__ == "__main__":
    main()
