"""第五轮研究数据：同时导出放量上涨与放量下跌两个方向的提醒，供做多 / 做空与组合层面（限仓、市场状态）研究。

与 r3_export.py 的区别：
- 下跌方向也判定（与 LaunchDetector 一样按（合约, 方向）分别记冷却与波次），launch_trade_backtest.detect 只做多；
- bars 只留提醒前 PRE=60、提醒后 POST 分钟的开高低收（没有成交额）：五年、两个方向、不设池下限的提醒有十万条量级，
  r3 那样留 24 小时前序和成交额要 10GB 以上；入场前 60 分钟平均振幅只用到前 60 根，其余提醒时的特征直接算进 meta。
输出 data/backtest/results/<--out>/：bars.npy (N, 4, PRE+POST) float64（--float32 时 float32）、
meta.npy (N, len(META))、symbol.npy、meta_names.txt。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r5_export.py --out r5_p30 --side up --set min_window_quote=300000 `
        --set baseline_minutes=360 --set require_breakout=false --set max_change_percent=none --min-volume 0
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_trade_backtest import REPO, RESULTS, PoolRule, _pool_membership  # noqa: E402
from load_data import load_contracts, load_series  # noqa: E402
from r3_export import _parse_value  # noqa: E402

PRE = 60
POST = int(os.environ.get("R5_POST", "1440"))
META = [
    "closed_at",
    "direction",  # +1 放量上涨，-1 放量下跌
    "wave",
    "change",  # 窗口涨跌幅（%），下跌为负
    "ratio",
    "window_quote",
    "volume_24h",
    "broke_out",  # 收盘是否越过基准期最高价（下跌为最低价）
    "start_price",
    "base_high",
    "base_low",
    "ret_1h",  # 提醒前的涨跌（按提醒价算），只用提醒时已收盘的 K 线
    "ret_24h",
    "ret_7d",
    "high_30d",  # 窗口开始前 30 天最高 / 最低价
    "low_30d",
    "range_60",  # 提醒前 60 分钟平均振幅（(高-低)/收）
    "ratio_base",  # 按成交量（币数）算的量比，与 ratio 的差别只是价格变化
]


def base_volume(series) -> np.ndarray:
    """每分钟成交量（币数 = 张数 × 面值）。

    下载站月份的成交额本来就是 张数 × 面值 × 典型价 估出来的，除回去就是币数；REST 月份的成交额是真实值，
    除以典型价是近似（分钟内价格变化很小）。
    """
    typical = (series.high + series.low + series.close) / 3
    return np.divide(series.quote, typical, out=np.zeros_like(series.quote), where=typical > 0)


def detect_both(series, rule, pool, sides=(1, -1), ratio_volume=None) -> list[tuple]:
    """两个方向的提醒：(分钟下标, 方向, 波次, 涨跌幅, 窗口成交额, 是否突破)，按时间排序。

    与 launch_trade_backtest.detect 相同的向量化预筛 + 重复提醒状态机，只是方向分开记录，下跌按最低价判突破、
    再推进按继续下跌的幅度算（LaunchDetector._repeat_outcome 的 sign）。
    ratio_volume 给出时量比按它算（例如 base_volume 的成交量），窗口成交额门槛仍按成交额。
    """
    n = len(series)
    window, baseline, history = rule.window_minutes, rule.baseline_minutes, rule.history_minutes
    if n < history:
        return []
    h, lo, c, q = series.high, series.low, series.close, series.quote
    cq = np.concatenate([[0.0], np.cumsum(q)])
    cv = cq if ratio_volume is None else np.concatenate([[0.0], np.cumsum(ratio_volume)])
    idx = np.arange(history - 1, n)
    window_quote = cq[idx + 1] - cq[idx - window + 1]
    window_volume = cv[idx + 1] - cv[idx - window + 1]
    baseline_sum = cv[idx - window + 1] - cv[idx - history + 1]
    change = (c[idx] / c[idx - window] - 1) * 100
    gaps = np.concatenate([[0], np.cumsum(~series.covered)])
    covered = gaps[idx + 1] - gaps[idx - history + 1] == 0
    base_ok = covered & (window_quote >= rule.min_window_quote) & _pool_membership(series, pool)[idx]
    found = []
    for sign in sides:
        # 放量下跌可以有自己的量比与涨跌幅门槛（LaunchRule.drop），与检测器一样按方向取。
        limits = rule.thresholds("surge" if sign > 0 else "drop")
        ratio_needed = limits.min_volume_ratio * baseline_sum * window
        ratio_ok = (baseline_sum <= 1e-6) | (window_volume * baseline >= ratio_needed)
        candidate = base_ok & ratio_ok & (change * sign >= limits.min_change_percent)
        last: tuple[int, float, int] | None = None
        for pos in np.flatnonzero(candidate):
            i = int(idx[pos])
            base = slice(i - history + 1, i - window + 1)
            broke = bool(c[i] > h[base].max()) if sign > 0 else bool(c[i] < lo[base].min())
            if rule.require_breakout and not broke:
                continue
            closed_at = series.t0 + (i + 1) * 60
            if last is not None and closed_at - last[0] < rule.cooldown_minutes * 60:
                continue
            if last is None or closed_at - last[0] >= rule.baseline_minutes * 60:
                wave = 1
            elif (c[i] / last[1] - 1) * 100 * sign < rule.realert_step_percent:
                continue
            else:
                wave = last[2] + 1
            if rule.overshoots(float(change[pos])):
                last = (closed_at, float(c[i]), wave - 1)
                continue
            last = (closed_at, float(c[i]), wave)
            found.append((i, sign, wave, float(change[pos]), float(window_quote[pos]), broke))
    found.sort()
    return found


def run(args):
    symbol, multiplier, rule, pool, sides, ratio_by = args
    series = load_series(symbol, multiplier)
    if series is None:
        return [], [], []
    n = len(series)
    o, h, lo, c, q = series.open, series.high, series.low, series.close, series.quote
    cq = np.concatenate([[0.0], np.cumsum(q)])
    v = base_volume(series)
    cv = np.concatenate([[0.0], np.cumsum(v)])
    w, history = rule.window_minutes, rule.history_minutes
    bars_list, meta_list = [], []
    found = detect_both(series, rule, pool, sides, v if ratio_by == "base" else None)
    for i, sign, wave, change, window_quote, broke in found:
        bars = np.full((4, PRE + POST), np.nan)
        lo_i = i - PRE + 1
        src_lo, src_hi = max(0, lo_i), min(n, i + 1 + POST)
        cov = series.covered[src_lo:src_hi]
        for row, values in enumerate((o, h, lo, c)):
            chunk = values[src_lo:src_hi].astype(np.float64).copy()
            chunk[~cov] = np.nan
            bars[row, src_lo - lo_i : src_hi - lo_i] = chunk
        # 提醒之后遇到缺口就截断：之后即使有数据也不能当作连续持仓。
        gap = np.isnan(bars[3, PRE:])
        if gap.any():
            bars[:, PRE + int(np.argmax(gap)) :] = np.nan
        base = slice(i - history + 1, i - w + 1)
        base_sum = cq[i - w + 1] - cq[i - history + 1]
        base_volume_sum = cv[i - w + 1] - cv[i - history + 1]
        pre_end = i - w + 1
        r60 = slice(max(0, i - 59), i + 1)

        def back(k: int, i: int = i) -> float:
            return float(c[i] / c[i - k] - 1) if i - k >= 0 else np.nan

        meta_list.append(
            [
                series.t0 + (i + 1) * 60,
                sign,
                wave,
                change,
                window_quote / (base_sum / rule.baseline_minutes * w) if base_sum > 0 else 1e6,
                window_quote,
                float(cq[i + 1] - cq[max(0, i + 1 - 1440)]),
                float(broke),
                float(c[i - w]),
                float(h[base].max()),
                float(lo[base].min()),
                back(60),
                back(1440),
                back(10080),
                float(h[max(0, i - 43200) : pre_end].max()),
                float(lo[max(0, i - 43200) : pre_end].min()),
                float(np.mean((h[r60] - lo[r60]) / c[r60])),
                float(cv[i + 1] - cv[i - w + 1]) / (base_volume_sum / rule.baseline_minutes * w)
                if base_volume_sum > 0
                else 1e6,
            ]
        )
        bars_list.append(bars)
    return bars_list, meta_list, [symbol] * len(meta_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="results 下的输出目录名")
    parser.add_argument("--post", type=int, default=1440, help="提醒后保留的分钟数")
    parser.add_argument("--set", action="append", default=[], metavar="字段=值", help="覆盖 LaunchRule 的字段，可多次")
    parser.add_argument("--min-volume", type=float, help="合约池 24 小时成交额下限，默认取配置")
    parser.add_argument("--side", choices=("both", "up", "down"), default="both", help="只导出一个方向可省一半空间")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "default.yaml")
    # 提醒后保留 15 天时 float64 要近 20GB、内存放不下；float32 的相对精度约 1e-7，只在价格恰好压线时可能判得不同。
    parser.add_argument("--float32", action="store_true", help="bars 按 float32 存，体积减半")
    parser.add_argument(
        "--ratio-by", choices=("quote", "base"), default="quote", help="量比按成交额（同检测器）还是成交量（币数）算"
    )
    args = parser.parse_args()
    # Windows 下子进程重新导入本模块，按环境变量取 POST；父进程启动进程池前设好。
    os.environ["R5_POST"] = str(args.post)
    global POST
    POST = args.post

    from price_alert.config import load_config

    config = load_config(args.config)
    overrides = {key: _parse_value(value) for key, value in (item.split("=", 1) for item in args.set)}
    # 两个方向都导出，alert_drops 不起作用。
    rule = replace(config.launch.to_rule(), **overrides)
    pool = PoolRule(
        config.launch.min_volume_24h_quote if args.min_volume is None else args.min_volume,
        config.launch.max_volume_24h_quote,
        config.gate.universe_exit_volume_ratio,
        config.gate.universe_refresh_seconds,
    )
    print(rule, pool, f"post={args.post} ratio_by={args.ratio_by}", flush=True)
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    sides = {"both": (1, -1), "up": (1,), "down": (-1,)}[args.side]
    jobs = [(symbol, multiplier(symbol), rule, pool, sides, args.ratio_by) for symbol in sorted(contracts)]
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.float32 else np.float64
    meta, names = [], []
    scratch = out / "bars.bin.part"
    count = 0
    with scratch.open("wb") as handle, ProcessPoolExecutor(max_workers=12) as executor:
        for b, m, s in executor.map(run, jobs, chunksize=4):
            for item in b:
                handle.write(np.ascontiguousarray(item, dtype=dtype).tobytes())
            count += len(b)
            meta.extend(m)
            names.extend(s)
    if not count:
        scratch.unlink()
        raise SystemExit("没有提醒")
    # 各合约内部按时间排序，合约之间不排；研究时按 closed_at 排序的地方自己排。
    shape = (count, 4, PRE + POST)
    raw = np.memmap(scratch, dtype=dtype, mode="r", shape=shape)
    stacked = np.lib.format.open_memmap(out / "bars.npy", mode="w+", dtype=dtype, shape=shape)
    for start in range(0, count, 4000):
        stacked[start : start + 4000] = raw[start : start + 4000]
    stacked.flush()
    del raw, stacked
    scratch.unlink()
    np.save(out / "meta.npy", np.array(meta))
    np.save(out / "symbol.npy", np.array(names))
    (out / "meta_names.txt").write_text("\n".join(META), encoding="utf-8")
    meta_arr = np.array(meta)
    print(f"导出 {count} 条提醒（拉升 {(meta_arr[:, 1] > 0).sum()}，下跌 {(meta_arr[:, 1] < 0).sum()}），bars {shape}")


if __name__ == "__main__":
    main()
