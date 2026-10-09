"""第三轮研究数据：与 r2_export.py 相同，但提醒后默认保留 48 小时（--post），并可覆盖提醒规则与合约池门槛。

- 第二轮的 bars 只到提醒后 12 小时，持 10 小时已经贴着上限，三年数据里要试更长的持仓；
- 交易用的信号不必与实时提醒的默认参数相同（如再推进幅度、成交额门槛），--set 覆盖 launch 段的字段，导出另一组提醒。
输出 data/backtest/results/<--out>/*.npy，格式与 r2 相同（只是列数不同），r2_lib.load(period, source) 读取。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r3_export.py
    & $P research/launch_trade/r3_export.py --post 1200 --out v_nobreak --set require_breakout=false
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
import r2_export  # noqa: E402

# 子进程（Windows 下是 spawn）会重新导入本模块，r2_export.run 读的是 r2_export.POST，所以在模块顶层按环境变量改，
# 父进程在启动进程池前设置好环境变量，父子进程就都生效。
r2_export.POST = int(os.environ.get("R3_POST", "2880"))


def _parse_value(text: str) -> object:
    if text.lower() in ("none", "null"):
        return None
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--post", type=int, default=2880, help="提醒后保留的分钟数")
    parser.add_argument("--out", default="r3", help="results 下的输出目录名")
    parser.add_argument("--set", action="append", default=[], metavar="字段=值", help="覆盖 LaunchRule 的字段，可多次")
    parser.add_argument("--min-volume", type=float, help="合约池 24 小时成交额下限，默认取配置")
    args = parser.parse_args()
    os.environ["R3_POST"] = str(args.post)
    r2_export.POST = args.post

    from launch_trade_backtest import REPO, RESULTS, PoolRule
    from load_data import load_contracts

    from price_alert.config import load_config

    config = load_config(REPO / "config" / "default.yaml")
    overrides = dict(item.split("=", 1) for item in args.set)
    rule = replace(config.launch.to_rule(), **{key: _parse_value(value) for key, value in overrides.items()})
    pool = PoolRule(
        config.launch.min_volume_24h_quote if args.min_volume is None else args.min_volume,
        config.launch.max_volume_24h_quote,
        config.gate.universe_exit_volume_ratio,
        config.gate.universe_refresh_seconds,
    )
    print(rule, pool, f"post={args.post}")
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    jobs = [(symbol, multiplier(symbol), rule, pool) for symbol in sorted(contracts)]
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    bars, meta, names = [], [], []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for b, m, s in executor.map(r2_export.run, jobs, chunksize=4):
            bars.extend(b)
            meta.extend(m)
            names.extend(s)
    if not bars:
        raise SystemExit("没有提醒，检查 --set / --min-volume 与数据")
    # 直接写进预分配的 memmap，避免 np.stack 再复制一份几 GB。
    shape = (len(bars), *bars[0].shape)
    stacked = np.lib.format.open_memmap(out / "bars.npy", mode="w+", dtype=np.float64, shape=shape)
    for k, b in enumerate(bars):
        stacked[k] = b
    stacked.flush()
    np.save(out / "meta.npy", np.array(meta))
    np.save(out / "symbol.npy", np.array(names))
    (out / "meta_names.txt").write_text("\n".join(r2_export.META), encoding="utf-8")
    print(f"导出 {len(meta)} 条提醒，bars {shape}")


if __name__ == "__main__":
    main()
