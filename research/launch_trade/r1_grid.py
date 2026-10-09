"""第一轮网格：在 r1_export 导出的价格路径上搜索出场（持仓、止损、止盈、回撤）与确认入场（等待分钟、确认涨幅）。

4~7 月为训练集、8 月起为检验集；按训练集均值排序，检验集只用来对照。结果存 data/backtest/results/r1_grid<轮次>.pkl。

用法：data/backtest/.venv/Scripts/python.exe research/launch_trade/r1_grid.py --round 2
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS  # noqa: E402
from r1_lib import SPLIT, load, simulate_trail  # noqa: E402

# 每组为 (持仓分钟, 止损, 止盈, 回撤, (等待分钟, 确认涨幅))；第 2 轮在第 1 轮发现的「确认入场 + 回撤止损」附近扩大范围。
GRIDS = {
    1: (
        [15, 30, 60, 120, 180, 300],
        [0.02, 0.03, 0.05, 0.07, 0.10],
        [None, 0.05, 0.10, 0.20],
        [None, 0.03, 0.05, 0.08, 0.12],
        [(0, None), (1, 0.0), (3, 0.0), (5, 0.0), (5, 0.02)],
    ),
    2: (
        [120, 180, 240, 300],
        [0.05, 0.07, 0.10, 0.15],
        [None],
        [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, None],
        [
            (0, None),
            (3, 0.0),
            (5, 0.0),
            (5, 0.02),
            (10, 0.0),
            (10, 0.02),
            (10, 0.05),
            (15, 0.0),
            (15, 0.03),
            (20, 0.0),
            (30, 0.0),
        ],
    ),
}

FEATURES, PATH, _ = load()
TRAIN = FEATURES["closed_at"] < SPLIT
FIRST_WAVE = FEATURES["wave"] == 1


def stats(net: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
    x = net[mask & ~np.isnan(net)]
    return len(x), x.mean() if len(x) else np.nan, x.sum()


def run(cfg: tuple) -> list[tuple]:
    hold, stop, take, trail, (delay, confirm) = cfg
    net = simulate_trail(PATH, hold, stop, take, trail, delay, confirm=confirm)
    return [
        (name, cfg, stats(net, sel & TRAIN), stats(net, sel & ~TRAIN))
        for name, sel in (("第1波", FIRST_WAVE), ("全部波次", np.ones_like(FIRST_WAVE)))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=sorted(GRIDS), default=2)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    with ProcessPoolExecutor(10) as executor:
        for result in executor.map(run, list(itertools.product(*GRIDS[args.round])), chunksize=10):
            rows.extend(result)
    with (RESULTS / f"r1_grid{args.round}.pkl").open("wb") as handle:
        pickle.dump(rows, handle)
    for name in ("第1波", "全部波次"):
        # 训练集至少 300 笔，免得少数几笔的组合靠运气排到前面。
        sub = sorted((r for r in rows if r[0] == name and r[2][0] >= 300), key=lambda r: -r[2][1])
        print(f"===== {name}：训练集均值最高的 15 组（检验集只作对照）")
        for _, cfg, train, test in sub[:15]:
            print(
                f"{cfg} 训练 n={train[0]} 均{train[1]:+.2%} 合{train[2] * 100:+.0f} | "
                f"检验 n={test[0]} 均{test[1]:+.2%} 合{test[2] * 100:+.0f}"
            )
        for dim, label in ((0, "持仓"), (1, "止损"), (3, "回撤"), (4, "入场")):
            values: dict = {}
            for _, cfg, train, test in sub:
                values.setdefault(cfg[dim], []).append((train[1], test[1]))
            marginal = "  ".join(
                f"{k}: 训{np.mean([a for a, _ in v]):+.2%}/检{np.mean([b for _, b in v]):+.2%}"
                for k, v in values.items()
            )
            print(f"  {label}边际：{marginal}")


if __name__ == "__main__":
    main()
