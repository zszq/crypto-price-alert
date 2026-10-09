"""在 research_alerts.npz 上快速模拟交易规则（向量化，一组参数几毫秒）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from paths import RESULTS

BEIJING = timezone(timedelta(hours=8))
START = int(datetime(2026, 4, 10, tzinfo=BEIJING).timestamp())
# 4~7 月找规则，8 月起只用来检验，不参与选参数。
SPLIT = int(datetime(2026, 8, 1, tzinfo=BEIJING).timestamp())


def load() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data = np.load(RESULTS / "research_alerts.npz")
    names = list(data["feature_names"])
    features = data["features"]
    keep = features[:, 0] >= START
    f = {name: features[keep, k] for k, name in enumerate(names)}
    return f, data["path"][keep], data["symbol"][keep]


def simulate(
    path: np.ndarray,
    hold: int,
    stop: float,
    take: float | None = None,
    delay: int = 0,
    fee: float = 0.0005,
    short: bool = False,
    entry_filter: np.ndarray | None = None,
) -> np.ndarray:
    """返回每笔扣费收益，无法完整模拟的为 NaN。delay 为提醒后等几分钟再按开盘价入场。"""
    o = path[:, 0, delay : delay + hold].astype(np.float64)
    h = path[:, 1, delay : delay + hold].astype(np.float64)
    low = path[:, 2, delay : delay + hold].astype(np.float64)
    c = path[:, 3, delay : delay + hold].astype(np.float64)
    entry = o[:, :1]
    o, h, low, c = o / entry, h / entry, low / entry, c / entry
    if short:
        # 做空把价格取倒数的方向翻过来：价格涨 = 亏损。用 2 - x 线性镜像，收益按入场价计算。
        o, h, low, c = 2 - o, 2 - low, 2 - h, 2 - c
    n = hold
    stop_hit = low <= 1 - stop
    take_hit = h >= 1 + take if take is not None else np.zeros_like(stop_hit)
    first_stop = np.where(stop_hit.any(1), stop_hit.argmax(1), n)
    first_take = np.where(take_hit.any(1), take_hit.argmax(1), n)
    rows = np.arange(len(path))
    exit_price = c[:, -1].copy()
    is_stop = first_stop < first_take
    is_take = first_take < first_stop
    same = (first_stop == first_take) & (first_stop < n)
    k = np.minimum(first_stop, n - 1)
    gap_take = same & (o[rows, k] >= 1 + (take or 0)) if take is not None else np.zeros(len(path), bool)
    is_stop |= same & ~gap_take
    is_take |= gap_take
    exit_price[is_stop] = np.minimum(o[rows, first_stop.clip(max=n - 1)], 1 - stop)[is_stop]
    if take is not None:
        kt = first_take.clip(max=n - 1)
        exit_price[is_take] = np.maximum(o[rows, kt], 1 + take)[is_take]
    gross = exit_price - 1
    net = gross - fee - fee * exit_price
    # 平仓前数据中断的无法判断结果。
    exit_minute = np.where(is_stop, first_stop, np.where(is_take, first_take, n - 1))
    valid = np.array([not np.isnan(c[r, : exit_minute[r] + 1]).any() for r in rows]) & ~np.isnan(entry[:, 0])
    if entry_filter is not None:
        valid &= entry_filter
    return np.where(valid, net, np.nan)


def report(net: np.ndarray, mask: np.ndarray, closed_at: np.ndarray, label: str = "") -> str:
    parts = []
    for name, period in (("训练", closed_at < SPLIT), ("检验", closed_at >= SPLIT), ("全部", np.ones_like(mask))):
        x = net[mask & period & ~np.isnan(net)]
        if len(x) == 0:
            parts.append(f"{name} 0 笔")
            continue
        parts.append(
            f"{name} {len(x)}笔 均{x.mean():+.2%} 中{np.median(x):+.2%} 胜{(x > 0).mean():.0%} 合{x.sum() * 100:+.0f}"
        )
    return f"{label} | " + " | ".join(parts)


def simulate_trail(
    path: np.ndarray,
    hold: int,
    stop: float,
    take: float | None = None,
    trail: float | None = None,
    delay: int = 0,
    fee: float = 0.0005,
    confirm: float | None = None,
    return_exit: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """逐分钟模拟（向量化到所有提醒）：固定止损、可选止盈、可选移动止损（从持仓期最高价回撤 trail 平仓）。
    confirm：等 delay 分钟后入场，且入场价要比提醒价高至少 confirm（如 0 表示不低于提醒价），否则不交易。
    同一分钟触及多条线时保守地按先出亏损的线算：开盘跳空 → 止损/移动止损 → 止盈。"""
    n_alerts = len(path)
    entry = path[:, 0, delay].astype(np.float64)
    active = ~np.isnan(entry)
    if confirm is not None:
        active &= entry >= 1 + confirm
    peak = entry.copy()
    result = np.full(n_alerts, np.nan)
    open_ = active.copy()
    # 平仓所在分钟（相对提醒后第一分钟的下标），供同一合约不重复持仓等组合层面的检查。
    exit_minute = np.full(n_alerts, -1)
    for k in range(delay, delay + hold):
        if not open_.any():
            break
        o, h, low, c = (path[:, row, k].astype(np.float64) for row in range(4))
        missing = open_ & np.isnan(c)
        open_ &= ~missing  # 数据中断：无法判断，保持 NaN
        line = entry * (1 - stop)
        if trail is not None:
            # 移动止损线只用上一分钟为止的最高价，本分钟的高点不能先拉高止损线再触发。
            line = np.maximum(line, peak * (1 - trail))
        gap = open_ & (o <= line)
        hit = open_ & ~gap & (low <= line)
        exit_price = np.where(gap, o, line)
        done = gap | hit
        result[done] = exit_price[done] / entry[done]
        exit_minute[done] = k
        open_ &= ~done
        if take is not None:
            target = entry * (1 + take)
            took = open_ & (h >= target)
            result[took] = np.maximum(o, target)[took] / entry[took]
            exit_minute[took] = k
            open_ &= ~took
        peak = np.where(open_, np.maximum(peak, h), peak)
        if k == delay + hold - 1:
            result[open_] = c[open_] / entry[open_]
            exit_minute[open_] = k
    net = result - 1 - fee - fee * result
    return (net, exit_minute) if return_exit else net
