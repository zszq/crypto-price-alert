"""在 record.py 的逐秒记录上按任意（更严的）参数重算提醒，并计算提醒质量指标。

确认、补齐秒、空档复核、冷却的处理与 src/price_alert/detectors/spike/detector.py 一致，verify.py 用真实检测器对拍。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

# 行情、记录与结果只放本机（data/backtest/move/，不进版本库），代码在这里
DATA = Path(__file__).resolve().parents[2] / "data" / "backtest" / "move"
MONTH_START = 1788192000  # 2026-09-01 00:00 UTC
DEV_END = MONTH_START + 20 * 86400  # 开发期 9-01~9-20，留出期 9-21~9-30
# 记录列
SEC, NOW, CARRIED, CHG, MATR, TRADES, PRICE, BASE, LCHG, LATR = range(10)


@dataclass(frozen=True)
class Win:
    change: float
    atr: float
    trades: int
    confirm: int
    cooldown: int


@dataclass(frozen=True)
class Params:
    short: Win
    long: Win | None
    min_volume: float = 8e6


CURRENT = Params(Win(1.2, 1.5, 15, 2, 30), Win(1.6, 1.8, 30, 3, 180))
# 第一轮（九月开发期）与第二轮（4~7 月开发期）的推荐，见 move_params.md 第三、八节
ROUND1 = Params(Win(2.0, 2.0, 15, 3, 60), CURRENT.long)
ROUND2 = Params(Win(2.0, 2.5, 15, 3, 30), CURRENT.long)
# 第三轮（以质量为目标）建议的一组：短窗口收紧、长窗口不动，见 move_params.md 第九节
ROUND3 = Params(Win(3.0, 2.5, 15, 5, 30), CURRENT.long)


OCT_START = 1790956800  # 2026-10-03 00:00 UTC，10 月留出期 10-03~10-11


def dataset(variant: str) -> dict:
    """九月数据（开发期 + 留出期）、10 月 REST 数据（variant 以 oct 开头，全部算作第二个留出期）
    或第二轮的 4~8 月下载站数据（variant 形如 m202604_s30_l180）。"""
    if variant.startswith("oct"):
        return {"start": OCT_START, "days": 9, "pool": "oct_pool.json", "symbols": "oct_symbols.json"}
    if variant.startswith("m"):
        month = variant[1:7]
        start = month_start(month)
        # hist 只有 4 月起的 K 线，4 月 1 日没有前 24 小时成交额，合约池从 4 月 2 日算
        if month == FIRST_MONTH:
            start += 86400
        days = (month_start(next_month(month)) - start) // 86400
        return {"start": start, "days": days, "pool": f"pool_{month}.json", "symbols": f"symbols_{month}.txt"}
    return {"start": MONTH_START, "days": 30, "pool": "pool.json", "symbols": "symbols.txt"}


FIRST_MONTH = "202604"
MONTHS = ("202604", "202605", "202606", "202607", "202608")  # 第二轮新增，九月沿用第一轮的数据


def month_variant(month: str, suffix: str = "s30_l180") -> str:
    return suffix if month == "202609" else f"m{month}_{suffix}"


def month_start(month: str) -> int:
    return int(datetime(int(month[:4]), int(month[4:]), 1, tzinfo=UTC).timestamp())


def next_month(month: str) -> str:
    y, m = int(month[:4]), int(month[4:])
    return f"{y + m // 12}{m % 12 + 1:02d}"


def previous_month(month: str) -> str:
    y, m = int(month[:4]), int(month[4:])
    return f"{y - (m == 1)}{(m - 2) % 12 + 1:02d}"


@cache
def pool(variant: str = "s30_l180"):
    return json.load(open(DATA / dataset(variant)["pool"]))


@cache
def volume_marks():
    return json.load(open(DATA / "volume.json")) if (DATA / "volume.json").exists() else None


@lru_cache(maxsize=256)
def load(symbol: str, variant: str = "s30_l180"):
    d = np.load(DATA / "rec" / variant / f"{symbol}.npz")
    return {k: d[k] for k in d.files}


def in_pool(symbol: str, t: float, min_volume: float = 8e6, variant: str = "s30_l180") -> bool:
    i = int((t - dataset(variant)["start"]) // 600)
    if i < 0:
        return False
    if min_volume == 8e6:
        member = pool(variant).get(symbol, {"member": []})["member"]
        return 0 <= i < len(member) and bool(member[i])
    marks = volume_marks()[symbol]
    return 0 <= i < len(marks) and _member_at(symbol, min_volume)[i]


@cache
def _member_at(symbol: str, min_volume: float):
    vol = volume_marks()[symbol]
    out, inside = [], False
    for v in vol:
        inside = v > min_volume * 0.8 if inside else v > min_volume
        out.append(inside)
    return out


def simulate_window(rec: np.ndarray, w: Win) -> list[int]:
    """返回触发提醒的记录下标。"""
    if len(rec) == 0:
        return []
    ok = (rec[:, TRADES] >= w.trades) & (np.abs(rec[:, CHG]) >= w.change) & (rec[:, MATR] >= w.atr)
    idx = np.flatnonzero(ok)
    alerts = []
    last_sec = -10.0
    last_dir = 0
    count = 0
    last_alert = -math.inf
    for i in idx:
        r = rec[i]
        sec = r[SEC]
        direction = 1 if r[CHG] > 0 else -1
        count = count + 1 if (direction == last_dir and sec - last_sec == 1) else 1
        last_dir, last_sec = direction, sec
        if count < w.confirm or r[CARRIED]:
            continue
        lchg = r[LCHG]
        if not math.isnan(lchg):
            # 空档后复核：触发结算的成交价须同方向且仍满足门槛
            if (lchg > 0) != (direction > 0) or abs(lchg) < w.change or r[LATR] < w.atr:
                continue
        if r[NOW] - last_alert >= w.cooldown:
            alerts.append(i)
            last_alert = r[NOW]
        last_dir, last_sec, count = 0, -10.0, 0
    return alerts


def alerts_for(symbol: str, params: Params, variant: str = "s30_l180") -> list[tuple]:
    """(提醒时间, 窗口, 方向, 提醒价, 基准价, 涨跌幅, ATR 倍数, 笔数)，只保留当时在合约池里的。"""
    data = load(symbol, variant)
    out = []
    for name, w in (("short", params.short), ("long", params.long)):
        if w is None:
            continue
        rec = data[name]
        for i in simulate_window(rec, w):
            r = rec[i]
            if in_pool(symbol, r[NOW], params.min_volume, variant):
                direction = 1 if r[CHG] > 0 else -1
                out.append((r[NOW], name, direction, r[PRICE], r[BASE], r[CHG], r[MATR], int(r[TRADES])))
    out.sort()
    return out


# ---------- 质量指标 ----------
HORIZON = 900  # 15 分钟
FOLLOW = 1.0  # 再顺势 1%


def outcomes(symbol: str, alerts: list[tuple], variant: str = "s30_l180") -> list[dict]:
    """每条提醒之后的走势：15 分钟内先再顺势 1%（good）还是先回到基准价（fake），以及 1/5/15 分钟收益。"""
    data = load(symbol, variant)
    sec, last, hi, lo = data["sec"], data["last"], data["hi"], data["lo"]
    res = []
    for t, name, d, p, ref, chg, matr, trades in alerts:
        t0 = int(math.floor(t))
        a = np.searchsorted(sec, t0, side="right")  # 提醒之后的秒
        b = np.searchsorted(sec, t0 + HORIZON, side="right")
        target = p * (1 + d * FOLLOW / 100)
        if d > 0:
            hit = np.flatnonzero(hi[a:b] >= target)
            back = np.flatnonzero(lo[a:b] <= ref)
            mfe = (hi[a:b].max() / p - 1) * 100 if b > a else 0.0
            mae = (lo[a:b].min() / p - 1) * 100 if b > a else 0.0
        else:
            hit = np.flatnonzero(lo[a:b] <= target)
            back = np.flatnonzero(hi[a:b] >= ref)
            mfe = (1 - lo[a:b].min() / p) * 100 if b > a else 0.0
            mae = (1 - hi[a:b].max() / p) * 100 if b > a else 0.0
        # 对称口径：先顺势 1% 还是先逆势 1%，随机游走下为 50%，不受提醒位移大小影响
        stop = p * (1 - d * FOLLOW / 100)
        against = np.flatnonzero(lo[a:b] <= stop) if d > 0 else np.flatnonzero(hi[a:b] >= stop)
        s_against = against[0] if len(against) else 1 << 30
        h = hit[0] if len(hit) else 1 << 30
        k = back[0] if len(back) else 1 << 30
        good = h < k
        fake = k < h or (k == h and k < 1 << 30)
        fwd = {}
        for lag in (60, 300, 900):
            j = np.searchsorted(sec, t0 + lag, side="right") - 1
            fwd[lag] = d * (last[j] / p - 1) * 100 if j >= 0 else 0.0
        # 5 分钟后仍保留至少一半位移
        held = d * (fwd_price(sec, last, t0 + 300) - ref) >= 0.5 * d * (p - ref)
        res.append(
            {
                "t": t,
                "sym": symbol,
                "win": name,
                "d": d,
                "chg": chg,
                "matr": matr,
                "trades": trades,
                "good": bool(good),
                "fake": bool(fake),
                "cont": bool(h < s_against),
                "rev": bool(s_against < h),
                "mfe": mfe,
                "mae": mae,
                "r1": fwd[60],
                "r5": fwd[300],
                "r15": fwd[900],
                "held": bool(held),
            }
        )
    return res


def fwd_price(sec, last, t):
    j = np.searchsorted(sec, t, side="right") - 1
    return last[j] if j >= 0 else last[0]
