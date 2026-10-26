"""逐秒复现 AtrMoveDetector 的窗口指标，输出每个窗口长度下「通过宽松门槛的秒」供网格扫描。

每个合约-月份处理一次，结果存 rows/<合约>-<月>.npz，键名以窗口变体为前缀（如 L30_chg）。
宽松门槛低于候选族的最小取值：没进表的秒在任何候选参数下都不满足门槛，连续确认的判断因此不受影响。

与检测器对应的语义（detector.py）：
- 秒桶价格为 VWAP（零成交量退化为均价）；一秒在下一秒第一笔成交到达时结算，结算时刻与入场价取那笔成交。
- 两次成交之间的空秒不超过补齐上限 F 时用最后成交价补齐，补齐秒参与确认、不能提醒；超过 F 不补。
- 基准桶为 ≤ 目标时刻的最后一个桶，距离不超过 L+2 秒用其 VWAP；否则空档不长于 L 时用其最后成交价。
- ATR 只用该秒所在分钟之前已收盘的 1 分钟 K 线（无成交分钟补平线）；空档后结算的真实秒要用触发成交价复核。
"""

from __future__ import annotations

import os
import sys
import zlib
from multiprocessing import Pool

import numba as nb
import numpy as np
import pandas as pd
from common import DATA, MONTHS, TRADES, contracts, load_klines, month_end, month_start, multiplier, previous_month
from pool import membership

# 窗口变体：名称 -> (窗口长度, 补齐上限)。单窗口时补齐上限等于窗口长度；L180F30 是现行双窗口里的长窗口
VARIANTS = {f"L{n}": (n, n) for n in (10, 20, 30, 60, 120, 180, 300)}
VARIANTS["L180F30"] = (180, 30)
LAX_CHANGE, LAX_ATR, LAX_TRADES = 0.8, 1.0, 5
POOL_LEVELS = (5e6, 8e6, 20e6)
ATR_PERIOD = 14
HORIZON = 10
ROWS = DATA / "rows"
NPY = DATA / "npy"


def load_trades(symbol: str, month: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = NPY / f"{symbol}-{month}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["p"], d["s"]
    df = pd.read_csv(TRADES / f"{symbol}-{month}.csv.gz", header=None, names=["t", "id", "p", "s"])
    # 实时推送按成交顺序到达，按成交 id 排序最接近；时间早于此前最大时间的成交检测器会丢弃
    df = df.sort_values("id", kind="stable")
    t = df["t"].to_numpy(np.float64)
    keep = t >= np.maximum.accumulate(t)
    t, p, s = t[keep], df["p"].to_numpy(np.float64)[keep], df["s"].to_numpy(np.float32)[keep]
    ok = (p > 0) & np.isfinite(p)
    t, p, s = t[ok], p[ok], s[ok]
    NPY.mkdir(parents=True, exist_ok=True)
    np.savez(cache, t=t, p=p, s=s)
    return t, p, s


@nb.njit(cache=True)
def _fill_bars(has, o, h, lo, c):
    # 无成交分钟按 Gate 口径补开高低收都等于上一收盘价的平线
    prev = np.nan
    for m in range(len(has)):
        if has[m]:
            prev = c[m]
        elif prev == prev:
            o[m] = prev
            h[m] = prev
            lo[m] = prev
            c[m] = prev
            has[m] = True


@nb.njit(cache=True)
def _atr_before(has, h, lo, c, period):
    """out[m] = 计入 m 之前全部 K 线后的 Wilder ATR，未就绪为 NaN。"""
    n = len(has)
    out = np.full(n + 1, np.nan)
    value = np.nan
    prev_close = np.nan
    warm = 0.0
    count = 0
    for m in range(n):
        out[m] = value
        if not has[m]:
            continue
        tr = h[m] - lo[m]
        if prev_close == prev_close:
            tr = max(tr, abs(h[m] - prev_close), abs(lo[m] - prev_close))
        if value != value:
            warm += tr
            count += 1
            if count == period:
                value = warm / period
        else:
            value = (value * (period - 1) + tr) / period
        prev_close = c[m]
    out[n] = value
    return out


@nb.njit(cache=True)
def _window_extreme(vwap, lo_idx, hi_idx, direction):
    # 窗口内（真实秒）最极端的 VWAP：急涨取最高、急跌取最低
    out = np.empty(len(lo_idx))
    for k in range(len(lo_idx)):
        best = vwap[hi_idx[k]]
        for r in range(lo_idx[k] + 1, hi_idx[k] + 1):
            v = vwap[r]
            if (direction[k] > 0 and v > best) or (direction[k] < 0 and v < best):
                best = v
        out[k] = best
    return out


@nb.njit(cache=True)
def _rolling_extreme(h, lo, width):
    """hi[m]、lo[m] 为 [m - width, m) 内 K 线的最高价、最低价。"""
    n = len(h)
    hi_out = np.full(n, np.nan)
    lo_out = np.full(n, np.nan)
    for m in range(width, n):
        a = -np.inf
        b = np.inf
        for k in range(m - width, m):
            if h[k] == h[k]:
                a = max(a, h[k])
                b = min(b, lo[k])
        if a > -np.inf:
            hi_out[m] = a
            lo_out[m] = b
    return hi_out, lo_out


def price_at(t: np.ndarray, p: np.ndarray, when: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t, when, side="right") - 1
    out = p[np.clip(idx, 0, None)]
    return np.where(idx >= 0, out, np.nan)


def minute_bars(symbol: str, month: str, t: np.ndarray, p: np.ndarray, start: int, end: int, warm: int = 60):
    """月初前 warm 根取下载站 K 线预热（与实时服务用 REST K 线预热相同），之后由成交聚合。"""
    m0 = start // 60 - warm
    n = end // 60 - m0
    has = np.zeros(n, bool)
    o, h, lo, c = (np.full(n, np.nan) for _ in range(4))
    prev = load_klines(symbol, previous_month(month))
    if len(prev):
        m = prev[:, 0].astype(np.int64) // 60 - m0
        ok = (m >= 0) & (m < warm)
        has[m[ok]] = True
        c[m[ok]], h[m[ok]], lo[m[ok]], o[m[ok]] = prev[ok, 2], prev[ok, 3], prev[ok, 4], prev[ok, 5]
    m = t.astype(np.int64) // 60 - m0
    first = np.r_[True, m[1:] != m[:-1]]
    starts = np.flatnonzero(first)
    mm = m[starts]
    has[mm] = True
    o[mm] = p[starts]
    c[mm] = p[np.r_[starts[1:] - 1, len(p) - 1]]
    h[mm] = np.maximum.reduceat(p, starts)
    lo[mm] = np.minimum.reduceat(p, starts)
    _fill_bars(has, o, h, lo, c)
    return m0, has, o, h, lo, c


# 第二轮新特征（PLAN.md 第二轮）：6 小时极值要 360 根之前的 K 线，月初多取一些预热
EXTRA_WARM = 400


def process(
    job: tuple[str, str], days: float | None = None, variants: dict | None = None, extras: bool = False
) -> dict:
    symbol, month = job
    start, end = month_start(month), month_end(month)
    t, p, s = load_trades(symbol, month)
    if days is not None:
        cut = t < start + days * 86400
        t, p, s = t[cut], p[cut], s[cut]
    out: dict[str, np.ndarray] = {}
    if len(t) < 100:
        return out
    mult = multiplier(contracts().get(symbol))
    m0, has, o, h, lo, c = minute_bars(symbol, month, t, p, start, end, EXTRA_WARM if extras else 60)
    atr_before = _atr_before(has, h, lo, c, ATR_PERIOD)
    hi60, lo60 = _rolling_extreme(h, lo, 60)
    if extras:
        hi6h, lo6h = _rolling_extreme(h, lo, 360)

    # 真实秒桶
    sec_all = np.floor(t).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, sec_all[1:] != sec_all[:-1]])
    ends = np.r_[starts[1:], len(t)]
    size = np.abs(s).astype(np.float64)
    wsum = np.add.reduceat(p * size, starts)
    ssum = np.add.reduceat(size, starts)
    psum = np.add.reduceat(p, starts)
    count = ends - starts
    vwap = np.where(ssum > 0, wsum / np.where(ssum > 0, ssum, 1), psum / count)
    last = p[ends - 1]
    sec_u = sec_all[starts]
    through = np.cumsum(count)
    quote = size * mult * p
    buy_q = np.add.reduceat(np.where(s > 0, quote, 0.0), starts)
    sell_q = np.add.reduceat(np.where(s < 0, quote, 0.0), starts)
    cum_buy = np.r_[0.0, np.cumsum(buy_q)]
    cum_sell = np.r_[0.0, np.cumsum(sell_q)]
    cum_quote = cum_buy + cum_sell
    # 最后一个真实秒没有下一笔成交来结算
    nreal = len(sec_u) - 1
    trig_t = t[starts[1:]]
    trig_p = p[starts[1:]]
    gap = sec_u[1:] - sec_u[:-1] - 1

    pool_bits = None
    vol = np.load(DATA / f"volume_{month}.npz")
    names = list(vol["symbols"])
    if symbol in names:
        row = vol["vols"][names.index(symbol)][None, :]
        pool_bits = [membership(row, level)[0] for level in POOL_LEVELS]

    for name, (window, fill) in (variants or VARIANTS).items():
        ncarry = np.where((gap > 0) & (gap <= fill), gap, 0)
        src = np.repeat(np.arange(nreal), 1 + ncarry)
        group_start = np.repeat(np.cumsum(np.r_[0, 1 + ncarry[:-1]]), 1 + ncarry)
        offset = np.arange(len(src)) - group_start
        bsec = sec_u[src] + offset
        real = offset == 0
        bprice = np.where(real, vwap[src], last[src])
        blast = last[src]
        bthrough = through[src]

        target = bsec - window
        j = np.searchsorted(bsec, target, side="right") - 1
        jj = np.clip(j, 0, None)
        near = (bsec - bsec[jj]) <= window + 2
        gapj = bsec[np.minimum(jj + 1, len(bsec) - 1)] - bsec[jj] - 1
        valid = (j >= 0) & (near | (gapj <= window))
        base = np.where(near, bprice[jj], blast[jj])
        atr = atr_before[np.clip(bsec // 60 - m0, 0, len(atr_before) - 1)]
        valid &= np.isfinite(atr) & (atr > 0)
        move = bprice - base
        chg = np.abs(move / base * 100.0)
        atrm = np.abs(move) / np.where(valid, atr, 1.0)
        trades = bthrough - bthrough[jj]
        keep = np.flatnonzero(valid & (chg >= LAX_CHANGE) & (atrm >= LAX_ATR) & (trades >= LAX_TRADES))
        if not len(keep):
            continue
        i = keep
        direction = np.where(move[i] > 0, 1, -1).astype(np.int8)
        r = src[i]
        rj = src[jj[i]]
        isreal = real[i]

        # 空档后结算的真实秒：用触发结算的成交价复核方向与幅度
        lmove = trig_p[r] - base[i]
        lok = np.sign(lmove) == direction
        lchg = np.where(lok, np.abs(lmove / base[i] * 100.0), -1.0)
        latrm = np.where(lok, np.abs(lmove) / atr[i], -1.0)
        recheck = isreal & (gap[r] > 0)
        lchg = np.where(recheck, lchg, np.inf)
        latrm = np.where(recheck, latrm, np.inf)

        # 新条件特征（只对真实秒有意义）
        wb = cum_buy[r + 1] - cum_buy[rj + 1]
        ws = cum_sell[r + 1] - cum_sell[rj + 1]
        wq = wb + ws
        imb = np.where(wq > 0, direction * (wb - ws) / np.where(wq > 0, wq, 1), 0.0)
        wstart = bsec[i] - window
        k0 = np.searchsorted(sec_u, wstart - 3600, side="left")
        k1 = np.searchsorted(sec_u, wstart, side="left")
        prior = (cum_quote[k1] - cum_quote[k0]) * window / 3600.0
        vr = np.where(prior > 0, wq / np.where(prior > 0, prior, 1), np.inf)
        ext = _window_extreme(vwap, rj, r, direction)
        retr = direction * (ext - bprice[i]) / np.abs(move[i])
        kpre = np.searchsorted(sec_u, wstart - 900, side="right") - 1
        pre_price = np.where(kpre >= 0, last[np.clip(kpre, 0, None)], np.nan)
        pre = direction * (base[i] / pre_price - 1) * 100.0
        mstart = np.clip(wstart // 60 - m0, 0, len(hi60) - 1)
        brk = np.where(direction > 0, bprice[i] / hi60[mstart] - 1, 1 - bprice[i] / lo60[mstart]) * 100.0
        dabs = np.r_[0.0, np.cumsum(np.abs(np.diff(bprice)))]
        path = dabs[i] - dabs[jj[i]]
        eff = np.where(path > 0, np.abs(move[i]) / np.where(path > 0, path, 1), 1.0)

        # 标签：入场价为触发结算的成交价，第 k 分钟取提醒时刻 + k 分钟的最新成交价
        tt = trig_t[r]
        entry = trig_p[r]
        marks = tt[:, None] + 60.0 * np.arange(1, HORIZON + 1)[None, :]
        future = price_at(t, p, marks.ravel()).reshape(marks.shape)
        ret = direction[:, None] * (future / entry[:, None] - 1) * 100.0
        ret[tt + 60 * HORIZON > t[-1]] = np.nan
        entry5 = price_at(t, p, tt + 5)
        marks5 = (tt + 5)[:, None] + 60.0 * np.arange(1, 6)[None, :]
        fut5 = price_at(t, p, marks5.ravel()).reshape(marks5.shape)
        ret5 = direction[:, None] * (fut5 / entry5[:, None] - 1) * 100.0
        hold5d = (ret5 > 0).all(axis=1)

        bits = np.zeros(len(i), np.uint8)
        if pool_bits is not None:
            slot = np.clip(((tt - start) // 600).astype(np.int64), 0, len(pool_bits[0]) - 1)
            for b, member in enumerate(pool_bits):
                bits |= member[slot].astype(np.uint8) << b

        f32 = np.float32
        out.update(
            {
                f"{name}_sec": bsec[i],
                f"{name}_t": tt,
                f"{name}_dir": direction,
                f"{name}_real": isreal,
                f"{name}_chg": chg[i].astype(f32),
                f"{name}_atrm": atrm[i].astype(f32),
                f"{name}_trades": trades[i].astype(np.int32),
                f"{name}_lchg": lchg.astype(f32),
                f"{name}_latrm": latrm.astype(f32),
                f"{name}_imb": imb.astype(f32),
                f"{name}_vr": vr.astype(f32),
                f"{name}_retr": retr.astype(f32),
                f"{name}_pre": pre.astype(f32),
                f"{name}_brk": brk.astype(f32),
                f"{name}_eff": eff.astype(f32),
                f"{name}_ret": ret.astype(f32),
                f"{name}_ret5d": ret5[:, -1].astype(f32),
                f"{name}_hold5d": hold5d,
                f"{name}_pool": bits,
                f"{name}_atrpct": (atr[i] / bprice[i] * 100).astype(f32),
            }
        )
        if extras:
            # 成交笔数暴增：最近 60 秒笔数 ÷（最近 3600 秒笔数 ÷ 60），取 ≤ 该时刻的最后一个真实秒的累计笔数
            def through_at(when):
                k = np.searchsorted(sec_u, when, side="right") - 1
                return np.where(k >= 0, through[np.clip(k, 0, None)], -1)

            now_n = bthrough[i]
            n60 = now_n - through_at(bsec[i] - 60)
            k3600 = through_at(bsec[i] - 3600)
            n3600 = now_n - k3600
            burst = np.where((k3600 >= 0) & (n3600 > 0), n60 / np.maximum(n3600, 1) * 60.0, np.nan)
            k300 = np.searchsorted(sec_u, wstart - 300, side="right") - 1
            pre300 = direction * (base[i] / np.where(k300 >= 0, last[np.clip(k300, 0, None)], np.nan) - 1) * 100.0
            # 离 6 小时极值：当前秒价格相对此前 360 根已收盘 1 分钟 K 线（不含当前分钟）的最高 / 最低价
            mnow = np.clip(bsec[i] // 60 - m0, 0, len(hi6h) - 1)
            ext6h = np.where(direction > 0, bprice[i] / hi6h[mnow] - 1, 1 - bprice[i] / lo6h[mnow]) * 100.0
            out.update(
                {
                    f"{name}_burst": burst.astype(f32),
                    f"{name}_pre300": pre300.astype(f32),
                    f"{name}_ext6h": ext6h.astype(f32),
                }
            )

    # 随机基准：每天 300 个随机时刻、随机方向
    rng = np.random.default_rng(zlib.crc32(f"{symbol}-{month}".encode()))
    span = (t[-1] - 60 * HORIZON) - max(t[0], start)
    if span > 0:
        rt = max(t[0], start) + rng.random(int(300 * span / 86400)) * span
        rdir = rng.choice(np.array([-1, 1], np.int8), len(rt))
        entry = price_at(t, p, rt)
        marks = rt[:, None] + 60.0 * np.arange(1, HORIZON + 1)[None, :]
        future = price_at(t, p, marks.ravel()).reshape(marks.shape)
        bits = np.zeros(len(rt), np.uint8)
        if pool_bits is not None:
            slot = np.clip(((rt - start) // 600).astype(np.int64), 0, len(pool_bits[0]) - 1)
            for b, member in enumerate(pool_bits):
                bits |= member[slot].astype(np.uint8) << b
        out.update(
            {
                "rand_t": rt,
                "rand_ret": (rdir[:, None] * (future / entry[:, None] - 1) * 100).astype(np.float32),
                "rand_pool": bits,
            }
        )
    return out


def rows_dir(variant: str | None = None):
    """主表放 rows/；事后补算的单个变体（如合并第二个窗口后的 L20F10）放 rows_<变体>/，不改动主表。"""
    return ROWS if variant is None or variant in VARIANTS else DATA / f"rows_{variant}"


def parse_variant(name: str) -> tuple[int, int]:
    length, _, fill = name[1:].partition("F")
    return int(length), int(fill or length)


def run(job: tuple[str, str], variant: str | None = None, extras: bool = False) -> tuple[str, str, int]:
    out_path = rows_dir(variant) / f"{job[0]}-{job[1]}.npz"
    if out_path.exists():
        return job[0], job[1], -1
    out = process(job, variants={variant: parse_variant(variant)} if variant else None, extras=extras)
    # 先写临时文件再改名：中断或两个进程碰巧处理同一文件时不会留下半个文件
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    return job[0], job[1], len(out.get(f"{variant or 'L30'}_sec", []))


def run_extra(args: tuple[tuple[str, str], str | None, bool]) -> tuple[str, str, int]:
    return run(*args)


def all_jobs(months: list[str]) -> list[tuple[str, str]]:
    jobs = [(p.name.split("-")[0], m) for m in months for p in sorted(TRADES.glob(f"*-{m}.csv.gz"))]
    # 大文件先跑，避免最后单进程拖尾
    return sorted(jobs, key=lambda j: -(TRADES / f"{j[0]}-{j[1]}.csv.gz").stat().st_size)


if __name__ == "__main__":
    # 用法：engine.py 进程数 [--asc] [--variant L20F10] [--extras] [月份 ...]
    # --extras 另算第二轮的成交笔数暴增、前置走势等特征
    args = sys.argv[2:]
    variant = None
    if "--variant" in args:
        k = args.index("--variant")
        variant = args[k + 1]
        del args[k : k + 2]
    rows_dir(variant).mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ascending = "--asc" in args
    jobs = all_jobs([a for a in args if a not in ("--asc", "--extras")] or MONTHS)
    if ascending:
        jobs.reverse()
    with Pool(workers) as pool:
        for n, (sym, month, rows) in enumerate(
            pool.imap_unordered(run_extra, [(j, variant, "--extras" in args) for j in jobs]), 1
        ):
            print(n, "/", len(jobs), sym, month, rows, flush=True)
