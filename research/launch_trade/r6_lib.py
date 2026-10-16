"""第六轮研究：不按持仓时间平仓，只用止损 + 移动止盈出场。

第五轮推荐的 S5 到 18 小时就按市价平仓，这一轮去掉时间出场，比较几种移动止盈的形态，看能不能让大行情多跑一段、
又不把小单的利润回吐掉。只做多（做空第五轮已排除）。数据来自 r5_export.py --post 21600 --float32（提醒后 15 天）。
按用户的要求，提醒只取到数据结束前 15 天（r6_studies.Book6），持仓满 15 天还没平仓的按那一分钟收盘价市价平仓，
这样每笔都有完整的 15 天，不会因为数据截止而丢掉还没走完的大单。

出场线 = 下面各条线中最高的一条，每分钟开盘跌破按开盘价、盘中跌破按线价平仓（与第五轮相同的保守口径）：
- 初始止损：max(等待期最低价 × (1 − stop_gap)，入场价 × (1 − stop_max))，贴着入场价时改用入场价 × (1 − stop_fallback)；
- 移动止盈：持仓以来最高价（到上一分钟）× (1 − 回撤幅度)，浮盈达到 trail_start 才启用；回撤幅度取
  max(trail，入场前 60 分钟平均振幅 × trail_atr)，dyn_atr 不为 NaN 时改为 max(trail，最近 dyn_n 分钟平均振幅 × dyn_atr)
  （行情走稳、振幅收窄后跟得更紧）；最大浮盈达到 tight_at 后幅度改为 tight_trail（可收紧也可放宽）；
- 锁定浮盈：最大浮盈达到 trail_start 后，线不低于 入场价 × (1 + lock × 最大浮盈)；
- 保本：最大浮盈达到 be_at 后，线不低于 入场价 × (1 + be_level)。
close_trigger 为真时，移动止盈、锁定浮盈与保本只看收盘价：按时钟对齐的 close_tf 分钟 K 线收盘低于线，
下一分钟开盘价平仓（避开插针与周期内的来回）；初始止损仍按盘中价格触发，hard_trail 不为 NaN 时
另有一条按盘中价格触发的宽回撤线（最高价 × (1 − hard_trail)），急跌时不必等收盘。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numba as nb
import numpy as np
from r5_lib import PRE, Data, Trades

# 持仓上限 15 天：用户定的兜底，不是出场规则（回撤线一般远早于它触发）。数据只导出到提醒后 21600 分钟，
# 所以实际上限还要减去入场等待的分钟数（等 5 分钟时为 21595 分钟），差几分钟不影响结论。
HOLD_CAP = 21600


@dataclass
class Exit:
    delay: int = 5  # 提醒后等这么多分钟，按下一分钟开盘价入场
    confirm: float = 0.01  # 入场价须比提醒价至少高这么多；NaN 不要求
    max_chase: float = np.nan  # 入场价比提醒价最多高这么多；NaN 不限
    stop_gap: float = 0.05
    stop_max: float = 0.20
    stop_fallback: float = 0.06
    trail: float = 0.10
    trail_atr: float = 3.0  # NaN 不按入场前振幅放宽
    dyn_atr: float = np.nan  # 动态振幅倍数；NaN 不用
    dyn_n: int = 60
    trail_start: float = 0.0  # 最大浮盈达到这么多才启用移动止盈与锁定浮盈
    tight_at: float = np.nan  # 最大浮盈达到这么多后回撤幅度改为 tight_trail
    tight_trail: float = np.nan
    lock: float = 0.0  # 锁定最大浮盈的比例，0 不用
    be_at: float = np.nan  # 保本触发的最大浮盈；NaN 不用
    be_level: float = 0.002  # 保本线相对入场价（略高于来回手续费）
    close_trigger: int = 0
    close_tf: int = 1  # 收盘触发看几分钟 K 线（按时钟对齐，如 15 即 :00/:15/:30/:45 收盘）
    hard_trail: float = np.nan  # 收盘触发时另设的盘中宽回撤；NaN 不设
    hold: int = HOLD_CAP
    fee: float = 0.0005

    def params(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, f.name)) for f in fields(self))


@nb.njit(cache=True)
def _soft_line(price, best, tr, trail_start, tight_at, tight_trail, lock, be_at, be_level):
    """移动止盈、锁定浮盈与保本合成的线；都没启用时为 -1。"""
    gain = best / price - 1
    if not np.isnan(tight_at) and gain >= tight_at:
        tr = tight_trail
    soft = -1.0
    if gain >= trail_start:
        soft = best * (1 - tr)
        if lock > 0:
            soft = max(soft, price * (1 + lock * gain))
    if not np.isnan(be_at) and gain >= be_at:
        soft = max(soft, price * (1 + be_level))
    return soft


@nb.njit(cache=True)
def _simulate(bars, minute0, p):
    (
        delay,
        confirm,
        max_chase,
        stop_gap,
        stop_max,
        stop_fallback,
        trail,
        trail_atr,
        dyn_atr,
        dyn_n,
        trail_start,
        tight_at,
        tight_trail,
        lock,
        be_at,
        be_level,
        close_trigger,
        close_tf,
        hard_trail,
        hold,
        fee,
    ) = p
    n = bars.shape[0]
    width = bars.shape[2]
    net = np.full(n, np.nan)
    entry_at = np.full(n, -1)
    exit_at = np.full(n, -1)
    peak = np.full(n, np.nan)  # 持仓期最大浮盈，用来看出场回吐了多少
    # 0 正常平仓，1 数据中断（下架、停牌），2 持满 hold；后两种按最后一个收盘价平仓
    status = np.zeros(n, dtype=np.int8)
    d = int(delay)
    dn = int(dyn_n)
    tf = int(close_tf)
    rr = np.zeros(width + 1)
    for a in range(n):
        o = bars[a, 0]
        h = bars[a, 1]
        lo = bars[a, 2]
        c = bars[a, 3]
        ap = c[PRE - 1]
        e = PRE + d
        if e >= width or np.isnan(o[e]):
            continue
        price = o[e]
        move = price / ap - 1
        if not np.isnan(confirm) and move < confirm:
            continue
        if not np.isnan(max_chase) and move > max_chase:
            continue
        lo_k = PRE if d > 0 else PRE - 1
        ext = lo[lo_k]
        for k in range(lo_k, e):
            ext = min(ext, lo[k])
        stop = max(ext * (1 - stop_gap), price * (1 - stop_max))
        if stop >= price * 0.995:
            stop = price * (1 - stop_fallback)
        base_tr = trail
        if not np.isnan(trail_atr):
            s = 0.0
            cnt = 0
            for k in range(e - 60, e):
                if k >= 0 and not np.isnan(c[k]) and c[k] > 0:
                    s += (h[k] - lo[k]) / c[k]
                    cnt += 1
            if cnt > 0:
                base_tr = max(trail, trail_atr * s / cnt)
        use_dyn = not np.isnan(dyn_atr)
        if use_dyn:
            # 振幅前缀和：rr[k] = 第 0..k-1 根的振幅之和，缺数据的分钟按 0（持仓中遇到缺数据会直接平仓，只影响入场前）
            rr[0] = 0.0
            for k in range(width):
                v = 0.0
                if not np.isnan(c[k]) and c[k] > 0:
                    v = (h[k] - lo[k]) / c[k]
                rr[k + 1] = rr[k] + v
        best = price
        px = np.nan
        k_out = -1
        last = e
        pending = False
        end = min(e + int(hold), width)
        for k in range(e, end):
            if np.isnan(c[k]):
                break
            last = k
            if pending:
                # 上一分钟收盘跌破移动线，本分钟开盘平仓
                px = o[k]
                k_out = k
                break
            tr = base_tr
            if use_dyn:
                a0 = max(0, k - dn)
                tr = max(trail, dyn_atr * (rr[k] - rr[a0]) / (k - a0))
            if close_trigger > 0:
                line = stop
                if not np.isnan(hard_trail):
                    line = max(line, best * (1 - hard_trail))
            else:
                line = max(stop, _soft_line(price, best, tr, trail_start, tight_at, tight_trail, lock, be_at, be_level))
            if o[k] <= line:
                px = o[k]
            elif lo[k] <= line:
                px = line
            elif close_trigger > 0 and (minute0[a] + k) % tf == 0:
                # 收盘时已知道本根最高价，收盘判定的线含本根（盘中的线只能用到上一分钟，分钟内先后顺序未知）
                bc = max(best, h[k])
                if c[k] <= _soft_line(price, bc, tr, trail_start, tight_at, tight_trail, lock, be_at, be_level):
                    pending = True
            best = max(best, h[k])
            if not np.isnan(px):
                k_out = k
                break
        if k_out < 0:
            k_out = last
            px = c[last]
            status[a] = 2 if last >= end - 1 else 1
        r = px / price
        net[a] = r - 1 - fee * (1 + r)
        entry_at[a] = e
        exit_at[a] = k_out
        peak[a] = best / price - 1
    return net, entry_at, exit_at, status, peak


@dataclass
class Trades6(Trades):
    status: np.ndarray
    peak: np.ndarray


def simulate(data: Data, rule: Exit) -> Trades6:
    # 第 k 根 K 线收盘于第 minute0 + k 分钟（Unix 分钟数），用来按时钟对齐大周期 K 线
    minute0 = data.closed_at // 60 - PRE + 1
    net, entry_at, exit_at, status, peak = _simulate(data.bars, minute0, rule.params())
    t_in = data.closed_at + (entry_at - PRE) * 60
    t_out = data.closed_at + (exit_at - PRE + 1) * 60
    # 数据截止前没走完的不算（结果未知）；中途中断的按最后收盘价平仓，代表下架、长时间停牌的真实损失。
    unfinished = (status == 1) & (t_out >= data.end - 60 * 60)
    net = np.where(unfinished, np.nan, net)
    return Trades6(net, t_in, t_out, (status > 0).astype(np.int8), status, peak)
