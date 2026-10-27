"""冷门启动两组门槛微调：现行一组涨幅 8% → 7% / 7.5%，慢启动后半成交额占比 0.5 → 0.4 / 0.45，口径与判据见 dormant_loosen.md。

现行一组逐分钟按 B3 实现（同 dormant_slow._b3_alerts，涨幅门槛可变），慢启动一组沿用 dormant_slow 的 W60 抽取与
dormant_waves 的状态机（含跟踪补报 5%/240 分钟）。「准」、冷门启动行情与开发期/样本外划分与 dormant_slow 相同。

用法（仓库根目录，$P 见 README.md）：
  & $P research/launch_trade/dormant_loosen.py run        # 逐合约跑 6 个变体（含两个对照），存每周计数
  & $P research/launch_trade/dormant_loosen.py evaluate   # 开发期选参、样本外检验
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from dormant_calm import _features
from dormant_early import _leg_position
from dormant_slow import (
    B3_ACTIVE,
    B3_BODY,
    B3_COOLDOWN,
    B3_JUMP,
    B3_LAST,
    B3_LATE,
    B3_MULTIPLE,
    B3_QUOTE,
    B3_STEP,
    B3_VOL,
    B3_W,
    CALM,
    COOLDOWN,
    FEATURES,
    GATE,
    _cumsum,
    _extract,
)
from dormant_solo import PRE, _dormant_legs
from dormant_waves import SLOW, UPPER, WINDOW, _run
from launch_alerts import BOOTSTRAP, DEV_WEEKS, FORWARD, GAINS, HOLDS, TARGETS, WEEK, _bootstrap_weights, _validity
from launch_grid import EPOCH, _multiplier, _rolling_extreme
from load_data import load_contracts
from numba import njit
from paths import RESULTS

OUT = RESULTS / "dormant_loosen"
# （问题, 取值）；每个问题第一个是现行对照，其余按从松到紧排，选参时先看最松的。
VARIANTS = [("A", 8.0), ("A", 7.0), ("A", 7.5), ("B", 0.5), ("B", 0.40), ("B", 0.45)]
CONTROL = {"A": 0, "B": 3}
FOLLOW_STEP = 5.0
FOLLOW_MINUTES = 240
P_SLACK = 0.01
NONINFERIOR = 0.02
MAX_RATIO = 1.5
# 每周计数的行：条数、计分条数、4 种设置的准确数、两种幅度的早报分与前半段召回、新增提醒的条数/计分/准确数。
ROWS = ("alerts", "scored", "c0", "c1", "c2", "c3", "credit10", "credit20", "caught10", "caught20",
        "new", "new_scored", "new_correct")


@njit(cache=True)
def _main_alerts(open_, high, low, close, quote, cum_quote, cum_active, cum_uncovered, cum_tr, jump_max, min_change):
    """与 dormant_slow._b3_alerts 相同，只有涨幅门槛作为参数。"""
    n = len(close)
    window = B3_W
    half = (window + 1) // 2
    out = []
    last_t = -1
    last_p = 0.0
    for t in range(window + PRE, n):
        s = t - window
        first = s + 1 - PRE
        if cum_uncovered[t + 1] - cum_uncovered[first - 1] != 0:
            continue
        if cum_quote[s + 1] - cum_quote[first] > GATE:
            continue
        q = cum_quote[t + 1] - cum_quote[s + 1]
        if q < B3_QUOTE:
            continue
        change = (close[t] / close[s] - 1) * 100
        if change < min_change or jump_max[t] > B3_JUMP:
            continue
        if cum_active[t + 1] - cum_active[s + 1] < B3_ACTIVE or quote[t] < B3_LAST:
            continue
        if (cum_quote[t + 1] - cum_quote[t + 1 - half]) / q < B3_LATE:
            continue
        vol = (cum_tr[s + 1] - cum_tr[s + 1 - CALM]) / CALM
        if vol > B3_VOL or (vol > 0 and change / vol < B3_MULTIPLE):
            continue
        body = 0.0
        span = 0.0
        for j in range(s + 1, t + 1):
            body += abs(close[j] - open_[j])
            span += high[j] - low[j]
        if span <= 0 or body / span < B3_BODY:
            continue
        if last_t >= 0 and t - last_t < B3_COOLDOWN and close[t] < last_p * (1 + B3_STEP / 100):
            continue
        last_t = t
        last_p = close[t]
        out.append(t)
    return np.array(out, np.int64)


def _symbol(args: tuple[str, float]) -> dict | None:
    symbol, multiplier = args
    prepared = _features(symbol, multiplier)
    if prepared is None:
        return None
    series, cum_quote, cum_active, cum_uncovered, cum_tr, minute_change = prepared
    open_, high, low, close = series.open, series.high, series.low, series.close
    offset = (series.t0 - EPOCH) // 60
    covered = (cum_uncovered[1:] - cum_uncovered[:-1]) == 0
    ok = covered & np.concatenate([[False], covered[:-1]])
    cum_move = _cumsum(np.nan_to_num(np.where(ok, np.abs(np.diff(close, prepend=close[0])), 0.0)))
    cum_body = _cumsum(np.nan_to_num(np.abs(close - open_)))
    cum_span = _cumsum(np.nan_to_num(high - low))
    index, f = _extract(open_, high, low, close, cum_quote, cum_active, cum_uncovered, cum_tr, cum_move, cum_body,
                        cum_span, _rolling_extreme(minute_change, WINDOW, True), _rolling_extreme(high, WINDOW, True),
                        WINDOW)
    slow_pass = np.ones(len(index), np.bool_)
    for name, value in SLOW.items():
        if name == "late":
            continue
        column = f[:, FEATURES.index(name)]
        slow_pass &= column <= value if name in UPPER else column >= value
    late = f[:, FEATURES.index("late")]
    jump15 = _rolling_extreme(minute_change, B3_W, True)
    legs = {gain: _dormant_legs(close, cum_quote, cum_uncovered, gain) for gain in GAINS}
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)

    def run(question: str, value: float) -> np.ndarray:
        if question == "A":
            t = _main_alerts(open_, high, low, close, series.quote, cum_quote, cum_active, cum_uncovered, cum_tr,
                             jump15, value)
            return t[t + offset >= 0]
        cond = np.zeros(len(close), np.bool_)
        cond[index[slow_pass & (late >= value)]] = True
        t, _ = _run(close, cond, covered, max(0, -offset), FOLLOW_STEP, FOLLOW_MINUTES, COOLDOWN)
        return t

    out: dict = {"events": {g: ((s + offset) * 60 // WEEK)[(s + offset) >= 0] for g, (s, _) in legs.items()}}
    alerts = [run(q, v) for q, v in VARIANTS]
    for k, (question, _) in enumerate(VARIANTS):
        t = alerts[k]
        valid = _validity(high, low, close, cum_uncovered, t, np.ones(len(t), np.int8), targets, holds, FORWARD)
        record = {"week": ((t + offset) * 60 + 60) // WEEK, "valid": valid,
                  "new": ~np.isin(t, alerts[CONTROL[question]])}
        for gain in GAINS:
            starts, ends = legs[gain]
            leg, frac = _leg_position(close, t, starts, ends)
            # 每段只算第一次提醒；按行情起点所在的周计。
            seen = set()
            first_week, first_frac = [], []
            for i in range(len(t)):
                if leg[i] >= 0 and leg[i] not in seen:
                    seen.add(leg[i])
                    w = (starts[leg[i]] + offset) * 60 // WEEK
                    if w >= 0:
                        first_week.append(w)
                        first_frac.append(frac[i])
            record[f"leg_week{int(gain * 100)}"] = np.array(first_week, np.int64)
            record[f"leg_frac{int(gain * 100)}"] = np.array(first_frac)
        out[k] = record
    return out


def _bincount(week, n_weeks, weights=None):
    keep = (week >= 0) & (week < n_weeks)
    return np.bincount(week[keep], weights=None if weights is None else weights[keep], minlength=n_weeks)


def cmd_run() -> None:
    contracts = load_contracts()
    jobs = [(s, _multiplier(contracts[s])) for s in sorted(contracts) if _multiplier(contracts[s])]
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=11) as executor:
        for done, result in enumerate(executor.map(_symbol, jobs, chunksize=2), 1):
            if result:
                results.append(result)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    last = max(int(r[k]["week"].max()) for r in results for k in range(len(VARIANTS)) if len(r[k]["week"]))
    n_weeks = last + 1
    weekly = np.zeros((len(VARIANTS), len(ROWS), n_weeks))
    events = np.zeros((len(GAINS), n_weeks))
    for r in results:
        for g, gain in enumerate(GAINS):
            events[g] += _bincount(r["events"][gain], n_weeks)
        for k in range(len(VARIANTS)):
            rec = r[k]
            week, valid, new = rec["week"], rec["valid"], rec["new"]
            scored = valid[:, 0] >= 0
            rows = [_bincount(week, n_weeks), _bincount(week[scored], n_weeks)]
            rows += [_bincount(week[scored & (valid[:, s] == 1)], n_weeks) for s in range(valid.shape[1])]
            for gain in GAINS:
                lw, lf = rec[f"leg_week{int(gain * 100)}"], rec[f"leg_frac{int(gain * 100)}"]
                rows.append(_bincount(lw, n_weeks, 1 - lf))
            for gain in GAINS:
                lw, lf = rec[f"leg_week{int(gain * 100)}"], rec[f"leg_frac{int(gain * 100)}"]
                rows.append(_bincount(lw[lf <= 0.5], n_weeks))
            new_scored = new & scored
            rows += [_bincount(week[new], n_weeks), _bincount(week[new_scored], n_weeks),
                     _bincount(week[new_scored], n_weeks, (valid[new_scored] == 1).mean(1))]
            weekly[k] += np.array(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "weekly.npz", weekly=weekly, events=events)
    print(f"完成 {time.time() - started:.0f}s，{n_weeks} 周")


def scores(x: np.ndarray, events: np.ndarray, days) -> dict:
    """x 的最后一维是 ROWS（已在时间上求和，可带前导维度），events 的最后一维是两种幅度的行情数。"""
    x = np.asarray(x, np.float64)
    r = {name: x[..., i] for i, name in enumerate(ROWS)}
    scored = r["scored"]
    correct = (r["c0"] + r["c1"] + r["c2"] + r["c3"]) / 4
    e10, e20 = events[..., 0], events[..., 1]
    return {
        "precision": np.divide(correct, scored, out=np.zeros(np.shape(scored)), where=scored > 0),
        "per_day": r["alerts"] / days,
        "re": 0.5 * (r["credit10"] / e10 + r["credit20"] / e20),
        "recall": 0.5 * (r["caught10"] / e10 + r["caught20"] / e20),
        "new_per_day": r["new"] / days,
        "new_precision": np.divide(r["new_correct"], r["new_scored"], out=np.zeros(np.shape(scored)),
                                   where=r["new_scored"] > 0),
    }


def _ok(s: dict, k: int, c: int) -> bool:
    return bool(s["precision"][k] >= s["precision"][c] - P_SLACK and s["per_day"][k] <= MAX_RATIO * s["per_day"][c])


def cmd_evaluate() -> None:
    with np.load(OUT / "weekly.npz") as data:
        weekly, events = data["weekly"], data["events"]
    n_weeks = weekly.shape[2]
    n_oos = n_weeks - DEV_WEEKS
    dev = scores(weekly[:, :, :DEV_WEEKS].sum(2), events[:, :DEV_WEEKS].sum(1), DEV_WEEKS * 7)
    oos = scores(weekly[:, :, DEV_WEEKS:].sum(2), events[:, DEV_WEEKS:].sum(1), n_oos * 7)
    print(f"开发期 {DEV_WEEKS} 周，样本外 {n_oos} 周；行情数（10%/20%）开发期 {events[:, :DEV_WEEKS].sum(1)}、"
          f"样本外 {events[:, DEV_WEEKS:].sum(1)}")
    print("变体：开发期 准确率 / 每天 / 早报分 / 前半召回 / 新增每天 / 新增准确率 | 样本外同")
    for k, variant in enumerate(VARIANTS):
        cells = []
        for s in (dev, oos):
            cells.append(f"{s['precision'][k]:.3f} {s['per_day'][k]:.2f} {s['re'][k]:.4f} {s['recall'][k]:.3f} "
                         f"{s['new_per_day'][k]:.2f} {s['new_precision'][k]:.3f}")
        print(f"  {variant}: {cells[0]} | {cells[1]}")
    weights = _bootstrap_weights(n_oos, BOOTSTRAP, seed=2711).astype(np.float64)
    boot = scores(np.einsum("bw,krw->bkr", weights, weekly[:, :, DEV_WEEKS:]),
                  np.einsum("bw,gw->bg", weights, events[:, DEV_WEEKS:])[:, None, :], (weights.sum(1) * 7)[:, None])
    result: dict = {"variants": VARIANTS,
                    "dev": {k: np.asarray(v).round(4).tolist() for k, v in dev.items()},
                    "oos": {k: np.asarray(v).round(4).tolist() for k, v in oos.items()}}
    for question in ("A", "B"):
        c = CONTROL[question]
        candidates = [k for k, (q, _) in enumerate(VARIANTS) if q == question and k != c]
        chosen = next((k for k in candidates if _ok(dev, k, c)), None)
        entry: dict = {"control": VARIANTS[c], "dev_ok": {str(VARIANTS[k]): _ok(dev, k, c) for k in candidates}}
        if chosen is None:
            entry["chosen"] = None
        else:
            x = weekly[:, :, DEV_WEEKS:].sum(2)
            setting = lambda k: x[k, 2:6] / max(x[k, 1], 1)  # noqa: E731
            d_p = boot["precision"][:, chosen] - boot["precision"][:, c]
            d_re = boot["re"][:, chosen] - boot["re"][:, c]
            entry.update({
                "chosen": VARIANTS[chosen],
                "oos": {name: float(oos[name][chosen]) for name in oos},
                "oos_control": {name: float(oos[name][c]) for name in oos},
                "by_setting": setting(chosen).round(4).tolist(),
                "control_by_setting": setting(c).round(4).tolist(),
                "p_inferior": float(np.mean(d_p < -NONINFERIOR)),
                "p_re_not_better": float(np.mean(d_re <= 0)),
            })
            entry["checks"] = {
                "precision": bool(oos["precision"][chosen] >= oos["precision"][c] - P_SLACK),
                "settings_ok": int(np.sum(setting(chosen) >= setting(c) - P_SLACK)),
                "noninferior": entry["p_inferior"] < 0.05,
                "earlier": bool(oos["re"][chosen] > oos["re"][c] and entry["p_re_not_better"] < 0.05),
                "per_day": bool(oos["per_day"][chosen] <= MAX_RATIO * oos["per_day"][c]),
            }
        result[question] = entry
    print(json.dumps({q: result[q] for q in ("A", "B")}, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = {"run": cmd_run, "evaluate": cmd_evaluate}
    parser.add_argument("command", choices=tuple(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
