"""冷门慢启动的跟踪补报：第一次提醒后只看价格，每比上次提醒再涨 S 就补报一波，口径与判据见 dormant_waves.md（跑数前写定）。

条件与特征沿用 dormant_slow.py 的抽取（W60），「准」与开发期/样本外划分与 dormant_slow 相同。

用法（仓库根目录，$P 见 README.md）：
  & $P research/launch_trade/dormant_waves.py run        # 逐合约跑 16 组跟踪参数，存每条提醒
  & $P research/launch_trade/dormant_waves.py evaluate   # 开发期选参、PBO、样本外检验
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from dormant_calm import _features
from dormant_slow import COOLDOWN, FEATURES, _cumsum, _extract
from launch_alerts import BOOTSTRAP, CSCV_BLOCKS, DEV_WEEKS, FORWARD, HOLDS, TARGETS, WEEK, _bootstrap_weights, _validity
from launch_grid import EPOCH, _multiplier, _rolling_extreme
from load_data import load_contracts
from numba import njit
from paths import RESULTS

OUT = RESULTS / "dormant_waves"
WINDOW = 60
# config/default/dormant.yaml 的 dormant.slow；上限类用 <=，其余用 >=。
SLOW = {"change": 6.0, "quote": 1e4, "active": 48.0, "jump": 3.0, "er": 0.3, "ratio": 3.0, "late": 0.5,
        "vol": 0.2, "retrace": 0.3}
UPPER = ("jump", "vol", "retrace")
STEPS = (3.0, 5.0, 8.0, 10.0)
FOLLOWS = (60, 120, 240, 360)
# 第一组不跟踪：对拍 dormant_slow.md 第三节第 2 小节第一行。
COMBOS = [(0.0, 0)] + list(itertools.product(STEPS, FOLLOWS))
P_SLACK = 0.01
NONINFERIOR = 0.02


@njit(cache=True)
def _run(close, cond, covered, start, step, follow, cooldown):
    """返回提醒的 K 线下标与波次；状态机见 dormant_waves.md 第二节第 2 小节。"""
    out_t = []
    out_w = []
    last_t = -1
    last_p = 0.0
    wave = 0
    for t in range(start, len(close)):
        if cond[t] and (last_t < 0 or t - last_t >= cooldown):
            wave = 1
        elif (follow > 0 and last_t >= 0 and t - last_t <= follow and covered[t]
              and close[t] >= last_p * (1 + step / 100)):
            wave += 1
        else:
            continue
        last_t = t
        last_p = close[t]
        out_t.append(t)
        out_w.append(wave)
    return np.array(out_t, np.int64), np.array(out_w, np.int64)


def _symbol(args: tuple[str, float]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    symbol, multiplier = args
    prepared = _features(symbol, multiplier)
    if prepared is None:
        return None
    series, cum_quote, cum_active, cum_uncovered, cum_tr, minute_change = prepared
    open_, high, low, close = series.open, series.high, series.low, series.close
    covered = (cum_uncovered[1:] - cum_uncovered[:-1]) == 0
    ok = covered & np.concatenate([[False], covered[:-1]])
    cum_move = _cumsum(np.nan_to_num(np.where(ok, np.abs(np.diff(close, prepend=close[0])), 0.0)))
    cum_body = _cumsum(np.nan_to_num(np.abs(close - open_)))
    cum_span = _cumsum(np.nan_to_num(high - low))
    index, f = _extract(open_, high, low, close, cum_quote, cum_active, cum_uncovered, cum_tr, cum_move, cum_body,
                        cum_span, _rolling_extreme(minute_change, WINDOW, True), _rolling_extreme(high, WINDOW, True),
                        WINDOW)
    passed = np.ones(len(index), np.bool_)
    for name, value in SLOW.items():
        column = f[:, FEATURES.index(name)]
        passed &= column <= value if name in UPPER else column >= value
    cond = np.zeros(len(close), np.bool_)
    cond[index[passed]] = True
    offset = (series.t0 - EPOCH) // 60
    # 2021-09 之前不计；状态从那时起算，与 dormant_slow 的网格（只含 2021-09 之后的候选）一致。
    start = max(0, -offset)
    targets = np.array(TARGETS)
    holds = np.array(HOLDS, np.int64)
    results = []
    for step, follow in COMBOS:
        t, wave = _run(close, cond, covered, start, step, follow, COOLDOWN)
        valid = _validity(high, low, close, cum_uncovered, t, np.ones(len(t), np.int8), targets, holds, FORWARD)
        week = ((t + offset) * 60 + 60) // WEEK
        results.append((week.astype(np.int64), wave, valid))
    return results


def cmd_run() -> None:
    contracts = load_contracts()
    jobs = [(s, _multiplier(contracts[s])) for s in sorted(contracts) if _multiplier(contracts[s])]
    parts: list[list] = [[] for _ in COMBOS]
    started = time.time()
    with ProcessPoolExecutor(max_workers=11) as executor:
        for done, result in enumerate(executor.map(_symbol, jobs, chunksize=2), 1):
            if result:
                for c, item in enumerate(result):
                    parts[c].append(item)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for c, chunks in enumerate(parts):
        arrays[f"week{c}"] = np.concatenate([x[0] for x in chunks])
        arrays[f"wave{c}"] = np.concatenate([x[1] for x in chunks])
        arrays[f"valid{c}"] = np.concatenate([x[2] for x in chunks])
    np.savez(OUT / "alerts.npz", combos=np.array(COMBOS), **arrays)
    print(f"完成 {time.time() - started:.0f}s")


# 每周计数的行：第 1 波（条数、计分条数、4 种设置的准确数），补报（同样 6 行）。
ROWS = 12


def _weekly(week: np.ndarray, wave: np.ndarray, valid: np.ndarray, n_weeks: int) -> np.ndarray:
    out = np.zeros((ROWS, n_weeks))
    keep = (week >= 0) & (week < n_weeks)
    for base, mask in ((0, wave == 1), (6, wave > 1)):
        m = keep & mask
        out[base] = np.bincount(week[m], minlength=n_weeks)
        scored = m & (valid[:, 0] >= 0)
        out[base + 1] = np.bincount(week[scored], minlength=n_weeks)
        for k in range(valid.shape[1]):
            out[base + 2 + k] = np.bincount(week[scored & (valid[:, k] == 1)], minlength=n_weeks)
    return out


def scores(x: np.ndarray, days) -> dict:
    """x 为各计数（ROWS 在倒数第一维之前或就是第一维，已在时间上求和）。"""
    x = np.asarray(x, np.float64)

    def precision(base):
        scored = x[..., base + 1]
        correct = x[..., base + 2 : base + 6].mean(-1)
        return np.divide(correct, scored, out=np.zeros(np.shape(scored)), where=scored > 0)

    p1, pw = precision(0), precision(6)
    wave_day = x[..., 6] / days
    return {"p1": p1, "first_per_day": x[..., 0] / days, "pw": pw, "wave_per_day": wave_day,
            "useful": pw * wave_day}


def eligible_of(s: dict) -> np.ndarray:
    return (s["pw"] >= s["p1"] - P_SLACK) & (s["wave_per_day"] <= s["first_per_day"])


def _pick(s: dict) -> int:
    # 第 0 组不跟踪（没有补报），不参与选择。
    useful = np.where(eligible_of(s), s["useful"], -np.inf)
    useful[0] = -np.inf
    return int(np.argmax(useful))


def _row(s: dict, i: int) -> dict:
    return {k: round(float(np.asarray(v)[i]), 4) for k, v in s.items()}


def cmd_evaluate() -> None:
    with np.load(OUT / "alerts.npz") as data:
        loaded = {name: data[name] for name in data.files}
    combos = [tuple(c) for c in loaded["combos"]]
    total = int(max(loaded[f"week{c}"].max() for c in range(len(combos)))) + 1
    weekly = np.array([_weekly(loaded[f"week{c}"], loaded[f"wave{c}"], loaded[f"valid{c}"], total)
                       for c in range(len(combos))])  # (组, ROWS, 周)
    dev = weekly[:, :, :DEV_WEEKS]
    oos = weekly[:, :, DEV_WEEKS:]
    n_oos = total - DEV_WEEKS
    s_dev = scores(dev.sum(2), DEV_WEEKS * 7)
    s_oos = scores(oos.sum(2), n_oos * 7)
    print("组（步长%, 跟踪分钟）：开发期 P1 / 首次每天 / Pw / 补报每天 / 有用 | 样本外同")
    for i, combo in enumerate(combos):
        d, o = _row(s_dev, i), _row(s_oos, i)
        mark = "✓" if eligible_of(s_dev)[i] else " "
        print(f"{mark} {combo}: {d['p1']:.3f} {d['first_per_day']:.2f} {d['pw']:.3f} {d['wave_per_day']:.2f} "
              f"{d['useful']:.3f} | {o['p1']:.3f} {o['first_per_day']:.2f} {o['pw']:.3f} {o['wave_per_day']:.2f} "
              f"{o['useful']:.3f}")
    eligible = eligible_of(s_dev)
    eligible[0] = False
    result: dict = {"combos": combos, "dev": {k: np.asarray(v).round(4).tolist() for k, v in s_dev.items()},
                    "oos": {k: np.asarray(v).round(4).tolist() for k, v in s_oos.items()},
                    "eligible": int(eligible.sum())}
    if not eligible.any():
        result["chosen"] = None
        print("开发期没有满足限制的组，不采用。")
    else:
        best = _pick(s_dev)
        result["chosen"] = combos[best]
        edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
        block = np.add.reduceat(dev, edges[:-1], axis=2)
        lengths = np.diff(edges) * 7
        logits = []
        picks = []
        for half in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
            mask = np.zeros(CSCV_BLOCKS, bool)
            mask[list(half)] = True
            s_is = scores(block[:, :, mask].sum(2), float(lengths[mask].sum()))
            s_os = scores(block[:, :, ~mask].sum(2), float(lengths[~mask].sum()))
            pick = _pick(s_is)
            picks.append(pick)
            perf = np.where(eligible_of(s_os), s_os["useful"], -np.inf)
            perf[0] = -np.inf
            perf = perf[1:]
            p = perf[pick - 1]
            below = np.sum(perf < p) + 0.5 * (np.sum(perf == p) - 1)
            omega = (below + 1) / (len(perf) + 1)
            logits.append(math.log(omega / (1 - omega)))
        result["pbo"] = float(np.mean(np.array(logits) <= 0))
        counts = np.bincount(np.array(picks), minlength=len(combos))
        result["cscv_picks"] = {str(combos[i]): float(counts[i] / len(picks)) for i in np.argsort(-counts)[:5]
                                if counts[i]}
        o = _row(s_oos, best)
        x = oos[best].sum(1)
        p1_setting = x[2:6] / max(x[1], 1)
        pw_setting = x[8:12] / max(x[7], 1)
        weights = _bootstrap_weights(n_oos, BOOTSTRAP, seed=2710).astype(np.float64)
        boot = scores(np.einsum("bw,rw->br", weights, oos[best]), weights.sum(1) * 7)
        p_inferior = float(np.mean(boot["pw"] - boot["p1"] < -NONINFERIOR))
        result.update({"chosen_dev": _row(s_dev, best), "chosen_oos": o,
                       "p1_by_setting": p1_setting.round(4).tolist(), "pw_by_setting": pw_setting.round(4).tolist(),
                       "bootstrap_p_inferior": p_inferior})
        result["checks"] = {
            "pbo": result["pbo"] < 0.5,
            "precision": bool(o["pw"] >= o["p1"] - P_SLACK),
            "settings_ok": int(np.sum(pw_setting >= p1_setting - P_SLACK)),
            "noninferior": p_inferior < 0.05,
            "per_day": bool(o["wave_per_day"] <= o["first_per_day"]),
        }
        # 只报告：样本外按波次的准确率。
        c = best
        week, wave, valid = loaded[f"week{c}"], loaded[f"wave{c}"], loaded[f"valid{c}"]
        sel = (week >= DEV_WEEKS) & (valid[:, 0] >= 0)
        by_wave = {}
        for label, m in (("2", wave == 2), ("3", wave == 3), ("4", wave == 4), ("5+", wave >= 5)):
            rows = valid[sel & m]
            if len(rows):
                by_wave[label] = {"n": int(len(rows)), "precision": round(float((rows == 1).mean()), 4)}
        result["oos_by_wave"] = by_wave
    print(json.dumps({k: v for k, v in result.items() if k not in ("dev", "oos")}, ensure_ascii=False, indent=1))
    (OUT / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = {"run": cmd_run, "evaluate": cmd_evaluate}
    parser.add_argument("command", choices=tuple(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
