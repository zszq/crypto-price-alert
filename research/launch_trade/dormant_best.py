"""冷门启动：放开全部形态条件重新选参，口径与判据见 dormant_best.md（跑数前写定）。

直接用 dormant_early.py extract 抽好的候选分钟（每个窗口一份，含全部形态特征），不重新抽取。

用法（仓库根目录，$P 见 README.md；先跑过 dormant_early.py extract）：
  & $P research/launch_trade/dormant_best.py search     # 全部候选逐周计数（全期，选参只用开发期）
  & $P research/launch_trade/dormant_best.py select     # B3、开发期选参、PBO
  & $P research/launch_trade/dormant_best.py evaluate   # 样本外：选中方案、B3、旧基准
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
from dormant_early import COOLDOWN, OUT, STEP, _data_weeks, event_counts, load
from launch_alerts import BOOTSTRAP, CSCV_BLOCKS, DEV_WEEKS, _bootstrap_weights
from numba import njit

BEST = OUT / "best"
MULTIPLE = 10.0  # 倍数下限固定：在 B3 上 0 与 10 结果相同，不值得占候选族
WINDOWS = (5, 10, 15)
GRID = {
    "change": (5.0, 6.0, 7.0, 8.0, 10.0),
    "quote": (5e3, 1e4, 2e4),
    "active": (0.5, 0.8),  # 占窗口的比例，按 ⌈比例 × W⌉ 换成分钟数
    "last_quote": (0.0, 1e3, 3e3),
    "late": (0.0, 0.4),
    "jump": (4.0, 5.0, 6.0, np.inf),
    "body": (0.0, 0.5, 0.6),
    "vol": (0.5, 1.0, np.inf),
    "pre": (5e5, 1e6),
}
NAMES = ("W", *GRID)
B3 = {"W": 15, "change": 8.0, "quote": 2e4, "active": 0.5, "last_quote": 1e3, "late": 0.4, "jump": 5.0, "body": 0.6,
      "vol": 0.5, "pre": 1e6}
OLD = {"W": 10, "change": 8.0, "quote": 5e3, "active": 0.5, "last_quote": 0.0, "late": 0.0, "jump": 5.0, "body": 0.6,
       "vol": 0.5, "pre": 1e6}
P_SLACK = 0.01
COUNTS = ("alerts", "scored", "correct", "credit10", "credit20", "caught10", "caught20")


def _need(window: int, share: float) -> int:
    return max(1, math.ceil(window * share - 1e-9))


def _ok(rec: dict, cfg: dict) -> np.ndarray:
    ok = rec["multiple"] >= MULTIPLE
    for name in GRID:
        ok &= _ok_single(rec, cfg["W"], name, cfg[name])
    return ok


@njit(cache=True)
def _weekly(sym, t, price, week, precision, leg10, leg20, frac10, frac20, lw10, lw20, ok, n_weeks, out, alert):
    """同 dormant_early._grid 的单组版本：冷却内只有比上次提醒价再涨 STEP 才报下一波；行情只算第一次提醒。"""
    last_s = -1
    last_t = -1
    last_p = 0.0
    l10 = -1
    l20 = -1
    for i in range(len(t)):
        if sym[i] != last_s:
            last_s = sym[i]
            last_t = -1
            l10 = -1
            l20 = -1
        if not ok[i]:
            continue
        if last_t >= 0 and t[i] - last_t < COOLDOWN and price[i] < last_p * (1 + STEP / 100):
            continue
        last_t = t[i]
        last_p = price[i]
        alert[i] = True
        w = week[i]
        if w < n_weeks:
            out[0, w] += 1
            if not np.isnan(precision[i]):
                out[1, w] += 1
                out[2, w] += precision[i]
        if leg10[i] >= 0 and leg10[i] != l10:
            l10 = leg10[i]
            if lw10[i] < n_weeks:
                out[3, lw10[i]] += 1 - frac10[i]
                out[5, lw10[i]] += frac10[i] <= 0.5
        if leg20[i] >= 0 and leg20[i] != l20:
            l20 = leg20[i]
            if lw20[i] < n_weeks:
                out[4, lw20[i]] += 1 - frac20[i]
                out[6, lw20[i]] += frac20[i] <= 0.5


def run_config(rec: dict, ok: np.ndarray, n_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(COUNTS), n_weeks))
    alert = np.zeros(len(ok), np.bool_)
    _weekly(rec["sym"], rec["t"], rec["price"], rec["week"], rec["precision"], rec["leg10"], rec["leg20"],
            rec["frac10"], rec["frac20"], rec["leg_week10"], rec["leg_week20"], ok, n_weeks, out, alert)
    return out, alert


def load_rec(window: int) -> dict:
    rec = load(window)
    rec.pop("f")
    for tag in (10, 20):
        rec[f"frac{tag}"] = rec[f"frac{tag}"].astype(np.float64)
    return rec


def configs(window: int) -> list[dict]:
    return [dict(zip(NAMES, (window, *combo), strict=True)) for combo in itertools.product(*GRID.values())]


def _search_window(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    window, n_weeks = args
    rec = load_rec(window)
    # 每个阈值的掩码只算一次，组合时逐项相与，比每组重新比较快一个数量级。
    masks = {name: {v: _ok_single(rec, window, name, v) for v in values} for name, values in GRID.items()}
    base = rec["multiple"] >= MULTIPLE
    rows = []
    counts = np.zeros((len(configs(window)), len(COUNTS), n_weeks), np.float32)
    for k, cfg in enumerate(configs(window)):
        ok = base.copy()
        for name in GRID:
            ok &= masks[name][cfg[name]]
        counts[k] = run_config(rec, ok, n_weeks)[0]
        rows.append([cfg[name] for name in NAMES])
    return np.array(rows), counts


def _ok_single(rec: dict, window: int, name: str, value: float) -> np.ndarray:
    if name == "active":
        return rec["active"] >= _need(window, value)
    if name in ("jump", "vol", "pre"):
        return rec[name] <= value
    return rec[name] >= value


def cmd_search() -> None:
    started = time.time()
    total = _data_weeks()
    with ProcessPoolExecutor(max_workers=len(WINDOWS)) as executor:
        parts = list(executor.map(_search_window, [(w, total) for w in WINDOWS]))
    BEST.mkdir(parents=True, exist_ok=True)
    np.savez(BEST / "grid.npz", rows=np.concatenate([p[0] for p in parts]),
             counts=np.concatenate([p[1] for p in parts]))
    print(f"{sum(len(p[0]) for p in parts)} 组，{total} 周，{time.time() - started:.0f}s")


# ---------------------------------------------------------------- 指标、选参与检验


def scores(c: np.ndarray, events: np.ndarray, days) -> dict:
    """c：(..., 7) 在时间上求和后的计数；events：(2, ...) 两种行情的段数。"""
    c = np.asarray(c, np.float64)
    precision = np.divide(c[..., 2], c[..., 1], out=np.zeros(c.shape[:-1]), where=c[..., 1] > 0)
    per_day = c[..., 0] / days
    return {"precision": precision, "per_day": per_day, "fpd": (1 - precision) * per_day,
            "re": 0.5 * (c[..., 3] / events[0] + c[..., 4] / events[1]),
            "recall": 0.5 * (c[..., 5] / events[0] + c[..., 6] / events[1])}


def objective(s: dict, b: int) -> np.ndarray:
    return 2 * (1 - s["fpd"] / s["fpd"][b]) + (s["re"] / s["re"][b] - 1)


def eligible(s: dict, b: int) -> np.ndarray:
    return (s["re"] >= s["re"][b]) & (s["precision"] >= s["precision"][b] - P_SLACK)


def pick(s: dict, b: int) -> int:
    return int(np.argmax(np.where(eligible(s, b), objective(s, b), -np.inf)))


def _index(rows: np.ndarray, cfg: dict) -> int:
    hit = np.flatnonzero(np.all(rows == np.array([cfg[n] for n in NAMES]), axis=1))
    assert len(hit) == 1, hit
    return int(hit[0])


def _describe(rows: np.ndarray, i: int) -> dict:
    return {n: float(v) for n, v in zip(NAMES, rows[i], strict=True)}


def _row(s: dict, i: int) -> dict:
    return {k: float(v[i]) for k, v in s.items()}


def cmd_select() -> None:
    data = np.load(BEST / "grid.npz")
    rows = data["rows"]
    dev = data["counts"][:, :, :DEV_WEEKS].astype(np.float64)
    events = event_counts(0, DEV_WEEKS)
    b = _index(rows, B3)
    s = scores(dev.sum(2), events.sum(1), DEV_WEEKS * 7)
    best = pick(s, b)
    j = objective(s, b)
    ok = eligible(s, b)
    summary: dict = {"trials": len(rows), "eligible": int(ok.sum()), "b3_dev": _row(s, b),
                     "old_dev": _row(s, _index(rows, OLD)), "chosen": _describe(rows, best),
                     "chosen_dev": {**_row(s, best), "J": float(j[best])}}
    edges = np.linspace(0, DEV_WEEKS, CSCV_BLOCKS + 1).round().astype(np.int64)
    block = np.add.reduceat(dev, edges[:-1], axis=2)
    ev_block = np.add.reduceat(events, edges[:-1], axis=1)
    lengths = np.diff(edges) * 7
    logits = []
    picks = []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        mask = np.zeros(CSCV_BLOCKS, bool)
        mask[list(combo)] = True
        part = {side: scores(block[:, :, m].sum(2), ev_block[:, m].sum(1), float(lengths[m].sum()))
                for side, m in (("is", mask), ("oos", ~mask))}
        chosen = pick(part["is"], b)
        picks.append(chosen)
        oos = part["oos"]
        perf = np.where(eligible(oos, b), objective(oos, b), -np.inf)
        below = np.sum(perf < perf[chosen]) + 0.5 * (np.sum(perf == perf[chosen]) - 1)
        omega = (below + 1) / (len(perf) + 1)
        logits.append(math.log(omega / (1 - omega)))
    picks = np.array(picks)
    summary["pbo"] = float(np.mean(np.array(logits) <= 0))
    summary["cscv_b3_picked"] = float(np.mean(picks == b))
    summary["cscv_window_of_pick"] = {int(w): float(np.mean(rows[picks, 0] == w)) for w in WINDOWS}
    order = np.argsort(-np.where(ok, j, -np.inf))
    summary["top10"] = [{**_describe(rows, int(i)), **_row(s, int(i)), "J": float(j[i])} for i in order[:10] if ok[i]]
    summary["best_by_window"] = {}
    for w in WINDOWS:
        inside = ok & (rows[:, 0] == w)
        if inside.any():
            i = int(np.flatnonzero(inside)[np.argmax(j[inside])])
            summary["best_by_window"][w] = {**_describe(rows, i), **_row(s, i), "J": float(j[i])}
    print(json.dumps({k: v for k, v in summary.items() if k != "top10"}, ensure_ascii=False, indent=1), flush=True)
    (BEST / "select.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _first_alerts(rec: dict, alert: np.ndarray, tag: int) -> dict:
    """每段行情的第一次提醒：(合约, 段号) → 候选下标。"""
    out: dict = {}
    for i in np.flatnonzero(alert & (rec[f"leg{tag}"] >= 0)):
        out.setdefault((int(rec["sym"][i]), int(rec[f"leg{tag}"][i])), i)
    return out


def _paired(runs: dict, tag: int) -> dict:
    """两组都报到的行情里，新方案第一次提醒早几分钟、提醒价低多少（只供解读）。"""
    firsts = {name: _first_alerts(run["rec"], run["alert"] & (run["rec"]["week"] >= DEV_WEEKS), tag)
              for name, run in runs.items()}
    common = sorted(set(firsts["baseline"]) & set(firsts["chosen"]))
    rb, rc = runs["baseline"]["rec"], runs["chosen"]["rec"]
    fb, fc = firsts["baseline"], firsts["chosen"]
    minutes = np.array([rb["t"][fb[k]] - rc["t"][fc[k]] for k in common], np.float64)
    price = np.array([rc["price"][fc[k]] / rb["price"][fb[k]] - 1 for k in common])
    frac = np.array([rb[f"frac{tag}"][fb[k]] - rc[f"frac{tag}"][fc[k]] for k in common], np.float64)
    summary = {"baseline_only": len(set(fb) - set(fc)), "chosen_only": len(set(fc) - set(fb)), "common": len(common)}
    if common:
        summary.update({"earlier_share": float(np.mean(minutes > 0)), "later_share": float(np.mean(minutes < 0)),
                        "median_minutes_earlier": float(np.median(minutes)),
                        "median_price_change": float(np.median(price)), "mean_frac_earlier": float(np.mean(frac))})
    return summary


def cmd_evaluate() -> None:
    chosen_cfg = json.loads((BEST / "select.json").read_text(encoding="utf-8"))["chosen"]
    chosen_cfg = {k: (int(v) if k == "W" else v) for k, v in chosen_cfg.items()}
    total = _data_weeks()
    n_weeks = total - DEV_WEEKS
    days = n_weeks * 7
    events = event_counts(DEV_WEEKS, n_weeks)
    runs = {}
    for name, cfg in (("baseline", B3), ("chosen", chosen_cfg), ("old", OLD)):
        rec = load_rec(cfg["W"])
        weekly, alert = run_config(rec, _ok(rec, cfg), total)
        valid = rec["valid"][alert & (rec["week"] >= DEV_WEEKS)]
        valid = valid[valid[:, 0] >= 0]
        runs[name] = {"weekly": weekly[:, DEV_WEEKS:], "rec": rec, "alert": alert,
                      "by_setting": [float(v) for v in valid.mean(0)]}
    flat = {name: scores(run["weekly"].sum(1), events.sum(1), days) for name, run in runs.items()}
    result: dict = {"chosen_config": chosen_cfg, "oos_weeks": n_weeks,
                    "events": [float(events[0].sum()), float(events[1].sum())]}
    for name, run in runs.items():
        result[name] = {k: float(v) for k, v in flat[name].items()} | {"by_setting": run["by_setting"]}
    base, chosen = result["baseline"], result["chosen"]

    def j_of(s: dict, ref: dict):
        return 2 * (1 - s["fpd"] / ref["fpd"]) + (s["re"] / ref["re"] - 1)

    chosen["J"] = float(j_of(flat["chosen"], flat["baseline"]))
    result["old"]["J_vs_b3"] = float(j_of(flat["old"], flat["baseline"]))
    weights = _bootstrap_weights(n_weeks, BOOTSTRAP, seed=2410).astype(np.float64)
    ev_boot = np.stack([weights @ events[0], weights @ events[1]])
    boot = {name: scores(np.einsum("bw,cw->bc", weights, run["weekly"]), ev_boot, weights.sum(1) * 7)
            for name, run in runs.items()}
    result["bootstrap_p_J_le_0"] = float(np.mean(j_of(boot["chosen"], boot["baseline"]) <= 0))
    # B3 相对旧基准（只报告）：准确率更高、每天假启动更少的概率。
    result["b3_vs_old_bootstrap"] = {
        "p_precision_le_old": float(np.mean(boot["baseline"]["precision"] <= boot["old"]["precision"])),
        "p_fpd_ge_old": float(np.mean(boot["baseline"]["fpd"] >= boot["old"]["fpd"]))}
    result["checks"] = {
        "re": chosen["re"] >= base["re"], "fpd": chosen["fpd"] < base["fpd"],
        "precision": chosen["precision"] >= base["precision"] - P_SLACK,
        "settings_ok": int(sum(a >= b - P_SLACK for a, b in zip(chosen["by_setting"], base["by_setting"],
                                                                 strict=True))),
        "bootstrap": result["bootstrap_p_J_le_0"] < 0.05, "per_day": 1 <= chosen["per_day"] <= 50,
    }
    result["paired"] = {tag: _paired({"baseline": runs["baseline"], "chosen": runs["chosen"]}, tag) for tag in (10, 20)}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (BEST / "evaluate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("search", "select", "evaluate"))
    {"search": cmd_search, "select": cmd_select, "evaluate": cmd_evaluate}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
