"""第八轮：放量上涨与放量下跌能否共用一套提醒参数——目标是提醒更早、提醒后方向延续性更高。

不看交易盈亏，只看提醒本身的质量，两个方向分开统计：
- 延续率：提醒价起，先顺着提醒方向走 X%、还是先逆向走 X%（X = 3/5/10），顺向记 1、逆向记 0、同一分钟都碰到记 0.5，
  24 小时内都没碰到不计。上下两条线按对数对称取（价格 × e^±x，x = ln(1 + X%)），随机游走时两个方向都是 50%；
  按百分比对称取的话 +5% / -5% 对上涨天然有利约 2.4 个百分点、对下跌不利同样多，两个方向就没法直接比；
- 提醒早晚：提醒时这段行情已经走了多少（提醒价相对前 60 分钟最低价 / 最高价的对数距离，越小越早），
  以及提醒后 4 小时内顺向最远还能走多少（对数），「剩余占比」= 之后 ÷（之前 + 之后）；
  另外把每个变体的提醒与基准变体按（合约, 方向, 前后 60 分钟内）配对，看同一段行情早报 / 晚报了几分钟。

检验协议沿用第五轮：2024-10-10 起两年是开发期（挑参数），2021-10-10 ~ 2024-09-30 三年只做样本外；
再按半年分 10 段，看变体相对基准的差在几段为正，只采用两段时期一致、各段多数同号、相邻取值单调或平稳的结论。
提醒按时间强烈聚集（全市场暴跌时一分钟几十条下跌提醒），另给「去掉提醒最多的 10 天」后的延续率。

    $P = "data/backtest/.venv/Scripts/python.exe"
    & $P research/launch_trade/r8_sides.py run            # 约 40 分钟，结果写 results/r8/
    & $P research/launch_trade/r8_sides.py report         # 单因素表（--first-wave 只看第 1 波）
    & $P research/launch_trade/r8_sides.py match          # 单因素变体与基准同段行情配对：早报 / 晚报
    & $P research/launch_trade/r8_sides.py control        # 随机时刻对照（同合约池），写 results/r8/control.npy
    & $P research/launch_trade/r8_sides.py run --set combo     # 两个方向各自的组合小网格，写 results/r8_combo/
    & $P research/launch_trade/r8_sides.py combo --set combo   # 组合排名
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_trade_backtest import REPO, RESULTS, PoolRule  # noqa: E402
from load_data import load_contracts, load_series  # noqa: E402
from r5_export import detect_both  # noqa: E402
from r5_lib import beijing  # noqa: E402

OUT = RESULTS / "r8"
SETS = {"single": RESULTS / "r8", "combo": RESULTS / "r8_combo", "final": RESULTS / "r8_final"}
HORIZON = 1440
LEVELS = (3, 5, 10)
FIELDS = [
    "closed_at",
    "direction",
    "wave",
    "change",
    "symbol",
    "pre_move",  # 提醒价离前 60 分钟反向极值的对数距离（已经走了多少）
    "mfe_60",  # 提醒后 60 / 240 分钟顺向最远（对数，用最高价 / 最低价）
    "mfe_240",
    "mae_60",  # 提醒后 60 分钟逆向最远（对数）
    "ret_60",  # 提醒后 60 / 240 分钟收盘的顺向对数收益
    "ret_240",
    *(f"cont_{level}" for level in LEVELS),
]

# 基准取当前 config/default.yaml 的 launch 段（池下限 50 万、基准期 360、窗口 10 万、不要求突破、冷却 10……），
# 单因素变体逐个改一项。池下限放进变体里，因为它决定哪些币能报，也影响两个方向的质量。
SINGLE: dict[str, list] = {
    "window_minutes": [3, 5, 10, 15],
    "baseline_minutes": [180, 360, 720, 1440],
    "min_volume_ratio": [2.0, 3.0, 4.0, 5.0, 8.0],
    "min_window_quote": [50_000, 100_000, 200_000, 300_000, 500_000],
    "min_change_percent": [2.0, 3.0, 4.0, 5.0],
    "require_breakout": [False, True],
    "max_change_percent": [8.0, 10.0, 15.0, 20.0, None],
    "pool": [0, 500_000, 1_000_000, 3_000_000],
}


def variants(base_rule, base_pool: float) -> dict[str, tuple]:
    found = {"base": (base_rule, base_pool, (1, -1))}
    for field, values in SINGLE.items():
        for value in values:
            if field == "pool":
                rule, pool = base_rule, value
            else:
                rule, pool = replace(base_rule, **{field: value}), base_pool
            if (rule, pool) == (base_rule, base_pool):
                continue
            found[f"{field}={value}"] = (rule, pool, (1, -1))
    return found


def combo_variants(base_rule, base_pool: float) -> dict[str, tuple]:
    """组合：两个方向各自只在单因素里两段时期都有利、各半年段多数同号的参数上取小网格，避免在大网格里挑到偶然的点。

    另加「只看价格」的对照（量比、窗口成交额不设门槛）：放量条件对某个方向没有作用的话，它和基准差不多。
    """
    found = {"base": (base_rule, base_pool, (1, -1))}
    for name, extra in {
        "price_only": {"min_volume_ratio": 0.0, "min_window_quote": 0.0},
        "no_ratio": {"min_volume_ratio": 0.0},
        "no_quote": {"min_window_quote": 0.0},
    }.items():
        found[name] = (replace(base_rule, **extra), base_pool, (1, -1))
    for baseline in (360, 1440):
        for window in (3, 5):
            for cap in (10.0, 15.0):
                for quote in (100_000, 200_000):
                    rule = replace(
                        base_rule,
                        baseline_minutes=baseline,
                        window_minutes=window,
                        max_change_percent=cap,
                        min_window_quote=quote,
                    )
                    found[f"U b{baseline} w{window} x{cap:g} q{quote // 1000}k"] = (rule, base_pool, (1,))
    for change in (2.0, 2.5, 3.0):
        for cap in (8.0, 10.0, 15.0):
            for window in (5, 10):
                for ratio in (3.0, 4.0):
                    rule = replace(
                        base_rule,
                        min_change_percent=change,
                        max_change_percent=cap,
                        window_minutes=window,
                        min_volume_ratio=ratio,
                    )
                    found[f"D c{change:g} x{cap:g} w{window} r{ratio:g}"] = (rule, base_pool, (-1,))
    return found


def final_variants(base_rule, base_pool: float) -> dict[str, tuple]:
    """组合排名后的复核：下跌涨跌幅下限在网格边上（2%），再往外取 1.5% 看是平台还是悬崖；量比再取 5；
    上涨只把上限改成两个方向共用的 10%，并看量比 4 对上涨是否无害（能否两个方向共用）。"""
    found = {"base": (base_rule, base_pool, (1, -1))}
    for ratio in (3.0, 4.0):
        rule = replace(base_rule, max_change_percent=10.0, min_volume_ratio=ratio)
        found[f"U x10 r{ratio:g}"] = (rule, base_pool, (1,))
    for change in (1.5, 2.0, 2.5):
        for ratio in (4.0, 5.0):
            rule = replace(base_rule, min_change_percent=change, max_change_percent=10.0, min_volume_ratio=ratio)
            found[f"D c{change:g} x10 w5 r{ratio:g}"] = (rule, base_pool, (-1,))
    return found


def _first(mask: np.ndarray) -> int:
    return int(np.argmax(mask)) if mask.any() else -1


def outcome(series, i: int, sign: int) -> list[float]:
    """提醒在 K 线 i 收盘时发出，结果从 i+1 开始算；遇到数据缺口截断。"""
    n = len(series)
    stop = min(n, i + 1 + HORIZON)
    covered = series.covered[i + 1 : stop]
    if not covered.all():
        stop = i + 1 + int(np.argmin(covered))
    price = series.close[i]
    high = np.log(series.high[i + 1 : stop] / price)
    low = np.log(series.low[i + 1 : stop] / price)
    close = np.log(series.close[i + 1 : stop] / price)
    fav, adv = (high, -low) if sign > 0 else (-low, high)
    pre = slice(max(0, i - 59), i + 1)
    if sign > 0:
        pre_move = float(np.log(price / series.low[pre].min()))
    else:
        pre_move = float(np.log(series.high[pre].max() / price))

    def best(values: np.ndarray, k: int) -> float:
        return float(values[:k].max()) if len(values) >= k else np.nan

    def at(k: int) -> float:
        return float(close[k - 1] * sign) if len(close) >= k else np.nan

    row = [pre_move, best(fav, 60), best(fav, 240), best(adv, 60), at(60), at(240)]
    for level in LEVELS:
        x = np.log1p(level / 100)
        hit_fav, hit_adv = _first(fav >= x), _first(adv >= x)
        if hit_fav < 0 and hit_adv < 0:
            # 24 小时内都没走出 ±X%：数据截断的也一样不计。
            row.append(np.nan)
        elif hit_adv < 0 or (hit_fav >= 0 and hit_fav < hit_adv):
            row.append(1.0)
        elif hit_fav < 0 or hit_adv < hit_fav:
            row.append(0.0)
        else:
            row.append(0.5)
    return row


def run_symbol(args):
    symbol, multiplier, sym_index, todo, exit_ratio, refresh = args
    series = load_series(symbol, multiplier)
    if series is None:
        return {}
    memo: dict[tuple[int, int], list[float]] = {}
    result = {}
    for name, (rule, pool_min, sides) in todo.items():
        pool = PoolRule(pool_min, None, exit_ratio, refresh)
        rows = []
        for i, sign, wave, change, _quote, _broke in detect_both(series, rule, pool, sides):
            key = (i, sign)
            if key not in memo:
                memo[key] = outcome(series, i, sign)
            rows.append([series.t0 + (i + 1) * 60, sign, wave, change, sym_index, *memo[key]])
        result[name] = np.array(rows, dtype=np.float64).reshape(-1, len(FIELDS))
    return result


def run(args) -> None:
    from price_alert.config import load_config

    config = load_config(REPO / "config" / "default.yaml")
    # drop 清掉：变体改的是两个方向共用的门槛，YAML 里的下跌单独门槛会盖住下跌方向的改动，研究就失去意义。
    base_rule = replace(config.launch.to_rule(), alert_drops=True, drop=None)
    build = {"single": variants, "combo": combo_variants, "final": final_variants}[args.set]
    todo = build(base_rule, config.launch.min_volume_24h_quote)
    print(f"基准 {base_rule} 池下限 {config.launch.min_volume_24h_quote:.0f}，共 {len(todo)} 个变体", flush=True)
    contracts = load_contracts()

    def multiplier(symbol: str) -> float | None:
        try:
            value = float(contracts[symbol].get("quanto_multiplier"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    symbols = sorted(contracts)
    jobs = [
        (s, multiplier(s), k, todo, config.gate.universe_exit_volume_ratio, config.gate.universe_refresh_seconds)
        for k, s in enumerate(symbols)
    ]
    collected: dict[str, list] = {name: [] for name in todo}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, part in enumerate(executor.map(run_symbol, jobs, chunksize=2), 1):
            for name, rows in part.items():
                collected[name].append(rows)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "symbols.npy", np.array(symbols))
    (OUT / "fields.txt").write_text("\n".join(FIELDS), encoding="utf-8")
    for name, parts in collected.items():
        rows = np.concatenate(parts) if parts else np.empty((0, len(FIELDS)))
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        np.save(OUT / f"{_safe(name)}.npy", rows)
    (OUT / "variants.txt").write_text("\n".join(todo), encoding="utf-8")
    print("完成", flush=True)


def _safe(name: str) -> str:
    return name.replace("=", "_")


# ---------- 统计 ----------

DEV = (beijing(2024, 10, 10), beijing(2026, 10, 15))
OOS = (beijing(2021, 10, 10), beijing(2024, 9, 30))
SEGMENTS = [
    (beijing(year, month), beijing(year + (month + 6 > 12), (month + 5) % 12 + 1))
    for year, month in [
        (2021, 10),
        (2022, 4),
        (2022, 10),
        (2023, 4),
        (2023, 10),
        (2024, 4),
        (2024, 10),
        (2025, 4),
        (2025, 10),
        (2026, 4),
    ]
]


def load(name: str) -> dict[str, np.ndarray]:
    rows = np.load(OUT / f"{_safe(name)}.npy")
    return {field: rows[:, k] for k, field in enumerate(FIELDS)}


def select(data: dict, start: int, end: int, sign: int, first_wave: bool = False) -> np.ndarray:
    mask = (data["closed_at"] >= start) & (data["closed_at"] < end) & (data["direction"] == sign)
    if first_wave:
        mask &= data["wave"] == 1
    return mask


def rate(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean() * 100) if len(values) else np.nan


def drop_top_days(data: dict, mask: np.ndarray, days: int = 10) -> np.ndarray:
    """去掉提醒最多的若干天（北京时间自然日）：一次全市场暴跌 / 普涨贡献几百条同向提醒，会主导平均值。"""
    day = ((data["closed_at"] + 8 * 3600) // 86400).astype(np.int64)
    picked = day[mask]
    if not len(picked):
        return mask
    values, counts = np.unique(picked, return_counts=True)
    top = values[np.argsort(counts)[::-1][:days]]
    return mask & ~np.isin(day, top)


def summary(data: dict, span: tuple[int, int], sign: int, first_wave: bool = False) -> dict[str, float]:
    mask = select(data, *span, sign, first_wave)
    days = (span[1] - span[0]) / 86400
    trimmed = drop_top_days(data, mask)
    pre, after = data["pre_move"][mask], data["mfe_240"][mask]
    share = after / (pre + after)
    return {
        "n": float(mask.sum()),
        "per_day": mask.sum() / days,
        **{f"c{level}": rate(data[f"cont_{level}"][mask]) for level in LEVELS},
        "c5_trim": rate(data["cont_5"][trimmed]),
        "pre": float(np.nanmedian(pre) * 100) if mask.any() else np.nan,
        "mfe240": float(np.nanmedian(after) * 100) if mask.any() else np.nan,
        "share": float(np.nanmedian(share) * 100) if mask.any() else np.nan,
        "ret240": float(np.nanmean(data["ret_240"][mask]) * 100) if mask.any() else np.nan,
    }


def segment_wins(data: dict, base: dict, sign: int, level: int = 5) -> tuple[int, int]:
    """变体在 10 个半年段里延续率高于基准的段数（两边都有提醒的段）。"""
    wins = total = 0
    for span in SEGMENTS:
        a = rate(data[f"cont_{level}"][select(data, *span, sign)])
        b = rate(base[f"cont_{level}"][select(base, *span, sign)])
        if np.isnan(a) or np.isnan(b):
            continue
        total += 1
        wins += a > b
    return wins, total


def report(args) -> None:
    names = (OUT / "variants.txt").read_text(encoding="utf-8").split("\n")
    base = load("base")
    side_names = {1: "放量上涨", -1: "放量下跌"}
    for sign in (1, -1):
        print(f"\n===== {side_names[sign]}（{'只看第 1 波' if args.first_wave else '所有波次'}）=====")
        header = (
            f"{'变体':<26}{'期':>4}{'条/天':>7}{'±3%':>6}{'±5%':>6}{'±10%':>6}{'去10天':>7}"
            f"{'已走%':>7}{'后4h%':>7}{'剩余%':>6}{'4h收%':>7}{'段胜':>6}"
        )
        print(header)
        for name in names:
            data = base if name == "base" else load(name)
            for label, span in (("开发", DEV), ("样本外", OOS)):
                s = summary(data, span, sign, args.first_wave)
                wins = "" if name == "base" or label == "样本外" else "{}/{}".format(*segment_wins(data, base, sign))
                print(
                    f"{name if label == '开发' else '':<26}{label:>4}{s['per_day']:>7.1f}{s['c3']:>6.1f}{s['c5']:>6.1f}"
                    f"{s['c10']:>6.1f}{s['c5_trim']:>7.1f}{s['pre']:>7.2f}{s['mfe240']:>7.2f}{s['share']:>6.1f}"
                    f"{s['ret240']:>7.2f}{wins:>6}"
                )


def pair(var: dict, base: dict, sign: int, span: tuple[int, int], within: int = 60) -> dict[str, float]:
    """按（合约, 方向）把两份第 1 波提醒配对（前后 within 分钟内最近的一条）。

    第 1 波就是一段行情的首报，比较它才能看出同一段行情谁报得早；只比所有提醒的「已走幅度」分不清
    是报得早，还是换了一批币。
    """

    def keys(data: dict) -> tuple[np.ndarray, np.ndarray]:
        mask = select(data, *span, sign, first_wave=True)
        key = data["symbol"][mask] * 1e10 + data["closed_at"][mask]
        order = np.argsort(key)
        return key[order], np.flatnonzero(mask)[order]

    base_key, base_rows = keys(base)
    var_key, var_rows = keys(var)
    if not len(base_key) or not len(var_key):
        return {"covered": np.nan, "dt": np.nan, "dprice": np.nan, "new": np.nan, "new_c5": np.nan, "lost_c5": np.nan}
    pos = np.clip(np.searchsorted(var_key, base_key), 1, len(var_key) - 1)
    left, right = var_key[pos - 1], var_key[pos]
    nearest = np.where(np.abs(left - base_key) <= np.abs(right - base_key), pos - 1, pos)
    dt = (var_key[nearest] - base_key) / 60
    ok = np.abs(dt) <= within
    # 同一段行情提醒时价格差多少（顺向对数 %，负数表示变体的提醒价更靠近起点，即更早）。
    price_gap = (var["pre_move"][var_rows[nearest[ok]]] - base["pre_move"][base_rows[ok]]) * 100
    pos_v = np.clip(np.searchsorted(base_key, var_key), 1, len(base_key) - 1)
    gap_v = np.minimum(np.abs(base_key[pos_v - 1] - var_key), np.abs(base_key[pos_v] - var_key)) / 60
    new = gap_v > within
    return {
        "covered": ok.mean() * 100,
        "dt": float(np.median(dt[ok])) if ok.any() else np.nan,
        "early": float((dt[ok] < 0).mean() * 100) if ok.any() else np.nan,
        "late": float((dt[ok] > 0).mean() * 100) if ok.any() else np.nan,
        "dprice": float(np.mean(price_gap)) if ok.any() else np.nan,
        "new": new.mean() * 100,
        "new_c5": rate(var["cont_5"][var_rows[new]]),
        "lost_c5": rate(base["cont_5"][base_rows[~ok]]),
    }


def match(args) -> None:
    names = (OUT / "variants.txt").read_text(encoding="utf-8").split("\n")
    base = load("base")
    for sign, title in ((1, "放量上涨"), (-1, "放量下跌")):
        print(f"\n===== {title}：第 1 波与基准配对（±60 分钟）=====")
        print(
            f"{'变体':<26}{'期':>4}{'覆盖%':>7}{'中位差分':>9}{'早报%':>7}{'晚报%':>7}{'价差%':>7}"
            f"{'新增%':>7}{'新增±5':>8}{'丢失±5':>8}"
        )
        for name in names[1:]:
            data = load(name)
            for label, span in (("开发", DEV), ("样本外", OOS)):
                m = pair(data, base, sign, span)
                print(
                    f"{name if label == '开发' else '':<26}{label:>4}{m['covered']:>7.1f}{m['dt']:>9.1f}"
                    f"{m.get('early', np.nan):>7.1f}{m.get('late', np.nan):>7.1f}{m['dprice']:>7.2f}"
                    f"{m['new']:>7.1f}{m['new_c5']:>8.1f}{m['lost_c5']:>8.1f}"
                )


def combo_report(args) -> None:
    """组合排名：每个方向只列该方向的变体（加基准与只看价格的对照），按开发期 ±3/5/10 平均延续率排序。

    「早晚」一栏是与基准第 1 波配对后的平均价差（提醒时已走幅度之差，负数为更早）与早报 / 晚报占比；
    提醒更早与延续更高通常此消彼长，两样都不差于基准的才算更好的配置。
    """
    names = (OUT / "variants.txt").read_text(encoding="utf-8").split("\n")
    base = load("base")
    control = load_control()
    for sign, title, prefix in ((1, "放量上涨", "U "), (-1, "放量下跌", "D ")):
        rows = []
        for name in names:
            if not (name in ("base", "price_only", "no_ratio", "no_quote") or name.startswith(prefix)):
                continue
            data = base if name == "base" else load(name)
            dev, oos = summary(data, DEV, sign), summary(data, OOS, sign)
            wins = [segment_wins(data, base, sign, level) for level in LEVELS]
            m_dev, m_oos = pair(data, base, sign, DEV), pair(data, base, sign, OOS)
            rows.append((name, dev, oos, wins, m_dev, m_oos))
        rows.sort(key=lambda row: -np.mean([row[1][f"c{level}"] for level in LEVELS]))
        print(f"\n===== {title}：组合（开发期 ±3/5/10 平均降序；段胜为 10 个半年段里 ±3/±5/±10 高于基准的段数）=====")
        if control is not None:
            null = [summary(control, span, sign) for span in (DEV, OOS)]
            print(
                "随机时刻对照（同合约池）：开发 "
                + " / ".join(f"{null[0][f'c{level}']:.1f}" for level in LEVELS)
                + "，样本外 "
                + " / ".join(f"{null[1][f'c{level}']:.1f}" for level in LEVELS)
            )
        print(
            f"{'变体':<26}{'条/天':>6}{'开发 ±3/5/10':>18}{'均':>6}{'样本外 ±3/5/10':>18}{'均':>6}{'去10天':>7}"
            f"{'段胜':>10}{'价差 开/外':>13}{'早/晚报% 开':>12}{'覆盖% 开':>9}"
        )
        for name, dev, oos, wins, m_dev, m_oos in rows:
            avg_dev = np.mean([dev[f"c{level}"] for level in LEVELS])
            avg_oos = np.mean([oos[f"c{level}"] for level in LEVELS])
            win_text = "" if name == "base" else "/".join(str(w) for w, _ in wins)
            gap = "" if name == "base" else f"{m_dev['dprice']:+.2f}/{m_oos['dprice']:+.2f}"
            timing = "" if name == "base" else f"{m_dev.get('early', np.nan):.0f}/{m_dev.get('late', np.nan):.0f}"
            cover = "" if name == "base" else f"{m_dev['covered']:.0f}"
            print(
                f"{name:<26}{dev['per_day']:>6.1f}{dev['c3']:>8.1f}{dev['c5']:>5.1f}{dev['c10']:>5.1f}{avg_dev:>6.1f}"
                f"{oos['c3']:>8.1f}{oos['c5']:>5.1f}{oos['c10']:>5.1f}{avg_oos:>6.1f}{dev['c5_trim']:>7.1f}"
                f"{win_text:>10}{gap:>13}{timing:>12}{cover:>9}"
            )


def load_control() -> dict | None:
    path = SETS["single"] / "control.npy"
    if not path.exists():
        return None
    rows = np.load(path)
    return {field: rows[:, k] for k, field in enumerate(FIELDS)}


def control_symbol(args):
    """随机对照：合约在池里、前后数据完整的分钟，每 STEP 分钟取一个，两个方向各算一次延续。"""
    symbol, multiplier, sym_index, pool_min, exit_ratio, refresh, step = args
    series = load_series(symbol, multiplier)
    if series is None:
        return np.empty((0, len(FIELDS)))
    from launch_trade_backtest import _pool_membership

    member = _pool_membership(series, PoolRule(pool_min, None, exit_ratio, refresh))
    # 起点按合约错开，免得所有合约都取同一批时刻（整点）。
    picks = np.arange(1440 + sym_index % step, len(series) - 1, step)
    picks = picks[member[picks] & series.covered[picks]]
    rows = []
    for i in picks:
        for sign in (1, -1):
            rows.append([series.t0 + (i + 1) * 60, sign, 0, 0, sym_index, *outcome(series, int(i), sign)])
    return np.array(rows, dtype=np.float64).reshape(-1, len(FIELDS))


def control(args) -> None:
    from price_alert.config import load_config

    config = load_config(REPO / "config" / "default.yaml")
    contracts = load_contracts()
    symbols = sorted(contracts)
    jobs = []
    for k, s in enumerate(symbols):
        try:
            mult = float(contracts[s].get("quanto_multiplier"))
        except (TypeError, ValueError):
            continue
        jobs.append(
            (
                s,
                mult,
                k,
                config.launch.min_volume_24h_quote,
                config.gate.universe_exit_volume_ratio,
                config.gate.universe_refresh_seconds,
                args.step,
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        parts = list(executor.map(control_symbol, jobs, chunksize=4))
    rows = np.concatenate(parts)
    np.save(OUT / "control.npy", rows[np.argsort(rows[:, 0], kind="stable")])
    data = load("control")
    for sign, title in ((1, "随机做多方向"), (-1, "随机做空方向")):
        for label, span in (("开发", DEV), ("样本外", OOS)):
            s = summary(data, span, sign)
            print(f"{title} {label}: n={s['n']:.0f} ±3 {s['c3']:.1f} ±5 {s['c5']:.1f} ±10 {s['c10']:.1f}")
        print("  半年段 ±5：", " ".join(f"{rate(data['cont_5'][select(data, *sp, sign)]):.1f}" for sp in SEGMENTS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=12)
    p_report = sub.add_parser("report")
    p_report.add_argument("--first-wave", action="store_true")
    p_match = sub.add_parser("match")
    p_combo = sub.add_parser("combo")
    p_control = sub.add_parser("control")
    p_control.add_argument("--workers", type=int, default=12)
    p_control.add_argument("--step", type=int, default=720, help="每隔多少分钟取一个随机对照时刻")
    for p in (p_run, p_report, p_match, p_combo):
        p.add_argument("--set", choices=tuple(SETS), default="single", help="单因素或组合（结果分目录存）")
    args = parser.parse_args()
    global OUT
    OUT = SETS[getattr(args, "set", "single")]
    {"run": run, "report": report, "match": match, "combo": combo_report, "control": control}[args.cmd](args)


if __name__ == "__main__":
    main()
