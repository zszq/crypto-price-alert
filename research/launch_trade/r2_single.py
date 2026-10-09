"""第二轮单因素扫描：在基准规则（等 10 分钟 +2% 入场、止损 7%、回撤 15%、持仓 5 小时）上每次只改一个条件。"""

import pickle
from dataclasses import replace

import numpy as np
from paths import RESULTS
from r2_lib import Engine, Rule, evaluate, fmt, load

SWEEPS = {
    # 入场确认
    "wait_volume_ratio": [1, 2, 3, 5, 8, 12],
    "wait_vs_window": [0.1, 0.2, 0.3, 0.5, 0.8],
    "above_support": [-0.02, 0.0, 0.02, 0.05],
    "near_high": [0.01, 0.02, 0.03, 0.05, 0.08],
    "btc_min": [-0.005, -0.002, 0.0, 0.002],
    "max_extension": [0.08, 0.10, 0.15, 0.20, 0.30],
    "confirm": [np.nan, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07],
    "delay": [3, 5, 7, 10, 15, 20, 30],
    # 出场
    "stop": [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
    "trail": [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    "hold": [60, 120, 180, 240, 300, 360, 480, 600],
    "trail_atr": [3, 5, 8, 12, 16],
    "trail_after": [0.03, 0.05, 0.10, 0.15],
    "trail_tighten_at": [0.2, 0.3, 0.5],
    "breakeven_at": [0.05, 0.08, 0.10, 0.15],
    "time_stop": [15, 30, 60, 90],
    "vol_exit_ratio": [0.1, 0.2, 0.3, 0.5],
    "take": [0.10, 0.15, 0.20, 0.30, 0.50],
}


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    base = Rule()
    rows = []

    def show(label: str, rule: Rule, key: str, value: object) -> None:
        res = evaluate(engine.run(rule)[0], first, data)
        rows.append((key, value, res))
        print(f"  {label}", fmt(res))

    print("基准", fmt(evaluate(engine.run(base)[0], first, data)))
    for name, values in SWEEPS.items():
        print("==", name)
        for value in values:
            show(f"{value!s:>6}", replace(base, **{name: value}), name, value)
    print("== 压力位：1 为 24 小时高点，2 为 7 天，3 为 30 天")
    for kind in (1, 2, 3):
        for blue_sky in (1, -1):
            label = f"压力{kind} {'已突破' if blue_sky == 1 else '未突破'}"
            show(label, replace(base, resistance_kind=kind, blue_sky=blue_sky), f"resist{kind}_blue_sky", blue_sky)
        for room in (0.05, 0.10, 0.20):
            show(
                f"压力{kind} 空间≥{room}",
                replace(base, resistance_kind=kind, room_min=room),
                f"resist{kind}_room",
                room,
            )
        for fraction in (1.0, 0.5):
            rule = replace(base, resistance_kind=kind, take_resistance=1, take_fraction=fraction)
            show(f"压力{kind} 止盈{fraction}", rule, f"resist{kind}_take", fraction)
    print("== 支撑止损：0 基准期高点，1 提醒价，2 等待期最低价")
    for kind in (0, 1, 2):
        for gap in (0.0, 0.01, 0.02):
            for stop_max in (0.07, 0.10):
                rule = replace(base, stop_support=kind, stop_support_gap=gap, stop_max=stop_max)
                show(f"支撑{kind} 余量{gap} 最多{stop_max}", rule, f"stop_support{kind}", (gap, stop_max))
    print("== 时间止损门槛")
    for minutes in (30, 60, 90):
        for gain in (0.0, 0.02, 0.05):
            show(f"{minutes}分钟未涨{gain}", replace(base, time_stop=minutes, time_gain=gain), "time", (minutes, gain))
    with (RESULTS / "r2_single.pkl").open("wb") as handle:
        pickle.dump(rows, handle)


if __name__ == "__main__":
    main()
