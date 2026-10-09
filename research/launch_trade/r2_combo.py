"""第二轮组合网格（按训练期排序）：把单因素里有效的条件组合起来。

结论是只按训练期挑出的前几名，检验期普遍不如基准（过拟合），所以之后改用 r2_robust.py 的三段评估。
"""

import collections
import itertools
import pickle

import numpy as np
from paths import RESULTS
from r2_lib import Engine, Rule, evaluate, fmt, load

GRID = dict(
    near_high=[np.nan, 0.02, 0.03, 0.05],
    max_extension=[np.nan, 0.15, 0.20],
    hold=[300, 420, 600],
    trail=[0.12, 0.15, 0.18],
    trail_atr=[np.nan, 5],
    stop=[0.06, 0.07, 0.08],
    entry=[(10, 0.02), (15, 0.02), (30, 0.02), (10, 0.015), (10, 0.025)],
    trail_tighten_at=[np.nan, 0.3],
    stop_support=[-1, 2],
)


def describe(rule: Rule) -> str:
    return (
        f"等{rule.delay}分+{rule.confirm:.1%} 贴高{rule.near_high} 不追{rule.max_extension} 持{rule.hold} "
        f"回撤{rule.trail} atr{rule.trail_atr} 损{rule.stop} 收紧{rule.trail_tighten_at} 支撑损{rule.stop_support}"
    )


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    rows = []
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo, strict=True))
        delay, confirm = kw.pop("entry")
        if kw["stop_support"] == 2:
            kw.update(stop_support_gap=0.02, stop_max=kw["stop"] + 0.03)
        rule = Rule(delay=delay, confirm=confirm, **kw)
        rows.append((rule, evaluate(engine.run(rule)[0], first, data)))
    with (RESULTS / "r2_combo.pkl").open("wb") as handle:
        pickle.dump(rows, handle)
    rows.sort(key=lambda x: -x[1]["train_sum"])
    print(f"{len(rows)} 组；训练期合计最高 25 组：")
    for rule, res in rows[:25]:
        print(describe(rule))
        print("    ", fmt(res))
    both = sum(1 for _, res in rows if res["train_sum"] > 350 and res["test_sum"] > 261)
    print(f"训练与检验都超过基准的：{both}/{len(rows)}")
    for key in [
        "near_high",
        "max_extension",
        "hold",
        "trail",
        "trail_atr",
        "stop",
        "delay",
        "confirm",
        "trail_tighten_at",
        "stop_support",
    ]:
        groups = collections.defaultdict(list)
        for rule, res in rows:
            groups[str(getattr(rule, key))].append((res["train_sum"], res["test_sum"]))
        marginal = "  ".join(
            f"{v}: 训{np.mean([a for a, _ in g]):.0f}/检{np.mean([b for _, b in g]):.0f}" for v, g in groups.items()
        )
        print(f"{key} {marginal}")


if __name__ == "__main__":
    main()
