"""第三轮最终排名：三年（2023-10-10 起）里比 J（第二轮推荐方案，三年 +1397）更赚钱的方案。

数据先用 r3_export.py 导出（--post 1200，见 README），提醒规则不同的方案用不同的导出目录：
- v_base：当前 config/default.yaml 的 launch 段；
- v_nobreak：同上但 require_breakout=false（不要求收盘突破基准期最高价）；
- v_nb_pool0_noover：再加合约池成交额下限 0、max_change_percent=null（不设插针上限）。
J4、J5、P5 的过热过滤还要先运行 r3_market.py（全市场逐小时数据）。
除单点合计外给出 6 段、去掉最好 10 笔、t 值、最大回撤、最多同时持仓、资金费率敏感性、
换等待分钟（确认涨幅不变）的平均与胜过同等待 J 的次数，以及入场 × 回撤的邻域平均（见 r3_lib.hood）。
"""

from __future__ import annotations

import sys
from dataclasses import replace

from r2_final import display_width, pad
from r2_lib import BEST
from r3_lib import BAD_HOURS, CAPS, FUNDING_PER_8H, HOT_BREADTH, Book, Scheme, X, report

X0 = replace(X, confirm=0.0)
# 在 v_nb_pool0_noover 上重扫出场得到的 Y：止损余量 8%、回撤 ≥ 5 倍振幅。与 X 只差约 15%，且是同一份数据上调的，偏乐观。
Y0 = replace(X0, stop_support_gap=0.08, trail_atr=5)
SCHEMES = [
    Scheme("J：第二轮推荐（参照）", "v_base", BEST),
    Scheme("J1：J 所有波次", "v_base", BEST, all_waves=True),
    Scheme("J2：J 避开7~10点", "v_base", BEST, skip_hours=BAD_HOURS),
    Scheme("J3：J 所有波次 避开7~10点", "v_base", BEST, all_waves=True, skip_hours=BAD_HOURS),
    Scheme("X1：出场X", "v_base", X),
    Scheme("X2：出场X 避开7~10点", "v_base", X, skip_hours=BAD_HOURS),
    Scheme("X3：出场X 所有波次 避开7~10点", "v_base", X, all_waves=True, skip_hours=BAD_HOURS),
    Scheme("X4：X3 确认降到0%", "v_base", X0, all_waves=True, skip_hours=BAD_HOURS),
    Scheme("N1：不要求突破 + J", "v_nobreak", BEST),
    Scheme("N2：不要求突破 + 出场X", "v_nobreak", X),
    Scheme("N3：N2 避开7~10点", "v_nobreak", X, skip_hours=BAD_HOURS),
    Scheme("N4：N2 确认降到0%", "v_nobreak", X0),
    Scheme("N5：N2 所有波次 避开7~10点", "v_nobreak", X, all_waves=True, skip_hours=BAD_HOURS),
    Scheme("N6：N5 确认降到0%", "v_nobreak", X0, all_waves=True, skip_hours=BAD_HOURS),
    Scheme("P1：不要求突破+不设池下限+不设插针上限 + 出场X", "v_nb_pool0_noover", X),
    Scheme("P2：P1 确认降到0%", "v_nb_pool0_noover", X0),
    Scheme("P3：P2 所有波次", "v_nb_pool0_noover", X0, all_waves=True),
    Scheme("P4：P3 出场换成Y（止损余量8%、回撤≥5倍振幅）", "v_nb_pool0_noover", Y0, all_waves=True),
    # 亏损期分析后加的过热过滤（2026-10-10），见 FINDINGS.md「亏损期分析与过热过滤」。
    Scheme("J4：J 过热不买", "v_base", BEST, hot_breadth=HOT_BREADTH),
    Scheme("J5：J3 过热不买", "v_base", BEST, all_waves=True, skip_hours=BAD_HOURS, hot_breadth=HOT_BREADTH),
    Scheme(
        "P5：P3 过热时不买J类",
        "v_nb_pool0_noover",
        X0,
        all_waves=True,
        hot_breadth=HOT_BREADTH,
        hot_core_only=True,
    ),
]
LEGEND = (
    "出场X = 止损在等待期最低价下方5%（离入场最多20%）、回撤取15%与入场前60分钟平均振幅×3的较大者、最长持18h；"
    "J = 等10分钟+2%入场、止损在等待期最低价下方2%（最多9%）、回撤≥5倍振幅、持10h；"
    "避开7~10点 = 不买北京时间7:00~9:59的提醒；不要求突破 = 提醒规则 require_breakout=false；"
    f"过热 = 全市场站上30天均线的合约占比 > {HOT_BREADTH:.0%}；"
    "J类 = 已突破基准期高点、24h成交额>100万、入场价比提醒价高≥2%"
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    books: dict[str, Book] = {}
    rows = []
    reference_wait = None
    for scheme in SCHEMES:
        book = books.setdefault(scheme.dataset, Book(scheme.dataset))
        row = report(book, scheme, reference_wait)
        if reference_wait is None:
            reference_wait = row["waits"]
            row = report(book, scheme, reference_wait)
        rows.append(row)
        print(f"  已算 {scheme.name}", file=sys.stderr, flush=True)
    labels = [label for label, _ in books["v_base"].data.period.folds]
    print(LEGEND)
    print(f"资金费率：每 8 小时持仓按 {FUNDING_PER_8H:.2%} 扣除后的合计见「扣资金费」列")
    width = max(display_width(r["name"]) for r in rows) + 1
    print(
        pad("方案", width)
        + f"{'笔数':>5s} {'合计':>6s} {'每笔':>6s} "
        + " ".join(f"{label:>9s}" for label in labels)
        + f" {'去10':>6s} {'t':>5s} {'胜率':>4s} {'回撤':>5s} {'亏损月':>4s} {'同时持仓':>4s} {'扣资金费':>6s}"
        + "".join(f" {f'限{c}笔':>6s}" for c in CAPS)
        + f" {'换等待平均':>6s} {'最差':>6s} {'胜J':>4s} {'邻域平均':>6s} {'10分位':>6s}"
    )
    for r in rows:
        print(
            pad(r["name"], width)
            + f"{r['n']:5d} {r['total']:+6.0f} {r['mean']:+5.2f}% "
            + " ".join(f"{v:+9.0f}" for v in r["folds"])
            + f" {r['minus10']:+6.0f} {r['t']:5.2f} {r['win']:4.0%} {r['dd']:5.0f}"
            + f" {r['neg_months']:4d} {r['concurrent']:6d}"
            + f" {r['funded']:+8.0f}"
            + "".join(f" {r['caps'][c]:+7.0f}" for c in CAPS)
            + f" {r['wait_mean']:+8.0f} {r['wait_min']:+6.0f} {r['wait_wins']:2d}/7"
            + f" {r['hood']:+8.0f} {r['hood_p10']:+6.0f}"
        )
    print("限N笔 = 同时最多持 N 笔（本金 N×100 USDT），满仓时跳过新提醒。")
    print("换等待：等 3/5/7/10/12/15/20 分钟、确认涨幅不变；胜J = 其中胜过同等待 J 的个数。")
    print("各方案换等待的合计：")
    for r in rows:
        print(pad(r["name"], width) + " ".join(f"{v:+6.0f}" for v in r["waits"]))


if __name__ == "__main__":
    main()
