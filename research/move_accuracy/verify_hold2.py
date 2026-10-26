"""第二轮样本外全量对拍：反弹回落检测器（src/price_alert/reversal，代码默认参数）与引擎逐条比较，不限合约池。

只取 2025-10 ~ 2026-03 里曾进 800 万合约池的合约-月份。用法：verify_hold2.py [进程数] [--resume]
逐条结果同时写入 data/backtest/move_acc/verify_hold2.log；全量要跑几个小时，中断后加 --resume 从日志接着跑。
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
from common import DATA, FRESH_MONTHS
from pool import membership
from verify_reversal import detector_alerts, engine_alerts


def job(args: tuple[str, str]) -> tuple[str, str, int, int, int]:
    symbol, month = args
    real, eng = detector_alerts(symbol, month), engine_alerts(symbol, month)
    return symbol, month, len(real), len(eng), len(real & eng)


if __name__ == "__main__":
    jobs = []
    for month in FRESH_MONTHS:
        vol = np.load(DATA / f"volume_{month}.npz")
        ever = membership(vol["vols"], 8e6).any(axis=1)
        jobs += [(str(s), month) for s, e in zip(vol["symbols"], ever, strict=True) if e]
    # --resume：已完成的合约-月份从上次的日志里读回（行格式见下方 print），只跑剩下的
    totals = np.zeros(3, np.int64)
    done = set()
    log = DATA / "verify_hold2.log"
    if "--resume" in sys.argv and log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[3] == "检测器" and (parts[1], parts[2]) not in done:
                done.add((parts[1], parts[2]))
                totals += (int(parts[4]), int(parts[6]), int(parts[8]))
    todo = [j for j in jobs if j not in done]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    # 续跑时追加，否则重写：日志里只能有本轮的结果，--resume 才不会把旧参数的结果读回来
    with Pool(workers) as pool, log.open("a" if done else "w", encoding="utf-8") as out:
        for n, (sym, month, a, b, both) in enumerate(pool.imap_unordered(job, todo), len(done) + 1):
            totals += (a, b, both)
            flag = "" if a == b == both else "  ← 不一致"
            line = f"{n}/{len(jobs)} {sym} {month} 检测器 {a} 引擎 {b} 一致 {both}{flag}"
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
    a, b, both = totals
    print(f"合计：检测器 {a} 条，引擎 {b} 条，一致 {both}，一致率 {both / max(a + b - both, 1):.4%}")
