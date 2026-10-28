"""第四轮（PLAN.md 第四轮）：为瞬时异动的窗口补算逐秒行，带 --extras 的成交暴增、前置 5 分钟、离 6 小时极值。

默认一组写 rows4/：短窗口 L30（补齐上限 30）与现行双窗口里的长窗口 L180F30（补齐上限跟随 30 秒短窗口）；
--long10 一组写 rows4_L180F10/：短窗口改为 10 秒时的长窗口（补齐上限跟随 10 秒）。10 秒短窗口沿用 rows_L10F10/。
第一轮的 rows/ 只有 2026-04~09 且没有这些列，所以另放目录，不改动旧表。
用法：engine4.py 进程数 [--long10] [月份 ...]（默认全部选参月份与样本外）
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
from common import ALL_MONTHS, DATA
from engine import all_jobs, process

ROWS4 = DATA / "rows4"
ROWS4_LONG10 = DATA / "rows4_L180F10"
SETS = {"default": (ROWS4, {"L30": (30, 30), "L180F30": (180, 30)}), "long10": (ROWS4_LONG10, {"L180F10": (180, 10)})}


def run(args: tuple[tuple[str, str], str]) -> tuple[str, str, int]:
    job, name = args
    folder, variants = SETS[name]
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{job[0]}-{job[1]}.npz"
    if out_path.exists():
        return job[0], job[1], -1
    out = process(job, variants=variants, extras=True)
    # 先写临时文件再改名，中断时不留半个文件
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    return job[0], job[1], len(out.get(f"{next(iter(variants))}_sec", []))


if __name__ == "__main__":
    args = sys.argv[2:]
    name = "long10" if "--long10" in args else "default"
    months = [a for a in args if a != "--long10"] or ALL_MONTHS
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(workers) as pool:
        for n, (sym, month, rows) in enumerate(pool.imap_unordered(run, [(j, name) for j in all_jobs(months)]), 1):
            print(n, "/", len(all_jobs(months)) if n == 1 else "", sym, month, rows, flush=True)
