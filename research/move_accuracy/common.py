"""秒级异动准确性研究的公共路径、月份与合约信息。

行情与中间结果只放本机 data/backtest/move_acc/（不进版本库），代码在这里。
"""

from __future__ import annotations

import calendar
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "backtest" / "move_acc"
RAW = ROOT / "data" / "backtest" / "raw"
CONTRACTS = ROOT / "data" / "backtest" / "contracts.json"
TRADES = DATA / "trades"

MONTHS = ["202604", "202605", "202606", "202607", "202608", "202609"]
# 开发期只用前四个月选参，后两个月只做验证（PLAN.md 写定，不改）
DEV_MONTHS = MONTHS[:4]
HOLD_MONTHS = MONTHS[4:]
# 第二轮的样本外（PLAN.md 第二轮）：第一轮没用过的六个月。追加在 MONTHS 之后编号，第一轮的月份下标不变
FRESH_MONTHS = ["202510", "202511", "202512", "202601", "202602", "202603"]
# 第三轮的样本外（PLAN.md 第三轮）：前两轮都没用过的六个月，同样追加在后面编号
ROUND3_MONTHS = ["202504", "202505", "202506", "202507", "202508", "202509"]
# 第三轮补充检验（PLAN.md 第三轮补充检验）：2025-09 往前一年里还没用过的六个月
ROUND4_MONTHS = ["202410", "202411", "202412", "202501", "202502", "202503"]
ALL_MONTHS = MONTHS + FRESH_MONTHS + ROUND3_MONTHS + ROUND4_MONTHS

# 下载的合约池门槛低于现行 800 万，研究里才能评估降低门槛；提高门槛只需取子集
DOWNLOAD_POOL_QUOTE = 5e6
EXIT_RATIO = 0.8
REFRESH_SECONDS = 600


def month_start(month: str) -> int:
    return int(datetime(int(month[:4]), int(month[4:]), 1, tzinfo=UTC).timestamp())


def month_end(month: str) -> int:
    year, mon = int(month[:4]), int(month[4:])
    return month_start(month) + calendar.monthrange(year, mon)[1] * 86400


def previous_month(month: str) -> str:
    year, mon = int(month[:4]), int(month[4:])
    return f"{year - 1}12" if mon == 1 else f"{year}{mon - 1:02d}"


def contracts() -> dict[str, dict]:
    return {item["name"]: item for item in json.load(open(CONTRACTS, encoding="utf-8"))}


def multiplier(info: dict | None) -> float:
    try:
        value = float((info or {}).get("quanto_multiplier") or 0)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def load_klines(symbol: str, month: str) -> np.ndarray:
    """下载站 1 分钟 K 线：列为 时间、张数、收、高、低、开；文件不存在返回空数组。"""
    path = RAW / month / f"{symbol}-{month}.csv.gz"
    if not path.exists():
        return np.empty((0, 6))
    with gzip.open(path, "rt") as f:
        text = f.read().strip()
    if not text:
        return np.empty((0, 6))
    return np.array([line.split(",") for line in text.splitlines()], dtype=float)
