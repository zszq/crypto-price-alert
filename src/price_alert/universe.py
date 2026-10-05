"""基于 Gate.io 24 小时计价成交额构建动态合约池。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from price_alert.gate import price_decimals, quanto_multiplier
from price_alert.models import ContractTicker


@dataclass(frozen=True, slots=True)
class VolumeRange:
    """一个检测器的合约池准入范围：24 小时计价成交额严格大于下限，且不超过上限（None 表示不设上限）。"""

    minimum: float
    maximum: float | None = None

    def describe(self) -> str:
        # 下限为 0 即不设门槛，写成「0M」反而像漏了单位。
        low = f"{self.minimum / 1_000_000:g}M" if self.minimum else "0"
        return f"> {low}" if self.maximum is None else f"{low} ~ {self.maximum / 1_000_000:g}M"


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    # inf 会让 "> 门槛" 恒为真而把异常合约放进合约池，统一按无效值处理。
    return number if math.isfinite(number) else 0.0


def select_liquid_contracts(
    tickers: Iterable[Mapping[str, Any]],
    contracts: Iterable[Mapping[str, Any]],
    min_volume_24h_quote: float,
    max_volume_24h_quote: float | None = None,
    retained_symbols: Iterable[str] = (),
    exit_volume_ratio: float = 1.0,
) -> list[ContractTicker]:
    # 保留合约元数据而不只是名字：报价精度要随筛选结果一起带给提醒展示。
    crypto_contracts = {
        str(contract.get("name", "")).upper(): contract
        for contract in contracts
        # Gate 会给股票、指数、贵金属等传统资产填写分类；只接受未分类的币本位标的，
        # 严格要求字段存在且为空，避免接口结构变化时误收非币类资产。
        if contract.get("contract_type") == ""
        and str(contract.get("status", "")).lower() == "trading"
        # 下架流程中的合约仍可能显示 trading，但流动性会迅速枯竭，继续监控只会制造噪声。
        and not contract.get("in_delisting")
    }
    retained = {symbol.upper() for symbol in retained_symbols}
    # 已在监控中的合约两侧都放宽退出线，避免成交额在下限或上限附近波动时被反复移除、重新预热。
    exit_low = min_volume_24h_quote * exit_volume_ratio
    exit_high = None if max_volume_24h_quote is None else max_volume_24h_quote / exit_volume_ratio
    selected: list[ContractTicker] = []
    for ticker in tickers:
        symbol = str(ticker.get("contract", "")).upper()
        last_price = _number(ticker.get("last"))
        # Gate 已弃用 volume_24h_usd；旧响应只在缺少新字段时作为兼容回退。
        volume = _number(ticker.get("volume_24h_quote", ticker.get("volume_24h_usd")))
        low, high = (exit_low, exit_high) if symbol in retained else (min_volume_24h_quote, max_volume_24h_quote)
        in_range = volume > low and (high is None or volume <= high)
        contract = crypto_contracts.get(symbol)
        if contract is not None and symbol.endswith("_USDT") and last_price > 0 and in_range:
            selected.append(
                ContractTicker(
                    symbol,
                    last_price,
                    volume,
                    price_decimals(contract.get("order_price_round")),
                    quanto_multiplier(contract.get("quanto_multiplier")),
                )
            )
    return sorted(selected, key=lambda item: item.volume_24h_quote, reverse=True)
