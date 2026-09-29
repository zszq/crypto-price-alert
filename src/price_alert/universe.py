"""基于 Gate.io 24 小时计价成交额构建动态合约池。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from price_alert.models import ContractTicker


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    # inf 会让 "> 门槛" 恒为真而把异常合约放进合约池，统一按无效值处理。
    return number if math.isfinite(number) else 0.0


def price_decimals(order_price_round: Any) -> int | None:
    """把 Gate 的报价步长（如 "0.01"）换算成展示用的小数位数，无法识别时返回 None。"""
    try:
        step = Decimal(str(order_price_round))
    except (InvalidOperation, ValueError):
        return None
    if not step.is_finite() or step <= 0:
        return None
    # normalize 去掉尾随零（"0.010" 与 "0.01" 同为两位）；步长为整数时指数非负，按 0 位小数显示。
    return max(0, -step.normalize().as_tuple().exponent)


def select_liquid_contracts(
    tickers: Iterable[Mapping[str, Any]],
    contracts: Iterable[Mapping[str, Any]],
    min_volume_24h_quote: float,
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
    # 已在监控中的合约用更低的退出门槛，避免成交额在门槛附近波动时被反复移除、重新预热。
    exit_volume = min_volume_24h_quote * exit_volume_ratio
    selected: list[ContractTicker] = []
    for ticker in tickers:
        symbol = str(ticker.get("contract", "")).upper()
        last_price = _number(ticker.get("last"))
        # Gate 已弃用 volume_24h_usd；旧响应只在缺少新字段时作为兼容回退。
        volume = _number(ticker.get("volume_24h_quote", ticker.get("volume_24h_usd")))
        threshold = exit_volume if symbol in retained else min_volume_24h_quote
        contract = crypto_contracts.get(symbol)
        if contract is not None and symbol.endswith("_USDT") and last_price > 0 and volume > threshold:
            selected.append(
                ContractTicker(symbol, last_price, volume, price_decimals(contract.get("order_price_round")))
            )
    return sorted(selected, key=lambda item: item.volume_24h_quote, reverse=True)
