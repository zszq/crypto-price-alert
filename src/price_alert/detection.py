"""服务主循环对检测器的约定。

实时监控可以同时运行多个检测器（ATR 秒级异动、K 线形态趋势……），它们共用同一路成交与同一个合约池。
服务只依赖这里的接口：新增或移除一种检测器只改 assembly.build_detectors，不必改动服务编排。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from price_alert.models import Candle, PriceAlert, PriceTick
from price_alert.trend.alerts import TrendAlert

Alert = PriceAlert | TrendAlert


class Detector(Protocol):
    @property
    def symbols(self) -> list[str]: ...

    @property
    def stale_symbols(self) -> list[str]: ...

    def add_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        volume_24h_quote: float,
        live_candle: Candle | None = None,
        price_decimals: int | None = None,
    ) -> None: ...

    def remove_symbols(self, symbols: set[str]) -> None: ...

    def mark_stream_gap(self) -> None: ...

    def mark_symbol_gap(self, symbol: str) -> None: ...

    def resync_symbol(
        self,
        symbol: str,
        candles: list[Candle],
        live_candle: Candle | None,
        fetched_at: datetime,
    ) -> bool: ...

    def add_tick(self, tick: PriceTick) -> Sequence[Alert]: ...


@dataclass(frozen=True, slots=True)
class MonitoredDetector:
    """检测器及其预热所需的 K 线：周期相同的检测器共用一次 REST 请求，按各自根数截取。"""

    name: str
    detector: Detector
    candle_interval: str
    # 请求的 K 线根数，含一根未收盘的当前 K 线。
    warmup_candles: int
