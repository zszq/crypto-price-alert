"""短时异动的配置：合约池门槛、共用的 K 线/ATR 参数、短/长观察窗口与连续提醒。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.common.models import INTERVAL_SECONDS


class WindowConfig(BaseModel):
    """观察窗口的公共字段与校验；短、长窗口结构相同，只在默认值、取值范围和冷却规则上不同。"""

    model_config = ConfigDict(extra="forbid")

    lookback_seconds: int
    trigger_atr_multiple: float
    min_change_percent: float
    confirmation_seconds: int
    min_window_trades: int
    cooldown_seconds: int

    @model_validator(mode="after")
    def validate_confirmation(self) -> WindowConfig:
        if self.confirmation_seconds > self.lookback_seconds:
            # 确认期长于观察窗口时，基准价会追上已经完成的跳变，持续性的异动反而永远无法确认。
            raise ValueError("confirmation_seconds 不能大于 lookback_seconds")
        return self


class ShortWindowConfig(WindowConfig):
    """主观察窗口，捕捉秒级急拉急砸，始终启用。"""

    lookback_seconds: int = Field(default=30, ge=5, le=3600)
    trigger_atr_multiple: float = Field(default=1.5, gt=0, le=20)
    min_change_percent: float = Field(default=1.0, gt=0, le=100)
    confirmation_seconds: int = Field(default=3, ge=1, le=10)
    min_window_trades: int = Field(default=10, ge=1)
    cooldown_seconds: int = Field(default=30, ge=0, le=86400)


class LongWindowConfig(WindowConfig):
    """较长的第二观察窗口，捕捉短窗口看不出的慢速单边行情；可关闭。"""

    enabled: bool = True
    lookback_seconds: int = Field(default=180, ge=60, le=3600)
    trigger_atr_multiple: float = Field(default=2.0, gt=0, le=20)
    min_change_percent: float = Field(default=2.0, gt=0, le=100)
    confirmation_seconds: int = Field(default=3, ge=1, le=10)
    min_window_trades: int = Field(default=30, ge=1)
    # 未填写时等于 lookback_seconds：冷却一旦短于观察窗口，同一段行情会在窗口滑过期间被反复报出。
    cooldown_seconds: int = Field(default=180, ge=60, le=86400)

    @model_validator(mode="after")
    def validate_cooldown(self) -> LongWindowConfig:
        if "cooldown_seconds" not in self.model_fields_set:
            self.cooldown_seconds = self.lookback_seconds
        if self.cooldown_seconds < self.lookback_seconds:
            raise ValueError("cooldown_seconds 不能小于 lookback_seconds，否则同一段行情会被重复提醒")
        return self


class StreakConfig(BaseModel):
    """短窗口同方向提醒连成串时追加的连续提醒。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 实际日志中同向连发 3 条的几乎都是累计 5% 以上的单边行情，2 条则大量是一次插针的余波。
    min_alerts: int = Field(default=3, ge=2, le=10)
    # 相邻同向提醒至少隔一个短窗口冷却，这里只配冷却之外的容差（重新确认、短暂停顿）；按冷却加容差，
    # 调短窗口冷却时连续提醒不会失效。实际日志里冷却 30 秒时，持续行情的间隔在 30~60 秒之间。
    gap_tolerance_seconds: int = Field(default=30, ge=1, le=3600)


class SpikeConfig(BaseModel):
    """短时异动：合约池门槛、共用的 K 线/ATR 参数与各观察窗口。"""

    model_config = ConfigDict(extra="forbid")

    # 只跑放量启动（例如给交易单独开一个实例）时关掉，不必为用不到的合约订阅成交、预热 K 线。
    enabled: bool = True
    # 合约池门槛与其他检测器一样放在本段：它只决定本检测器监控哪些合约，与交易所连接无关。
    min_volume_24h_quote: float = Field(default=10_000_000, gt=0)
    candle_interval: Literal["1m", "5m", "15m"] = "1m"
    atr_period: int = Field(default=14, ge=2, le=200)
    warmup_candles: int = Field(default=50, ge=15, le=500)
    max_atr_age_seconds: int = Field(default=180, ge=30, le=3600)
    # 各窗口共用 K 线、ATR 与秒级数据，门槛、确认与冷却按窗口独立配置。
    short_window: ShortWindowConfig = Field(default_factory=ShortWindowConfig)
    long_window: LongWindowConfig = Field(default_factory=LongWindowConfig)
    streak: StreakConfig = Field(default_factory=StreakConfig)

    @model_validator(mode="after")
    def validate_warmup_and_freshness(self) -> SpikeConfig:
        if self.warmup_candles <= self.atr_period:
            raise ValueError("warmup_candles 必须大于 atr_period，以便排除尚未收盘的 K 线")
        interval_seconds = INTERVAL_SECONDS[self.candle_interval]
        if "max_atr_age_seconds" not in self.model_fields_set:
            # 未显式配置时跟随 K 线周期，只改 candle_interval 就不会因固定默认值过小而校验失败。
            self.max_atr_age_seconds = interval_seconds * 3
        if self.max_atr_age_seconds < interval_seconds * 2:
            raise ValueError("max_atr_age_seconds 不能小于两个 K 线周期")
        if self.long_window.enabled and self.long_window.lookback_seconds <= self.short_window.lookback_seconds:
            # 长窗口不比短窗口长时，它能看到的行情短窗口都已覆盖，只会多出一套重复门槛。
            raise ValueError("long_window.lookback_seconds 必须大于 short_window.lookback_seconds")
        return self
