"""K 线形态趋势提醒的配置模型（YAML 的 trend 段）。

各周期结构相同、默认值不同，沿用短/长窗口的做法：公共字段放在基类，每个周期一个子类只改默认值，
这样 YAML 里只写某个周期的部分字段时，其余字段取的仍是该周期自己的默认值。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.trend.pattern import PeriodName, TrendRule


class TrendPeriodConfig(BaseModel):
    """一个周期的形态门槛：最近 candles 根已收盘 K 线满足全部条件才算趋势。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    candles: int = Field(ge=3, le=60)
    min_change_percent: float = Field(gt=0, le=100)
    max_counter_candles: int = Field(ge=0, le=59)
    max_rebound_ratio: float = Field(default=0.3, ge=0, le=1)
    min_body_ratio: float = Field(default=0.5, ge=0, le=1)
    # 最大一根顺势 K 线实体 ÷ 累计位移的上限：排除一根急拉急砸加几根小 K 线凑出来的“趋势”。
    max_single_candle_ratio: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def validate_counter_candles(self) -> TrendPeriodConfig:
        if self.max_counter_candles >= self.candles:
            raise ValueError("max_counter_candles 必须小于 candles")
        return self

    def to_rule(self, period: PeriodName) -> TrendRule:
        return TrendRule(
            period=period,
            candles=self.candles,
            min_change_percent=self.min_change_percent,
            max_counter_candles=self.max_counter_candles,
            max_rebound_ratio=self.max_rebound_ratio,
            min_body_ratio=self.min_body_ratio,
            max_single_candle_ratio=self.max_single_candle_ratio,
        )


class Trend3mConfig(TrendPeriodConfig):
    candles: int = Field(default=5, ge=3, le=60)
    min_change_percent: float = Field(default=5.0, gt=0, le=100)
    max_counter_candles: int = Field(default=1, ge=0, le=59)


class Trend5mConfig(TrendPeriodConfig):
    candles: int = Field(default=4, ge=3, le=60)
    min_change_percent: float = Field(default=6.0, gt=0, le=100)
    max_counter_candles: int = Field(default=1, ge=0, le=59)


class Trend15mConfig(TrendPeriodConfig):
    candles: int = Field(default=3, ge=3, le=60)
    min_change_percent: float = Field(default=8.0, gt=0, le=100)
    # 只有 3 根时夹一根反向就只剩两根顺势，算不上连续走势。
    max_counter_candles: int = Field(default=0, ge=0, le=59)


class TrendPeriodsConfig(BaseModel):
    # YAML 键直接写周期名（3m、5m、15m），Python 里以 m3、m5、m15 访问。
    # 1 分钟周期噪声太大，不提供；1 分钟 K 线只作为合成各周期的单位。
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    m3: Trend3mConfig = Field(default_factory=Trend3mConfig, alias="3m")
    m5: Trend5mConfig = Field(default_factory=Trend5mConfig, alias="5m")
    m15: Trend15mConfig = Field(default_factory=Trend15mConfig, alias="15m")

    def items(self) -> list[tuple[PeriodName, TrendPeriodConfig]]:
        return [("3m", self.m3), ("5m", self.m5), ("15m", self.m15)]


class TrendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 形态持续成立时，每根周期 K 线收盘都再报一条「延续」提醒；关闭后同一段走势只在形成时报一次。
    alert_continuing: bool = True
    periods: TrendPeriodsConfig = Field(default_factory=TrendPeriodsConfig)

    @model_validator(mode="after")
    def validate_periods(self) -> TrendConfig:
        if self.enabled and not self.rules():
            raise ValueError("启用趋势提醒时至少要启用一个周期")
        return self

    def rules(self) -> list[TrendRule]:
        return [period.to_rule(name) for name, period in self.periods.items() if period.enabled]
