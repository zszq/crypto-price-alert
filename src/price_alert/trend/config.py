"""K 线形态趋势提醒的配置模型（YAML 的 trend 段）。

各周期结构相同、默认值不同，沿用短/长窗口的做法：公共字段放在基类，每个周期一个子类只改默认值，
这样 YAML 里只写某个周期的部分字段时，其余字段取的仍是该周期自己的默认值。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.trend.detector import MAX_WARMUP_CANDLES, EpisodeSettings
from price_alert.trend.pattern import PeriodName, TrendRule


class TrendPeriodConfig(BaseModel):
    """一个周期的形态门槛：最近 candles 根已收盘 K 线整体满足全部条件才算趋势。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    candles: int = Field(ge=3, le=60)
    min_change_percent: float = Field(gt=0, le=100)
    max_rebound_ratio: float = Field(default=0.3, ge=0, le=1)
    min_trend_candle_ratio: float = Field(default=0.7, gt=0, le=1)
    min_step_ratio: float = Field(default=0.7, gt=0, le=1)
    min_body_ratio: float = Field(default=0.5, ge=0, le=1)
    # 可选：窗口平均成交量 ÷ 基准成交量中位数的下限，null 关闭。
    min_volume_ratio: float | None = Field(default=1.5, gt=0, le=100)
    baseline_candles: int = Field(ge=5, le=200)

    def to_rule(self, period: PeriodName) -> TrendRule:
        return TrendRule(
            period=period,
            candles=self.candles,
            min_change_percent=self.min_change_percent,
            max_rebound_ratio=self.max_rebound_ratio,
            min_trend_candle_ratio=self.min_trend_candle_ratio,
            min_step_ratio=self.min_step_ratio,
            min_body_ratio=self.min_body_ratio,
            min_volume_ratio=self.min_volume_ratio,
            baseline_candles=self.baseline_candles,
        )


class Trend1mConfig(TrendPeriodConfig):
    candles: int = Field(default=8, ge=3, le=60)
    min_change_percent: float = Field(default=4.0, gt=0, le=100)
    baseline_candles: int = Field(default=30, ge=5, le=200)


class Trend3mConfig(TrendPeriodConfig):
    candles: int = Field(default=6, ge=3, le=60)
    min_change_percent: float = Field(default=5.0, gt=0, le=100)
    baseline_candles: int = Field(default=20, ge=5, le=200)


class Trend5mConfig(TrendPeriodConfig):
    candles: int = Field(default=6, ge=3, le=60)
    min_change_percent: float = Field(default=6.0, gt=0, le=100)
    baseline_candles: int = Field(default=20, ge=5, le=200)


class Trend15mConfig(TrendPeriodConfig):
    candles: int = Field(default=6, ge=3, le=60)
    min_change_percent: float = Field(default=8.0, gt=0, le=100)
    baseline_candles: int = Field(default=16, ge=5, le=200)


class TrendPeriodsConfig(BaseModel):
    # YAML 键直接写周期名（1m、3m……），Python 里以 m1、m3 访问。
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    m1: Trend1mConfig = Field(default_factory=Trend1mConfig, alias="1m")
    m3: Trend3mConfig = Field(default_factory=Trend3mConfig, alias="3m")
    m5: Trend5mConfig = Field(default_factory=Trend5mConfig, alias="5m")
    m15: Trend15mConfig = Field(default_factory=Trend15mConfig, alias="15m")

    def items(self) -> list[tuple[PeriodName, TrendPeriodConfig]]:
        return [("1m", self.m1), ("3m", self.m3), ("5m", self.m5), ("15m", self.m15)]


class TrendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 同方向两次“开始”提醒的最短间隔。
    cooldown_minutes: int = Field(default=30, ge=0, le=1440)
    escalation_step_percent: float = Field(default=5.0, gt=0, le=100)
    end_rebound_ratio: float = Field(default=0.5, gt=0, le=1)
    stall_minutes: int = Field(default=30, ge=1, le=1440)
    notify_end: bool = True
    periods: TrendPeriodsConfig = Field(default_factory=TrendPeriodsConfig)

    @model_validator(mode="after")
    def validate_periods(self) -> TrendConfig:
        if not self.enabled:
            return self
        rules = self.rules()
        if not rules:
            raise ValueError("启用趋势提醒时至少要启用一个周期")
        needed = max(rule.history_minutes for rule in rules) + 1
        if needed > MAX_WARMUP_CANDLES:
            # 预热一次请求拿不到这么多 1 分钟 K 线，最长周期启动后很久都无法判定。
            raise ValueError(
                f"最长周期的 (candles + baseline_candles) × 周期分钟数为 {needed - 1}，"
                f"不能超过 {MAX_WARMUP_CANDLES - 1}"
            )
        return self

    def rules(self) -> list[TrendRule]:
        return [period.to_rule(name) for name, period in self.periods.items() if period.enabled]

    def episode_settings(self) -> EpisodeSettings:
        return EpisodeSettings(
            cooldown_minutes=self.cooldown_minutes,
            escalation_step_percent=self.escalation_step_percent,
            end_rebound_ratio=self.end_rebound_ratio,
            stall_minutes=self.stall_minutes,
            notify_end=self.notify_end,
        )
