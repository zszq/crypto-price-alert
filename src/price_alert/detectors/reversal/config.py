"""反弹回落提醒的配置模型（YAML 的 reversal 段）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.detectors.reversal.rule import MAX_BURST_RATIO, ReversalRule


class ReversalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 合约池门槛再低，提醒变多但更不准。
    min_volume_24h_quote: float = Field(default=8_000_000, ge=0)
    max_volume_24h_quote: float | None = Field(default=None, gt=0)
    lookback_seconds: int = Field(default=10, ge=5, le=600)
    min_change_percent: float = Field(default=1.5, gt=0, le=100)
    trigger_atr_multiple: float = Field(default=1.0, gt=0, le=20)
    confirmation_seconds: int = Field(default=1, ge=1, le=10)
    min_window_trades: int = Field(default=5, ge=1)
    cooldown_seconds: int = Field(default=30, ge=0, le=86400)
    prior_seconds: int = Field(default=900, ge=60, le=86400)
    min_prior_move_percent: float = Field(default=6.0, gt=0, le=100)
    min_trade_burst_ratio: float = Field(default=16.0, gt=0, lt=MAX_BURST_RATIO)
    # 离 24 小时反向极值的距离上限：急拉已远离此前低点（急砸已远离此前高点）时不算刚开始的反转；null 即不限。
    max_extreme_distance_percent: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def validate_ranges(self) -> ReversalConfig:
        if self.confirmation_seconds > self.lookback_seconds:
            # 确认期长于窗口时，基准价会追上已经完成的急动，永远确认不了。
            raise ValueError("confirmation_seconds 不能大于 lookback_seconds")
        extreme = self.max_extreme_distance_percent
        if extreme is not None and extreme <= self.min_change_percent:
            # 急动的基准价落在已收盘 K 线里时，24 小时极值不会比它更近，距离大致不小于急动幅度
            # （距离按触发价、幅度按那一秒均价算）。上限不高于急动门槛就只剩急跌、急拉都挤在同一分钟里的个例，
            # 等于把这一提醒关掉了。
            raise ValueError("max_extreme_distance_percent 必须大于 min_change_percent")
        if self.max_volume_24h_quote is not None and self.max_volume_24h_quote <= self.min_volume_24h_quote:
            raise ValueError("max_volume_24h_quote 必须大于 min_volume_24h_quote")
        return self

    def to_rule(self) -> ReversalRule:
        return ReversalRule(
            lookback_seconds=self.lookback_seconds,
            min_change_percent=self.min_change_percent,
            trigger_atr_multiple=self.trigger_atr_multiple,
            confirmation_seconds=self.confirmation_seconds,
            min_window_trades=self.min_window_trades,
            cooldown_seconds=self.cooldown_seconds,
            prior_seconds=self.prior_seconds,
            min_prior_move_percent=self.min_prior_move_percent,
            min_trade_burst_ratio=self.min_trade_burst_ratio,
            max_extreme_distance_percent=self.max_extreme_distance_percent,
        )
