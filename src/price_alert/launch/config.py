"""放量启动提醒的配置模型（YAML 的 launch 段）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.launch.rule import LaunchRule


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 启动前的币往往成交极少（RLC 启动前 24 小时只有约 7 万 USDT），按 24 小时成交额设门槛就会等它涨完才入池，
    # 所以默认不设下限，监控全部虚拟币合约；冷门币的噪声靠 min_window_quote 的绝对成交额兜底。
    min_volume_24h_quote: float = Field(default=0, ge=0)
    max_volume_24h_quote: float | None = Field(default=None, gt=0)
    window_minutes: int = Field(default=5, ge=1, le=60)
    # 基准期加窗口不超过 1500 分钟，预热一次请求就能取完（Gate 单次上限 2000 根）。
    baseline_minutes: int = Field(default=360, ge=30, le=1440)
    min_volume_ratio: float = Field(default=5.0, gt=1, le=1000)
    min_window_quote: float = Field(default=30_000, ge=0)
    min_change_percent: float = Field(default=3.0, gt=0, le=100)
    require_breakout: bool = True
    alert_drops: bool = False
    cooldown_minutes: int = Field(default=15, ge=1, le=1440)
    realert_step_percent: float = Field(default=5.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_ranges(self) -> LaunchConfig:
        if self.baseline_minutes <= self.window_minutes:
            raise ValueError("baseline_minutes 必须大于 window_minutes")
        if self.cooldown_minutes < self.window_minutes:
            # 冷却短于窗口时，同一段放量还在窗口里就冷却完了，只要价格再推进一截就会被当作新的一波。
            raise ValueError("cooldown_minutes 不能小于 window_minutes")
        if self.max_volume_24h_quote is not None and self.max_volume_24h_quote <= self.min_volume_24h_quote:
            raise ValueError("max_volume_24h_quote 必须大于 min_volume_24h_quote")
        return self

    def to_rule(self) -> LaunchRule:
        return LaunchRule(
            window_minutes=self.window_minutes,
            baseline_minutes=self.baseline_minutes,
            min_volume_ratio=self.min_volume_ratio,
            min_window_quote=self.min_window_quote,
            min_change_percent=self.min_change_percent,
            require_breakout=self.require_breakout,
            alert_drops=self.alert_drops,
            cooldown_minutes=self.cooldown_minutes,
            realert_step_percent=self.realert_step_percent,
        )
