"""冷门启动提醒的配置模型（YAML 的 dormant 段）。默认值见 research/launch_trade/dormant_calm.md 的开发期选参。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.dormant.rule import DormantRule


class DormantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 合约池按交易所的 24 小时成交额筛选，冷门与否由检测器按自己的 K 线判断（窗口之前 24 小时，见 max_prior_quote）。
    # 下限 0 即所有有成交的币；上限要明显高于 max_prior_quote：拉升会把 24 小时成交额推高，
    # 上限贴着冷门门槛的话，刚启动的币可能在合约池刷新时被移出。
    min_volume_24h_quote: float = Field(default=0, ge=0)
    max_volume_24h_quote: float | None = Field(default=3_000_000, gt=0)
    window_minutes: int = Field(default=10, ge=1, le=60)
    # 冷门：窗口之前 24 小时的成交额上限。
    max_prior_quote: float = Field(default=1_000_000, gt=0)
    min_window_quote: float = Field(default=5_000, ge=0)
    # 窗口内有成交的分钟数下限：冷门币一两笔单子就能拉出涨幅，要求多分钟有成交才算有人在持续买。
    min_active_minutes: int = Field(default=5, ge=1, le=60)
    min_change_percent: float = Field(default=8.0, gt=0, le=100)
    # 窗口内单分钟收盘涨幅上限（百分比）：一两分钟跳上去的多是几笔单子拉的。
    max_minute_change_percent: float = Field(default=5.0, gt=0, le=100)
    # 窗口总实体占比（实体和 ÷ 振幅和）下限：上影线多说明冲高被砸回，0 即不限。
    min_body_ratio: float = Field(default=0.6, ge=0, le=1)
    # 窗口涨幅至少是涨前波动（窗口之前 240 分钟平均每分钟真实波幅）的多少倍：涨前越平稳，同样的涨幅倍数越高。
    min_calm_multiple: float = Field(default=10.0, ge=0)
    # 涨前波动（窗口之前 240 分钟平均每分钟真实波幅，百分比）上限：涨前来回乱跳的币不报；null 即不限。
    max_pre_volatility_percent: float | None = Field(default=0.5, gt=0)
    # 窗口最后一分钟成交额下限（USDT）：拉到最后一分钟没人买了的多是几笔单子拉完就停；0 即不限。
    min_last_minute_quote: float = Field(default=0, ge=0)
    # 后半窗口（最后 ⌈窗口/2⌉ 分钟）成交额占窗口成交额的比例下限：成交集中在前半段的容易回落；0 即不限。
    # 必须小于 1：等于 1 要求前半窗口一笔成交都没有，那就不是「拉升中持续有人买」了。
    min_late_quote_share: float = Field(default=0, ge=0, lt=1)
    cooldown_minutes: int = Field(default=60, ge=1, le=1440)
    # 冷却期内价格比上次提醒价再涨这么多（百分比）就再报「第 N 波」；null 即冷却内不再报。
    realert_step_percent: float | None = Field(default=3.0, gt=0, le=100)

    @model_validator(mode="after")
    def validate_ranges(self) -> DormantConfig:
        if self.min_active_minutes > self.window_minutes:
            raise ValueError("min_active_minutes 不能大于 window_minutes")
        if self.cooldown_minutes < self.window_minutes:
            # 冷却短于窗口时，同一段拉升还在窗口里就冷却完了，会被重复提醒。
            raise ValueError("cooldown_minutes 不能小于 window_minutes")
        if self.min_change_percent > self.window_minutes * self.max_minute_change_percent:
            # 每分钟都涨到上限也凑不够窗口涨幅（按单利近似），这组门槛永远不会提醒。
            raise ValueError("min_change_percent 不能大于 window_minutes × max_minute_change_percent")
        if self.max_volume_24h_quote is not None and self.max_volume_24h_quote <= self.min_volume_24h_quote:
            raise ValueError("max_volume_24h_quote 必须大于 min_volume_24h_quote")
        return self

    def to_rule(self) -> DormantRule:
        return DormantRule(
            window_minutes=self.window_minutes,
            max_prior_quote=self.max_prior_quote,
            min_window_quote=self.min_window_quote,
            min_active_minutes=self.min_active_minutes,
            min_change_percent=self.min_change_percent,
            max_minute_change_percent=self.max_minute_change_percent,
            min_body_ratio=self.min_body_ratio,
            min_calm_multiple=self.min_calm_multiple,
            max_pre_volatility_percent=self.max_pre_volatility_percent,
            min_last_minute_quote=self.min_last_minute_quote,
            min_late_quote_share=self.min_late_quote_share,
            cooldown_minutes=self.cooldown_minutes,
            realert_step_percent=self.realert_step_percent,
        )
