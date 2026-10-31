"""冷门启动提醒的配置模型（YAML 的 dormant 段：共用的开关与合约池、main 与 slow 两组参数）。"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.detectors.dormant.rule import DormantGroup, DormantRule


class DormantRuleConfig(BaseModel):
    """一组判定门槛与冷却的公共字段与校验；现行、慢启动两组结构相同，只在默认值、取值范围上不同，慢启动另有几项。"""

    model_config = ConfigDict(extra="forbid")
    group: ClassVar[DormantGroup]

    window_minutes: int
    # 冷门：窗口之前 24 小时的成交额上限。
    max_prior_quote: float
    min_window_quote: float
    # 窗口内有成交的分钟数下限：冷门币一两笔单子就能拉出涨幅，要求多分钟有成交才算有人在持续买。
    min_active_minutes: int
    min_change_percent: float
    # 窗口内单分钟收盘涨幅上限（百分比）：一两分钟跳上去的多是几笔单子拉的。
    max_minute_change_percent: float
    # 窗口总实体占比（实体和 ÷ 振幅和）下限：上影线多说明冲高被砸回，0 即不限。
    min_body_ratio: float
    # 窗口涨幅至少是涨前波动（窗口之前 240 分钟平均每分钟真实波幅）的多少倍：涨前越平稳，同样的涨幅倍数越高。
    min_calm_multiple: float
    # 涨前波动（窗口之前 240 分钟平均每分钟真实波幅，百分比）上限：涨前来回乱跳的币不报；null 即不限。
    max_pre_volatility_percent: float | None
    # 窗口最后一分钟成交额下限（USDT）：最后一分钟没人买的多是几笔单子拉完就停；0 即不限。
    min_last_minute_quote: float
    # 后半窗口（最后 ⌈窗口/2⌉ 分钟）成交额占比下限：成交集中在前半段的容易回落；0 即不限。
    min_late_quote_share: float
    cooldown_minutes: int
    # 冷却期内价格比上次提醒价再涨这么多（百分比）就再报「第 N 波」；null 即冷却内不再报。
    realert_step_percent: float | None

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.min_active_minutes > self.window_minutes:
            raise ValueError("min_active_minutes 不能大于 window_minutes")
        if self.cooldown_minutes < self.window_minutes:
            # 冷却短于窗口时，同一段拉升还在窗口里就冷却完了，会被重复提醒。
            raise ValueError("cooldown_minutes 不能小于 window_minutes")
        if self.min_change_percent > self.window_minutes * self.max_minute_change_percent:
            # 每分钟都涨到上限也凑不够窗口涨幅（按单利近似），这组门槛永远不会提醒。
            raise ValueError("min_change_percent 不能大于 window_minutes × max_minute_change_percent")
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
            group=self.group,
        )


class DormantMainConfig(DormantRuleConfig):
    """现行一组（YAML 的 dormant.main），始终启用。"""

    group: ClassVar[DormantGroup] = DormantGroup.MAIN

    window_minutes: int = Field(default=10, ge=1, le=60)
    max_prior_quote: float = Field(default=1_000_000, gt=0)
    min_window_quote: float = Field(default=5_000, ge=0)
    min_active_minutes: int = Field(default=5, ge=1, le=60)
    min_change_percent: float = Field(default=8.0, gt=0, le=100)
    max_minute_change_percent: float = Field(default=5.0, gt=0, le=100)
    min_body_ratio: float = Field(default=0.6, ge=0, le=1)
    min_calm_multiple: float = Field(default=10.0, ge=0)
    max_pre_volatility_percent: float | None = Field(default=0.5, gt=0)
    min_last_minute_quote: float = Field(default=0, ge=0)
    # 必须小于 1：等于 1 要求前半窗口一笔成交都没有，那就不是「拉升中持续有人买」了。
    min_late_quote_share: float = Field(default=0, ge=0, lt=1)
    cooldown_minutes: int = Field(default=60, ge=1, le=1440)
    realert_step_percent: float | None = Field(default=3.0, gt=0, le=100)


class DormantSlowConfig(DormantRuleConfig):
    """慢启动一组（YAML 的 dormant.slow）：涨前近乎一条直线、60 分钟里量价缓慢拉升的冷门币。

    比现行一组更不可靠，所以代码默认关闭，只由 YAML 开启。
    实体占比、涨前波动倍数、最后一分钟成交额与第 N 波补报默认不限。
    效率、量比、回落与跟踪补报（follow_*）只在这一组开放，跟踪补报代码默认关闭。
    """

    group: ClassVar[DormantGroup] = DormantGroup.SLOW

    enabled: bool = False
    # 慢启动要看一两个小时的缓慢拉升，上限比现行一组的 60 分钟宽。
    window_minutes: int = Field(default=60, ge=1, le=120)
    max_prior_quote: float = Field(default=1_000_000, gt=0)
    min_window_quote: float = Field(default=10_000, ge=0)
    min_active_minutes: int = Field(default=48, ge=1, le=120)
    min_change_percent: float = Field(default=6.0, gt=0, le=100)
    max_minute_change_percent: float = Field(default=3.0, gt=0, le=100)
    min_body_ratio: float = Field(default=0, ge=0, le=1)
    min_calm_multiple: float = Field(default=0, ge=0)
    max_pre_volatility_percent: float | None = Field(default=0.2, gt=0)
    min_last_minute_quote: float = Field(default=0, ge=0)
    min_late_quote_share: float = Field(default=0.5, ge=0, lt=1)
    # 效率（净涨幅的价差 ÷ 逐分钟收盘价差绝对值之和）下限：来回拉扯的不报；0 即不限。
    min_efficiency: float = Field(default=0.3, ge=0, le=1)
    # 量比（窗口成交额 ÷ 之前 24 小时同长度均值）下限；0 即不限。
    min_volume_ratio: float = Field(default=3.0, ge=0)
    # 从窗口最高价回落的比例（占窗口起点到最高价涨幅）上限；null 即不限。
    max_retrace: float | None = Field(default=0.3, ge=0, le=1)
    cooldown_minutes: int = Field(default=120, ge=1, le=1440)
    realert_step_percent: float | None = Field(default=None, gt=0, le=100)
    # 跟踪补报：提醒（含补报）后收盘价比上次提醒价再涨这么多（百分比）就补报下一波，不看其他条件；null 即不跟踪。
    follow_step_percent: float | None = Field(default=None, gt=0, le=100)
    # 距上次提醒（含补报）超过这么久（分钟）没再涨够就停止跟踪。
    follow_minutes: int = Field(default=240, ge=1, le=1440)

    def to_rule(self) -> DormantRule:
        return replace(
            super().to_rule(),
            min_efficiency=self.min_efficiency,
            min_volume_ratio=self.min_volume_ratio,
            max_retrace=self.max_retrace,
            follow_step_percent=self.follow_step_percent,
            follow_minutes=self.follow_minutes,
        )


class DormantConfig(BaseModel):
    """冷门启动：两组共用的开关与合约池，现行一组（main）与慢启动一组（slow）各自的门槛与冷却。"""

    model_config = ConfigDict(extra="forbid")

    # 关掉整个冷门启动（两组都不报）。
    enabled: bool = True
    # 合约池按交易所 24 小时成交额筛选，冷门与否由检测器按自己的 K 线判断（见 max_prior_quote）。
    # 上限要明显高于 max_prior_quote：拉升会推高 24 小时成交额，贴着冷门门槛的话刚启动的币可能被移出合约池。
    min_volume_24h_quote: float = Field(default=0, ge=0)
    max_volume_24h_quote: float | None = Field(default=3_000_000, gt=0)
    # 两组共用合约池与 K 线，门槛、冷却按组独立配置。
    main: DormantMainConfig = Field(default_factory=DormantMainConfig)
    slow: DormantSlowConfig = Field(default_factory=DormantSlowConfig)

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        if self.max_volume_24h_quote is not None and self.max_volume_24h_quote <= self.min_volume_24h_quote:
            raise ValueError("max_volume_24h_quote 必须大于 min_volume_24h_quote")
        return self

    def rules(self) -> tuple[DormantRule, ...]:
        """检测器要判定的各组；整个冷门启动关闭时为空。"""
        if not self.enabled:
            return ()
        groups = [self.main, self.slow] if self.slow.enabled else [self.main]
        return tuple(group.to_rule() for group in groups)
