"""放量启动提醒的配置模型（YAML 的 launch 段）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.launch.rule import LaunchRule, Thresholds


class LaunchDropConfig(BaseModel):
    """放量下跌单独的门槛（YAML 的 launch.drop）；没写的项沿用 launch 段顶层的值，什么都不写就与上涨完全相同。

    只开放量比与涨跌幅上下限；窗口、基准期、成交额与冷却两边共用，
    窗口和基准期还决定要保留多少根 K 线，分开会让两个方向的历史长度不一致。
    """

    model_config = ConfigDict(extra="forbid")

    min_volume_ratio: float | None = Field(default=None, gt=1, le=1000)
    min_change_percent: float | None = Field(default=None, gt=0, le=100)
    # 显式写 null 表示下跌不设上限；不写才沿用顶层，两者靠 model_fields_set 区分。
    max_change_percent: float | None = Field(default=None, gt=0, le=1000)


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 24 小时成交额不足 100 万的币，提醒后 30 分钟的成交额还不如提醒窗口本身，多是没人跟进的假启动；
    # 下限定在 100 万而不更高，是因为启动中的币成交额涨得很快，多数第一波刚过就会入池，
    # 6.6 天回测中报到的大行情没有减少；再提高就开始漏掉之后大涨的提醒。
    min_volume_24h_quote: float = Field(default=1_000_000, ge=0)
    max_volume_24h_quote: float | None = Field(default=None, gt=0)
    window_minutes: int = Field(default=5, ge=1, le=60)
    # 基准期加窗口不超过 1500 分钟，预热一次请求就能取完（Gate 单次上限 2000 根）。
    baseline_minutes: int = Field(default=360, ge=30, le=1440)
    # 量比：最近一个窗口（默认 5 分钟）的成交额，是它平时同样长度成交额的多少倍；
    # 「平时」取基准期的均值，按相对放大衡量，不必给成交额悬殊的各个币分别定门槛。
    # 半年回测中从 5 降到 3，多报到的都是基准期本来就活跃的币，冷门币已由窗口成交额门槛挡住，假启动不增反降。
    min_volume_ratio: float = Field(default=3.0, gt=1, le=1000)
    # 拉升后成交随即冷清的假启动几乎都出自窗口成交额只有几万的冷门币；回测中门槛提到 10 万，
    # 与「拉升后再观察几分钟成交」去掉的假启动相当，却不必延迟提醒，大行情也一段不漏。
    min_window_quote: float = Field(default=100_000, ge=0)
    min_change_percent: float = Field(default=3.0, gt=0, le=100)
    require_breakout: bool = True
    alert_drops: bool = False
    cooldown_minutes: int = Field(default=15, ge=1, le=1440)
    realert_step_percent: float = Field(default=5.0, ge=0, le=100)
    # 涨跌幅上限：窗口内就走完这么多的多是插针式拉盘，提醒时已在高点；None 表示不设上限。
    max_change_percent: float | None = Field(default=15.0, gt=0, le=1000)
    drop: LaunchDropConfig = Field(default_factory=LaunchDropConfig)

    @model_validator(mode="after")
    def validate_ranges(self) -> LaunchConfig:
        if self.baseline_minutes <= self.window_minutes:
            raise ValueError("baseline_minutes 必须大于 window_minutes")
        if self.cooldown_minutes < self.window_minutes:
            # 冷却短于窗口时，同一段放量还在窗口里就冷却完了，只要价格再推进一截就会被当作新的一波。
            raise ValueError("cooldown_minutes 不能小于 window_minutes")
        if self.max_volume_24h_quote is not None and self.max_volume_24h_quote <= self.min_volume_24h_quote:
            raise ValueError("max_volume_24h_quote 必须大于 min_volume_24h_quote")
        if self.max_change_percent is not None and self.max_change_percent <= self.min_change_percent:
            raise ValueError("max_change_percent 必须大于 min_change_percent")
        merged = self._drop_values()
        if merged is not None:
            _, low, high = merged
            if high is not None and high <= low:
                # 只写了一边时另一边取自顶层，报错里写明是合并后的值，免得以为是顶层两项冲突。
                raise ValueError(
                    f"放量下跌的 max_change_percent（{high:g}）必须大于 min_change_percent（{low:g}），"
                    "未在 drop 中设置的项取自顶层"
                )
        return self

    def _drop_values(self) -> tuple[float, float, float | None] | None:
        given = self.drop.model_fields_set
        if not given:
            return None
        ratio, change = self.drop.min_volume_ratio, self.drop.min_change_percent
        return (
            self.min_volume_ratio if ratio is None else ratio,
            self.min_change_percent if change is None else change,
            self.drop.max_change_percent if "max_change_percent" in given else self.max_change_percent,
        )

    def drop_thresholds(self) -> Thresholds | None:
        """合并出放量下跌的门槛；drop 段什么都没写时返回 None，规则里两个方向共用顶层门槛。"""
        merged = self._drop_values()
        return None if merged is None else Thresholds(*merged)

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
            max_change_percent=self.max_change_percent,
            drop=self.drop_thresholds(),
        )
