"""YAML 配置读取与严格校验。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.trend.config import TrendConfig

# K 线周期映射只保留这一份，避免服务编排与配置校验各自维护而出现分歧。
INTERVAL_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rest_url: str = "https://api.gateio.ws/api/v4"
    websocket_url: str = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    settle: Literal["usdt"] = "usdt"
    min_volume_24h_quote: float = Field(default=10_000_000, gt=0)
    universe_exit_volume_ratio: float = Field(default=0.8, gt=0, le=1)
    universe_refresh_seconds: int = Field(default=600, ge=60, le=86400)
    rest_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    rest_retries: int = Field(default=3, ge=1, le=10)
    # 所有 REST 出口共用的长期平均速率上限。合约池再大、重连再频繁，都只是拉长一轮预热/回补的耗时。
    rest_rate_limit_per_second: float = Field(default=10.0, gt=0, le=1000)
    # 攒满后可一次放行的瞬时额度：让启动预热不必逐个等令牌。它是任意一刻的绝对上限，
    # 不应超过交易所单个窗口的配额；长期平均速率仍由上面的值决定。
    rest_rate_limit_burst: int = Field(default=20, ge=1, le=1000)
    warmup_concurrency: int = Field(default=8, ge=1, le=32)
    subscription_chunk_size: int = Field(default=100, ge=1, le=500)
    receive_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_data_lag_seconds: float = Field(default=10.0, gt=0, le=300)
    reconnect_initial_seconds: float = Field(default=1.0, gt=0, le=60)
    reconnect_max_seconds: float = Field(default=30.0, gt=0, le=300)
    status_interval_seconds: int = Field(default=60, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_reconnect_backoff(self) -> GateConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError("reconnect_max_seconds 不能小于 reconnect_initial_seconds")
        return self


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


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candle_interval: Literal["1m", "5m", "15m"] = "1m"
    atr_period: int = Field(default=14, ge=2, le=200)
    warmup_candles: int = Field(default=50, ge=15, le=500)
    max_atr_age_seconds: int = Field(default=180, ge=30, le=3600)
    # 各窗口共用 K 线、ATR 与秒级数据，门槛、确认与冷却按窗口独立配置。
    short_window: ShortWindowConfig = Field(default_factory=ShortWindowConfig)
    long_window: LongWindowConfig = Field(default_factory=LongWindowConfig)

    @model_validator(mode="after")
    def validate_warmup_and_freshness(self) -> IndicatorConfig:
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


class AlertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_size: int = Field(default=1000, ge=1, le=100_000)
    console: bool = True
    console_colors: bool = True
    beep: bool = True
    jsonl_path: Path | None = Path("data/alerts/alerts.jsonl")
    jsonl_max_bytes: int = Field(default=10_000_000, ge=0)
    jsonl_backup_count: int = Field(default=5, ge=1, le=100)
    webhook_url: str | None = None
    webhook_timeout_seconds: float = Field(default=5.0, gt=0, le=30)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: GateConfig = Field(default_factory=GateConfig)
    indicator: IndicatorConfig = Field(default_factory=IndicatorConfig)
    # K 线形态趋势提醒，独立于秒级 ATR 异动检测；模型定义在 trend 包内，删除该功能时只需去掉这一项。
    trend: TrendConfig = Field(default_factory=TrendConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)


def load_config(path: str | Path) -> AppConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)

    # Webhook 常包含密钥，环境变量覆盖可避免敏感地址进入配置文件。
    webhook_url = os.getenv("PRICE_ALERT_WEBHOOK_URL")
    if webhook_url:
        config.alerts.webhook_url = webhook_url
    return config
