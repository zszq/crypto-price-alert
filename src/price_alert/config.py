"""YAML 配置读取与严格校验。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from price_alert.detectors.dormant.config import DormantConfig
from price_alert.detectors.launch.config import LaunchConfig
from price_alert.detectors.move.config import MoveConfig
from price_alert.detectors.reversal.config import ReversalConfig
from price_alert.detectors.trend.config import TrendConfig
from price_alert.exchange.config import GateConfig
from price_alert.notify.config import AlertConfig


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: GateConfig = Field(default_factory=GateConfig)
    # 秒级 ATR 异动提醒；模型定义在 move 包内。
    move: MoveConfig = Field(default_factory=MoveConfig)
    # K 线形态趋势提醒，独立于秒级 ATR 异动检测；模型定义在 trend 包内，删除该功能时只需去掉这一项。
    trend: TrendConfig = Field(default_factory=TrendConfig)
    # 放量启动提醒，与上面两者相互独立；模型定义在 launch 包内，删除该功能时只需去掉这一项。
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    # 冷门启动提醒，与上面三者相互独立；模型定义在 dormant 包内，删除该功能时只需去掉这一项。
    dormant: DormantConfig = Field(default_factory=DormantConfig)
    # 反弹回落提醒，与上面四者相互独立；模型定义在 reversal 包内，删除该功能时只需去掉这一项。
    reversal: ReversalConfig = Field(default_factory=ReversalConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)

    @model_validator(mode="after")
    def validate_any_detector(self) -> AppConfig:
        detectors = (self.move, self.trend, self.launch, self.dormant, self.reversal)
        if not any(detector.enabled for detector in detectors):
            # 没有检测器时 run 会一直按合约池刷新失败重试，不如在加载配置时直接指出。
            raise ValueError("move、trend、launch、dormant、reversal 至少要启用一个")
        return self


def load_config(path: str | Path) -> AppConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)

    # Webhook 常包含密钥，环境变量覆盖可避免敏感地址进入配置文件。
    webhook_url = os.getenv("PRICE_ALERT_WEBHOOK_URL")
    if webhook_url:
        config.alerts.webhook_url = webhook_url
    return config
