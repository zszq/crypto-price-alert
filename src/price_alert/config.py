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
    # 五种提醒相互独立，模型定义在各自的检测器子包内；删除某种提醒时只需去掉对应一项。
    move: MoveConfig = Field(default_factory=MoveConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    dormant: DormantConfig = Field(default_factory=DormantConfig)
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
    """读取单个 YAML 文件，或一个目录下全部 .yaml 文件（各文件的顶层键合并为一份配置）。"""
    source = Path(path)
    raw = _read_directory(source) if source.is_dir() else _read_file(source)
    config = AppConfig.model_validate(raw)

    # Webhook 常包含密钥，环境变量覆盖可避免敏感地址进入配置文件。
    webhook_url = os.getenv("PRICE_ALERT_WEBHOOK_URL")
    if webhook_url:
        config.alerts.webhook_url = webhook_url
    return config


def _read_file(source: Path) -> dict:
    # 传文件对象而不是字符串，YAML 语法错误的提示里才带文件名。
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} 的顶层必须是键值对")
    return raw


def _read_directory(directory: Path) -> dict:
    # 只做顶层合并：同一段拆在两个文件里时一处会被另一处悄悄覆盖，所以重复的顶层键直接报错。
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"配置目录 {directory} 下没有 .yaml 文件")
    merged: dict = {}
    origins: dict[str, Path] = {}
    for file in files:
        for key, value in _read_file(file).items():
            if key in origins:
                raise ValueError(f"顶层键 {key} 同时出现在 {origins[key].name} 与 {file.name}")
            origins[key] = file
            merged[key] = value
    return merged
