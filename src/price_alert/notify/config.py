"""alerts 配置段：各提醒通道的开关、队列与输出位置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
