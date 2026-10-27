"""gate 配置段：交易所地址、合约池刷新、REST 限速与重试、WebSocket 订阅与重连。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rest_url: str = "https://api.gateio.ws/api/v4"
    websocket_url: str = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    settle: Literal["usdt"] = "usdt"
    universe_exit_volume_ratio: float = Field(default=0.8, gt=0, le=1)
    universe_refresh_seconds: int = Field(default=600, ge=60, le=86400)
    rest_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    rest_retries: int = Field(default=3, ge=1, le=10)
    # 所有 REST 出口共用的平均速率上限：合约池再大、重连再频繁，都只是拉长预热/回补的耗时。
    rest_rate_limit_per_second: float = Field(default=10.0, gt=0, le=1000)
    # 攒满后可一次放行的瞬时额度，让启动预热不必逐个等令牌；不应超过交易所单个窗口的配额。
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
