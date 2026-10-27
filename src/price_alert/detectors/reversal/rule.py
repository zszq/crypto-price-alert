"""反弹回落的判定门槛（纯数据），检测器、提醒文本与模拟共用。"""

from __future__ import annotations

from dataclasses import dataclass

# 成交笔数暴增：最近 BURST_SECONDS 秒的成交笔数与最近 BURST_BASE_SECONDS 秒的每分钟平均相比。
# 两个长度固定：研究只验证过 1 分钟对 1 小时（research/move_accuracy/move_accuracy.md），不开放成配置。
BURST_SECONDS = 60
BURST_BASE_SECONDS = 3600
# 全部成交都落在最近 1 分钟时倍数最大，等于两个长度之比；门槛不小于它就永远不会提醒。
MAX_BURST_RATIO = BURST_BASE_SECONDS / BURST_SECONDS
# ATR 口径与短时异动的代码默认值相同（研究用的就是这一套），不单独开放配置。
ATR_PERIOD = 14
CANDLE_INTERVAL = "1m"
CANDLE_INTERVAL_SECONDS = 60
WARMUP_CANDLES = 50
MAX_ATR_AGE_SECONDS = 180
# 反向极值距离回看此前多少分钟的已收盘 1 分钟 K 线：研究第三轮 720 / 1440 / 2880 分钟里选中 24 小时，不开放配置。
EXTREME_MINUTES = 1440


@dataclass(frozen=True, slots=True)
class ReversalRule:
    # 急动窗口：与短时异动相同的判定（秒级 VWAP、基准桶、ATR、连续确认、空档复核、冷却）。
    lookback_seconds: int
    min_change_percent: float
    trigger_atr_multiple: float
    confirmation_seconds: int
    min_window_trades: int
    cooldown_seconds: int
    # 前置走势：窗口开始前 prior_seconds 秒到窗口基准价，逆着急动方向走了至少 min_prior_move_percent。
    prior_seconds: int
    min_prior_move_percent: float
    # 最近 1 分钟成交笔数 ÷ 最近 1 小时每分钟平均笔数的下限。
    min_trade_burst_ratio: float
    # 入场价离此前 24 小时反向极值（急拉看最低价、急砸看最高价）的距离上限，顺急动方向为正；None 不限。
    max_extreme_distance_percent: float | None = None

    @property
    def warmup_candles(self) -> int:
        """预热要的 1 分钟 K 线根数：判反向极值时要整 24 小时已收盘 K 线，另留当前 K 线与一根余量。"""
        if self.max_extreme_distance_percent is None:
            return WARMUP_CANDLES
        return max(WARMUP_CANDLES, EXTREME_MINUTES + 2)

    @property
    def history_seconds(self) -> int:
        """判定要回看的成交历史：前置走势的起点与成交暴增的 1 小时基准，取更早的那个。"""
        return max(self.lookback_seconds + self.prior_seconds, BURST_BASE_SECONDS)
