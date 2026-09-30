"""K 线形态趋势提醒：由 1 分钟 K 线合成 1m/3m/5m/15m 周期，按形态识别持续单边行情。

与秒级 ATR 异动检测相互独立，只通过 assembly.build_detectors 接入实时监控。
"""
