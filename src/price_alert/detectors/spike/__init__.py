"""瞬时异动与短时异动提醒：短时价格位移同时超过百分比与 ATR 倍数门槛、连续确认数秒后提醒。

同一个 AtrSpikeDetector 按两组参数各跑一个检测器：瞬时异动（instant 段）急拉急砸都报，短窗口同向连发时追加连续提醒；
短时异动（spike 段）另要求此前逆向走过一段、成交突然涌入，提醒后更常沿提醒方向继续走。
覆盖成交活跃的主流合约，与 K 线趋势、放量启动、冷门启动相互独立，只通过 assembly.build_detectors 接入实时监控；
反弹回落（reversal）继承这里的 AtrSpikeDetector，改它的内部实现时要一并跑 tests/test_reversal.py。
"""
