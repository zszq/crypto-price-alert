"""冷门启动提醒：之前 24 小时几乎没人交易的币，最近几分钟突然有成交、价格明显拉升。

面向连放量上涨的合约池都进不去的冷门币，与短时异动、K 线趋势、放量启动相互独立，
只通过 assembly.build_detectors 接入实时监控。准确率低于放量上涨，是用户知情后作为例外采用的提醒，
依据见 research/launch_trade/dormant_solo.md。可选的第二组参数「冷门慢启动」（60 分钟里量价缓慢拉升，更不准）
见 research/launch_trade/dormant_slow.md。
"""
