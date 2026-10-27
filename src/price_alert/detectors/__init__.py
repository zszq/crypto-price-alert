"""各类提醒的检测器子包，以及它们共用的提醒类型汇总与历史回放取数。

每个子包（spike、trend、launch、dormant、reversal）自带配置段、判定、提醒文本、回放与模拟，
只通过 assembly 接入实时监控；子包之间互不依赖，reversal 继承 spike 的检测器除外。
"""
