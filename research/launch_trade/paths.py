"""回测研究的路径约定。

代码在版本库（research/launch_trade/），行情数据与结果只放本机（data/backtest/，已在 .gitignore 中忽略）。
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# 历史行情、缓存、研究导出与结果都在这里（两年数据合计约 7.5GB），不进版本库。
# 删掉后运行 update_data.py 可重新下载月度文件，但 REST 补的部分只能取回最近约 7 天，更早的要等下载站发布那个月的文件。
DATA = REPO / "data" / "backtest"
RESULTS = DATA / "results"
