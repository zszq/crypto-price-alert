# 放量启动研究的数据工具

这里只放下载、读取历史 1 分钟 K 线的工具，供放量启动（`src/price_alert/detectors/launch/`）相关的回测研究使用。
以前的研究脚本与结论已全部删除，新研究的脚本也放在本目录。

不参与实时监控，也不在 `ruff check src tests` 与 pytest 的范围内（提交前可单独运行 `ruff check research`）。

## 目录约定

- 代码：本目录，进版本库；所有路径由 `paths.py` 统一给出。
- 数据：`data/backtest/`，**不进版本库**（`.gitignore` 已忽略）。删掉后运行 `update_data.py` 可重新下载月度文件，但 REST 补的部分只能取回最近约 7 天，更早的要等下载站发布那个月的文件：
  - `raw/YYYYMM/<合约>-YYYYMM.csv.gz`：Gate 历史数据下载站的月度 1 分钟 K 线（列：时间、张数、收、高、低、开，省略无成交分钟），下载过的不再请求；`raw_missing.json` 记录确认不存在的（合约, 月份）。
  - `recent/<合约>.csv`：下载站还没发布的月份用 REST 补（列：时间、开、高、低、收、计价成交额），每次只追加新分钟；`.spans` 记录每次实际覆盖的区间。
  - `contracts.json`：合约面值等（合并保存，之后下架的合约也保留）。
  - `cache/<合约>.npz`：`load_data.py` 合并后的分钟序列缓存，来源文件变化或加载规则改版（`load_data.CACHE_VERSION`）时自动重建。
  - `oi_1h/`、`oi_5m/<合约>.npz`：以前下载的历史持仓（Gate `contract_stats`，REST 只保留最近 180 天，过期的取不回来）。下载脚本已随旧研究删除，需要时重写。
  - `results/`：研究输出。
  - `.venv/`：回测专用虚拟环境。

## 环境

回测需要 numpy 与 numba，项目本身不依赖，所以单独建虚拟环境（在仓库根目录执行）：

```powershell
python -m venv data/backtest/.venv
data/backtest/.venv/Scripts/python.exe -m pip install numpy numba -e .
```

下文命令都在仓库根目录的 PowerShell 中执行，先设置（PowerShell 里运行变量中的程序要加调用运算符 `&`）：

```powershell
$P = "data/backtest/.venv/Scripts/python.exe"
```

## 数据下载与更新：update_data.py

```powershell
& $P research/launch_trade/update_data.py            # 默认取约半年前的月份起
& $P research/launch_trade/update_data.py --start 202409 --workers 16   # 两年；补很多月份时调高下载并发
& $P research/launch_trade/update_data.py --start 202109 --workers 16 --discover   # 五年，并补进期间已下架的币
```

- 下载站的 1 分钟 K 线最早到 2020-01（BTC_USDT；各合约从上线月份开始），另有 5 分钟 / 1 小时 K 线、逐笔成交（`trades`）、资金费率结算（`funding_applies`）、标记价格（`mark_prices`），这里只下载 1 分钟 K 线；
- `--discover`：合约列表默认只取当前还在交易的合约（再合并此前下载过的），期间下架的币不会进样本。加上它会按下载站背后 S3 存储桶的月度文件列表，把 `--start` 以来存在过、合约列表里没有的币类合约（合约详情接口对已下架合约仍返回面值，按 `contract_type` 为空筛掉股票等）补进 `contracts.json` 再下载。面值按当前值，期间改过面值的合约要用 `check_jumps.py` 检查；
- 已结束的月份先按存储桶的月度文件列表把不存在的（合约, 月份）记入 `raw_missing.json`，只请求实际存在的文件；列表取不到时退回逐个请求；
- 已结束的月份从下载站取，没有计价成交额，加载时按 张数 × 合约面值 × 典型价 估算；
- 下载站还没有的月份用 REST 补，带精确成交额。REST 只保留最近 10000 根 1 分钟 K 线（约 6.9 天），**至少每周更新一次**，否则会留下缺口（加载时缺口按无数据处理）；
- 下载站发布新月份后再运行一次，自动下载并以月度文件为准。

## 读取：load_data.py

`load_contracts()` 读 `contracts.json`；`load_series(symbol, multiplier)` 把 `raw/` 与 `recent/` 合并成该合约一条连续的分钟序列（`Series`：开高低收、计价成交额、`covered`），并缓存到 `cache/`：

- 有成交的分钟取原始 K 线；数据源覆盖范围内、没有成交的分钟按上一收盘价补平线、成交额 0（与检测器口径一致）；
- 数据源没有覆盖的分钟，以及连续一整天以上完全没有成交的空档（停牌、下架后重新上线），`covered=False`，研究里应按无数据处理。

## 数据检查：check_jumps.py

```powershell
& $P research/launch_trade/check_jumps.py
```

列出价格整体跳变（改面值、停牌后重新上线、旧名字给了新币）的位置，供人工判断数据是否可用：月度文件只有张数，面值改过的合约，改之前的成交额会算错。
