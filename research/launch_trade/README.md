# 放量启动提醒的交易回测

研究「放量启动」提醒（`src/price_alert/launch/`）能不能直接拿来做多：提醒后买入，按不同的入场确认、止损、回撤、止盈与持仓时间平仓，统计半年的收益。**结论见 [FINDINGS.md](FINDINGS.md)。**

只是研究代码，不参与实时监控，也不在 `ruff check src tests` 与 pytest 的范围内（提交前可单独运行 `ruff check research`）。

## 目录约定

- 代码：本目录，进版本库；所有路径由 `paths.py` 统一给出。
- 数据：`data/backtest/`，**不进版本库**（`.gitignore` 已忽略），两年数据合计约 7.5GB（行情约 2.4GB、缓存约 3.4GB、研究导出与结果约 1.5GB、虚拟环境约 0.2GB）。删掉后运行 `update_data.py` 可重新下载月度文件，但 REST 补的部分只能取回最近约 7 天，更早的要等下载站发布那个月的文件：
  - `raw/YYYYMM/<合约>-YYYYMM.csv.gz`：Gate 历史数据下载站的月度 1 分钟 K 线（列：时间、张数、收、高、低、开，省略无成交分钟），下载过的不再请求；`raw_missing.json` 记录确认不存在的（合约, 月份）。
  - `recent/<合约>.csv`：下载站还没发布的月份用 REST 补（列：时间、开、高、低、收、计价成交额），每次只追加新分钟；`.spans` 记录每次实际覆盖的区间。
  - `contracts.json`：合约面值等（合并保存，之后下架的合约也保留）。
  - `cache/<合约>.npz`：合并后的分钟序列缓存，来源文件变化或加载规则改版（`load_data.CACHE_VERSION`）时自动重建。连续一整天以上完全没有成交的空档（停牌、下架后重新上线）按无数据处理。
  - `results/`：回测与研究的输出（汇总、逐笔明细、导出的研究数据、网格结果）。
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

## 数据下载与更新

```powershell
& $P research/launch_trade/update_data.py            # 默认取约半年前的月份起
& $P research/launch_trade/update_data.py --start 202409 --workers 16   # 两年；补很多月份时调高下载并发
```

- 已结束的月份从下载站取，没有计价成交额，加载时按 张数 × 合约面值 × 典型价 估算；
- 下载站还没有的月份用 REST 补，带精确成交额。REST 只保留最近 10000 根 1 分钟 K 线（约 6.9 天），**至少每周更新一次**，否则会留下缺口（回测把缺口当作无数据，不判定、不持仓）；
- 下载站发布新月份后再运行一次，自动下载并以月度文件为准。

## 正式回测：launch_trade_backtest.py

与实时监控同一套判定：规则取 `config/default.yaml` 的 `launch` 段，合约池按 `gate.universe_refresh_seconds` 定时刷新、24 小时成交额严格大于下限才入池（已在池中的按 `universe_exit_volume_ratio` 放宽）。判定用 numpy 向量化预筛后按时间顺序走与 `LaunchDetector` 相同的冷却/波次状态机，`--validate` 可与真检测器逐条对照：

```powershell
& $P research/launch_trade/launch_trade_backtest.py --validate RLC_USDT
```

交易参数（百分比都写数字，如 `--stop-loss 7` 表示 7%）：

| 参数 | 含义 |
| --- | --- |
| `--days` | 回测最近多少天的提醒，默认 183（半年），一年 365、两年 730、三年 1096（中间有 2024-02-29，1095 会晚一天开始）；更早的数据只用于预热 |
| `--hold-hours` | 最长持仓小时数，到时按收盘价平仓 |
| `--stop-loss` | 固定止损 |
| `--take-profit`、`--take-fraction` | 止盈；只平一部分时剩余仓位继续按回撤止损跑 |
| `--trail`、`--trail-atr` | 从持仓期最高价回撤平仓；`--trail-atr k` 把回撤幅度放宽到至少「入场前 60 分钟平均振幅 × k」 |
| `--entry-delay`、`--confirm` | 提醒后等 N 分钟，那一分钟开盘价比提醒价至少高 X% 才买，否则放弃 |
| `--min-dip` | 等待期最低价须比提醒价至少低这么多（先回踩）才买 |
| `--stop-wait-low`、`--stop-max` | 止损放在等待期最低价下方这么多，离入场价最多 `--stop-max` |
| `--max-extension` | 入场价比启动前（窗口开始前一根收盘）高出超过这么多就不追 |
| `--fee` | 单边手续费，默认 0.05 |
| `--all-waves` | 月度与逐笔明细按所有波次统计（默认只看第 1 波） |
| `--no-pool` | 不按合约池筛选 |

成交口径：入场按那一分钟开盘价；每分钟开盘就越过止盈价的，先按开盘价止盈；再看亏损的线（止损、回撤，开盘跳空按开盘价，否则按线价），最后看盘中止盈；回撤线只用上一分钟为止的最高价；开盘之后同一分钟两条线都碰到时按保守的一边。不计资金费率与滑点。

结果写入 `data/backtest/results/launch_first_wave_*`（或 `launch_all_waves_*`）。

## 研究脚本

为了一组参数几毫秒地试大量方案，研究脚本先把提醒前后的 K 线导出，再用向量化 / numba 模拟。第二轮（`r2_*`，float64）与正式回测逐笔一致，FINDINGS.md 第二轮排名中的 A、B、D、F、G 已用正式回测复核，结果相同；第一轮（`r1_*`）的价格路径按 float32 存储，止损、回撤很紧时价格恰好压线的分钟可能判得不同，合计会差几 USDT（如止损 5% / 回撤 5% / 1 小时：-39 对正式回测的 -29），只用于找方向。

| 脚本 | 内容 |
| --- | --- |
| `r1_export.py`、`r1_lib.py` | 第一轮：导出提醒后 6 小时价格路径；向量化模拟固定止损/止盈/回撤与确认入场 |
| `r1_grid.py --round 1/2` | 第一轮网格（4~7 月训练、8 月起检验），发现「确认入场 + 回撤止损」 |
| `r2_export.py`、`r2_lib.py` | 第二轮：导出提醒前 24 小时、后 12 小时的开高低收与成交额及各类价位；numba 模拟器支持全部入场确认与出场方式，`fold_sums` 按 4~5 月、6~7 月、8 月起三段评估 |
| `r2_single.py` | 单因素扫描（量能、压力/支撑、不追高、BTC、止损/回撤/持仓、保本、时间止损、量能衰竭、止盈……） |
| `r2_entry_modes.py` | 突破等待期高点追入、回踩支撑挂单 |
| `r2_filters.py` | 按提醒时的量价特征分档 |
| `r2_exits.py` | 收盘价回撤、收盘跌破支撑离场 |
| `r2_combo.py` | 组合网格（按训练期排序，展示过拟合） |
| `r2_robust.py` | 稳健组合搜索（三段都超过基准） |
| `r2_dip.py` | 回踩阈值的单调性与最稳组附近的平稳性 |
| `r2_more.py` | 部分止盈、所有波次、成本压力、第二次入场、回踩守突破位、回踩缩量、再加过滤 |
| `r2_delays.py` | 各项改进在 12 种等待设置下是否都胜过基准 |
| `r2_final.py --period half/year/two_years/three_years` | 最终排名：14 个方案在半年（选参数用）、一年、两年、三年（检验）上的结果；三年需先用 `update_data.py --start 202309` 补数据 |
| `check_jumps.py` | 检查价格整体跳变（改面值、停牌后重新上线、旧名字给了新币），确认数据可用 |
| `r3_export.py` | 第三轮：与 `r2_export.py` 相同，但提醒后保留的分钟数可调（`--post`，默认 48 小时），`--set 字段=值` 覆盖 `launch` 段、`--min-volume` 覆盖合约池下限，导出到 `results/<--out>` |
| `r3_market.py` | 第三轮：全市场逐小时收盘价与成交额（上涨币占比、活跃度等市场状态的原料），输出 `results/r3_market.npz` |
| `r3_lib.py` | 第三轮：方案定义（`Scheme`）、出场 X、邻域平均、限同时持仓、资金费率敏感性 |
| `r3_final.py` | 第三轮最终排名：三年里比 J 更赚钱的 18 个方案 |
| `r3_studies.py <研究>` | 第三轮排名表以外的研究：波次、出场网格、持仓、出场形态、特征与时段分档、前推检验、提醒门槛变体、突破拆分、随机对照、确认扫描、市场状态、滚动重选、做空（子命令见脚本开头） |

运行顺序：`update_data.py` → `r1_export.py` / `r2_export.py` → 其余脚本。

第三轮（三年，先用 `update_data.py --start 202309` 补数据）要三份导出，每份约 2 分钟、2~3.5GB：

```powershell
& $P research/launch_trade/r3_export.py --post 1200 --out v_base
& $P research/launch_trade/r3_export.py --post 1200 --out v_nobreak --set require_breakout=false
& $P research/launch_trade/r3_export.py --post 1200 --out v_nb_pool0_noover --set require_breakout=false --set max_change_percent=none --min-volume 0
& $P research/launch_trade/r3_final.py
```

`r3_studies.py` 的多数研究用 48 小时的默认导出 `r3`（`r3_export.py` 不带参数），`control`、`market` 还要先运行 `r3_market.py`，`variants` 要先按其中 `VARIANTS` 的目录名导出各门槛变体（如 `--out v_step3 --set realert_step_percent=3`）。

`r2_lib.load(period, source)` 可以读任意一份导出；模拟器按导出的列数判断持仓是否超出数据，12 小时（r2）与更长的导出通用。
