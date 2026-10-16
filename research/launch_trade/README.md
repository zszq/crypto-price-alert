# 放量启动提醒的交易回测

研究「放量启动」提醒（`src/price_alert/launch/`）能不能直接拿来做多（第五轮还试了做空下跌放量；第八轮不看交易，比较放量上涨与放量下跌两个方向的提醒参数）：提醒后买入，按不同的入场确认、止损、回撤、止盈与持仓时间平仓，统计半年的收益。**结论见 [FINDINGS.md](FINDINGS.md)。**

只是研究代码，不参与实时监控，也不在 `ruff check src tests` 与 pytest 的范围内（提交前可单独运行 `ruff check research`）。

## 目录约定

- 代码：本目录，进版本库；所有路径由 `paths.py` 统一给出。
- 数据：`data/backtest/`，**不进版本库**（`.gitignore` 已忽略），两年数据合计约 7.5GB（行情约 2.4GB、缓存约 3.4GB、研究导出与结果约 1.5GB、虚拟环境约 0.2GB）。删掉后运行 `update_data.py` 可重新下载月度文件，但 REST 补的部分只能取回最近约 7 天，更早的要等下载站发布那个月的文件：
  - `raw/YYYYMM/<合约>-YYYYMM.csv.gz`：Gate 历史数据下载站的月度 1 分钟 K 线（列：时间、张数、收、高、低、开，省略无成交分钟），下载过的不再请求；`raw_missing.json` 记录确认不存在的（合约, 月份）。
  - `recent/<合约>.csv`：下载站还没发布的月份用 REST 补（列：时间、开、高、低、收、计价成交额），每次只追加新分钟；`.spans` 记录每次实际覆盖的区间。
  - `contracts.json`：合约面值等（合并保存，之后下架的合约也保留）。
  - `cache/<合约>.npz`：合并后的分钟序列缓存，来源文件变化或加载规则改版（`load_data.CACHE_VERSION`）时自动重建。连续一整天以上完全没有成交的空档（停牌、下架后重新上线）按无数据处理。
  - `oi_1h/`、`oi_5m/<合约>.npz`：历史持仓（`oi_fetch.py`），REST 只保留最近 180 天，过期的取不回来，删掉就只能从最近 180 天重新开始。
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
& $P research/launch_trade/update_data.py --start 202109 --workers 16 --discover   # 五年，并补进期间已下架的币
```

- 下载站的 1 分钟 K 线最早到 2020-01（BTC_USDT；各合约从上线月份开始），另有 5 分钟 / 1 小时 K 线、逐笔成交（`trades`）、资金费率结算（`funding_applies`）、标记价格（`mark_prices`），本研究只用 1 分钟 K 线；
- `--discover`：合约列表默认只取当前还在交易的合约（再合并此前下载过的），期间下架的币不会进样本。加上它会按下载站背后 S3 存储桶的月度文件列表，把 `--start` 以来存在过、合约列表里没有的币类合约（合约详情接口对已下架合约仍返回面值，按 `contract_type` 为空筛掉股票等）补进 `contracts.json` 再下载。只算币类合约，2021-10 ~ 2025-04 每个月有 39%~47% 原来不在样本里（全是之后下架的），2025-10 约 28%，2026 年以后 2%~11%；补齐后 1145 个合约（原 588 个），各月全部覆盖。面值按当前值，期间改过面值的合约要用 `check_jumps.py` 检查；
- 已结束的月份先按存储桶的月度文件列表把不存在的（合约, 月份）记入 `raw_missing.json`，只请求实际存在的文件（补五年时请求从约 4.4 万降到约 1 万）；列表取不到时退回逐个请求；

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
| `--hot-breadth`、`--hot-core-only` | 市场过热（全市场站上 30 天均线的合约占比，取提醒前最后一个已收盘小时）超过这个百分比时不买；`--hot-core-only` 只跳过 J 类提醒（已突破基准期高点、24h 成交额 > 100 万、入场价比提醒价高 ≥ 2%）。需先运行 `r3_market.py` |
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
| `r3_lib.py` | 第三轮：方案定义（`Scheme`）、出场 X、过热过滤（全市场站上 30 天均线占比）、邻域平均、限同时持仓、资金费率敏感性 |
| `r3_final.py [--period three_years/five_years]` | 第三轮最终排名：12 个方案（含过热过滤的 J4、P5；含时段过滤的方案是过拟合，已删除）；五年要先用 `update_data.py --start 202109 --discover` 补数据 |
| `r3_studies.py <研究>` | 第三轮排名表以外的研究：波次、出场网格、持仓、出场形态、特征分档、前推检验、提醒门槛变体、突破拆分、随机对照、确认扫描、市场状态、滚动重选、做空、亏损期与过热过滤（子命令见脚本开头） |
| `r5_export.py` | 第五轮：两个方向的提醒（`--side up/down/both`，下跌方向与开启 `alert_drops` 的 `LaunchDetector` 一致），只留提醒前 60、后 `--post`（默认 1440）分钟的开高低收，提醒时的特征写进 meta；`--set`、`--min-volume` 同 `r3_export.py`；`--ratio-by base` 量比改按成交量（币数）算（默认 `quote` 与检测器一致）；`--float32` bars 存 float32（第六轮导出 15 天时用，体积减半） |
| `r5_lib.py` | 第五轮：多空双向逐笔模拟（与 `simulate_trade` 逐笔一致）、限仓组合（先到先买、同一币不叠仓）、季度 / 去掉最好 N 天 / 夏普 / 回撤，全市场状态（宽度、BTC 均线、中位涨跌、全市场放量） |
| `r5_studies.py <研究>` | 第五轮：以稳定为目标的方案评估，开发期（2024-10 起）与样本外（2021-10 ~ 2024-09）分开给出；子命令 base、exits、slots、regime、filters、quality、core、hood、costs、hot、final |
| `r5_scan.py <数据集>` | 第五轮：提醒特征分档，每档在五年各半年段每笔为正的段数 |
| `r5_regime.py` | 第五轮：开发期多空两个方向的市场状态分档（找做空何时有效时用） |
| `r5_ratio.py diff/scan <前缀>` | 量比按成交额还是成交量：同门槛下的提醒差异与 S5 结果（`diff`），两种口径 × 量比门槛 2.5~3.5（`scan`）；导出命令见 FINDINGS.md 对应一节 |
| `r5_rank.py [--dataset 数据集]` | 历次方案统一重排：第三、四轮里不含时间段条件的 12 个方案与第五轮 S5 等共 19 个，按第五轮口径排名；`--dataset` 只用一份提醒（如 `r5_nobreak`）比较各交易规则；需先导出 `r5_nobreak`（命令见 FINDINGS.md 对应一节） |
| `oi_fetch.py <数据集...>` | 取导出里最近 180 天上涨提醒涉及合约的历史持仓（REST `contract_stats`，只保留 180 天）：1 小时粒度整段存 `oi_1h/`，5 分钟粒度只取提醒前 90 分钟存 `oi_5m/`；`--only-missing` 补上次失败的合约。**每周运行一次**才能积累超过 180 天的数据 |
| `r5_oi.py <研究> [数据集...]` | 持仓价值过滤：持仓特征分档（`buckets`）、各方案加持仓过滤（`filters`）、与 24h 成交额同保留比例对照（`proxy`）、随机对照（`control`）、五年 24h 成交额对照（`volume`）、J4 与 P5（`jp`）；结论见 FINDINGS.md「持仓价值过滤」 |
| `r6_lib.py` | 第六轮：不按时间平仓的做多模拟器，出场线为初始止损、移动止盈（盘中或按时钟对齐的 N 分钟收盘价触发，可另设盘中硬回撤；回撤幅度可按入场前或持仓中近期振幅放宽、浮盈后改变）、锁定浮盈、保本中最高的一条；S5 参数下与 `r5_lib` 逐笔一致 |
| `r6_studies.py <研究...> [--dataset r6_p30]` | 第六轮：提醒只取到数据结束前 15 天、持满 15 天市价平仓；单因素（base、trail、dyn、start、tight、lock、be、close、closetf、slots、entry、stop）、组合网格（`grid`）、12 个入场设置平均（`robust`）、候选邻域与回撤（`hood`、`drawdown`）、推荐 T6 的平原与最终检验（`plateau`、`final`）；一次可跑多个研究（数据近 10GB，只加载一次）；结论见 FINDINGS.md「第六轮」 |
| `r7_studies.py <研究...> [--dataset r5_p30]` | 第七轮：持仓 ≤ 60 分钟的超短线，用 `r6_lib` 的出场、第五轮的导出（或 `r5_export.py --post 180` 的 `r7_*` 变体）；出场网格（`grid`）、持仓上限（`hold`）、收紧手段（`extras`）、平原（`plateau`）、入场（`entry`）、信号与提醒前特征过滤（`signal`、`features`）、限仓、成本与按 24h 成交额分档的滑点（`slots`、`costs`、`liquidity`）、推荐 U7 的单点与逐年（`final`）；`signals --datasets ...` 比较多份提醒导出；结论见 FINDINGS.md「第七轮」 |
| `r8_sides.py <命令> [--set single/combo/final]` | 第八轮：放量上涨与放量下跌能否共用提醒参数，不看交易、只看提醒质量：直接读分钟序列，同一合约上一次跑多组 `LaunchRule`（两个方向，下跌与开启 `alert_drops` 的检测器一致），算对数对称的 ±3/5/10% 先到概率（延续率）、提醒时已走幅度与之后 4 小时顺向空间；`run` 单因素（`single`）/ 两个方向各自的组合网格（`combo`）/ 网格边界复核（`final`），`report`、`match`（与基准第 1 波同段配对看早报晚报）、`combo` 输出表格，`control` 同合约池随机时刻对照；结论见 FINDINGS.md「第八轮」 |
| `r9_pump.py <命令>` | 第九轮：合约池下限调高、最近拉升过的币，放量提醒是否更可靠；`run` 导出各合约池下限的提醒与随机时刻，附提醒前状态（24h 成交额、活跃度、1 小时 ~ 30 天涨跌、离区间高低点、振幅），`pool`、`buckets` 按下限与五分位分档比较，`tails`、`nearlow` 看拉升过 / 贴近低点的固定阈值，`long` 算 1 / 3 / 7 天，`long-report`（顺向对数收益）、`long-arith`（按币价的算术均值、中位数、下跌占比）输出；结论见 FINDINGS.md「第九轮」 |

运行顺序：`update_data.py` → `r1_export.py` / `r2_export.py` → 其余脚本。

第三轮（三年，先用 `update_data.py --start 202309` 补数据）要三份导出，每份约 2 分钟、2~3.5GB：

```powershell
& $P research/launch_trade/r3_export.py --post 1200 --out v_base
& $P research/launch_trade/r3_export.py --post 1200 --out v_nobreak --set require_breakout=false
& $P research/launch_trade/r3_export.py --post 1200 --out v_nb_pool0_noover --set require_breakout=false --set max_change_percent=none --min-volume 0
& $P research/launch_trade/r3_market.py   # 过热过滤要用的全市场逐小时数据
& $P research/launch_trade/r3_final.py
```

`r3_studies.py` 的多数研究用 48 小时的默认导出 `r3`（`r3_export.py` 不带参数），`control`、`market`、`regime` 还要先运行 `r3_market.py`，`variants` 要先按其中 `VARIANTS` 的目录名导出各门槛变体（如 `--out v_step3 --set realert_step_percent=3`）。

第五轮（五年，先 `update_data.py --start 202109`、`r3_market.py`）的导出，每份约 1 分钟、1.4~2.8GB：

```powershell
& $P research/launch_trade/r5_export.py --out r5_p30 --side up --set baseline_minutes=360 --set min_window_quote=300000 --set require_breakout=false --set max_change_percent=none --min-volume 0
& $P research/launch_trade/r5_export.py --out r5_p30_1440 --side up --set baseline_minutes=1440 --set min_window_quote=300000 --set require_breakout=false --set max_change_percent=none --min-volume 0
& $P research/launch_trade/r5_export.py --out r5_p --side up --set baseline_minutes=360 --set min_window_quote=100000 --set require_breakout=false --set max_change_percent=none --min-volume 0
& $P research/launch_trade/r5_export.py --out r5_base --side up --set baseline_minutes=360 --set min_window_quote=100000 --set require_breakout=true
& $P research/launch_trade/r5_export.py --out r5_user --side up   # 当前 config/default.yaml 的 launch 段
& $P research/launch_trade/r5_studies.py final --dataset r5_p30
```

第六轮（不按时间平仓）要提醒后 15 天的数据，每份约 3 分钟、8~9GB，float32 存储，加载需约 10GB 内存，导出命令见 FINDINGS.md「第六轮」。

`r2_lib.load(period, source)` 可以读任意一份导出；模拟器按导出的列数判断持仓是否超出数据，12 小时（r2）与更长的导出通用。
