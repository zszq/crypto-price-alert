# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Gate.io 虚拟币 USDT 永续合约的实时价格异动监控服务（Python 3.11+）。只监控和提醒，不交易。用户可见文本、日志、注释和 README 均为简体中文。

## 常用命令

所有命令都应在仓库根目录执行：`config/default.yaml`、`data/price-alert.lock`、`data/alerts/alerts.jsonl` 都是相对当前工作目录解析的。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest                                   # 全部测试（离线）
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py            # 单个文件
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py -k cooldown # 单个用例
.\.venv\Scripts\python.exe -m pytest --cov=price_alert                 # 覆盖率
.\.venv\Scripts\ruff.exe check src tests                               # lint（E/F/I/UP/B，行宽 120）

.\.venv\Scripts\python.exe -m price_alert.cli check-config  # 校验配置
.\.venv\Scripts\python.exe -m price_alert.cli simulate      # 合成行情跑一遍检测，不联网、不写文件
.\.venv\Scripts\python.exe -m price_alert.cli universe      # 联网列出当前合约池
.\.venv\Scripts\python.exe -m price_alert.cli replay QNT --start "2026-09-29 09:25" --end "2026-09-29 09:36"  # 历史成交逐秒回放，解释为什么提醒/没提醒（--all 输出每一秒）
.\.venv\Scripts\python.exe -m price_alert.cli trend-replay ARK --start "2026-09-30 21:30" --end "2026-10-01 00:30"  # 用历史 1 分钟 K 线回放趋势提醒（--all 输出每根 K 线）
.\.venv\Scripts\python.exe -m price_alert.cli launch-replay RLC --start "2026-10-05 17:30" --end "2026-10-06 03:00"  # 用历史 1 分钟 K 线回放放量启动提醒（--all 输出每一分钟）
.\.venv\Scripts\python.exe -m price_alert.cli run           # 实时监控（或双击 start-monitor.bat）
```

`run` 持有 `data/price-alert.lock` 进程锁，已有实例运行时再次启动会直接退出。

## 架构

数据流：`universe.py` 筛选合约池 → `gate.py` REST 拉 K 线预热 → `gate.py` WebSocket `futures.trades` 推送成交 → 各检测器判定（`detector.py` 秒级 ATR 异动、`trend/` K 线形态趋势、`launch/` 放量启动）→ `notifier.py` 分发（控制台 / JSONL / Webhook）。`service.py::run_monitor` 负责把这些串起来。

### 服务主循环（service.py）

- 服务同时驱动多个检测器，只依赖 `detection.Detector` 约定（`symbols`、`stale_symbols`、`add_symbol`（带报价精度与合约面值 `quanto_multiplier`，用不到的检测器忽略即可）、`remove_symbols`、`mark_stream_gap`、`mark_symbol_gap`、`resync_symbol`、`add_tick`），检测器列表由 `assembly.build_detectors` 给出，每个包装成 `MonitoredDetector`（名称、预热 K 线周期与根数、成交额范围）。同一周期的检测器共用一次 REST 请求（取最多的根数，再各自截取尾部，与单独请求结果相同，且只为需要该合约预热/回补的检测器请求）；`MonitoredDetector.volume`（`universe.VolumeRange`）是各检测器自己的成交额范围，每个检测器各自筛选、增删合约池，行情订阅取并集（`_Monitor.symbols`），检测器忽略不在自己池里的成交；每笔成交依次交给所有检测器，单个检测器抛出异常只跳过这笔成交，不断线、不影响其他检测器：出错可能发生在更新状态的中途，所以由 `_DetectorFaults` 记账后调用该检测器的 `mark_symbol_gap` 把这个合约标记失效、并在没有回补任务时启动 `resync_stale_symbols` 用 REST K 线重建（同一检测器同一合约 60 秒内最多修复一次；带堆栈的错误日志每个检测器 60 秒最多一次；出错次数附在「监控正常」状态日志里，汇报后清零）；断线时全部 `mark_stream_gap`，回补覆盖任一检测器失效的合约。新增/删除一种检测器只改 `build_detectors`，不改服务。
- 检测器、REST 客户端、订阅、配置和预热信号量这组共享依赖装在 `_Monitor` 里，各循环是它的方法；首次初始化、断线回补与重连三处的指数退避共用 `_Backoff`。
- `run_monitor` 在 `TaskGroup` 中并行跑两个任务：一个先用 `initial_universe` 带退避地完成首次合约池初始化、再进入 `universe_loop` 定期刷新；另一个等 `_Monitor.subscribed`（首次订阅到合约时置位）后进入 `stream_loop` 维持 WebSocket 长连接。预热与回补都按 `_Monitor._tiers` 分批：合约归入需要它的最靠前的检测器（`build_detectors` 的顺序：ATR、趋势、放量），`sync_universe` 每预热完一批（最后一批除外）就 `_subscribe` 一次，所以启动几秒后 ATR/趋势就开始收成交，不必等放量启动的几百个合约；`resync_stale_symbols` 按同样顺序排队。状态日志里的合约数是实际订阅数。合约池变化通过 `GateTradeFeed.set_symbols` 增量订阅/退订，**不会断线**。
- `stream_loop` 中任何异常（包括握手/TCP 超时抛出的 `TimeoutError`）都按故障处理：先 `detector.mark_stream_gap()` 清空秒级窗口并把全部合约的 ATR 标记为失效，再按指数退避重连；收到第一笔成交后退避重置；若此前断过线，会再次 `mark_stream_gap()`（覆盖断线退避期间合约池刷新新增的合约），并在后台启动 `resync_stale_symbols` 为失效合约回补 K 线（失败按退避重试，断线时取消）。回补必须在实时成交恢复后发起，这样请求前的缺口由 REST 覆盖、请求后的成交由实时流覆盖。`GateTradeFeed` 的接收超时被转换成 `ConnectionError`。
- `GateTradeFeed` 的订阅请求带自增 `id`（Gate 会原样回传），Gate 一批中只要有一个无效合约就整批失败，所以批量失败时会逐个重订阅，被拒绝的合约记录在 `rejected_symbols`。订阅发送与集合变更由 `_subscription_lock` 串行化。
- 单笔坏成交或无法解析的消息只跳过并限流打日志（`_ThrottledLogger`），不能向上抛出导致断线。
- `sync_universe` 按检测器分别做增量同步：移除的合约丢弃检测状态（冷却记录保留），保留的合约只更新成交额，新增合约并发（`warmup_concurrency`）拉 K 线预热（同一合约同时新进多个检测器时只预热一次）；只有所有检测器都选不出合约时才按刷新失败处理；`_split_candles` 把已收盘 K 线用于 seed ATR，未收盘的当前 K 线作为 `live_candle` 初始化实时 K 线。预热失败的合约仍加入，靠实时 K 线自行就绪。
- 提醒通过 `AlertDispatcher.publish` 非阻塞投递，每个通道独立队列和后台任务，绝不能在成交循环里 `await` 通知发送。
- `GateRestClient` 是同步 `urllib` 实现，异步代码中通过 `asyncio.to_thread` 调用。只重试 408/429/5xx 与网络/读取/JSON 解析类临时错误，其他 4xx 直接抛出。

### 检测器（detector.py + windows.py）

检测器持有一组有序的 `MoveWindow`（`windows.py`，目前是 `short` 与可关闭的 `long`），每个窗口有独立的 lookback、门槛、确认秒数、最低笔数和冷却时长；所有窗口共用同一份秒级桶与 ATR。每个合约一个 `_SymbolState`，每个窗口一个 `_Candidate` 确认进度。核心语义：

- 成交按秒聚合为 `_SecondBucket`（VWAP，零成交量时退化为算术均价）。**只有在下一秒的第一笔成交到达时才会评估刚结束的那一秒**，所以没有新成交就不会产生判定；测试和 `simulate` 都要多送一秒的成交来“结算”最后一个桶。
- 两笔成交之间的空秒（不超过最短窗口的 `lookback_seconds` 个）会用最后成交价生成补齐桶（`_SecondBucket.carried`）并依次评估：补齐桶可以推进确认计数，但**只有真实成交的秒才能触发提醒**。空档更长则不补齐。
- 空档前那个真实秒是被迟到的成交结算的，它的 VWAP 已经过期，因此还要用这笔成交价按同一套门槛（`MoveWindow.exceeds_thresholds`）复核方向与幅度（`_still_moving`），不成立就只推进确认计数、不提醒——否则价格已回落时仍会发出携带旧价格的提醒。紧邻结算只迟一秒，不复核，以免新一秒的单笔离群成交否掉本该发出的提醒。
- 乱序（时间早于上一笔）的成交直接丢弃。
- ATR 由 `indicators.WilderAtr` 计算：预热 K 线 seed 后，实时成交维护 `_LiveBar`（可由 `live_candle` 初始化），跨入下一个 K 线周期时才把上一根 bar 喂给 ATR；若跨过了多个周期，中间按 Gate 的口径补开高低收都等于上一收盘价的平线 K 线（最多 `atr_period × 4` 根）。**每一秒（含补齐秒）都只用它所在周期之前已收盘 K 线的 ATR 判定**：触发结算的成交在结算之后才计入 K 线（同一秒的后续成交也要计入）；补齐秒跨入实时 K 线之后的周期时，由 `_atr_before` 在 `WilderAtr.copy()` 副本上推演，不改实时状态，所以每笔成交处理完后的 ATR 与实时 K 线与结算顺序无关。ATR 年龄按最后计入 K 线的**收盘时间**（开盘时间 + 周期）计算，超过 `max_atr_age_seconds` 则不判定。
- 单个窗口的触发条件（全部满足）：基准桶在 `lookback_seconds` 前且间隔不超过 lookback+2 秒；窗口内成交笔数 ≥ `min_window_trades`；`|涨跌幅| ≥ min_change_percent` **且** `位移/ATR ≥ trigger_atr_multiple`；同方向连续 `confirmation_seconds` 个相邻秒满足（任一条件不满足就重置该窗口的候选）；冷却期已过。
- 多窗口互不影响（`_evaluate`）：每秒各窗口独立评估、独立提醒，同一秒多个窗口满足就各报一条（按窗口顺序，短窗口在前）；冷却按（合约, 窗口）记录 `_last_alert`，只从本窗口自己的上次提醒算起、按本窗口的冷却计时长（`indicator.short_window.cooldown_seconds`、`indicator.long_window.cooldown_seconds`，后者 ≥ 其窗口长度，配置层强制；冷却属于判定逻辑，不在 `alerts` 下）；提醒或冷却拦下后只重置该窗口自己的候选。
- 秒级桶按最长窗口保留；空秒补齐上限是**最短**窗口的 lookback，保证短窗口行为与单窗口时一致。窗口起点落在更长的空档里时，`_find_baseline` 沿用空档前最后成交价作基准，前提是空档不长于该窗口（短窗口的这类空档都已补齐，走不到这里）。基准定位用二分查找，窗口笔数用桶上的累计笔数 `trades_before` 相减，长窗口不必逐桶扫描。
- 构造时可传 `observer`，每秒每个窗口回调一条 `WindowEvaluation`（含 `Outcome` 判定结果与涨跌幅、ATR 倍数、笔数等指标），回放靠它解释原因；实时监控不传。
- `remove_symbols` 不清除冷却记录。`mark_stream_gap` 清空秒级窗口和确认进度并置 `atr_stale`：失效期间不判定、不向 ATR 喂 K 线（也不补平线），直到 `resync_symbol` 用 REST K 线重建 ATR。重建时若本地实时 K 线与交易所当前 K 线同一周期，高低点取并集；若本地已跨入新周期，交易所的“当前 K 线”按已收盘计入 ATR。`resync_symbol` 只作用于仍处于失效状态的合约。回补保留秒级桶，回补前还没结算、且早于重建后实时 K 线周期的秒，其所在周期的 ATR 已被替换，`_atr_before` 返回 None，按 `ATR_UNAVAILABLE` 放弃判定，不拿含之后周期的 ATR 充数。

### K 线形态趋势（trend/）

独立子包，只经 `assembly.build_trend_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.trend`、`notifier` 中的 `TrendAlert` 分派与 `cli` 的 `trend-replay`/`simulate` 调用即可。只描述已形成的健康单边走势，不跟踪、不预测后续。

- `pattern.py`：纯函数。`TrendRule` 描述一个周期的门槛；`evaluate_pattern(window, rule)` 对最近 `candles` 根周期 K 线判定，返回 `PatternMetrics`（含未满足的 `Condition`）。七个条件：累计涨跌幅（第一根开盘到最后一根收盘）、反向 K 线根数（实体 ÷ 本根振幅低于 `min_candle_body_ratio` 的小实体 K 线与十字星也算反向，≤ `max_counter_candles`；判定在 `_is_counter`，首尾顺势、反向后突破共用这份标记）、首尾顺势（反向 K 线不能是第一根或最后一根）、反向后突破（每段连续反向 K 线之后的第一根，收盘要严格突破这段之前那根的收盘价，没有配置项）、最大反弹比例（按收盘价从滚动极值算，起点之后先反向走也算反弹）、总实体占比（实体和 ÷ 振幅和）、最大实体占比（最大一根顺势 K 线实体 ÷ 整段累计位移，反向 K 线不计）。`PERIOD_MINUTES` 是趋势周期到分钟数的唯一映射；`aggregate_candles` 把对齐的 1 分钟 K 线合成周期 K 线。
- `detector.py::TrendDetector`：由成交聚合 1 分钟 K 线，**下一分钟第一笔成交到达时**才结算上一分钟；空分钟按上一收盘价补平线。保留最长周期所需的 `candles × 分钟数` 根 1 分钟 K 线（预热请求 `warmup_candles` 为其 + 1）。每结算一分钟，对收盘时刻落在周期边界的各周期判定。**每个周期各自独立**：`_holding` 记录各周期上一次判定时成立的方向，形态从不成立（或反向）变为成立时提醒，持续成立时按 `trend.alert_continuing`（默认开启）发 `continuing=True` 的延续提醒（`CONTINUING_ALERT`）或只记 `CONTINUING`，形态被破坏（判定不成立、K 线不足或没有方向）即清空，再次成立就是新的一段；不同周期互不影响，同一分钟多个周期新成立会各报一条（短周期在前）。`_holding` 不随合约移出清除。`add_candle` 直接喂已收盘 1 分钟 K 线（回放、模拟用），不能与 `add_tick` 混用。断线：`mark_stream_gap` 后只维护当前分钟、不结算不判定；`resync_symbol` 用 REST 1 分钟 K 线重建历史（当前分钟与交易所同一分钟时高低点取并集；本地已跨入新分钟则交易所当前 K 线按已收盘计入），`_holding` 保留。
- `alerts.py`：`TrendAlert`（方向、周期、根数、起止时间与 `PatternMetrics`）与文本格式；`to_dict()` 带 `kind: "trend"`。
- `config.py`：`TrendConfig`（`enabled`、`alert_continuing`、合约池成交额范围 `min_volume_24h_quote`/`max_volume_24h_quote` 与 `periods`）；各周期是 `TrendPeriodConfig` 的子类（`Trend3mConfig` 等），只改默认值，YAML 键为 `3m`/`5m`/`15m`（alias；1 分钟周期不提供，`pattern.PeriodName` 仍保留 `1m` 供检测器与测试直接使用），部分覆盖时其余字段仍取该周期默认值；校验 `max_counter_candles < candles`、设置了上限时 `max_volume_24h_quote > min_volume_24h_quote`。
- `replay.py`：`cli trend-replay` 用 REST 1 分钟 K 线（`gate.fetch_minute_candles` 分段请求，每段 ≤ 1000 根，与放量回放共用）驱动同一个检测器，`plan_trend_replay` 预先检查预热起点是否超出 Gate 只保留的最近 10000 根 1 分钟 K 线（测试须传入固定的 `now`）；`simulate.py` 为每个启用周期各跑一段合成单边行情，每个周期应恰好提醒一次。

### 放量启动（launch/）

独立子包，只经 `assembly.build_launch_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.launch`、`detection.Alert` 与 `notifier` 中的 `LaunchAlert` 分派、`cli` 的 `launch-replay`/`simulate`/`check-config` 调用即可（`Detector.add_symbol` 的 `quanto_multiplier` 参数可保留）。面向启动前成交极少、进不了另两者合约池的币，默认 `launch.min_volume_24h_quote` 为 0，监控全部虚拟币合约。

- `rule.py`：纯函数。`LaunchRule` 是全部门槛；`evaluate_launch(history, rule)` 对最近 `baseline_minutes + window_minutes` 根连续 1 分钟 K 线判定，返回 `LaunchMetrics`（含未满足的 `Condition`）。条件：窗口成交额 ≥ `min_window_quote`、量比（窗口成交额 ÷ 基准期同长度**均值**，基准为 0 时 `volume_ratio=None` 视为满足）≥ `min_volume_ratio`、窗口开始前一根收盘到最后一根收盘的涨跌幅 ≥ `min_change_percent`、`require_breakout` 时收盘越过基准期最高价/最低价、未开启 `alert_drops` 时下跌记为 `DIRECTION`。
- `detector.py::LaunchDetector`：成交额 = 张数 × `quanto_multiplier` × 价格，面值为 None 的合约保留在池里但不聚合、不判定。与趋势检测器相同，**下一分钟第一笔成交到达时**结算上一分钟，空分钟按上一收盘价补成交额为 0 的平线；补齐的多根 K 线**逐根追加、逐根判定**（`_filled` + `_close_minute`），判定某一分钟时看不到它之后的 K 线。保留 `history_minutes` 根，预热请求 `warmup_candles` 为其 + 1。重复提醒按（合约, 方向）记在 `_last_alerts`（不随合约移出清除）：距上次不足 `cooldown_minutes` → `COOLDOWN`（先于基准期重置判断，冷却长于基准期也生效）；距上次已满一个基准期则重新从第 1 波算；否则价格须比上次提醒价再推进 `realert_step_percent` → 不够为 `NOT_EXTENDED`，满足则 `wave + 1`。隔了空档才结算的分钟（触发结算的成交不在紧邻的下一分钟）用这笔成交价经 `rule.still_launching` 复核涨跌幅与突破，不成立记 `EXPIRED`、不提醒也不记录；回放用 `add_candle` 不复核。断线处理同趋势检测器，`resync_symbol` 同一分钟时高低点取并集、成交额取两边较大值（两边成交有重叠，不能相加）。`add_candle` 供回放、模拟，不能与 `add_tick` 混用。
- `alerts.py`：`LaunchAlert`（`direction` 取自 metrics，`wave`、`label` 为「放量拉升/放量下跌」加「·第N波」），`to_dict()` 带 `kind: "launch"`；成交额按万/亿显示、基准期按小时显示，分别用 `formatting.format_amount`、`formatting.describe_minutes`。
- `config.py`：`LaunchConfig`，校验 `baseline_minutes > window_minutes`、`cooldown_minutes ≥ window_minutes`、设了上限时上限大于下限。
- `replay.py`：`cli launch-replay` 用 `gate.fetch_minute_candles`（与趋势回放共用）取 REST 1 分钟 K 线（自带计价成交额，面值传 1）驱动同一个检测器，`plan_launch_replay` 同样检查 Gate 10000 根的保留范围（测试须传入固定的 `now`）；`simulate.py` 跑一段基准横盘后放量拉升的合成行情，冷却覆盖整段，恰好提醒一次。
- 默认参数来自 6 天全市场 1 分钟 K 线回测（README「放量启动」一节有方案对比与数据），改默认值或判定逻辑时同步更新那里的数字。

### 回放（replay.py）

`cli replay` 用 Gate REST 历史成交离线重放一段行情：`plan_replay` 推算喂入起点（开始时间减最长窗口，向下对齐 K 线边界）与 K 线预热区间，`fetch_replay_data` 联网取合约精度、K 线和成交，`run_replay` 用 `build_detector(observer=...)` 驱动同一个检测器，`render_replay` 输出规则、提醒、各窗口统计与「最接近触发」的秒。回放从空白冷却开始、不模拟断线。

`GateRestClient.fetch_trades` 按 `[from, to)` 取成交：Gate 从新到旧分页、每页 1000 笔，翻页靠把 `to` 收缩到本页最早秒 + 1 并按 id 去重；整页落在同一秒时才用 `offset`（offset 在十万级会被拒绝）。REST 成交的 `create_time_ms` 实际单位是秒，只能用 `create_time`（`parse_rest_trade`），与 WebSocket 的解析不同。

### 配置（config.py + config/default.yaml）

- pydantic 模型全部 `extra="forbid"`，YAML 中出现未知键会直接报错。
- `PRICE_ALERT_WEBHOOK_URL` 环境变量覆盖 `alerts.webhook_url`。
- `config.py` 中的代码默认值固定不变，没有特别要求不要改；它只在 YAML 缺少对应项时生效，实际运行以 YAML 为准。
- `config/default.yaml` 每项注释中的「默认 X」标注的是代码默认值。调参时只改 YAML 的取值，注释里的默认值和 `config.py` 都不动，两者允许不一致；配置测试只校验 YAML 能通过校验。
- README 中的参数、例子和回放结果一律按代码默认值写（复现时给 `--config` 传只含 `{}` 的配置文件），调 YAML 不需要改 README；只有改代码默认值或判定逻辑时才同步更新。
- `IndicatorConfig` 只放各窗口共用的 K 线/ATR 参数（`candle_interval`、`atr_period`、`warmup_candles`、`max_atr_age_seconds`），窗口参数在结构相同的 `short_window`（`ShortWindowConfig`）与 `long_window`（`LongWindowConfig`，多一个 `enabled`）里，二者继承 `WindowConfig`，只在默认值、取值范围和冷却规则上不同。校验：`warmup_candles > atr_period`、`max_atr_age_seconds ≥ 2 个 K 线周期`（未显式设置时自动取 3 个周期）；`WindowConfig` 校验确认秒数 ≤ 窗口长度；启用长窗口时 `long_window.lookback_seconds > short_window.lookback_seconds`；`LongWindowConfig` 额外校验冷却 ≥ 窗口长度（未显式设置时等于窗口长度）。`GateConfig` 校验 `reconnect_max_seconds ≥ reconnect_initial_seconds`。

### 新增/修改检测参数时需要同步的位置

`AtrMoveDetector` 统一由 `assembly.build_detector` 组装、`TrendDetector` 由 `assembly.build_trend_detector` 组装（`run_monitor`、`cli.simulate`、`replay` 共用，REST 客户端同样由 `assembly.build_rest_client` 组装），窗口参数由 `assembly.build_windows` 从配置生成。各入口只依赖 `assembly`，不为了拿检测器去依赖实时编排模块 `service`。新增指标参数需要同时改 `config.py` 模型、`config/default.yaml`、`build_windows`/`build_detector`、README 的配置说明。`simulate` 为每个启用的窗口各用一个只含该窗口的检测器跑一遍，再为每个启用的趋势周期、放量启动各跑一遍，正常时提醒数等于窗口数加趋势周期数再加 1（放量启动启用时，`cli.expected_simulated_alerts`）。趋势参数改在 `trend/config.py`（周期子类的默认值）、`config/default.yaml` 的 `trend` 段、README；放量参数改在 `launch/config.py`、`LaunchRule`、`config/default.yaml` 的 `launch` 段、README。K 线周期到秒数的映射只有 `config.INTERVAL_SECONDS` 一份。

### 合约池筛选（universe.py）

筛选结果 `ContractTicker` 带 `price_decimals`（由 `gate.price_decimals` 把合约的 `order_price_round` 换算成小数位数，仅用于展示；回放取精度也用它），经 `detector.add_symbol` 传到 `PriceAlert`；还带 `quanto_multiplier`（`gate.quanto_multiplier` 解析合约面值，非正或无法识别为 None），供放量检测把成交张数换算成计价成交额。只接受 `contract_type == ""`（字段必须存在且为空，非币类资产会有分类值）、`status == "trading"`、`in_delisting` 不为真、以 `_USDT` 结尾、`volume_24h_quote` **严格大于**下限、且不超过可选上限（含等于）的合约。秒级异动的下限是 `gate.min_volume_24h_quote`，趋势提醒的范围是 `trend.min_volume_24h_quote` / `trend.max_volume_24h_quote`（上限默认不设），放量启动是 `launch.min_volume_24h_quote` / `launch.max_volume_24h_quote`（下限默认 0），由 `assembly.build_detectors` 装进各自的 `VolumeRange`。已在监控中的合约（`retained_symbols`，按各检测器自己的池）退出线两侧放宽：下限 × `universe_exit_volume_ratio`、上限 ÷ 该比例。`volume_24h_usd` 已被 Gate 弃用，仅作缺字段时的回退。`is_internal=true` 的成交在 `gate.parse_trade_payload` 中被忽略。

### 通知（notifier.py）

提醒类型是 `detection.Alert = PriceAlert | TrendAlert | LaunchAlert`，`format_alert`/`colorize_alert` 按类型分派（趋势提醒的标签用亮紫色 `TREND_LABEL_COLOR`，放量启动用亮蓝色 `LAUNCH_LABEL_COLOR`）。`AlertDispatcher` 为每个通道建立独立的有界队列（`alerts.queue_size`）和后台任务：`publish` 非阻塞，队列满时丢弃并记错误日志；单个通道失败只记日志；退出时最多等待 `drain_timeout` 秒把积压发完。`build_notifiers` 按配置返回通道列表。`JsonlNotifier` 按 `jsonl_max_bytes` 整文件轮转。价格、窗口长度（整分钟显示为「N分钟」）和北京时间的格式都在 `formatting.py`，提醒与回放共用；价格按合约报价精度显示，缺失时按 8 位有效数字、不截断整数部分。提醒时间统一转为北京时间；控制台急涨绿色、急跌红色，涨跌幅用同方向亮色高亮，长窗口提醒的窗口长度用不分方向的亮青色（`LONG_WINDOW_COLOR`）高亮（短窗口保持正文色）；JSONL/Webhook 保留完整结构化字段（`PriceAlert.to_dict()`）。控制台提醒附带的交易地址由 `gate.futures_trade_url` 生成，Gate 网页地址只维护在 `gate.py`。修改提醒文本格式时注意 README 中的示例。

## 约定

- 测试必须确定、离线；文件名 `tests/test_<module>.py`，覆盖边界条件、异常交易所数据、冷却、通知失败等场景。
- 修改 `config/default.yaml` 中的阈值属于行为变更，提交说明中需写明运营影响。
- 提交信息风格：简洁祈使句，常用 `config:`、`docs:`、`feat(...)`、`chore:` 前缀，也可用中文标题。
