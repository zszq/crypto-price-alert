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
.\.venv\Scripts\python.exe -m pytest tests/test_move_detector.py       # 单个文件
.\.venv\Scripts\python.exe -m pytest tests/test_move_detector.py -k cooldown # 单个用例
.\.venv\Scripts\python.exe -m pytest --cov=price_alert                 # 覆盖率
.\.venv\Scripts\ruff.exe check src tests                               # lint（E/F/I/UP/B，行宽 120）

.\.venv\Scripts\python.exe -m price_alert.cli check-config  # 校验配置
.\.venv\Scripts\python.exe -m price_alert.cli simulate      # 合成行情跑一遍检测，不联网、不写文件
.\.venv\Scripts\python.exe -m price_alert.cli universe      # 联网列出当前合约池
.\.venv\Scripts\python.exe -m price_alert.cli replay QNT --start "2026-09-29 09:25" --end "2026-09-29 09:36"  # 历史成交逐秒回放，解释为什么提醒/没提醒（--all 输出每一秒）
.\.venv\Scripts\python.exe -m price_alert.cli trend-replay ARK --start "2026-09-30 21:30" --end "2026-10-01 00:30"  # 用历史 1 分钟 K 线回放趋势提醒（--all 输出每根 K 线）
.\.venv\Scripts\python.exe -m price_alert.cli launch-replay RLC --start "2026-10-05 17:30" --end "2026-10-06 03:00"  # 用历史 1 分钟 K 线回放放量启动提醒（--all 输出每一分钟）
.\.venv\Scripts\python.exe -m price_alert.cli dormant-replay IRYS --start "2026-10-21 14:00" --end "2026-10-21 18:00"  # 用历史 1 分钟 K 线回放冷门启动提醒（--all 输出每一分钟）
.\.venv\Scripts\python.exe -m price_alert.cli reversal-replay CAP --start "2026-10-26 17:00" --end "2026-10-26 17:35"  # 用历史成交逐秒回放反弹回落提醒（--all 输出每一秒）
.\.venv\Scripts\python.exe -m price_alert.cli sounds        # 依次试听八种提示音（Windows/macOS）
.\.venv\Scripts\python.exe -m price_alert.cli run           # 实时监控（或双击 start-monitor.bat）
```

`run` 按配置文件持有进程锁（`cli.lock_path`：`default.yaml` 用 `data/price-alert.lock`，其他配置用 `data/price-alert-<文件名>.lock`），同一份配置已有实例运行时再次启动会直接退出，不同配置可同时运行。`config/launch-trade.yaml` 是交易用的放量实例（只开放量上涨，提醒与交易规则取 S5：窗口成交额 30 万、最长持 18 小时，交易规则写在文件开头注释里）；`config/launch-f1.yaml` 存待用户验证的放量参数（`research/launch_trade/launch_grid.md` 第二轮 F1，只开放量启动、单独的 JSONL）；两者都关掉了 move、trend、dormant、reversal，由 `tests/test_config.py` 校验。

## 架构

### 代码布局

`src/price_alert/` 按依赖方向分层：`common/` ← `exchange/` ← `detectors/` ← `notify/` ← 顶层。检测器的判定代码（`detector.py`、`rule.py`/`pattern.py`/`windows.py`、`alerts.py`、`config.py`）只依赖 `common/` 与其他检测器的同类模块；离线入口例外：各子包的 `replay.py`、`simulate.py` 通过顶层 `assembly` 组装检测器（与实时监控同一套组装），`move/replay.py` 还用 `notify.notifier.format_alert` 输出提醒原文。`assembly` 不能反过来导入这些 `replay`/`simulate` 模块，否则会循环导入。

- `common/`：不依赖交易所与检测器的基础。`models.py`（`Candle`、`PriceTick`、`ContractTicker` 与 K 线周期秒数 `INTERVAL_SECONDS`）、`formatting.py`、`indicators.py`、`minute_bars.py`（1 分钟 K 线检测器骨架，见下文）。
- `exchange/`：`gate.py`（REST / WebSocket 与解析）、`universe.py`（合约池筛选）、`config.py`（`GateConfig`）。
- `detectors/`：五个检测器子包 `move/`、`trend/`、`launch/`、`dormant/`、`reversal/`，各自带 `config.py`、判定、`alerts.py`、`replay.py`、`simulate.py`；`alerts.py` 汇总全部提醒类型 `Alert`；`replay.py` 是各回放共用的时间解析、报价精度与取数。子包之间互不依赖（`reversal` 继承 `move` 的检测器除外）。
- `notify/`：`notifier.py`、`sounds.py`、`config.py`（`AlertConfig`）。
- 顶层：`config.py`（`AppConfig` 汇总各段与 `load_config`）、`detection.py`（服务对检测器的约定 `Detector` / `MonitoredDetector`）、`assembly.py`、`service.py`、`instance.py`、`cli.py`（入口仍是 `python -m price_alert.cli`）。

下文的 `gate.py`、`notifier.py`、`trend/` 等都按这个布局定位。

数据流：`universe.py` 筛选合约池 → `gate.py` REST 拉 K 线预热 → `gate.py` WebSocket `futures.trades` 推送成交 → 各检测器判定（`move/` 秒级 ATR 异动、`trend/` K 线形态趋势、`launch/` 放量启动、`dormant/` 冷门启动、`reversal/` 反弹回落）→ `notifier.py` 分发（控制台 / JSONL / Webhook）。`service.py::run_monitor` 负责把这些串起来。

### 服务主循环（service.py）

- 服务同时驱动多个检测器，只依赖 `detection.Detector` 约定（`symbols`、`stale_symbols`、`add_symbol`（带报价精度与合约面值 `quanto_multiplier`，用不到的检测器忽略即可）、`remove_symbols`、`mark_stream_gap`、`mark_symbol_gap`、`resync_symbol`、`add_tick`），检测器列表由 `assembly.build_detectors` 给出（顺序与 `config/default.yaml` 一致），每个包装成 `MonitoredDetector`（名称、预热 K 线周期与根数、成交额范围）。同一周期的检测器共用一次 REST 请求（取最多的根数，再各自截取尾部，与单独请求结果相同，且只为需要该合约预热/回补的检测器请求）；`MonitoredDetector.volume`（`universe.VolumeRange`）是各检测器自己的成交额范围，每个检测器各自筛选、增删合约池，行情订阅取并集（`_Monitor.symbols`），检测器忽略不在自己池里的成交；每笔成交依次交给所有检测器，单个检测器抛出异常只跳过这笔成交，不断线、不影响其他检测器：出错可能发生在更新状态的中途，所以由 `_DetectorFaults` 记账后调用该检测器的 `mark_symbol_gap` 把这个合约标记失效、并在没有回补任务时启动 `resync_stale_symbols` 用 REST K 线重建（同一检测器同一合约 60 秒内最多修复一次；带堆栈的错误日志每个检测器 60 秒最多一次；出错次数附在「监控正常」状态日志里，汇报后清零）；断线时全部 `mark_stream_gap`，回补覆盖任一检测器失效的合约。新增/删除一种检测器只改 `build_detectors`，不改服务。
- 检测器、REST 客户端、订阅、配置和预热信号量这组共享依赖装在 `_Monitor` 里，各循环是它的方法；首次初始化、断线回补与重连三处的指数退避共用 `_Backoff`。
- `run_monitor` 在 `TaskGroup` 中并行跑两个任务：一个先用 `initial_universe` 带退避地完成首次合约池初始化、再进入 `universe_loop` 定期刷新；另一个等 `_Monitor.subscribed`（首次订阅到合约时置位）后进入 `stream_loop` 维持 WebSocket 长连接。预热与回补都按 `_Monitor._tiers` 分批：合约归入需要它的最靠前的检测器（按各检测器成交额下限从高到低，相同时按 `build_detectors` 的顺序：ATR、放量上涨、放量下跌、趋势、冷门启动、反弹回落，与 YAML 一致；默认即 ATR、趋势、反弹回落、放量、冷门），`sync_universe` 每预热完一批（最后一批除外）就 `_subscribe` 一次，所以启动几秒后 ATR/趋势就开始收成交，不必等放量启动的约 100 个、冷门启动的约 550 个合约；`resync_stale_symbols` 按同样顺序排队。状态日志里的合约数是实际订阅数。合约池变化通过 `GateTradeFeed.set_symbols` 增量订阅/退订，**不会断线**。
- `stream_loop` 中任何异常（包括握手/TCP 超时抛出的 `TimeoutError`）都按故障处理：先 `detector.mark_stream_gap()` 清空秒级窗口并把全部合约的 ATR 标记为失效，再按指数退避重连；收到第一笔成交后退避重置；若此前断过线，会再次 `mark_stream_gap()`（覆盖断线退避期间合约池刷新新增的合约），并在后台启动 `resync_stale_symbols` 为失效合约回补 K 线（失败按退避重试，断线时取消）。回补必须在实时成交恢复后发起，这样请求前的缺口由 REST 覆盖、请求后的成交由实时流覆盖。`GateTradeFeed` 的接收超时被转换成 `ConnectionError`。
- `GateTradeFeed` 的订阅请求带自增 `id`（Gate 会原样回传），Gate 一批中只要有一个无效合约就整批失败，所以批量失败时会逐个重订阅，被拒绝的合约记录在 `rejected_symbols`。订阅发送与集合变更由 `_subscription_lock` 串行化。
- 单笔坏成交或无法解析的消息只跳过并限流打日志（`_ThrottledLogger`），不能向上抛出导致断线。
- `sync_universe` 按检测器分别做增量同步：移除的合约丢弃检测状态（冷却记录保留），保留的合约只更新成交额，新增合约并发（`warmup_concurrency`）拉 K 线预热（同一合约同时新进多个检测器时只预热一次）；只有所有检测器都选不出合约时才按刷新失败处理；`_split_candles` 把已收盘 K 线用于 seed ATR，未收盘的当前 K 线作为 `live_candle` 初始化实时 K 线。预热失败的合约仍加入，靠实时 K 线自行就绪。
- 提醒通过 `AlertDispatcher.publish` 非阻塞投递，每个通道独立队列和后台任务，绝不能在成交循环里 `await` 通知发送。
- `GateRestClient` 是同步 `urllib` 实现，异步代码中通过 `asyncio.to_thread` 调用。只重试 408/429/5xx 与网络/读取/JSON 解析类临时错误，其他 4xx 直接抛出。

### 秒级异动（move/）

子包，与其他检测器一样只经 `assembly.build_detector` / `build_windows` / `build_detectors` 接入（这两个函数收 `MoveConfig`，与 `build_trend_detector` 收 `TrendConfig` 一致）：`detector.py::AtrMoveDetector`、`windows.py`（窗口规则、`Outcome`、`WindowEvaluation`）、`streak.py`（连续提醒）、`alerts.py`（`PriceAlert` 与文本格式，`notifier` 只做分派与着色）、`config.py`（`MoveConfig` 等）、`replay.py`（`cli replay`）、`simulate.py`。`reversal/` 继承 `AtrMoveDetector`，所以它不能像其他子包那样整个删掉。

检测器持有一组有序的 `MoveWindow`（`windows.py`，目前是 `short` 与可关闭的 `long`），每个窗口有独立的 lookback、门槛、确认秒数、最低笔数和冷却时长；所有窗口共用同一份秒级桶与 ATR。每个合约一个 `_SymbolState`，每个窗口一个 `_Candidate` 确认进度。核心语义：

- 成交按秒聚合为 `_SecondBucket`（VWAP，零成交量时退化为算术均价）。**只有在下一秒的第一笔成交到达时才会评估刚结束的那一秒**，所以没有新成交就不会产生判定；测试和 `simulate` 都要多送一秒的成交来“结算”最后一个桶。
- 两笔成交之间的空秒（不超过最短窗口的 `lookback_seconds` 个）会用最后成交价生成补齐桶（`_SecondBucket.carried`）并依次评估：补齐桶可以推进确认计数，但**只有真实成交的秒才能触发提醒**。空档更长则不补齐。
- 空档前那个真实秒是被迟到的成交结算的，它的 VWAP 已经过期，因此还要用这笔成交价按同一套门槛（`MoveWindow.exceeds_thresholds`）复核方向与幅度（`_still_moving`），不成立就只推进确认计数、不提醒——否则价格已回落时仍会发出携带旧价格的提醒。紧邻结算只迟一秒，不复核，以免新一秒的单笔离群成交否掉本该发出的提醒。
- 乱序（时间早于上一笔）的成交直接丢弃。
- ATR 由 `indicators.WilderAtr` 计算：预热 K 线 seed 后，实时成交维护 `_LiveBar`（可由 `live_candle` 初始化），跨入下一个 K 线周期时才把上一根 bar 喂给 ATR；若跨过了多个周期，中间按 Gate 的口径补开高低收都等于上一收盘价的平线 K 线（最多 `atr_period × 4` 根）。**每一秒（含补齐秒）都只用它所在周期之前已收盘 K 线的 ATR 判定**：触发结算的成交在结算之后才计入 K 线（同一秒的后续成交也要计入）；补齐秒跨入实时 K 线之后的周期时，由 `_atr_before` 在 `WilderAtr.copy()` 副本上推演，不改实时状态，所以每笔成交处理完后的 ATR 与实时 K 线与结算顺序无关。ATR 年龄按最后计入 K 线的**收盘时间**（开盘时间 + 周期）计算，超过 `max_atr_age_seconds` 则不判定。
- 单个窗口的触发条件（全部满足）：基准桶在 `lookback_seconds` 前且间隔不超过 lookback+2 秒；窗口内成交笔数 ≥ `min_window_trades`；`|涨跌幅| ≥ min_change_percent` **且** `位移/ATR ≥ trigger_atr_multiple`；同方向连续 `confirmation_seconds` 个相邻秒满足（任一条件不满足就重置该窗口的候选）；冷却期已过。
- 多窗口互不影响（`_evaluate`）：每秒各窗口独立评估、独立提醒，同一秒多个窗口满足就各报一条（按窗口顺序，短窗口在前）；冷却按（合约, 窗口）记录 `_last_alert`，只从本窗口自己的上次提醒算起、按本窗口的冷却计时长（`move.short_window.cooldown_seconds`、`move.long_window.cooldown_seconds`，后者 ≥ 其窗口长度，配置层强制；冷却属于判定逻辑，不在 `alerts` 下）；提醒或冷却拦下后只重置该窗口自己的候选。
- 秒级桶按最长窗口保留；空秒补齐上限是**最短**窗口的 lookback，保证短窗口行为与单窗口时一致。窗口起点落在更长的空档里时，`_find_baseline` 沿用空档前最后成交价作基准，前提是空档不长于该窗口（短窗口的这类空档都已补齐，走不到这里）。基准定位用二分查找，窗口笔数用桶上的累计笔数 `trades_before` 相减，长窗口不必逐桶扫描。
- 构造时可传 `observer`，每秒每个窗口回调一条 `WindowEvaluation`（含 `Outcome` 判定结果与涨跌幅、ATR 倍数、笔数等指标），回放靠它解释原因；实时监控不传。
- 连续提醒（`move/streak.py`）：构造时传 `streak: StreakRule`（`assembly.build_detector` 按 `move.streak` 生成，`simulate` 传 `streak=False`）后，每条**短窗口**提醒交给 `StreakTracker.observe`：同一合约同方向、相邻间隔 ≤ `StreakRule.max_gap_seconds`（= `short_window.cooldown_seconds + streak.gap_tolerance_seconds`，跟随冷却，没有跨字段校验）的短窗口提醒连成一串，第 `min_alerts` 条时紧跟其后追加一条 `StreakAlert`（`kind: "streak"`，累计涨跌幅从第一条提醒的窗口起点算），每串只报一次；累计涨跌方向与提醒方向不一致（冷却期内急跌又拉回没触发反向提醒）时先不报、计数照常累加，之后某条提醒让方向一致时再报；反向短窗口提醒或间隔超限重新计数，长窗口提醒不参与也不打断，记录不随合约移出清除。默认值（3 条、容差 30 秒即间隔 ≤ 60 秒）来自实际提醒日志，README「连续提醒」一节有数据。`replay` 的连续提醒随触发它的短窗口提醒一起按区间取舍。
- `remove_symbols` 不清除冷却记录。`mark_stream_gap` 清空秒级窗口和确认进度并置 `atr_stale`：失效期间不判定、不向 ATR 喂 K 线（也不补平线），直到 `resync_symbol` 用 REST K 线重建 ATR。重建时若本地实时 K 线与交易所当前 K 线同一周期，高低点取并集；若本地已跨入新周期，交易所的“当前 K 线”按已收盘计入 ATR。`resync_symbol` 只作用于仍处于失效状态的合约。回补保留秒级桶，回补前还没结算、且早于重建后实时 K 线周期的秒，其所在周期的 ATR 已被替换，`_atr_before` 返回 None，按 `ATR_UNAVAILABLE` 放弃判定，不拿含之后周期的 ATR 充数。

### 1 分钟 K 线检测器骨架（common/minute_bars.py）

趋势、放量启动、冷门启动三个检测器都继承 `MinuteBarDetector`，K 线聚合、空分钟补平线、断线失效与回补、`symbols`/`add_symbol`/`remove_symbols`/`mark_*_gap`/`resync_symbol`/`add_tick`/`add_candle` 都在基类，子类只实现 `_evaluate(symbol, state, closed_at, latest_price)`（`closed_at` 那一分钟已追加进 `state.minutes`，返回提醒列表）。差异用类属性表达：`requires_quote`（放量、冷门为 True：没有合约面值的合约不聚合、不判定）、`state_class`（冷门启动用带滚动累计值的 `MinuteState` 子类，覆盖 `append`/`clear`）。`latest_price` 只在隔了空档才结算时不为 None，放量、冷门用它复核，趋势忽略。下面三节里关于结算时机、补平线、回补合并的描述都是基类的行为，改动时三者一起受影响，要跑三个检测器的测试。

### K 线形态趋势（trend/）

独立子包，只经 `assembly.build_trend_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.trend`、`detectors/alerts.py` 的 `Alert` 与 `notifier` 中的 `TrendAlert` 分派、`cli` 的 `trend-replay`/`simulate` 调用即可。只描述已形成的健康单边走势，不跟踪、不预测后续。

- `pattern.py`：纯函数。`TrendRule` 描述一个周期的门槛；`evaluate_pattern(window, rule)` 对最近 `candles` 根周期 K 线判定，返回 `PatternMetrics`（含未满足的 `Condition`）。七个条件：累计涨跌幅（第一根开盘到最后一根收盘）、反向 K 线根数（实体 ÷ 本根振幅低于 `min_candle_body_ratio` 的小实体 K 线与十字星也算反向，≤ `max_counter_candles`；判定在 `_is_counter`，首尾顺势、反向后突破共用这份标记）、首尾顺势（反向 K 线不能是第一根或最后一根）、反向后突破（每段连续反向 K 线之后的第一根，收盘要严格突破这段之前那根的收盘价，没有配置项）、最大反弹比例（按收盘价从滚动极值算，起点之后先反向走也算反弹）、总实体占比（实体和 ÷ 振幅和）、最大实体占比（最大一根顺势 K 线实体 ÷ 整段累计位移，反向 K 线不计）。`PERIOD_MINUTES` 是趋势周期到分钟数的唯一映射；`aggregate_candles` 把对齐的 1 分钟 K 线合成周期 K 线。
- `detector.py::TrendDetector`：由成交聚合 1 分钟 K 线，**下一分钟第一笔成交到达时**才结算上一分钟；空分钟按上一收盘价补平线，一次补出的多根平线逐根追加、逐根判定（`_filled` + `_close_minute`），判定某一分钟时窗口里没有它之后的 K 线。保留最长周期所需的 `candles × 分钟数` 根 1 分钟 K 线（预热请求 `warmup_candles` 为其 + 1）。每结算一分钟，对收盘时刻落在周期边界的各周期判定。**每个周期各自独立**：`_holding` 记录各周期上一次判定时成立的方向，形态从不成立（或反向）变为成立时提醒，持续成立时按 `trend.alert_continuing`（默认开启）发 `continuing=True` 的延续提醒（`CONTINUING_ALERT`）或只记 `CONTINUING`，形态被破坏（判定不成立、K 线不足或没有方向）即清空，再次成立就是新的一段；不同周期互不影响，同一分钟多个周期新成立会各报一条（短周期在前）。`_holding` 不随合约移出清除。`add_candle` 直接喂已收盘 1 分钟 K 线（回放、模拟用），不能与 `add_tick` 混用。断线：`mark_stream_gap` 后只维护当前分钟、不结算不判定；`resync_symbol` 用 REST 1 分钟 K 线重建历史（当前分钟与交易所同一分钟时高低点取并集；本地已跨入新分钟则交易所当前 K 线按已收盘计入），`_holding` 保留。
- `alerts.py`：`TrendAlert`（方向、周期、根数、起止时间与 `PatternMetrics`）与文本格式；`to_dict()` 带 `kind: "trend"`。
- `config.py`：`TrendConfig`（`enabled`、`alert_continuing`、合约池成交额范围 `min_volume_24h_quote`/`max_volume_24h_quote` 与 `periods`）；各周期是 `TrendPeriodConfig` 的子类（`Trend3mConfig` 等），只改默认值，YAML 键为 `3m`/`5m`/`15m`（alias；1 分钟周期不提供，`pattern.PeriodName` 仍保留 `1m` 供检测器与测试直接使用），部分覆盖时其余字段仍取该周期默认值；校验 `max_counter_candles < candles`、设置了上限时 `max_volume_24h_quote > min_volume_24h_quote`。
- `replay.py`：`cli trend-replay` 用 REST 1 分钟 K 线（`gate.fetch_minute_candles` 分段请求，每段 ≤ 1000 根，与放量回放共用）驱动同一个检测器，`plan_trend_replay` 预先检查预热起点是否超出 Gate 只保留的最近 10000 根 1 分钟 K 线（测试须传入固定的 `now`）；`simulate.py` 为每个启用周期各跑一段合成单边行情，每个周期应恰好提醒一次。

### 放量启动（launch/）

独立子包，只经 `assembly.build_launch_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.launch`、`detectors/alerts.py` 的 `Alert` 与 `notifier` 中的 `LaunchAlert` 分派、`cli` 的 `launch-replay`/`simulate`/`check-config` 调用即可（`Detector.add_symbol` 的 `quanto_multiplier` 参数可保留）。面向启动前成交较少、进不了另两者合约池的币，默认合约池下限 100 万（低于它的币拉升后多半没人跟进），约 100 个合约。**放量上涨与放量下跌是两个检测器**（`build_detectors` 为 `LaunchConfig.sides()` 中每个启用方向建一个 `LaunchDetector`，名称「放量上涨」「放量下跌」），全部参数、合约池（各自的 `VolumeRange`）、K 线保留长度、冷却与波次互不影响。

- `rule.py`：纯函数。`LaunchRule` 是一个方向（`direction`：`surge`/`drop`）的全部门槛；`evaluate_launch(history, rule)` 对最近 `baseline_minutes + window_minutes` 根连续 1 分钟 K 线判定，返回 `LaunchMetrics`（含未满足的 `Condition`）。条件：窗口成交额 ≥ `min_window_quote`、量比（窗口成交额 ÷ 基准期同长度**均值**，基准为 0 时 `volume_ratio=None` 视为满足）≥ `min_volume_ratio`、窗口开始前一根收盘到最后一根收盘的涨跌幅 ≥ `min_change_percent`、`require_breakout` 时收盘越过基准期最高价/最低价、涨跌幅（绝对值）不超过 `max_change_percent`（`LaunchRule.overshoots`，None 不限，否则记 `OVERSHOOT`）、涨跌方向与规则相反记 `DIRECTION`（此时突破按规则方向判定，通常也一并失败）。`LaunchMetrics.direction` 是规则的方向，不是行情实际涨跌方向。
- `detector.py::LaunchDetector`：成交额 = 张数 × `quanto_multiplier` × 价格，面值为 None 的合约保留在池里但不聚合、不判定。与趋势检测器相同，**下一分钟第一笔成交到达时**结算上一分钟，空分钟按上一收盘价补成交额为 0 的平线；补齐的多根 K 线**逐根追加、逐根判定**（`_filled` + `_close_minute`），判定某一分钟时看不到它之后的 K 线。保留 `history_minutes` 根，预热请求 `warmup_candles` 为其 + 1。重复提醒按（合约, 方向）记在 `_last_alerts`（不随合约移出清除）：距上次不足 `cooldown_minutes` → `COOLDOWN`（先于基准期重置判断，冷却长于基准期也生效）；距上次已满一个基准期则重新从第 1 波算；否则价格须比上次提醒价再推进 `realert_step_percent` → 不够为 `NOT_EXTENDED`，满足则 `wave + 1`。只因 `OVERSHOOT` 失败的分钟由 `_record_overshoot` 走同一个 `_repeat_outcome`，判定为 `ALERT` 时静默写入 `_last_alerts`（波次不加），该分钟判定记为 `LaunchOutcome.OVERSHOOT`（回放默认列出），防止窗口滑过插针后在高位补报；`still_launching` 只防回落，不查上限。隔了空档才结算的分钟（触发结算的成交不在紧邻的下一分钟）用这笔成交价经 `rule.still_launching` 复核涨跌幅与突破，不成立记 `EXPIRED`、不提醒也不记录；回放用 `add_candle` 不复核。断线处理同趋势检测器，`resync_symbol` 同一分钟时高低点取并集、成交额取两边较大值（两边成交有重叠，不能相加）。`add_candle` 供回放、模拟，不能与 `add_tick` 混用。
- `alerts.py`：`LaunchAlert`（`direction` 取自 metrics，`wave`、`label` 为「放量上涨/放量下跌」加「·第N波」），`to_dict()` 带 `kind: "launch"`；成交额按万/亿显示、基准期按小时显示，分别用 `formatting.format_amount`、`formatting.describe_minutes`。
- `config.py`：`LaunchConfig` 只有 `rise`（`LaunchRiseConfig`）与 `fall`（`LaunchFallConfig`）两段（YAML 键用 rise/fall，方向值仍是 `surge`/`drop`，JSONL 字段不变），二者继承 `LaunchSideConfig`、键完全相同（含 `enabled` 与合约池 `min_volume_24h_quote`/`max_volume_24h_quote`），只有 `enabled` 默认值不同（上涨开、下跌关），其余代码默认值相同；没写的项取代码默认值，不沿用另一个方向。`LaunchConfig.enabled` 是两者之一启用，`configured()` 是键与方向值的唯一对应（含已关闭的，`check-config` 用），`sides()` 返回其中启用的（方向, 配置）、上涨在前，`rules()` 为对应的 `LaunchRule`。每段校验 `baseline_minutes > window_minutes`、`cooldown_minutes ≥ window_minutes`、设了上限时上限大于下限、`max_change_percent > min_change_percent`。旧版写在 `launch` 顶层的键（`enabled`、`alert_drops` 等）与曾用的 `launch.drop` 会被 `extra="forbid"` 拒绝。`default.yaml` 上涨保持原取值，下跌取 `research/launch_trade/launch_grid.md` 第三轮。
- `replay.py`：`cli launch-replay` 用 `gate.fetch_minute_candles`（与趋势回放共用）取 REST 1 分钟 K 线（自带计价成交额，面值传 1），为每个启用方向各驱动一个检测器（`LaunchSideReplay`），提醒按时间合并、判定按方向分段输出；`plan_launch_replay` 按较长的基准期预热，同样检查 Gate 10000 根的保留范围（测试须传入固定的 `now`）；`simulate.py` 为每个启用方向跑一段基准横盘后放量上涨（下跌）的合成行情，冷却覆盖整段，每个方向恰好提醒一次。
- 默认参数来自 6 天全市场 1 分钟 K 线回测（README「放量启动」一节有方案对比与数据），改默认值或判定逻辑时同步更新那里的数字。

### 冷门启动（dormant/）

独立子包，只经 `assembly.build_dormant_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.dormant`、`detectors/alerts.py` 的 `Alert` 与 `notifier`/`sounds` 中的 `DormantAlert` 分派、`cli` 的 `dormant-replay`/`simulate`/`check-config` 调用即可。面向连放量上涨合约池都进不去的冷门币（合约池默认 24 小时成交额 0~300 万，约 550 个合约）。准确率低于放量上涨，是用户知情后作为例外采用的：代码默认值是 `research/launch_trade/dormant_calm.md` 的选中方案加用户指定的总实体占比 ≥ 60% 与涨前波动 ≤ 0.5%，`default.yaml` 取 `dormant_early.md` 第三轮（研究经过见「回测研究」一节）；改默认值或判定逻辑时同步 README「冷门启动」一节。

- `rule.py`：纯函数。`DormantRule`；`evaluate_dormant(window, start_price, prior_quote, pre_volatility_percent, rule)` 对最近 `window_minutes` 根 1 分钟 K 线判定。窗口之前 `PRIOR_MINUTES`（1440）根只用到成交额合计，窗口之前 `CALM_MINUTES`（240）根只用到 `minute_range_percent`（真实波幅，含跳空，平线为 0）的平均值（涨前波动），两个长度都固定、不开放配置；两者由检测器滚动维护后传入（`_SymbolState.total_quote` 与和 `minutes` 一一对应的累计和 `range_totals`，两次取值相减即得一段之和；每分钟对上千根重新求和，沉寂很久的币一笔成交补出上千根平线时会卡住成交循环）。返回 `DormantMetrics`（含未满足的 `Condition`）。条件：窗口之前 24 小时成交额 ≤ `max_prior_quote`（`PRIOR`）、窗口成交额 ≥ `min_window_quote`、窗口涨幅（窗口前一根收盘 → 最后一根收盘）≥ `min_change_percent`（只看上涨）、窗口内有成交的分钟数 ≥ `min_active_minutes`、涨幅 ÷ 涨前波动 ≥ `min_calm_multiple`（`CALM`，涨前波动为 0 时倍数为无穷大）、涨前波动 ≤ `max_pre_volatility_percent`（`VOLATILE`，None 不限；倍数只看相对大小，这一项挡掉涨前乱跳但涨得够多的币，用户要求，`dormant_calm.md` 第六节）、窗口内单分钟收盘涨幅 ≤ `max_minute_change_percent`（`JUMP`）、窗口总实体占比（实体和 ÷ 振幅和，与趋势形态同一口径，振幅和为 0 时为 0）≥ `min_body_ratio`（`BODY`，用户指定的形态，未经选参，`dormant_calm.md` 第四节）、窗口最后一根成交额 ≥ `min_last_minute_quote`（`LAST`）、最后 `late_minutes(window)` = ⌈W/2⌉ 根成交额 ÷ 窗口成交额 ≥ `min_late_quote_share`（`LATE`，窗口无成交时为 0）；这两项代码默认 0 不限，`default.yaml` 连同窗口 15、成交额 2 万、8 分钟有成交按 `dormant_early.md` 第三轮「假启动更少」开启（判据未通过，用户知情后的例外）。量比、突破、回吐等研究中没让提醒更准的条件都不设。
- `detector.py::DormantDetector`：K 线聚合、空分钟补平线逐根判定、空档后用触发结算的成交价复核（`still_rising`，不成立记 `EXPIRED`、不记冷却）、断线与回补都与 `LaunchDetector` 相同。重复提醒按合约记上次提醒（`_LastAlert`：时间、价格、波次，`_last_alerts` 不随合约移出清除）：距上次（含补报）不足 `cooldown_minutes` 时，收盘价比上次提醒价再涨 `realert_step_percent` 才报下一波（`wave + 1`，标签「冷门启动·第N波」），否则记 `COOLDOWN`（`realert_step_percent` 为 None 时冷却内一律不报）；超过冷却就从第 1 波重新算；空档后复核同时检查这笔成交价仍比上次提醒价高够步长。没有涨幅上限。`simulate` 报出第一条即停（继续上涨会报第 N 波）。`add_candle` 供回放、模拟，不能与 `add_tick` 混用。
- `alerts.py`：`DormantAlert`（方向固定 `surge`、标签「冷门启动」），`to_dict()` 带 `kind: "dormant"`。
- `config.py`：`DormantConfig`（`enabled`、合约池 `min_volume_24h_quote`/`max_volume_24h_quote` 与上述门槛、`cooldown_minutes`），校验有成交分钟数 ≤ 窗口、冷却 ≥ 窗口、涨幅门槛 ≤ 窗口 × 单分钟涨幅上限（否则永远不会提醒）、设了上限时上限大于下限。
- `replay.py`：`cli dormant-replay` 用 `gate.fetch_minute_candles` 驱动同一个检测器，`plan_dormant_replay` 同样检查 Gate 10000 根的保留范围（测试须传入固定的 `now`）；`simulate.py` 用之前 24 小时少量成交的横盘加一段拉升，恰好提醒一次。

### 反弹回落（reversal/）

独立子包，只经 `assembly.build_reversal_detector` / `build_detectors` 接入，删除时去掉这两处、`config.AppConfig.reversal`、`detectors/alerts.py` 的 `Alert` 与 `notifier`/`sounds` 中的 `ReversalAlert` 分派、`cli` 的 `reversal-replay`/`simulate`/`check-config` 调用即可。只报「此前已反向走了一大段、急动时成交突然涌入」的急动：急拉报「下跌反弹」、急砸报「冲高回落」，提醒之后几分钟更可能沿急动方向继续走。来自 `research/move_accuracy/move_accuracy.md` 第二轮：秒级异动提醒后的方向保持率与随机相同，只有这两个条件同时满足才明显有效（样本外 hold5 51%、约七成提醒 1~10 分钟后在提醒方向一侧、每天约 3 条），代码默认值就是选中的一组；改默认值或判定逻辑时同步 README「反弹回落」一节与研究报告。

- `rule.py`：`ReversalRule` 与固定常量：成交暴增的 1 分钟对 1 小时（`BURST_SECONDS`/`BURST_BASE_SECONDS`，倍数理论上限 `MAX_BURST_RATIO` = 60，配置校验门槛必须小于它）、ATR 口径（14 根 1 分钟 K 线、预热 50 根、过期 180 秒，与研究一致，不开放配置）；`history_seconds` 为窗口 + 前置回看与 1 小时中较长者。
- `detector.py::ReversalDetector` **继承 `AtrMoveDetector`**（单个 `short` 窗口、不带连续提醒），急动判定、空秒补齐、ATR、空档复核、冷却全部沿用基类，研究也是按这套判定选的参数；不改 `move/detector.py`，只覆盖两个私有方法：`_settle_through` 在结算真实秒前把（秒、最后成交价、累计笔数 `trades_through`）记入 `_history`（按 `history_seconds` 保留，留一个不晚于最早所需时刻的点；`remove_symbols`、`mark_stream_gap`、`mark_symbol_gap` 时丢弃：断线期间成交缺失、基类清空秒桶后累计笔数也从 0 重新计数，重连后重新攒满 1 小时才判定）；`_evaluate` 调基类 `_evaluate_window`，得到 `ALERT` 后再判前置走势（窗口开始前 `prior_seconds` 秒那一刻 ≤ 该时刻最后一个真实秒的最后成交价，到基准价顺急动方向的涨跌幅 ≤ −`min_prior_move_percent`）与成交暴增（最近 60 秒笔数 ÷（最近 3600 秒笔数 ÷ 60）≥ `min_trade_burst_ratio`，没有 1 小时前的记录不报），任一不满足只拦这一秒、不清零确认；都满足后按基类同样的方式处理冷却并清零确认。依赖基类的私有成员（`_SymbolState`、`_SecondBucket`、`_evaluate_window`、`_last_alert`、`_cooldowns`），改 `AtrMoveDetector` 内部时要跑 `tests/test_reversal.py`。构造时可传 `observer`，每秒回调一条 `ReversalEvaluation`（急动窗口的 `WindowEvaluation`、`ReversalOutcome`：急动未达标 / 前置走势 / 成交暴增 / 冷却 / 提醒，以及两个条件的取值和提醒本身）；传了观察者时，急动达到涨跌幅与 ATR 门槛的秒（确认中、补齐、复核失败）也会算两个条件的取值供回放展示，判定结果不受影响。
- `alerts.py`：`ReversalAlert`（`direction` 为急动方向，`prior_change_percent` 为带符号的实际前置涨跌幅、与急动方向相反），`to_dict()` 带 `kind: "reversal"`；文本把急动涨跌幅写在前置涨跌幅之前（着色按首次出现替换）。
- `config.py`：`ReversalConfig`（`enabled`、合约池 `min_volume_24h_quote`/`max_volume_24h_quote`、急动窗口六项、`prior_seconds`、`min_prior_move_percent`、`min_trade_burst_ratio`），校验确认秒数 ≤ 窗口、设了上限时上限大于下限。
- `replay.py`：`cli reversal-replay` 用 REST 历史成交逐秒回放，取数复用 `detectors.replay.fetch_replay_data`（预热周期固定传 `"1m"`，不跟随 `move.candle_interval`）；`plan_reversal_replay` 从开始时间往前留 `history_seconds`（1 小时）再对齐到分钟；提醒按判定所属的秒在区间内取舍；输出急动窗口判定统计（窗口层面的 `ALERT` 写成「急动达标、可以提醒」，不是真正的提醒）、急动可以提醒之后各条件拦下的秒数、急动最接近门槛的一秒，以及急动过门槛的每一秒。
- `simulate.py`：平稳 1 小时 → 从前置回看起点开始的一段下跌（门槛的 1.5 倍）→ 窗口内停在低位 → 急拉前 50 秒起成交涌入（每秒笔数加倍搜索到倍数达门槛的 1.2 倍）→ 急拉，恰好一条「下跌反弹」。

### 回放（detectors/replay.py + 各子包 replay.py）

`detectors/replay.py` 放各回放共用的部分：`parse_time`、`normalize_symbol`、`fetch_price_decimals`；成交逐秒回放（秒级异动、反弹回落）的 `ReplayPlan`、`ReplayData`、`fetch_replay_data`（按传入的预热周期取合约精度、K 线和成交，区间末尾不够结算时继续向后找一笔）；1 分钟 K 线回放（趋势、放量、冷门）的 `MinuteReplayPlan` 与 `plan_minute_replay`（起点对齐到分钟、往前留判定所需根数，并预先检查是否超出 Gate 只保留的最近 10000 根，`now` 供测试固定）。`cli` 的五个回放命令共用 `_plan_or_exit`（解析 `--start`/`--end`、推算区间，`ValueError` 转成一行错误退出）、`_fetch_trades_or_exit`、`_fetch_minutes_or_exit`。`cli replay` 在 `move/replay.py`：`plan_replay` 推算喂入起点（开始时间减最长窗口，向下对齐 K 线边界）与 K 线预热区间，`run_replay` 用 `build_detector(move, observer=...)` 驱动同一个检测器，`render_replay` 输出规则、提醒、各窗口统计与「最接近触发」的秒。回放从空白冷却开始、不模拟断线。

`GateRestClient.fetch_trades` 按 `[from, to)` 取成交：Gate 从新到旧分页、每页 1000 笔，翻页靠把 `to` 收缩到本页最早秒 + 1 并按 id 去重；整页落在同一秒时才用 `offset`（offset 在十万级会被拒绝）。REST 成交的 `create_time_ms` 实际单位是秒，只能用 `create_time`（`parse_rest_trade`），与 WebSocket 的解析不同。

### 配置（config.py + config/default.yaml）

- 顶层 `config.py` 只有 `AppConfig` 与 `load_config`，各段模型放在所属的包里：`gate` 在 `exchange/config.py`，`alerts` 在 `notify/config.py`，`move`/`trend`/`launch`/`dormant`/`reversal` 在各检测器子包的 `config.py`。
- pydantic 模型全部 `extra="forbid"`，YAML 中出现未知键会直接报错。
- `PRICE_ALERT_WEBHOOK_URL` 环境变量覆盖 `alerts.webhook_url`。
- `config.py` 中的代码默认值固定不变，没有特别要求不要改；它只在 YAML 缺少对应项时生效，实际运行以 YAML 为准。
- `config/default.yaml` 每项注释中的「默认 X」标注的是代码默认值。调参时只改 YAML 的取值，注释里的默认值和 `config.py` 都不动，两者允许不一致；配置测试只校验 YAML 能通过校验。
- README 中的参数、例子和回放结果一律按代码默认值写（复现时给 `--config` 传只含 `{}` 的配置文件），调 YAML 不需要改 README；只有改代码默认值或判定逻辑时才同步更新。
- 秒级异动的配置段是 `move`（`move/config.py::MoveConfig`），放合约池门槛 `min_volume_24h_quote` 与各窗口共用的 K 线/ATR 参数（`candle_interval`、`atr_period`、`warmup_candles`、`max_atr_age_seconds`），窗口参数在结构相同的 `short_window`（`ShortWindowConfig`）与 `long_window`（`LongWindowConfig`，多一个 `enabled`）里，二者继承 `WindowConfig`，只在默认值、取值范围和冷却规则上不同。校验：`warmup_candles > atr_period`、`max_atr_age_seconds ≥ 2 个 K 线周期`（未显式设置时自动取 3 个周期）；`WindowConfig` 校验确认秒数 ≤ 窗口长度；启用长窗口时 `long_window.lookback_seconds > short_window.lookback_seconds`；`LongWindowConfig` 额外校验冷却 ≥ 窗口长度（未显式设置时等于窗口长度）。`GateConfig` 校验 `reconnect_max_seconds ≥ reconnect_initial_seconds`。

### 新增/修改检测参数时需要同步的位置

`AtrMoveDetector` 统一由 `assembly.build_detector` 组装、`TrendDetector` 由 `assembly.build_trend_detector` 组装（`run_monitor`、`cli.simulate`、`replay` 共用，REST 客户端同样由 `assembly.build_rest_client` 组装），窗口参数由 `assembly.build_windows` 从配置生成。各入口只依赖 `assembly`，不为了拿检测器去依赖实时编排模块 `service`。新增指标参数需要同时改 `move/config.py` 模型、`config/default.yaml`、`build_windows`/`build_detector`、README 的配置说明。`simulate` 为每个启用的窗口各用一个只含该窗口的检测器跑一遍（`move/simulate.py`）（`move.enabled: false` 时跳过），再为每个启用的趋势周期、放量方向、冷门启动和反弹回落各跑一遍，正常时提醒数等于窗口数加趋势周期数、启用的放量方向数，冷门启动、反弹回落启用时各再加一（`cli.expected_simulated_alerts`）。`build_detectors` 只创建启用的检测器，`AppConfig` 校验五者至少启用一个。趋势参数改在 `trend/config.py`（周期子类的默认值）、`config/default.yaml` 的 `trend` 段、README；放量参数改在 `launch/config.py`（`LaunchSideConfig`）、`LaunchRule`、`config/default.yaml` 的 `launch.rise`/`launch.fall` 段、README；冷门启动参数改在 `dormant/config.py`、`DormantRule`、`config/default.yaml` 的 `dormant` 段、README；反弹回落参数改在 `reversal/config.py`、`ReversalRule`、`config/default.yaml` 的 `reversal` 段、README。K 线周期到秒数的映射只有 `common/models.py` 的 `INTERVAL_SECONDS` 一份（放在 `common` 而不是顶层 `config`：`move/config.py` 校验要用它，`config.py` 又要导入 `move/config.py`，放 `config` 会循环导入）。

### 合约池筛选（exchange/universe.py）

筛选结果 `ContractTicker` 带 `price_decimals`（由 `gate.price_decimals` 把合约的 `order_price_round` 换算成小数位数，仅用于展示；回放取精度也用它），经 `detector.add_symbol` 传到 `PriceAlert`；还带 `quanto_multiplier`（`gate.quanto_multiplier` 解析合约面值，非正或无法识别为 None），供放量检测把成交张数换算成计价成交额。只接受 `contract_type == ""`（字段必须存在且为空，非币类资产会有分类值）、`status == "trading"`、`in_delisting` 不为真、以 `_USDT` 结尾、`volume_24h_quote` **严格大于**下限、且不超过可选上限（含等于）的合约。秒级异动的下限是 `move.min_volume_24h_quote`，冷门启动的范围是 `dormant.min_volume_24h_quote` / `dormant.max_volume_24h_quote`（默认 0~300 万），反弹回落的范围是 `reversal.min_volume_24h_quote` / `reversal.max_volume_24h_quote`（默认 > 800 万、上限不设），趋势提醒的范围是 `trend.min_volume_24h_quote` / `trend.max_volume_24h_quote`（上限默认不设），放量上涨、放量下跌分别是 `launch.rise.*` / `launch.fall.*` 的 `min_volume_24h_quote` / `max_volume_24h_quote`（下限默认 100 万），由 `assembly.build_detectors` 装进各自的 `VolumeRange`。已在监控中的合约（`retained_symbols`，按各检测器自己的池）退出线两侧放宽：下限 × `universe_exit_volume_ratio`、上限 ÷ 该比例。`volume_24h_usd` 已被 Gate 弃用，仅作缺字段时的回退。`is_internal=true` 的成交在 `gate.parse_trade_payload` 中被忽略。

### 通知（notify/notifier.py）

提醒类型是 `detectors/alerts.py` 的 `Alert = PriceAlert | StreakAlert | TrendAlert | LaunchAlert | DormantAlert | ReversalAlert`，`format_alert`/`colorize_alert` 按类型分派（趋势提醒的标签用亮紫色 `TREND_LABEL_COLOR`，放量启动用金黄色 `LAUNCH_LABEL_COLOR`，冷门启动用亮蓝色 `DORMANT_LABEL_COLOR`，反弹回落用亮白色 `REVERSAL_LABEL_COLOR`，连续提醒的标签沿用方向色、不另外着色）。`AlertDispatcher` 为每个通道建立独立的有界队列（`alerts.queue_size`）和后台任务：`publish` 非阻塞，队列满时丢弃并记错误日志；单个通道失败只记日志；退出时最多等待 `drain_timeout` 秒把积压发完。`build_notifiers` 按配置返回通道列表。`JsonlNotifier` 按 `jsonl_max_bytes` 整文件轮转。价格、窗口长度（整分钟显示为「N分钟」）和北京时间的格式都在 `formatting.py`，提醒与回放共用；价格按合约报价精度显示，缺失时按 8 位有效数字、不截断整数部分。提醒时间统一转为北京时间；控制台急涨绿色、急跌红色，涨跌幅用同方向亮色高亮，长窗口提醒的窗口长度用不分方向的亮青色（`LONG_WINDOW_COLOR`）高亮（短窗口保持正文色）；JSONL/Webhook 保留完整结构化字段（`PriceAlert.to_dict()`）。控制台提醒附带的交易地址由 `gate.futures_trade_url` 生成，Gate 网页地址只维护在 `gate.py`。修改提醒文本格式时注意 README 中的示例。

提示音（`sounds.py`）：`beep` 开启时 Windows（`winsound` 异步播放）与 macOS（`afplay`）播放同一套代码合成的 WAV（`synthesize` 纯函数，`write_sound_files` 写到 `default_sound_dir()`、文件名带内容哈希；Windows 用 `%LOCALAPPDATA%`、macOS 用 `~/Library/Caches`，不用系统临时目录——存储感知会在运行中清掉它，而 winsound 找不到文件时静默不出声），其他平台或生成失败时退回终端响铃。`ConsoleNotifier._sound_path` 每次发声前检查文件，被删就重新生成（失败则本次不发声，警告按 `SOUND_REGENERATE_LOG_INTERVAL_SECONDS` 限流）。`sound_for` 按提醒类型选八种音效：音色分大类（秒级异动/连续提醒为木琴且音量最小、趋势为无 4kHz 以上分音的钟音、放量为号角琶音且音量最大、冷门启动为只有奇次谐波的簧管音三步上行，时长与放量启动相同、音色与节奏不同、音量稍小，反弹回落为拨弦音先落后起 E6-B5-E6），敲两下表示连续/延续。正在播放时按 `SOUND_PRIORITY`（放量启动 > 冷门启动 = 反弹回落 > 连续提醒 > 趋势 > 秒级异动）只让更高优先级打断，同级或更低跳过；`MUST_PLAY`（放量启动，优先级必须最高）例外：播放中再来的同类提醒若晚于开播 `MUST_PLAY_MERGE_SECONDS`，播完后补响一次（合并）。Windows 没有播放结束通知，按音效时长加 `WINDOWS_PLAYBACK_MARGIN_SECONDS` 等待来判断是否仍在播放。各音效结尾淡出（`fade_out_seconds`）要让最后 50 毫秒低于峰值 -30 dB，有测试约束。改音效时同步 README「提示音」一节。

## 回测研究（research/）

`research/launch_trade/`，不参与实时监控，也不在 `ruff check src tests` 与 pytest 范围内，用法见其 README.md，判据与结果都写在各自的 .md：

- 数据工具：`update_data.py` 下载/更新历史 1 分钟 K 线、`load_data.py` 合并为分钟序列、`check_jumps.py` 检查面值变更造成的跳价、`paths.py` 路径。
- 放量启动：`drop_thresholds.py`/`.md`（放量下跌门槛）、`surge_params.py`/`.md`（放量上涨参数，维持现值）、`launch_grid.py`/`launch_alerts.py`/`launch_grid.md`（两个方向全量搜索：第一轮按跟单收益已作废，第二轮 F1、第三轮 F0.5 按准确率与召回；`default.yaml` 的放量下跌取第三轮）。
- 冷门启动（依赖链 `dormant_best` → `dormant_early` → `dormant_calm` → `dormant_solo`）：`dormant_solo`（单独的冷门启动检测器）、`dormant_calm`（涨前平稳、不靠跳涨，代码默认值）、`dormant_early`（更早、更少假启动，`default.yaml` 取其第三轮）、`dormant_best`（放开全部条件重新选参，维持现行，并给出现行的样本外：准确率与原参数相同，只是少报）。都未通过判据，采用的都是用户知情后的例外。

`research/move_accuracy/`：秒级异动「提醒后方向能否保持」（`PLAN.md` 预先写定两轮、`move_accuracy.md` 结果），按用户要求不参考 `move_params`、从头做。下载站逐笔成交（数据在 `data/backtest/move_acc/`），`engine.py` 逐秒复现 `AtrMoveDetector`（与真实检测器逐条对拍一致），`sim.py`/`fastgrid.py`/`grid2.py` 在宽松门槛的行上重算任意参数，主指标是提醒后 1~5 分钟每个整分钟点都在提醒方向一侧的比例（hold5）。第一轮：现行秒级参数与随机相同（约 23%），只加「前置走势 ≤ 0」样本外 27.8%，不够。第二轮在 2026-04~09 探索（`features.py`/`model.py` 估可预测上限、`rules.py` 规则化）后选出「前置走势反向 ≥ 6%、成交笔数暴增 ≥ 16 倍」，在从没用过的 2025-10~2026-03 上 hold5 51.1%；用户认定这是新信号，落地为独立的 `reversal/` 检测器，秒级异动不改。`verify_reversal.py`/`verify_hold2.py` 对拍 `ReversalDetector` 与引擎。
`research/move_params/`：秒级异动参数（`move_params.md`）。用下载站 4~9 月逐笔成交与 REST 十月逐笔成交（数据在 `data/backtest/move/`），`record.py` 用真实检测器按宽松门槛逐笔重放并记录逐秒指标，`sim.py` 在记录上按更严的参数离线重算确认、补齐秒、空档复核与冷却（`verify.py` 与检测器逐条对拍一致），`half.py` 汇总半年，`quality.py` 是第三轮。第一轮只用九月；第二轮（4~7 月开发期）以减少数量为主；第三轮按用户要求以质量为目标（5 分钟保留率 − 假提醒率）重新选参，建议短窗口 3.0%、2.5 ATR、确认 5 秒，相对第二轮质量只高约 0.4 个百分点。对比后用户选用第二轮（2.0%、2.5 ATR、确认 3 秒，笔数、冷却不变，提醒约 −33%）；之后用户又把 `default.yaml` 短窗口放宽到 1.3%、1.5 ATR、确认 2 秒，这组不是研究推荐的取值。

行情数据在 `data/backtest/`（不进版本库），回测用单独的虚拟环境 `data/backtest/.venv`（多装 numpy、numba；`move_accuracy` 还要 pandas、lightgbm）。新研究的方案要先通过预先写定的过拟合判据（只用开发期选参数、多个入场设置平均、Deflated Sharpe、PBO 等）才能推荐或写进配置。

## 约定

- 测试必须确定、离线；文件名 `tests/test_<module>.py`，覆盖边界条件、异常交易所数据、冷却、通知失败等场景。
- 修改 `config/default.yaml` 中的阈值属于行为变更，提交说明中需写明运营影响。
- 提交信息风格：简洁祈使句，常用 `config:`、`docs:`、`feat(...)`、`chore:` 前缀，也可用中文标题。
