# price_alert

一个只面向 Gate.io 虚拟币 USDT 永续合约的实时价格异动监控服务。股票、指数、外汇、贵金属和商品合约不会进入监控。项目只负责监控和提醒，不执行交易。

## 检测逻辑

项目不使用固定涨跌幅作为触发门槛，而是使用 ATR 对短时价格位移进行标准化：

```text
ATR 异动强度 = |当前完整秒 VWAP - N 秒前完整秒 VWAP| / Wilder ATR
实际涨跌幅 = |当前完整秒 VWAP / N 秒前完整秒 VWAP - 1| × 100%
```

默认配置为：

- 从 Gate.io 全部虚拟币 USDT 永续合约中，自动选择 24 小时计价成交额严格大于 10,000,000 USDT 的合约；
- 使用最近 14 根 1 分钟 K 线计算 Wilder ATR；
- 比较当前完整秒的成交量加权均价与 30 秒前的完整秒均价；
- 实际涨跌幅至少达到 1%，同时位移至少达到 1.5 ATR；
- 两道门槛连续满足 3 秒，且窗口内至少有 5 笔成交时提醒；
- 同一合约无论涨跌方向，在 30 秒内只提醒一次；合约移出合约池后重新入池，冷却记录仍然有效；
- 每 10 分钟刷新交易对池，通过增量订阅/退订生效，不会断开实时连接；已在监控中的合约成交额跌破门槛的 80% 才会被移除，避免在门槛附近反复进出。

触发距离等于 `max(基准价格 × 1%, ATR × 1.5)`。最低 1% 保证提醒具有足够的实际价格幅度，ATR 门槛则随着近期波动率动态变化。完整秒 VWAP 和连续确认用于过滤单笔离群成交与瞬时价格尖刺。

成交稀疏的合约在快速行情中常出现没有成交的空秒。空秒（不超过观察窗口长度）会沿用最后成交价参与连续确认，但提醒只会在有真实成交的秒触发，所以一笔离群成交后恰好无人成交，不会被误判为持续异动。实时连接断开重连后，秒级窗口和确认进度会清空，不会把断线前后的价格当作连续行情比较；同时全部合约暂停异动判定，连接恢复后自动从 REST 回补最近的 K 线并重建 ATR，回补完成的合约才恢复判定（失败的合约按退避重试）。连接正常时，某个 K 线周期内没有成交，会像 Gate 官方 K 线一样按上一收盘价补一根平线，保证实时 ATR 与预热数据口径一致。

交易所返回的价格、数量、成交额和 K 线数值中出现 NaN 或 Infinity 时一律视为无效数据丢弃。ATR 预热时，REST 返回的当前未收盘 K 线会作为实时 K 线的起点，避免第一根实时 K 线只包含订阅之后的成交而低估 ATR。

## 数据流程

```text
Gate REST /futures/usdt/contracts + /tickers
        ↓ 仅虚拟币、交易中、24h quote volume > 10M USDT
动态交易对池
        ↓
Gate REST candlesticks 预热 ATR
        ↓
Gate WebSocket futures.trades 实时成交
        ↓
完整秒 VWAP 窗口 + 实时分钟 K 线 + Wilder ATR
        ↓
连续确认 / 成交笔数 / ATR 新鲜度 / 统一冷却过滤
        ↓
控制台声音 + JSONL + 可选 Webhook
```

项目通过 Gate 合约元数据的 `contract_type` 排除股票、指数、外汇、贵金属和商品等非币类合约，并且只监控 `status=trading` 且不在下架流程（`in_delisting`）中的品种。Gate 官方将 `volume_24h_usd` 标记为弃用，因此项目使用 `volume_24h_quote`。`is_internal=true` 的内部成交可能偏离正常盘口且不会进入 K 线，检测时会主动忽略。

Gate 在一批订阅中只要有一个无效合约，整批都会失败。程序会识别订阅失败的响应，把该批合约改为逐个订阅，只放弃真正被拒绝的合约并记录错误日志。单笔无法解析的成交数据只会被跳过（同类日志每分钟最多输出一次），不会断开整个连接。

## 安装

需要 Python 3.11 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 启动

Windows 可以双击 `start-monitor.bat`，或者运行：

```powershell
.\.venv\Scripts\python.exe -m price_alert.cli run
```

停止监控请按 `Ctrl+C`。

一个监控进程会同时订阅全部符合条件的交易对。程序使用进程锁阻止重复启动；如果已有监控进程运行，再次启动会直接退出，从而避免同一行情被重复提醒。

查看当前满足成交额条件的合约：

```powershell
.\.venv\Scripts\python.exe -m price_alert.cli universe
```

校验配置：

```powershell
.\.venv\Scripts\python.exe -m price_alert.cli check-config
```

使用合成数据验证 ATR 提醒，不联网、不写告警文件：

```powershell
.\.venv\Scripts\python.exe -m price_alert.cli simulate
```

如果当前配置下模拟行情没有产生提醒，命令会以非零退出码结束并给出提示。配置文件缺失或校验失败时，会列出出错的配置键路径。

## 提醒示例

```text
[急涨提醒] 2026-09-16 16:20:30 | BTC_USDT | 30秒内价格上涨 1.24% | 75800 → 76739.92 | 异动强度 2.84 ATR
交易地址：https://www.gate.com/zh/futures/USDT/BTC_USDT
```

提醒时间使用北京时间，格式为 `YYYY-MM-DD HH:MM:SS`。控制台中急涨提醒显示为绿色，急跌提醒显示为红色，交易对以亮黄色突出，涨跌幅百分比以同方向的亮绿/亮红色突出；`alerts.beep` 开启时，macOS 使用系统 `Glass` 音效，其他平台使用终端响铃。文本明确显示价格涨跌百分比，不显示原始 ATR 数值和 24 小时成交额。JSONL 和 Webhook 记录仍保留完整结构化字段，并包含值为 `green` 或 `red` 的 `color` 字段。

控制台每条提醒下方会以灰色显示对应合约的 Gate 中文交易页完整地址。支持网址识别的终端可点击打开（部分终端需要按住 `Ctrl` 或 `Cmd` 再点击）；不支持时可复制到浏览器打开。交易地址仅附加在控制台输出中。

实时提醒默认追加到 `data/alerts/alerts.jsonl`。每行是一条完整 JSON，即使程序异常退出，也不会破坏之前的记录。文件超过 `alerts.jsonl_max_bytes` 后整体轮转为 `alerts.jsonl.1`、`alerts.jsonl.2` 等，最多保留 `alerts.jsonl_backup_count` 个历史文件。

## Webhook

可以在 `config/default.yaml` 填写通用 HTTP Webhook，也可以用环境变量覆盖：

```powershell
$env:PRICE_ALERT_WEBHOOK_URL = "https://example.com/your-webhook"
.\.venv\Scripts\python.exe -m price_alert.cli run
```

环境变量优先，适合包含密钥的地址。Webhook 会收到可直接展示的 `text`，以及 ATR、ATR 倍数、起止价格、成交额和时间等结构化字段。每个通知通道都有独立的队列和后台任务，Webhook 响应慢既不会阻塞行情处理，也不会拖慢控制台提醒；单个通道失败只记录日志，不会中断行情监控。某个通道积压超过 `alerts.queue_size` 时，会丢弃该通道的新提醒并记录错误。

## 配置说明

`config/default.yaml` 的主要参数：

- `gate.min_volume_24h_quote`：24 小时 USDT 计价成交额门槛；
- `gate.universe_exit_volume_ratio`：已在监控中的合约的退出门槛比例，成交额低于或等于 `min_volume_24h_quote × 该比例` 即移除（准入与退出都是严格大于门槛才保留），默认 0.8；
- `gate.universe_refresh_seconds`：交易对池刷新周期；
- `gate.reconnect_initial_seconds` / `gate.reconnect_max_seconds`：断线重连的指数退避初始值与上限，上限不能小于初始值；
- `gate.max_data_lag_seconds`：实时成交的最大允许滞后，默认 10 秒。网络拥塞时成交会在链路上积压，推送过来的已是几十秒前的行情，此时秒级判定失去意义；逐笔校验成交时间戳，超过该值即主动断开重连以清空积压，滞后的成交在判定之前就被拦下，不会产生提醒。若日志频繁出现「行情数据滞后」，说明到 Gate 的网络链路不稳，应先排查网络而不是调高该值；
- `gate.rest_rate_limit_per_second` / `gate.rest_rate_limit_burst`：所有 REST 请求共用的令牌桶速率与突发额度，默认 10 次/秒、突发 20 次。交易所限频按单位时间的请求数计算，而合约池大小只决定单轮请求数，因此上限必须加在时间维度上：合约池再大、断线重连再频繁，长期平均速率都收敛到该值，只会拉长一轮预热或回补的耗时。注意突发额度是瞬时放行的额度——桶攒满时可以一次发出 20 个请求，限制的是长期平均速率而非任意一秒内的绝对上限，所以突发额度不应超过交易所在一个限频窗口内的配额——Gate 公共接口的官方口径是每个端点每 10 秒 200 次（响应头 `X-Gate-RateLimit-Limit` 实测同为 200），折合平均 20 次/秒，而本项目的令牌桶是所有端点共用的，比按端点计算更保守。若日志出现 429，应下调速率而不是调高重试次数；
- `gate.rest_retries`：单个请求的最大尝试次数。429 按交易所告知的恢复时刻等待（封顶 30 秒）：优先读 HTTP 标准头 `Retry-After`，Gate 实际不返回该头，恢复时刻放在专有头 `X-Gate-RateLimit-Reset-Timestamp`（Unix 秒级绝对时间戳）里；两者都没有时才按 2 秒起的指数退避，明显长于其他临时故障的 0.5 秒，避免在限频期间继续加码请求；
- `gate.warmup_concurrency`：ATR 预热与断线回补共用的并发上限，两者共用同一个信号量，峰值并发不会叠加；
- `indicator.candle_interval`：ATR K 线周期；
- `indicator.atr_period`：Wilder ATR 周期；
- `indicator.lookback_seconds`：短时位移观察窗口；
- `indicator.trigger_atr_multiple`：触发所需 ATR 倍数；
- `indicator.min_change_percent`：触发所需的最低实际涨跌幅；
- `indicator.confirmation_seconds`：超过动态门槛后需要连续确认的秒数，不能大于 `lookback_seconds`；
- `indicator.min_window_trades`：窗口内最低成交笔数；
- `indicator.max_atr_age_seconds`：ATR 过期保护，按最近一根计入 ATR 的 K 线的收盘时间计算，不能小于两个 K 线周期；不填写时默认为三个 K 线周期；
- `alerts.cooldown_seconds`：同一合约的统一提醒冷却时间；
- `alerts.queue_size`：每个通知通道允许积压的提醒数量；
- `alerts.console_colors`：是否启用控制台颜色，默认开启；
- `alerts.beep`：是否在控制台提醒时播放提示音；macOS 使用系统音效，不依赖终端响铃设置；
- `alerts.jsonl_max_bytes` / `alerts.jsonl_backup_count`：JSONL 轮转大小（0 表示不轮转）与保留的历史文件数；

## 工程结构

```text
config/default.yaml          Gate、ATR、提醒配置
src/price_alert/
├── config.py                配置模型与严格校验
├── models.py                领域模型
├── universe.py              动态高成交额合约池
├── gate.py                  Gate REST / WebSocket 适配器
├── indicators.py            Wilder ATR
├── detector.py              ATR 标准化异动检测
├── instance.py              防止重复提醒的跨平台进程锁
├── notifier.py              控制台、JSONL、Webhook 与独立队列分发
├── service.py               预热、增量刷新、重连和服务编排
└── cli.py                   run/universe/check-config/simulate
tests/                       指标、筛选、解析、检测、通知、服务编排和命令行测试
data/alerts/                 本地告警记录
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
```

该工具只提供行情提醒，不构成投资建议。
