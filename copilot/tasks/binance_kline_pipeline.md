## 任务

创建数据管道`src/pipelines/binance_klines`，下载和更新币安永续合约的历史k线(1分钟)。

## 背景和目标

数据管道需要解决两个问题：1. 下载全部永续合约的1分钟历史k线；2. 增量更新1分钟最新数据。

这些数据将作为后续量化研究和回溯检验的基础。

`src/pipelines/binance_klines`目前包含两个模块：
- `downloader.py`：提供下载数据的函数，从历史仓库下载历史数据，从api获取历史数据。
- `manage.py`: 开发测试模块。

回填历史数据：`uv run python -m src.pipelines.binance_klines.task backfill`

增量更新数据：`uv run python -m src.pipelines.binance_klines.task update`

## 数据分区策略

- 按照交易对 + 时间（月）进行分区
- 交易对作为一级分区，时间作为二级分区
- k线数据的最常见查询模式是："获取某个交易对过去一段时间的k线"。
- 时间分区：
  - 按年分区，重写代价很大，更新20251225当天数据时，需要覆盖2025年整个分区。
  - 按天分区，碎片化文件太多，不利于查询。
  - 月份通常是比较好的平衡点（经验）。

```text
data/cleaned/binance_klines_perp_m1/
  ├── symbol=BTCUSDT/
  │    ├── period=2023-01/data.parquet
  │    ├── period=2023-02/data.parquet
  │    └── ...
  ├── symbol=ETHUSDT/
  │    ├── period=2023-01/data.parquet
  │    └── ...
```

## 核心逻辑

回调历史数据：
- 从`data/cleaned/binance_tickers_perp.parquet`读取活跃交易对列表。
  - `symbol`字段是交易对名称，例如BTCUSDT
  - `onboard_date`字段表示上市日期，datetime对象
  - `status`表示交易状态，`TRADING`表示能够交易
- 遍历交易对，从上市日期开始到今天，获取月数据，清洗数据，写入parquet（覆盖原始文件）。
- 实现多线程加速。

增量更新：
- 从`data/cleaned/binance_tickers_perp.parquet`读取活跃交易对列表。
- 获取交易对最近的时间戳，下载最新数据。假设BTCUSDT的最后一条记录的时间戳是`2026-01-18 14:25:00+00`，从`2026-01-18 00:00:00+00`开始下载最新数据，目的是覆盖最后一天的数据，以确保分钟级数据没有遗漏。
- 下载数据时优先从历史仓库下载，如果仓库没有更新则改用api下载。
- 获取数据后读取对应月份的parquet数据，合并和去重，再写入parquet，实现批量更新。
- 使用多线程加速，每条线程处理一个交易对。

清洗数据核心字段：
- datetime: 日期时间，包含时区，使用UTC
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- symbol: 交易对名称，不包含分隔符

## 技术约束

- 使用`requests`处理网络请求，`tenacity`处理重试逻辑
- 使用`ccxt`从官方API获取1分钟k线
- 使用`duckdb`作为查询引擎，操作parquet数据库
- 使用`pandas`操作数据
- 使用`src.core.logger`创建的自定义logger
- 使用`typer`添加命令行参数
- 使用现代化的类型注释，例如`list[str]`而不是`List[str]`
- 文档字符串，注释和日志均使用中文而不是英文（强制要求）。

## 验证标准

因为要处理的交易对非常多（几百个），因此需要创建命令行参数来提供测试，例如：

`uv run python -m src.pipelines.binance_klines.task backfill -s btcusdt,ethusdt --start-date 20260101 --end-date 20260201`


## 阅读和测试

### 回填历史数据

单个交易对1个月的分钟k线约为42000行，只有2mb，如果按年分区，才24mb，按年分区是否更好？

使用marimo notebook建立查询机制，检查parquet文件的数据是否正确。

结果符合预期，没有时区更换的问题，通过测试。

### 更新数据

月数据有缺失，可能遗漏1天的数据，对于创建量化模型可能影响不大，但是不能用于回溯检验。

如果每天下载数据，虽然精度更高，但是会耗费大量时间。

能够成功更新数据，除了数据精度外，程序能够正常运行。
