## 数据存储方案

parquet + duckdb，parquet存储历史数据，duckdb作为查询引擎。

这种方案同时满足本地开发和云端部署的需求。

GCS(parquet) -> API(duckdb) server -> Website

## 如何更新parquet数据？

parquet并不支持`upsert`功能，即行存在时更新，不存在时插入。

解决方案：分区覆盖模式，管理分区（块，partiion）而不是行。

例如架构设计：
```text
gcs://my-quant-bucket/market_data/
    ├── dt=2023-10-25/data.parquet
    ├── dt=2023-10-26/data.parquet
    └── dt=2023-10-27/data.parquet
```

1. 增量数据更新
10月28号收盘后，将28号的数据添加到新文件，不修改旧文件。

2. 修正历史数据
假设发现10-25号的数据有误，使用辅助脚本重新计算当天的数据，然后覆盖`dt=2023-10-25/data.parquet`分区。

核心原理：通过“覆盖”文件实现批量"upsert"。

## 下载和更新分钟k线

考虑应用场景：下载和更新数百个交易对的历史k线（1分钟）。

假设可以通过api下载某个交易对1个月或者1天的历史数据。

### 分区策略
- 按照交易对 + 时间（月）进行分区
- 交易对作为一级分区，时间作为二级分区
- k线数据的最常见查询模式是："获取某个交易对过去一段时间的k线"。
- 时间分区：
  - 按年分区，重写代价很大，更新20251225当天数据时，需要覆盖2025年整个分区。
  - 按天分区，碎片化文件太多，不利于查询。
  - 月份通常是比较好的平衡点（经验）。

```text
gcs://quant-data/klines_m1/
  ├── symbol=BTCUSDT/
  │    ├── period=2023-01/data.parquet
  │    ├── period=2023-02/data.parquet
  │    └── ...
  ├── symbol=ETHUSDT/
  │    ├── period=2023-01/data.parquet
  │    └── ...
```

### 工作流设计

创建数据管道，先回填历史数据（backfill），然后定期更新(update)。

核心逻辑：读 -> 改 -> 写

1. 当前交易日结束后，下载当天数据。
2. 从parquet读取本月数据，
3. 合并数据，去重。
4. 重写写入parquet文件，覆盖旧文件，相当于实现批量'upsert'。

如果要获取数十个或者数百个交易对的小时图或者四小时图数据，需要建立聚合模块，并存储到新的parquet文件，按照时间进行分区。

```text
data/aggregated/
  klines_h1/
    year=2021
    year=2022
    ...
  klines_h4/
    year=2021
    year=2022
    ...
```
