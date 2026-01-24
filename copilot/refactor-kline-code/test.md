## 问题

重点测试增量更新模块。

两种模式：更新全部交易对的数据，针对性更新指定交易对或指定时间范围的数据。

1. 更新指定交易对和时间范围的数据。
2. 更新全部交易对，不指定时间范围。

更新全部交易对，线程数=20，报错：2026-01-24 14:32:40 | ERROR | kline | 查询ONTUSDT最后时间戳失败: IO Error: Cannot open file "data/cleaned/binance_klines_perp_m1/symbol=ONTUSDT/period=2020-02/data.parquet": Too many open files

使用5条线程没有问题。
