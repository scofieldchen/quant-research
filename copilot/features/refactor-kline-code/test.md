## 问题

1. 更新指定交易对和时间范围的数据。
2. 更新全部交易对，不指定时间范围。

更新全部交易对，线程数=20，报错：2026-01-24 14:32:40 | ERROR | kline | 查询ONTUSDT最后时间戳失败: IO Error: Cannot open file "data/cleaned/binance_klines_perp_m1/symbol=ONTUSDT/period=2020-02/data.parquet": Too many open files

使用5条线程没有问题。

`_execute_with_retry`：处理失败任务的逻辑有点难理解，如果使用类对任务进行建模是否会更好？

`run_backfill`,`run_update`两个方法的参数都相同，都需要先对参数进行验证或者调整，逻辑有重复，将逻辑提取到统一的函数是否更好？

`UpdateTask`：我认为这个类的定义非常不明确，它应该代表一个最基础的更新任务，应该和交易对和日期范围对应，他的属性应该是`symbol`,`dates`，表示更新该交易对在这些时间范围内的数据。增量更新时不管是全部日期失败，还是部分失败，都可以直接更新dates属性，并进行下一轮重试，这样简化是否更合理？

`generate_months`函数一定要使用dt.datetime吗？

能否整合failedtask和任务类？
