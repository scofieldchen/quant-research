### duckdb 查询 parquet 时间序列并进行聚合

- `date_trunc('day', datetime AT TIME ZONE 'UTC')` -> 将 datetime 列截断为日期，构建一个分组对象，`AT TIME ZONE 'UTC'`是处理带时区列的标准做法，在截断之前强制转换时区为目标时区。
- `arg_max(A, B)` -> 将 B 列作为索引，找出 A 列中最大索引对应的值。
- `read_parquet('data_dir/*/*/data.parquet', hive_partitioning=1)`，使用通配符来匹配使用 hive 分区的数据集，duckdb 会自动过滤不需要的子目录，实现快速查询。
- `GROUP BY 1` -> 按照 select 的第一列进行分组汇总。
- `ORDER BY 1` -> 按照 select 的第一列进行排序。

如果要重采样为任意时间周期，应该使用更强大的`time_bucket`: `time_bucket(INTERVAL 'n units', datetime)`

```sql
SELECT 
    time_bucket(INTERVAL '{interval_str}', datetime AT TIME ZONE 'UTC') as bucket,
    symbol,
    arg_max(close, datetime) as close
FROM read_parquet('{DATA_PATH}/*/*/data.parquet', hive_partitioning=1)
WHERE symbol IN ({symbols_str})
  AND datetime >= '{query_start}'
  AND datetime <= '{end_date}'
GROUP BY bucket, symbol
ORDER BY bucket ASC
```
