## 1. 数据获取模块

### 目标 

创建函数 `get_hourly_agg_trades(ticker: str, date: datetime.date, hour: int)`，实现：  
**输入**：交易对符号（如BTCUSDT），UTC日期，小时
**输出**：包含指定时间的全部聚合交易的Pandas DataFrame  

### 核心需求
1. 精准时间范围
   - 自动计算目标日期的UTC毫秒时间戳范围（00:00:00.000至23:59:59.999）  

2. 分页策略
   - 按照请求参数endTime和limit的组合不断向后追溯以获取完整的历史数据

3. 容错与性能  
   - 重试机制：对网络/API错误实施指数退避重试（最多4次）  
   - 动态限速：使用请求间隔（默认50ms/次）  

4. 数据完整性 
   - 校验字段完整性、交易时间戳是否在目标日期内  
   - 时间戳按照升序排列

### 技术约束

- 使用requests库负责网络请求
- 使用tenacity库管理重试逻辑

### 输出数据结构 
| 字段            | 类型                | 来源              |  
|-----------------|---------------------|-------------------|  
| trade_id        | int64               | API字段 `a`       |  
| timestamp       | datetime64[ns, UTC] | API字段 `T`       |  
| price           | float64             | API字段 `p`       |  
| quantity        | float64             | API字段 `q`       |  
| is_buyer_maker  | bool                | API字段 `m` 转换  |  

API接口：`GET /api/v3/aggTrades`

## 2. 数据管理模块

### 目标

设计一个高性能的本地数据存储管理系统，支持高频时间序列交易数据的写入、更新和查询，具备可扩展性和容错能力。

### 核心需求

1. 分层存储结构
    - 按交易对符号，年，月，日四级目录分区存储  
    - 文件名包含交易对符号和具体时间（如 `BTCUSDT_20240101_01.parquet`）  

2. 写入优化
   - 去重机制：基于 `trade_id` 字段自动跳过重复数据  
   - 压缩策略：使用 **ZSTD 压缩算法**（压缩比与性能平衡最优）  
   - 事务性写入：通过预写日志（WAL）确保数据完整性  

3. 查询优化
   - 时间范围查询：支持毫秒级时间窗口过滤（如 `2023-10-05 12:30:00.500` 至 `2023-10-05 12:35:00.000`）  
   - 元数据索引：  
     - 维护每个文件的统计信息（最小/最大时间戳、trade_id范围）  
     - 布隆过滤器（Bloom Filter）加速 `trade_id` 查询  

4. 增量更新
   - 冲突解决：  
     - 新数据覆盖旧数据（基于 `trade_id` 版本）  
     - 记录数据变更日志（Change Data Capture）  

### 存储目录结构
```  
data/  
├── metadata/  
│   ├── wal/                  # 预写日志（用于崩溃恢复）  
│   ├── schema_version=1/     # Schema定义文件（Avro格式）  
│   └── stats/                # 统计信息（JSON格式）  
└── symbol=BTCUSDT/  
    └── year=2023/  
        └── month=10/  
          └── day=1/
            └── BTCUSDT_20231001_01.parquet
            ├── BTCUSDT_20231001_02.parquet
            ...
```

## 3. 命令行界面

### 核心需求

1. 下载命令 (`download`) 
- 参数
  - `SYMBOLS`：支持多交易对（如 `BTCUSDT ETHUSDT`）
- 智能特性
  - 自动跳过已存在完整数据的日期
  - 失败日期记录到 `.retry` 文件供后续重试  

2. 更新命令 (`update`)
- 逻辑流程
  1. 读取现有数据的最新时间戳
  2. 仅抓取该时间戳之后的新数据 
  3. `--force` 强制全量更新（覆盖模式）  

3. 查询命令 (`query`)
- 输出模式
  - 默认：表格预览（显示前10行）  
  - `--stats`：显示统计摘要（行数、时间范围、成交量总和）  
  - `--output csv/json`：导出格式支持  

### 可视化组件
- 进度显示 
  - 多层级进度条（总进度 + 单个交易对进度）  
  - 动态指标：下载速度(MB/s)、剩余时间预估  
- 状态面板  
  - 实时显示：已处理日期/总日期、当前交易对  
  - 存储空间占用动态更新  

#### 错误恢复机制 
- 网络中断时保留临时文件，续传时自动检测  
- 提供 `cli retry` 命令重试所有失败任务  

### 技术栈约束
- 命令行：使用 `typer` 库创建命令行界面
- UI框架：使用 `rich` 库实现终端美化  

### 示例场景

```bash
# 下载BTCUSDT 2023年数据（显示彩色进度条）
cli download BTCUSDT --start 2023-01-01 --end 2023-12-31

# 增量更新ETHUSDT到最新
cli update ETHUSDT

# 查询BTCUSDT 10月数据并显示统计
cli query BTCUSDT --start 2023-10-01 --end 2023-10-31 --stats
```