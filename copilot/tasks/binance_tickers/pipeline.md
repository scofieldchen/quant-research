解决问题：获取币安永续合约的全部交易对信息

创建新的数据管道，`src/pipelines/binance_tickers`

1. 从官方API获取永续合约全部交易对的数据，将原始数据存储到`data/raw/binance_tickers`
2. 清洗数据，将清洗数据存储到`data/cleaned/binance_tickers_perp.parquet`

阅读代码，测试

1. 重试机制应该使用指数退避策略
2. 日志信息应该使用中文
3. 原始数据文件的时间戳应该采用日期时间而不是时间戳，例如`exchange_info_20260119.json`
4. 类型提示应该使用`dict`而不是`typing.Dict`
5. 清洗数据时保留以下字段，且转化为小写和下滑线分隔
  - symbol
  - pair
  - contractType
  - onboardDate -> 需转化为日期时间格式
  - status
  - maintMarginPercent
  - requiredMarginPercent
  - baseAsset
  - quoteAsset
  - marginAsset
  - pricePrecision
  - quantityPrecision

6. 将数据目录作为全局变量是否更好？
