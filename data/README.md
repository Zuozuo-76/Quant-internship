# 日频 K 线数据说明

文件：`daily_kline.csv`

每一行是一只股票在一个交易日的日频 K 线。

字段：

- `date`：交易日，格式 `YYYYMMDD`
- `code`：股票代码，已保留前导零
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `volume`：成交量
- `amount`：成交额
- `trade_count`：成交笔数
- `buy_volume`：主买量
- `sell_volume`：主卖量
- `buy_amount`：主买额
- `sell_amount`：主卖额

价格字段已经从原始逐笔数据中的整数价格恢复为真实价格，即原始 `Price / 100`。

补充文件：

- `minute_kline_20250401.csv`：分钟频 K 线样例长表，包含 20250401 当天 300 只股票的分钟数据。
- `factor_values.csv`：因子构建结果长表。
- `factor_formulas_zh.csv`：因子公式说明。
- `output_structure.csv`：每个字段一张表的输出结构清单。
