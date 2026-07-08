# 作业输出补充说明

本文档补齐以下内容：主买量、主卖量、主买额、主卖额，分钟频数据，因子构建结果，因子公式，缺失值填充说明，以及“每个字段一张表”的输出结构。

## 1. 输出目录总览

```text
processed/
  daily/
    open.csv
    high.csv
    low.csv
    close.csv
    volume.csv
    trade_count.csv
    amount.csv
    buy_volume.csv
    sell_volume.csv
    buy_amount.csv
    sell_amount.csv

  minute/
    open/YYYYMMDD.csv
    high/YYYYMMDD.csv
    low/YYYYMMDD.csv
    close/YYYYMMDD.csv
    volume/YYYYMMDD.csv
    trade_count/YYYYMMDD.csv
    amount/YYYYMMDD.csv
    buy_volume/YYYYMMDD.csv
    sell_volume/YYYYMMDD.csv
    buy_amount/YYYYMMDD.csv
    sell_amount/YYYYMMDD.csv

factors/
  daily/
    amount_mean_sd_log.csv
    buy_sell_imbalance_5d.csv
    intraday_range_10d.csv
    reversal_5d.csv
  factor_long.csv
  factor_long.rds
  factor_metadata.csv
  factor_formulas_zh.csv
```

## 2. 每个字段一张表的结构

### 日频结构

日频数据在 `processed/daily/` 下。

每个字段一张 CSV 表。例如：

- `processed/daily/open.csv`
- `processed/daily/buy_volume.csv`
- `processed/daily/sell_amount.csv`

表结构：

```text
date,000012,000014,000019,...
20250401,...
20250402,...
...
```

含义：

- 行：交易日
- 列：股票代码
- 单元格：该股票在该交易日对应字段的值

### 分钟频结构

分钟频数据在 `processed/minute/` 下。

每个字段一个文件夹，文件夹内每天一张表。例如：

- `processed/minute/open/20250401.csv`
- `processed/minute/buy_volume/20250401.csv`
- `processed/minute/sell_amount/20250401.csv`

表结构：

```text
minute,000012,000014,000019,...
915,...
916,...
...
1500,...
```

含义：

- 行：分钟
- 列：股票代码
- 单元格：该股票在该分钟对应字段的值

每个分钟字段文件夹都有 302 个交易日文件。

## 3. 主买主卖字段

原始数据字段 `BSFlag` 的含义：

- `BSFlag == 0`：主买成交
- `BSFlag == 1`：主卖成交
- `BSFlag == 2`：集合竞价成交

本项目中：

- 主买量 `buy_volume`：`sum(Volume where BSFlag == 0)`
- 主卖量 `sell_volume`：`sum(Volume where BSFlag == 1)`
- 主买额 `buy_amount`：`sum((Price / 100) * Volume where BSFlag == 0)`
- 主卖额 `sell_amount`：`sum((Price / 100) * Volume where BSFlag == 1)`

集合竞价成交 `BSFlag == 2` 计入总成交量 `volume` 和总成交额 `amount`，但不计入主买量、主卖量、主买额、主卖额。

## 4. 基础字段计算公式

价格字段使用真实价格，原始数据中的 `Price` 已除以 100。

日频：

- 开盘价 `open`：当日第一笔成交价格
- 最高价 `high`：当日最高成交价格
- 最低价 `low`：当日最低成交价格
- 收盘价 `close`：当日最后一笔成交价格
- 成交量 `volume`：当日 `Volume` 求和
- 成交笔数 `trade_count`：当日逐笔成交记录数
- 成交额 `amount`：当日 `sum((Price / 100) * Volume)`

分钟频：

- 开盘价 `open`：该分钟第一笔成交价格
- 最高价 `high`：该分钟最高成交价格
- 最低价 `low`：该分钟最低成交价格
- 收盘价 `close`：该分钟最后一笔成交价格
- 成交量 `volume`：该分钟 `Volume` 求和
- 成交笔数 `trade_count`：该分钟逐笔成交记录数
- 成交额 `amount`：该分钟 `sum((Price / 100) * Volume)`

## 5. 缺失值填充说明

### 日频价格字段

日频 `open/high/low/close` 如果某只股票某天没有成交价格，则使用该股票前一交易日的收盘价填充。

对应字段：

- `open`
- `high`
- `low`
- `close`

### 分钟频价格字段

分钟频 `open/high/low/close` 如果某一分钟没有成交，则使用上一分钟的收盘价填充。

如果是当天开头阶段且没有上一分钟收盘价，则使用前一交易日收盘价填充。

如果是整个样本第一天的开头分钟，且没有前一交易日收盘价，则保留为缺失值。

### 非价格字段

非价格字段没有成交时填 0。

对应字段：

- `volume`
- `trade_count`
- `amount`
- `buy_volume`
- `sell_volume`
- `buy_amount`
- `sell_amount`

### 因子缺失值

滚动窗口因子在历史数据不足时保留为缺失值，不做向前填充。例如 20 日窗口因子前 19 个交易日无法计算完整窗口，因此为 `NA`。

因子评价和回测时，会删除因子值或未来收益率缺失的样本。

## 6. 因子构建结果

因子结果位于 `factors/`。

宽表结果：

- `factors/daily/amount_mean_sd_log.csv`
- `factors/daily/buy_sell_imbalance_5d.csv`
- `factors/daily/intraday_range_10d.csv`
- `factors/daily/reversal_5d.csv`

长表结果：

- `factors/factor_long.csv`
- `factors/factor_long.rds`

RStudio 推荐读取：

```r
factors <- readRDS("factors/factor_long.rds")
```

## 7. 因子公式

### amount_mean_sd_log

成交额滚动均值波动因子：

```text
amount_mean_sd_log = log(1 + sd(mean_1d(amount), mean_5d(amount), mean_10d(amount), mean_20d(amount)))
```

其中：

- `mean_1d(amount)`：1 日成交额均值
- `mean_5d(amount)`：过去 5 日成交额均值
- `mean_10d(amount)`：过去 10 日成交额均值
- `mean_20d(amount)`：过去 20 日成交额均值

### buy_sell_imbalance_5d

5 日主买主卖不平衡因子：

```text
buy_sell_imbalance = (buy_volume - sell_volume) / max(volume, 1)
buy_sell_imbalance_5d = mean_5d(buy_sell_imbalance)
```

### intraday_range_10d

10 日日内振幅因子：

```text
intraday_range = (high - low) / close
intraday_range_10d = mean_10d(intraday_range)
```

### reversal_5d

5 日反转因子：

```text
reversal_5d = -(close / close_lag_5d - 1)
```

含义：过去 5 日涨得越多，因子值越低；过去 5 日跌得越多，因子值越高。

## 8. 已生成的数据规模

- 股票数量：300
- 交易日数量：302
- 日频字段表：11 张
- 分钟频字段：11 个文件夹
- 每个分钟频字段文件夹：302 张日表
- 因子数量：4 个
- 因子长表行数：90,600 行，即 302 日 x 300 股票
