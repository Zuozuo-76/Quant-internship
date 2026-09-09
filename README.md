# 逐笔成交降采样、因子评价与策略回测项目

本项目研究逐笔数据降采样、日频因子与执行约束回测，以及独立的分钟线方向分类。09/09/2026 修正了历史 IC 和组合波动率的收益可用时序，以及 LSTM 多数类 baseline。当前结果与验证边界见 [时序修正与重跑报告](reports/timing-correction-2026-09-09.md)。

1. 将 2025-04-01 至 2026-06-30 的 300 只股票逐笔成交数据降采样为日频和分钟频；
2. 使用 `adjfactor.pkl` 对 OHLC 价格复权；
3. 构建 18 个因子（含下行波动率、特质波动率和下行 Beta 风险因子），计算 IC / IR / ICIR / Rank IC / Rank IR / Rank ICIR，并做五分组评价；
4. 以 1,000 万元初始资金完成信号日后一个交易日执行的回测，加入股票池等权基准、超额收益、换手率、滑点、平方根冲击成本、涨跌停与停牌限制，并保留前视方案作为错误对照；
5. 提供行业和对数市值横截面中性化接口；当前匿名股票数据没有真实行业、市值字段，因此只输出明确的跳过状态，不伪造暴露；
6. 完成三档成本敏感性、四步因子消融、因子相关性和分组图，并比较每日调仓、5 日调仓和 Top20 缓冲区换手控制；
7. 用 PyTorch `torch.nn.LSTM`、自动求导和 Adam，对 6 只股票、140 个交易日做 3 折扩展窗口 walk-forward 样本外预测，并与多数类和逻辑回归基线比较。

早期项目总报告与 Overleaf 导出为历史材料，版本对应关系见 [版本说明](reports/VERSIONS.md)。当前低风险比较见 [`reports/降低波动与回撤策略报告.md`](reports/降低波动与回撤策略报告.md)，字段与文件口径见 [`reports/数据与字段说明.md`](reports/数据与字段说明.md)。

## 主要结果

- K 线 CSV：`kline_data/daily_kline.csv`，按 `date, code` 展开的 90,600 行日频长表，包含 OHLC、成交量/额、主买卖量/额；
- 日频：`processed/daily/<字段>.csv`，每个字段一张 `302 × 300` 表；
- 分钟频：`processed/minute/<字段>/<YYYYMMDD>.csv`，每个字段每天一张 `253 × 300` 表；
- 因子：`factors/`；
- 因子评价、分组汇总、相关性、中性化状态和诊断图：`evaluation/`；
- 回测净值、等权基准、超额收益、换手和成本拆分、成本敏感性、因子消融、换手控制、逐笔审计与图表：`backtest/`；
- 低风险版本：`backtest/low_risk_strategy_metrics.csv`、`low_risk_strategy_results.csv`、持仓、因子权重和净值回撤图；
- PyTorch LSTM 模型（`model.pt`）、各折检查点、样本外预测、完整分类指标、混淆矩阵、逻辑回归基线和比较图：`lstm/`；
- 中文报告：`reports/`；历史中文 Overleaf 材料：`overleaf_upload/`；历史英文材料：`overleaf_upload_english_latest/`。

11 个基础字段为：`open, high, low, close, volume, trade_count, amount, buy_volume, sell_volume, buy_amount, sell_amount`。

## 复现

环境为 Python 3.10+。建议先创建独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

原始逐笔包 `task.tar.gz` 未公开提供。完整重建需要自行取得有权使用的原始数据和对应 `adjfactor.pkl`；仓库本身不构成原始输入的完整分发，也不提供不存在的下载地址。有原始输入后再执行：

```bash
tar -xzf task.tar.gz
python pipeline_code/downsample_tick_data_and_apply_adjustment.py --input data --adjfactor adjfactor.pkl --output processed
python scripts/build_kline_csv.py
python scripts/analyze_factors_backtest.py
python scripts/train_lstm_pytorch.py --max-codes 6 --max-dates 140 --epochs 3 --device auto
python scripts/validate_outputs.py
python -m unittest discover -s tests -v
```

无需原始数据的确定性演示：

```bash
python scripts/run_synthetic_demo.py --output tmp/synthetic-demo
```

该命令生成 40 个虚拟代码、110 个工作日的合成日频输入，运行实际因子构建、清洗和次日开盘回测，输出交易、持仓、权重及汇总。固定 seed=42。它验证日频管线可运行，不证明历史收益、不包含逐笔重建或 LSTM 训练；输出目录必须不存在，避免覆盖已有实验。

仓库中已保存的 CSV/图片可直接查看；真实数据重跑与合成 demo 的结果不可混用。当前网站跟踪配置为 `vol_target_15_baseline_signal`，对应 `historical_20d_ic`、原 15 个基线因子、`five_day_buffer60_top30`、次日开盘执行及 15% 波动率目标。18 是已评价因子总数。最新 [指标](backtest/low_risk_strategy_metrics.csv) 与 [报告](reports/timing-correction-2026-09-09.md) 应一起阅读。 [作品集详情](https://github.com/Zuozuo-76/my-website/blob/main/projects/quant-research.html)。

若另有真实行业和市值暴露文件，可在分析脚本中启用中性化：

```bash
python scripts/analyze_factors_backtest.py --risk-exposures data/risk_exposures.csv
```

暴露文件必须包含 `code, industry, market_cap`，也可以增加 `date` 做逐日暴露；模板见 [`examples/risk_exposures.example.csv`](examples/risk_exposures.example.csv)（仅演示字段，不能用于真实研究）。在缺少真实暴露时，程序会将状态写入 `evaluation/neutralization_status.json`，不会根据匿名代码或流动性代理伪造行业、市值。

`--device auto` 会在 Apple Silicon Mac 上优先使用 MPS，其他环境依次选择 CUDA 或 CPU。

核心流程代码统一放在 `pipeline_code/`，文件名直接说明用途。`downsample_tick_data_and_apply_adjustment.py` 是标准的原始逐笔重建入口；`apply_adjustment_factor_to_existing_tables.py` 仅用于已有同源未复权中间表时的替代路径。

## 关键口径

- 复权价：`Price / 100 × adjfactor`；成交额按实际成交价格计算，不随复权价变化。
- 主买为 `BSFlag=0`，主卖为 `BSFlag=1`，集合竞价 `BSFlag=2` 计入总量/总额但不归入主买主卖。
- 分钟索引含 `09:15–09:25`、`09:30–11:30`、`13:00–15:00`，共 253 个时间点；`15:00` 不可遗漏。
- 无成交分钟的 OHLC 使用前一分钟收盘价；日频无成交价格使用前一交易日收盘价；量、笔数和金额填 0。
- 信号在交易日 t 收盘后形成，最早在 t+1 收盘或开盘执行，随后持有至 t+2；不再假设收盘信号能按同一收盘价成交。
- 交易成本包括卖出手续费 5bp、买卖双方基础滑点各 5bp，以及按订单金额占当日成交额平方根计算、单边不超过 2% 的冲击成本。
- 停牌、跌停卖不出和涨停买不进会冻结或阻止相应交易；限制明细和成本逐笔记录在 `backtest/backtest_trades.csv`。
- 市场基准为入场日可买股票的每日等权收益；`backtest/benchmark_excess_returns.csv` 保存策略毛收益、净收益、基准收益和主动收益路径。
- `backtest/cost_sensitivity.csv` 包含乐观、基准和悲观三档成本；`backtest/factor_ablation.csv` 包含一至四因子的逐步组合结果。
- `backtest/turnover_control_comparison.csv` 比较每日 Top10、5 日 Top10、每日缓冲区和 5 日缓冲区；当前对外跟踪政策 `five_day_buffer60_top30` 每 5 日调仓，目标 Top30、退出缓冲 Top60。
- `leaky_same_day_ic` 使用未来收益计算当前权重，仅作为前视偏差对照；历史方案 `historical_20d_ic` 按完整交易日历滞后 2 期：信号日 s 的收益在 s+2 才结束，因此 t 收盘仅可使用 s≤t−2 的 IC。
- 低风险版本使用截至信号日已结束持有期的 60 日 EWMA ICIR（交易日历滞后 2 期）（单因子上限 25%）、正向得分乘逆波动率权重（单股目标上限 4%）和 15%/18% 波动率目标（总敞口 0.5–1.0）；组合波动率同样只纳入信号日已结束的持有期；本次保持既有配置，不重新按留出期择优。
- LSTM 使用按日期扩展的 walk-forward：每一折只用过去训练和验证，并在互不重叠的未来日期上测试；多数类概率仅由每折训练标签估计并写入逐条测试预测，合并比较保留各折概率；多数类、逻辑回归和 LSTM 使用相同测试窗口，报告 Accuracy、Balanced Accuracy、Precision、Recall、F1、ROC AUC、PR AUC 和混淆矩阵。
- 所有股票代码在长表 CSV 中按字符串并带引号输出，避免丢失前导零。
