# 时序修正与重跑记录 — 09/09/2026

## 修改与口径

- IC：收益索引为信号日 s，持有期 s+1 → s+2；滚动均值与 EWMA 均按交易日历滞后 2 期，缺失因子日期不会压缩时间轴。
- 波动率目标：仅纳入信号日已经结束的组合收益，排除尚未结束的上一持有期。
- LSTM：每折用训练标签均值估计 majority probability，逐条保存后合并评估。模型预测目标为 next-bar direction classification。
- 保持既有 `vol_target_15_baseline_signal`、Top30/Top60 缓冲、5 日调仓、15% 波动率目标和费用配置，不根据新留出期结果重新选策略。

## 修正前后：同名配置

| 样本 | 原成本后累计收益 | 修正后累计收益 | 修正后基准 | 修正后 Sharpe | 修正后最大回撤 |
|---|---:|---:|---:|---:|---:|
| full | 27.52% | 25.90% | 3.25% | 1.378 | -10.34% |
| research80 | 32.97% | 33.33% | 15.28% | 2.169 | -7.74% |
| holdout20 | -4.10% | -5.58% | -10.44% | -1.370 | -8.75% |

收益为累计收益，不是年化收益。完整可交易区间由 IC warm-up 后开始；修正后为 13/05/2025–26/06/2026，共 274 个持有期。末 20% 区间为 07/04/2026–26/06/2026，共 55 个持有期。warm-up 日期变化也改变了完整样本基准，前后差异不应全部归因于权重数值变化。

网站 18 个因子表示全部评价范围；当前配置使用 15 个基线因子。`research80`/`holdout20` 为同一条路径切片；该留出期参与过历史版本选择，仍是 Quasi-OOS。

## LSTM

CPU、seed=42、6 只股票、140 个交易日、3 折、每折 3 epochs 重跑完成，83,880 条 OOS 预测。ROC AUC 为 0.646451；Accuracy 68.9819%，Recall 8.1014%。训练集 majority baseline 的合并 Accuracy 为 68.9056%，ROC AUC 为 0.499184。不同折概率不同，合并 AUC 不必等于 0.5。

## 复现与验证

```bash
python -m unittest discover -s tests -v
python scripts/run_synthetic_demo.py --output tmp/synthetic-demo
python scripts/analyze_factors_backtest.py --processed /path/to/processed
python scripts/train_lstm_pytorch.py --processed /path/to/processed --output lstm --max-codes 6 --max-dates 140 --epochs 3 --device cpu
python scripts/validate_outputs.py --root .
```

- 24 项回归测试通过，包括未来价格扰动、缺失因子日期、IC 到期边界、组合波动率收益到期检查和训练/测试类别比例反转。
- 完整输出校验 `PASS`：11 张 302×300 日频表、3,322 张 253×300 分钟表、90,600 行 K 线、6 个低风险版本与 83,880 条 LSTM 测试预测。旧的收益必须超过 25% 断言已改为日收益复利与汇总指标一致性检查。
- 合成 demo 已实际运行，通过真实因子及回测函数输出有限日收益。仅为管线验证，不代表市场表现。
- 本次从既有 processed 表重跑全部因子、主回测、成本敏感性、因子消融、换手比较、6 个低风险版本及 LSTM；未重新聚合原始逐笔数据，也未重跑旧 30% 固定杠杆目标或重制历史 PDF/Overleaf。
- [输入哈希与配置](timing-correction-provenance.json)；[完整指标](../backtest/low_risk_strategy_metrics.csv)；[低风险比较](降低波动与回撤策略报告.md)。

执行仍是日频 OHLC/成交额模型：入场日全天状态用于停牌、涨跌停与冲击成本近似，不能声称逐笔撮合或所有执行假设均为开盘前已知。本次回归验证的是 IC 与波动率预算信息边界。
