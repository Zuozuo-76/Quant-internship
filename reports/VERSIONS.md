# 版本与结果来源

- 当前：09/09/2026 时序修正版，见 [重跑记录](timing-correction-2026-09-09.md)。网站跟踪 `vol_target_15_baseline_signal`，不是因子池全部 18 个因子的组合。
- `backtest/low_risk_strategy_metrics.csv`：该配置与其他风险版本的 `research80`、`holdout20`、`full`；网站指标仅对应指定配置。留出期参与过历史版本选择，因此为 Quasi-OOS。
- `lstm/`：3 折 next-bar direction classification；每折 majority baseline 只用训练标签估计。严格 OOS 仅指这些测试预测。
- 旧七因子项目总报告、30% 目标与固定杠杆报告、所有 `overleaf_upload*`、`output/` 中 PDF/TeX/ZIP 为修正前历史快照。它们保留供追溯，不代表当前结果。
- `scripts/build_report.py` 是旧汇总器，含历史文字模板；当前报告不依赖它。当前回测脚本生成分析报告和低风险比较，修正记录给出重跑与发布口径。

合成 demo 与示例暴露文件仅用于接口和执行验证，不是市场数据或收益证据。
