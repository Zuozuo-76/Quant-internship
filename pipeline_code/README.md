# 核心数据与因子代码

这个文件夹集中保存降采样、复权、因子构建和因子评价代码。文件名直接说明脚本用途；因子评价同时输出可复算的 CSV 和诊断图。

## 文件说明

- `downsample_tick_data_and_apply_adjustment.py`：读取逐笔成交 CSV，降采样为日频和分钟频，并在同一流程中给 OHLC 乘复权因子。
- `apply_adjustment_factor_to_existing_tables.py`：已有未复权日频/分钟宽表时，单独添加复权因子；它是替代路径，不需要在上一脚本之后重复执行。
- `construct_factors.py`：构建 4 个日频因子，输出因子宽表、长表和因子公式说明 CSV。
- `evaluate_factors.py`：读取已构建因子，按日做缩尾和标准化；如果提供真实风险暴露，则先对行业哑变量和对数市值做横截面 OLS 中性化，再计算 IC、Rank IC、IR、ICIR、五分组、多空收益、因子相关性，并生成每个因子的分组收益、累计净值和滚动 Rank IC 图。

## 推荐运行顺序

```bash
python3 pipeline_code/downsample_tick_data_and_apply_adjustment.py \
  --input data --adjfactor adjfactor.pkl --output processed

python3 pipeline_code/construct_factors.py \
  --processed processed --output factors

python3 pipeline_code/evaluate_factors.py \
  --processed processed --factors factors --output evaluation
```

若有真实行业和市值数据，可使用：

```bash
python3 pipeline_code/evaluate_factors.py \
  --processed processed --factors factors --output evaluation \
  --risk-exposures data/risk_exposures.csv
```

暴露表支持静态的 `code,industry,market_cap`，或带 `date` 的逐日版本。每个交易日覆盖率必须至少为 90%，市值必须为正。缺少真实暴露时会保留原有缩尾和 z-score 处理，并在 `evaluation/neutralization_status.json` 中记录未启用原因；字段模板见 `data/risk_exposures.example.csv`。

如果手里已有同口径但未复权的 `processed` 表，第一步改用：

```bash
python3 pipeline_code/apply_adjustment_factor_to_existing_tables.py \
  --source <未复权目录> --adjfactor adjfactor.pkl --output processed
```
