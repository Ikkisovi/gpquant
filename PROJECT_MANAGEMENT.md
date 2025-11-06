# 项目管理指南：时序自适应线策略

## 📋 项目概览

**项目名称**: GPQuant 时序自适应线策略
**类型**: 私有策略研发项目
**基于**: gpquant (UePG-21/gpquant) fork
**主分支**: `claude/gquant-strategy-daily-data-011CUr94QhRtYosuzyxNs2iz`

---

## 📂 目录结构

```
gpquant/
├── gpquant/                      # 原始GPQuant核心（不要修改）
│   ├── Backtester.py
│   ├── Fitness.py
│   ├── Function.py
│   ├── SyntaxTree.py
│   └── SymbolicRegressor.py
│
├── gpquant_ext/                  # 你的策略扩展（可修改）⭐
│   ├── __init__.py
│   ├── ops_ts.py                 # 时序算子
│   ├── features.py               # 特征工程
│   ├── strategy_ts.py            # 固定策略模板
│   ├── fitness.py                # 信息夏普+约束
│   ├── benchmark.py              # 基准计算
│   ├── dataloader.py             # 数据加载
│   └── trainer.py                # 训练循环
│
├── train_ts_adaptive.py          # 训练入口
├── config_ts_adaptive.yaml       # 配置文件
├── demo_ts_adaptive.py           # 快速演示
│
├── data/                         # 数据目录（自行创建）
│   └── daily_am_pm_data.csv
│
├── results/                      # 结果目录（自动生成）
│   └── results_ts_adaptive.csv
│
├── experiments/                  # 实验记录（推荐创建）
│   ├── exp_001_baseline/
│   ├── exp_002_tuned_params/
│   └── ...
│
└── docs/                         # 文档
    ├── TS_ADAPTIVE_STRATEGY_README.md
    ├── QUICK_START_TS.md
    ├── GPQUANT_PIPELINE_EXPLAINED.md
    └── ...
```

---

## 🔄 开发工作流

### 1. 日常开发

```bash
# 1. 确保在正确的分支
git checkout claude/gquant-strategy-daily-data-011CUr94QhRtYosuzyxNs2iz

# 2. 查看状态
git status

# 3. 修改代码（在 gpquant_ext/ 中）
# 编辑你的策略文件...

# 4. 测试
python demo_ts_adaptive.py

# 5. 提交
git add gpquant_ext/
git commit -m "描述你的修改"
git push

# 6. 完整训练
python train_ts_adaptive.py
```

### 2. 创建新实验

```bash
# 为每个实验创建配置
cp config_ts_adaptive.yaml experiments/exp_002_config.yaml

# 编辑配置
vim experiments/exp_002_config.yaml

# 运行实验
python train_ts_adaptive.py --config experiments/exp_002_config.yaml \
    --output experiments/exp_002_results.csv
```

### 3. 版本管理

```bash
# 创建功能分支（可选）
git checkout -b feature/new-indicator
# 开发新功能...
git add .
git commit -m "Add new indicator: XXX"

# 合并回主分支
git checkout claude/gquant-strategy-daily-data-011CUr94QhRtYosuzyxNs2iz
git merge feature/new-indicator
git push
```

---

## 🧪 实验管理建议

### 创建实验记录系统

```bash
mkdir -p experiments
```

每个实验一个文件夹：
```
experiments/
├── exp_001_baseline/
│   ├── config.yaml
│   ├── results.csv
│   ├── notes.md
│   └── formulas.txt
│
├── exp_002_high_momentum/
│   ├── config.yaml
│   ├── results.csv
│   └── notes.md
│
└── ...
```

**notes.md 模板**：
```markdown
# Experiment 002: High Momentum Focus

## Date
2024-01-XX

## Hypothesis
增加动量特征权重可能提升信息夏普

## Configuration Changes
- population_size: 200 → 500
- Added: mom_rank_ts_slow (120 window)

## Results
- Information Sharpe: 1.85 (vs baseline 1.65)
- Constraint satisfaction: 90% (vs baseline 85%)

## Conclusion
✅ 成功，采纳此配置

## Next Steps
- 尝试结合波动率过滤
```

---

## 🔧 常见开发任务

### 添加新特征

1. 编辑 `gpquant_ext/features.py`:
```python
# 在 make_features() 中添加
def make_features(panel, lookbacks=None):
    # ... 现有代码 ...

    # 添加你的新特征
    my_new_feature = calculate_my_feature(close, volume)

    features = pd.concat([
        # ... 现有特征 ...
        my_new_feature.rename('my_new_feature'),
    ], axis=1)
```

2. 测试：
```bash
python demo_ts_adaptive.py
```

### 添加新算子

1. 编辑 `gpquant_ext/ops_ts.py`:
```python
# 定义新函数
def _ts_my_operator(x, n: int):
    """你的算子逻辑"""
    # ... 实现 ...
    return result

# 创建Function对象
ts_my_op_func = Function(
    function=_ts_my_operator,
    name="ts_my_op",
    arity=2,
    is_ts=1
)

# 添加到映射
TS_OPERATOR_MAP = {
    # ... 现有算子 ...
    "ts_my_op": ts_my_op_func,
}
```

2. 重新注册：
```python
# 会自动在 trainer 中注册
```

### 调整策略参数

编辑 `config_ts_adaptive.yaml`:
```yaml
# 修改你想调整的参数
k_tanh: 2.0        # 原来 1.0
z_L: 30            # 原来 40
```

### 修改约束条件

编辑 `gpquant_ext/fitness.py`:
```python
# 在 fitness_info_sharpe_constrained() 中修改
cap_mdd_rel=0.08,      # 原来 0.05 (5% → 8%)
lambda_mdd=2.0,        # 原来 3.0 (降低惩罚)
```

---

## 📊 结果分析工作流

### 1. 查看训练结果

```bash
# CSV
cat results_ts_adaptive.csv

# 或用pandas
python -c "import pandas as pd; df=pd.read_csv('results_ts_adaptive.csv'); print(df)"
```

### 2. 分析最佳公式

```python
import pandas as pd

results = pd.read_csv('results_ts_adaptive.csv')

# 找出最佳fold
best = results.sort_values('information_sharpe', ascending=False).iloc[0]

print(f"Best formula: {best['formula']}")
print(f"Info Sharpe: {best['information_sharpe']:.4f}")
print(f"Relative MDD: {best['relative_mdd']:.2%}")
```

### 3. 可视化（可选）

创建 `analyze_results.py`:
```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results_ts_adaptive.csv')

# 信息夏普分布
plt.figure(figsize=(10, 6))
plt.hist(results['information_sharpe'], bins=20)
plt.xlabel('Information Sharpe')
plt.ylabel('Frequency')
plt.title('Information Sharpe Distribution')
plt.savefig('results/sharpe_distribution.png')

# 约束满足率
satisfaction_rate = results['meets_all_constraints'].mean()
print(f"Constraint satisfaction: {satisfaction_rate:.1%}")
```

---

## 🔐 数据安全

### Git忽略敏感文件

创建 `.gitignore`（如果还没有）:
```bash
# 数据文件
*.csv
data/
raw_data/

# 结果
results/
experiments/*/results.csv

# Python
__pycache__/
*.pyc
*.pyo

# 临时文件
*.tmp
*.log

# 环境
.env
venv/
```

### 备份重要结果

```bash
# 定期备份到安全位置
rsync -av results/ /path/to/backup/results_$(date +%Y%m%d)/
```

---

## 🚀 性能优化建议

### 1. 快速迭代（开发阶段）

```yaml
# config_fast.yaml
train_months: 6        # 12 → 6
gp_config:
  population_size: 100 # 200 → 100
  generations: 15      # 30 → 15
```

```bash
python train_ts_adaptive.py --config config_fast.yaml
```

### 2. 生产级训练（最终）

```yaml
# config_production.yaml
train_months: 18       # 12 → 18
gp_config:
  population_size: 500 # 200 → 500
  generations: 50      # 30 → 50
```

### 3. 并行训练（如果需要）

```python
# 修改 trainer.py 支持多进程
# 或在不同机器上运行不同fold
```

---

## 📝 最佳实践

### ✅ 推荐做的

1. **每次实验记录**：日期、假设、配置、结果、结论
2. **版本控制**：频繁commit，清晰的commit message
3. **配置管理**：每个实验独立config文件
4. **结果备份**：定期保存重要结果
5. **代码注释**：自定义功能要写清楚
6. **测试先行**：改动后先run demo再full training

### ❌ 避免做的

1. **直接修改原始gpquant代码**：保持在gpquant_ext/扩展
2. **忽略约束检查**：如果全fail，要分析原因
3. **过度优化**：小心过拟合
4. **忽略交易成本**：成本模型要真实
5. **跳过验证**：每次改动都要验证
6. **不记录实验**：否则忘记哪些尝试过

---

## 🛠️ 故障排除快速参考

| 问题 | 解决方案 |
|-----|---------|
| 导入错误 | `pip install pandas numpy pyyaml numba` |
| 数据未找到 | 检查 `daily_am_pm_data.csv` 路径 |
| 所有fitness=-999 | 放宽约束或增大population |
| 训练太慢 | 降低population和generations |
| 策略不交易 | 增大k_tanh或减小z_L |
| 过拟合 | 增大parsimony_coefficient |

---

## 📞 快速命令参考

```bash
# 测试系统
python demo_ts_adaptive.py

# 完整训练
python train_ts_adaptive.py

# 自定义训练
python train_ts_adaptive.py --config my_config.yaml --data my_data.csv

# 查看结果
cat results_ts_adaptive.csv

# 提交代码
git add gpquant_ext/
git commit -m "Update: XXX"
git push

# 查看状态
git status
git log --oneline -5
```

---

## 🎯 下一步建议

### 立即可做

1. ✅ 运行 `python demo_ts_adaptive.py` 验证系统
2. 📊 准备你的 `daily_am_pm_data.csv`
3. 🚀 运行第一次完整训练
4. 📝 创建 `experiments/` 目录结构

### 短期（1-2周）

1. 🧪 运行3-5个不同配置的实验
2. 📈 分析哪些特征最有效
3. 🔧 调优参数（k_tanh, z_L, 约束等）
4. 📚 记录实验笔记

### 中期（1-2月）

1. 🆕 添加你自己的特征和算子
2. 📊 实现结果可视化
3. 🧮 对比不同策略变体
4. 📖 优化文档和注释

### 长期

1. 🏭 部署到生产环境
2. 🔄 建立自动化训练pipeline
3. 📈 实盘验证
4. 🔬 继续研究改进

---

## 📚 相关文档

- **完整技术文档**: `TS_ADAPTIVE_STRATEGY_README.md`
- **快速入门**: `QUICK_START_TS.md`
- **GPQuant原理**: `GPQUANT_PIPELINE_EXPLAINED.md`
- **训练结果案例**: `REAL_TRAINING_RESULTS.md`

---

**你现在拥有一个完整的私有策略研发平台！开始训练吧！** 🚀
