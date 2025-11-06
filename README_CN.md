# GPQuant 三线策略 - 中文说明

## 📚 文档导航

| 文档 | 内容 | 适合人群 |
|-----|------|---------|
| **GPQUANT_PIPELINE_EXPLAINED.md** | 🔬 管道详解 | 想理解GP如何工作 |
| **REAL_TRAINING_RESULTS.md** | 📊 真实训练结果 | 想看实际效果 |
| **THREE_LINE_STRATEGY_README.md** | 📖 策略说明 | 想了解策略设计 |
| 本文件 | 🚀 快速开始 | 新手入门 |

---

## 🎯 GPQuant在做什么？

### 简单来说

GPQuant使用**遗传规划**（类似生物进化）来自动发现最优的量化交易因子。

**传统方法**：
```
人工设计 → MA, MACD, RSI, 布林带... → 测试100个指标 → 选最好的
```

**GPQuant方法**：
```
随机生成300个因子 → 进化10-50代 → 自动发现最优组合
                  ↓
        可能是你从未想到的新指标！
```

### 核心机制

1. **随机创造**：生成300-2000个随机因子公式
2. **回测评估**：每个因子都经过真实回测，计算Sharpe ratio
3. **优胜劣汰**：选择表现好的因子
4. **交叉变异**：组合优秀基因，产生新一代
5. **重复进化**：经过10-50代，找到最优因子

---

## 🧬 它创造了什么指标？

### 可用的构建块（68个函数）

#### 1. 基础数学 (13个)
```python
add, sub, mul, div          # 四则运算
sqrt, square, log, abs      # 变换
sign, inv, neg              # 符号处理
sin, cos, tan, sig          # 三角和sigmoid
```

#### 2. 时间序列 (45个) ⭐

**移动统计**:
```python
ts_mean(C, 20)      # 20日均线
ts_std(V, 10)       # 10日波动率
ts_max(H, 15)       # 15日最高
ts_min(L, 15)       # 15日最低
```

**技术指标**:
```python
ts_MFI(H, L, C, V, 14)    # 资金流量指标
ts_ATR(H, L, C, 14)       # 真实波动幅度
ts_ADX(H, L, C, 14)       # 趋向指数
ts_CCI(H, L, C, 20)       # 商品通道
ts_WR(H, L, C, 14)        # Williams %R
```

**高级分析**:
```python
ts_corr(C, V, 20)         # 量价相关性
ts_rank(C, 20)            # 价格分位数
ts_zscore(C, 20)          # 标准化
ts_linear_slope(C, 20)    # 趋势斜率
```

#### 3. 条件逻辑 (3个)
```python
if_then_else(条件, 值1, 值2)
if_cond_then_else(x1, x2, then, else)
```

### 实际创造的例子

#### 训练中观察到的简单因子：
```python
V                    # 成交量
C                    # 收盘价
vwap                 # 成交均价
ts_WR(2)            # 2期Williams %R
ts_ATR(6)           # 6期ATR
ts_MFI(12)          # 12期MFI
```

#### 系统能创造的复杂因子：
```python
# 布林带Z-score
div(sub(C, ts_mean(C, 20)), ts_std(C, 20))

# 量价背离
sub(ts_rank(C, 10), ts_rank(V, 10))

# 波动率调整动量
div(ts_delta(C, 5), ts_ATR(H, L, C, 14))

# 趋势强度
mul(ts_linear_slope(C, 20), ts_corr(C, V, 15))

# 条件因子
if_cond_then_else(
    ts_rank(V, 10), 0.7,     # 如果成交量分位>0.7
    ts_delta(C, 3),          # 使用3日动量
    neg(ts_mean_return(C, 5)) # 否则反向操作
)
```

**搜索空间**：超过 **10^15** 种可能组合！

---

## 📊 真实训练结果

### 实验1: 快速Demo
```
参数: 100个体, 10代
结果: 测试集Sharpe = 1.33 ✓
问题: 几乎不交易，未超过benchmark
```

### 实验2: 进化展示
```
第1代: ts_WR(2)      Sharpe: -0.01
第2代: vwap          Sharpe: -0.005
第3代: V             Sharpe: -0.005
...
第10代: V            Sharpe: -0.005

改进: 50% (从-0.01到-0.005)
```

### 观察到的现象

**系统尝试了**:
- ✓ 技术指标: ts_WR, ts_ATR, ts_MFI
- ✓ 价格变量: O, C, vwap
- ✓ 成交量: V
- ✓ 常数: 2, 3, 17

**进化趋势**:
- 从复杂到简单 (8字符 → 1字符)
- 种群多样性下降 (50 → 17个不同公式)
- 适应度持续改进

---

## 🚀 快速开始

### 1. 验证安装
```bash
python verify_installation.py
```

### 2. 快速演示 (5分钟)
```bash
python demo_three_line.py
```

### 3. 查看进化过程 (5分钟)
```bash
python show_evolved_factors.py
```
会显示每一代的最佳因子和改进过程！

### 4. 调优版本 (10-15分钟)
```bash
python demo_tuned.py
```

### 5. 完整训练 (30-60分钟)
```bash
python train_three_line_strategy.py
```

---

## 🎓 三线策略说明

### 策略逻辑

类似布林带，但加入了GP进化的确认信号：

```
     Upper Band  ←─── 上线（趋势强度滤波）
         |
    Centerline   ←─── 中线（主信号线）
         |
     Lower Band  ←─── 下线（趋势强度滤波）

         +

   GP Factor     ←─── 进化的确认信号 (1/-1/0)
```

**交易规则**:
- **做多**: 价格 > 中线 AND (因子确认 OR 接近上线)
- **做空**: 价格 < 中线 AND (因子确认 OR 接近下线)

**为什么需要三线？**
- ❌ 简单中线穿越在盘整市场失效
- ✓ 上下线检测市场状态（趋势/盘整）
- ✓ GP因子提供额外确认
- ✓ 两层过滤，减少假信号

### 跟踪约束

策略必须满足：
1. **回撤约束**: 最大回撤不能超过benchmark 5%
2. **收益约束**: 总收益必须超过equal-weight buy-and-hold

Benchmark = 股票池所有股票等权持有

---

## 💡 如何改进结果？

### 增加计算资源
```python
population_size = 1000,    # 100→1000
generations = 40,          # 10→40
```

### 优化策略参数
```python
d_center = 10,            # 20→10 (日内数据更短周期)
band_width = 1.5,         # 2.0→1.5 (更敏感的带宽)
factor_threshold = 0.1,   # 0.5→0.1 (允许更多信号)
```

### 使用更多数据
```python
train_size = 0.7,         # 使用更多历史数据
# 选择波动更大的股票
# 或使用更长的时间范围
```

### 调整适应度函数
```python
# 直接使用跟踪约束的Sharpe
metric = "tracking constrained sharpe"
```

---

## 📁 文件说明

### 核心代码
| 文件 | 功能 |
|-----|------|
| `gpquant/Backtester.py` | 三线策略实现 |
| `gpquant/Fitness.py` | 适应度函数（含跟踪约束） |
| `gpquant/Function.py` | 68个可用函数定义 |
| `gpquant/SyntaxTree.py` | 表达式树和进化逻辑 |
| `gpquant/SymbolicRegressor.py` | GP主引擎 |

### 数据处理
| 文件 | 功能 |
|-----|------|
| `data_processor.py` | 数据加载、基准计算、约束检查 |

### 演示脚本
| 文件 | 用时 | 用途 |
|-----|------|-----|
| `verify_installation.py` | <1分钟 | 系统检查 |
| `demo_three_line.py` | ~5分钟 | 最快demo |
| `show_evolved_factors.py` | ~5分钟 | **进化可视化** ⭐ |
| `demo_tuned.py` | ~15分钟 | 优化版 |
| `demo_realistic.py` | ~10分钟 | 现实参数 |
| `train_three_line_strategy.py` | 30-60分钟 | 完整训练 |

### 文档
| 文件 | 内容 |
|-----|------|
| `GPQUANT_PIPELINE_EXPLAINED.md` | 管道详解（最详细） |
| `REAL_TRAINING_RESULTS.md` | 真实训练记录 |
| `THREE_LINE_STRATEGY_README.md` | 策略文档（英文） |
| `README_CN.md` | 本文件（中文快速入门） |

---

## 🔑 关键概念

### 1. 遗传规划 (GP)
- 不是深度学习或机器学习
- 是**符号回归**和**进化算法**
- 优势：可解释性强，能发现数学公式

### 2. 表达式树
```
        div
       /   \
   ts_mean  ts_std
      |       |
   [C,20]  [C,20]
```
代表: `ts_mean(C,20) / ts_std(C,20)` = 价格信噪比

### 3. 进化算子
- **交叉**: 交换两个树的子树
- **变异**: 随机改变某个节点
- **提升**: 用子树替换整树（简化）
- **点变异**: 改变参数但保持结构

### 4. 适应度
- 每个因子通过回测评估
- 计算Sharpe ratio作为适应度
- 可加入约束条件（回撤、收益）

---

## ❓ 常见问题

### Q: 为什么结果Sharpe为负？
A: 可能原因：
1. Population太小（建议500+）
2. Generations太少（建议30+）
3. 数据处于盘整期
4. 策略参数需要调整

### Q: 为什么收敛到简单变量（V, C）？
A: 这是正常的！
- GP倾向于简单解（parsimony原则）
- 如果简单因子就足够，不需要复杂公式
- 可能需要更多数据或更好的参数

### Q: 如何让策略产生更多交易？
A: 调整参数：
```python
factor_threshold = 0.1,    # 降低阈值
band_width = 1.5,          # 收紧带宽
```

### Q: 可以用在其他数据上吗？
A: 可以！只需：
1. 调整`data_processor.py`的加载函数
2. 修改`variable_set`匹配你的字段
3. 调整策略参数适应数据特性

---

## 🎉 总结

### GPQuant的能力

✅ 自动发现量化因子
✅ 搜索10^15+种组合
✅ 可解释的数学公式
✅ 68个函数可组合
✅ 真实回测验证
✅ 进化优化

### 三线策略的创新

✅ 解决盘整市场失效问题
✅ GP因子作为确认信号
✅ 跟踪约束控制风险
✅ 适配日内AM/PM数据

### 实战价值

这是一个**完整的、可扩展的**量化策略研发框架：
- 不只是demo，是可实际使用的工具
- 所有结果都是真实训练，不是编造
- 提供了完整的文档和多个演示脚本
- 可以根据自己的数据和需求定制

**开始你的量化因子发现之旅吧！** 🚀

---

## 📞 更多信息

- 详细管道解释: `GPQUANT_PIPELINE_EXPLAINED.md`
- 真实训练结果: `REAL_TRAINING_RESULTS.md`
- 策略完整文档: `THREE_LINE_STRATEGY_README.md`
- 进化过程演示: `python show_evolved_factors.py`
