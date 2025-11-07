# GPQuant 管道详解：它在创造什么指标？

## 🧬 核心概念：遗传规划 (Genetic Programming)

GPQuant 使用**遗传规划**来自动发现最优的量化因子。这不是简单地测试预定义的指标，而是**自动创造和进化新的指标组合**。

---

## 🔬 管道做什么？

### 整体流程

```
初始化 → 随机生成种群 → 评估适应度 → 选择 → 交叉/变异 → 新一代 → 重复
   ↓                                                                    ↑
 指标树                                                                  |
   ↓                                                                    |
执行计算 → 生成因子序列 → 转换为交易信号 → 回测 → 计算Sharpe → ────────┘
```

### 关键步骤详解

#### 1️⃣ **初始化：创建随机指标树**

系统随机生成 `population_size` (如300个) 表达式树，每棵树代表一个**候选指标**。

**示例树结构**：
```
        div
       /   \
    ts_mean  ts_std
      |        |
    [C, 10]  [V, 15]
```

这代表指标：`ts_mean(C, 10) / ts_std(V, 15)`
即：**10日收盘价均值 除以 15日成交量标准差**

---

## 📊 可用的指标构建块

GPQuant有**68个函数**可以组合，分为以下类别：

### 1. **基础数学函数** (13个)
用于变换数据的基本操作

| 函数 | 作用 | 示例 |
|-----|------|------|
| `square(x)` | 平方 | `square(C)` = 收盘价平方 |
| `sqrt(x)` | 平方根 (符号保护) | `sqrt(C-O)` |
| `log(x)` | 对数 (符号保护) | `log(V)` = 成交量对数 |
| `abs(x)` | 绝对值 | `abs(C-O)` = 实体大小 |
| `sign(x)` | 符号函数 | `sign(C-O)` = 涨跌方向 |
| `inv(x)` | 倒数 | `inv(C)` = 1/价格 |
| `neg(x)` | 取负 | `neg(C)` |
| `sin/cos/tan` | 三角函数 | `sin(ts_rank(C, 10))` |
| `sig(x)` | Sigmoid | `sig(momentum)` 归一化 |

### 2. **双变量运算** (7个)

| 函数 | 作用 | 示例 |
|-----|------|------|
| `add(x1, x2)` | 加法 | `add(H, L)` = 最高+最低 |
| `sub(x1, x2)` | 减法 | `sub(C, O)` = 涨跌幅 |
| `mul(x1, x2)` | 乘法 | `mul(C, V)` = 成交额 |
| `div(x1, x2)` | 除法 (安全) | `div(C, vwap)` |
| `max(x1, x2)` | 最大值 | `max(H, ts_mean(H, 5))` |
| `min(x1, x2)` | 最小值 | `min(L, ts_mean(L, 5))` |
| `mean(x1, x2)` | 平均 | `mean(O, C)` = 中间价 |

### 3. **条件逻辑** (3个)

| 函数 | 作用 | 示例 |
|-----|------|------|
| `if_then_else(x1, x2, x3)` | 条件选择 | 如果x1非零则x2，否则x3 |
| `if_cond_then_else(x1, x2, x3, x4)` | 条件比较 | 如果x1<x2则x3，否则x4 |
| `clear_by_cond(x1, x2, x3)` | 条件清零 | 如果x1<x2则0，否则x3 |

### 4. **时间序列函数** (45个) ⭐ 最强大

这是量化策略的核心！所有带 `ts_` 前缀的函数。

#### 4.1 差分和变化 (4个)
```python
ts_delay(x, d)        # d天前的值
ts_delta(x, d)        # d天变化量: x[t] - x[t-d]
ts_pct_change(x, d)   # d天涨跌幅: (x[t] - x[t-d]) / x[t-d]
ts_mean_return(x, d)  # d天平均收益率
```

**示例**：
- `ts_delta(C, 1)` = 今日涨跌
- `ts_pct_change(C, 5)` = 5日涨跌幅

#### 4.2 移动统计 (10个)
```python
ts_max(x, d)      # d天最高
ts_min(x, d)      # d天最低
ts_sum(x, d)      # d天总和
ts_mean(x, d)     # d天均值 (MA)
ts_std(x, d)      # d天标准差 (波动率)
ts_median(x, d)   # d天中位数
ts_skew(x, d)     # d天偏度
ts_kurt(x, d)     # d天峰度
ts_midpoint(x, d) # (max + min) / 2
ts_inverse_cv(x, d) # 均值/标准差 (信噪比)
```

**示例组合**：
- `div(ts_mean(C, 10), ts_std(C, 10))` = 10日价格信噪比
- `div(sub(C, ts_mean(C, 20)), ts_std(C, 20))` = **布林带Z-score**

#### 4.3 协方差和相关 (3个)
```python
ts_cov(x1, x2, d)     # x1和x2的d天协方差
ts_corr(x1, x2, d)    # x1和x2的d天相关系数
ts_autocorr(x, d, i)  # x自身滞后i期的d天自相关
```

**示例**：
- `ts_corr(C, V, 20)` = 20日量价相关性
- `ts_autocorr(C, 30, 1)` = 价格序列相关（趋势强度）

#### 4.4 归一化 (2个)
```python
ts_maxmin(x, d)   # (x - min) / (max - min) 归一化到[0,1]
ts_zscore(x, d)   # (x - mean) / std 标准化
```

#### 4.5 回归 (3个)
```python
ts_regression_beta(x1, x2, d)  # x1对x2回归的beta
ts_linear_slope(x, d)          # x的d天线性趋势斜率
ts_linear_intercept(x, d)      # x的d天线性趋势截距
```

**示例**：
- `ts_linear_slope(C, 20)` = 20日价格趋势强度
- `ts_regression_beta(C, vwap, 10)` = 价格对VWAP的回归系数

#### 4.6 位置和排名 (4个)
```python
ts_argmax(x, d)     # d天内最大值的位置
ts_argmin(x, d)     # d天内最小值的位置
ts_argmaxmin(x, d)  # 最大值位置 - 最小值位置
ts_rank(x, d)       # 当前值在d天窗口中的分位数
```

**示例**：
- `ts_rank(C, 20)` = 当前价格在20日内的位置（0-1）
- `ts_argmaxmin(C, 10)` = 高低点相对位置（趋势方向）

#### 4.7 技术指标均线 (3个)
```python
ts_ema(x, d)         # 指数移动平均 EMA
ts_dema(x, d)        # 双重指数移动平均 DEMA
ts_kama(x, d1, d2, d3) # Kaufman自适应移动平均 KAMA
```

#### 4.8 专业技术指标 (7个) 🎯

这些是经典的技术分析指标：

```python
ts_AROONOSC(H, L, d)     # Aroon振荡器
ts_WR(H, L, C, d)        # Williams %R
ts_CCI(H, L, C, d)       # 商品通道指数 CCI
ts_ATR(H, L, C, d)       # 平均真实波动 ATR
ts_NATR(H, L, C, d)      # 标准化ATR
ts_ADX(H, L, C, d)       # 平均趋向指数 ADX
ts_MFI(H, L, C, V, d)    # 资金流量指标 MFI
```

**训练中看到的例子**：
```
Generation 1: ts_MFI(12)
Generation 2: ts_ATR(6)
```
这些都是系统自动尝试的技术指标！

---

## 🔄 遗传规划如何遍历指标？

### 第1代：随机初始化

系统生成300个随机表达式，比如：

```python
Individual 1:   ts_mean(C, 15)                      # 简单15日均线
Individual 2:   div(ts_max(H, 20), ts_min(L, 20))   # 20日价格范围比
Individual 3:   ts_MFI(12)                          # 12日资金流
Individual 4:   sub(ts_ema(C, 10), ts_ema(C, 20))   # MACD类型
Individual 5:   mul(ts_rank(V, 10), sign(ts_delta(C, 1)))  # 量价配合
...
Individual 300: log(div(vwap, C))                   # VWAP偏离度
```

每个都会被评估：
1. 计算因子值
2. 通过三线策略生成交易信号
3. 回测计算Sharpe ratio
4. 记录适应度

### 第2代：进化

通过**锦标赛选择**选出优秀个体：
- 随机抽20个个体比较
- 选择Sharpe最高的作为父代

然后应用遗传算子：

#### 1. **交叉 (Crossover, 70%)**
交换两个表达式的子树

```
父代A:  div(ts_mean(C, 10), ts_std(C, 10))
                    ↓
父代B:  mul(ts_rank(V, 15), sign(C))
                    ↓
子代:   div(ts_rank(V, 15), ts_std(C, 10))  # 混合了两者
```

#### 2. **子树变异 (Subtree Mutation, 15%)**
替换子树为新随机子树

```
原始:   div(ts_mean(C, 10), ts_std(V, 5))
                             ↓ 变异
变异后:  div(ts_mean(C, 10), ts_corr(C, V, 20))  # V的标准差 → C-V相关性
```

#### 3. **提升变异 (Hoist Mutation, 10%)**
用子树替换整个树（简化）

```
原始:   div(sub(C, ts_mean(C, 20)), ts_std(C, 20))  # 布林带Z-score
              ↓ 提升
简化:   sub(C, ts_mean(C, 20))                      # 只保留价格偏离
```

#### 4. **点变异 (Point Mutation, 5%)**
改变节点但保持结构

```
原始:   ts_mean(C, 10)
              ↓
变异:   ts_median(C, 10)   # 均值 → 中位数
或:     ts_mean(C, 15)     # 参数改变
```

### 第3-20代：持续进化

每一代都重复这个过程，逐渐发现更好的因子组合。

**实际训练输出解读**：
```
Generation  1: ts_ATR(6)         fitness: -0.002
Generation  2: 17                fitness: -0.001
Generation  3: vwap              fitness: -0.001
...
Generation 10: V                 fitness: -0.001
```

这显示：
- **Generation 1**: 尝试了ATR指标 (真实波动)
- **Generation 2**: 进化出常数17 (几乎不交易)
- **Generation 3**: 尝试vwap
- **Generation 10**: 最终收敛到成交量V

---

## 🎯 实际创造的指标例子

### 训练过程中系统尝试过的指标：

#### 简单指标：
```python
C                           # 收盘价本身
V                           # 成交量
vwap                        # VWAP
17                          # 常数（不交易）
```

#### 技术指标：
```python
ts_MFI(12)                  # 12期资金流量指标
ts_ATR(6)                   # 6期平均真实波动
```

#### 可能的复杂组合（系统能创造）：
```python
# 趋势强度指标
div(ts_linear_slope(C, 20), ts_std(C, 20))

# 动量指标
mul(ts_delta(C, 5), ts_rank(V, 10))

# 均值回归指标
div(sub(C, ts_mean(C, 20)), ts_std(C, 20))  # 布林带Z-score

# 量价背离
sub(ts_rank(C, 10), ts_rank(V, 10))

# 波动率归一化动量
div(ts_delta(C, 5), ts_ATR(H, L, C, 14))

# 相对强度
div(
    ts_ema(C, 10),
    ts_ema(C, 30)
)

# 复杂组合
if_cond_then_else(
    ts_corr(C, V, 20),     # 如果量价相关性
    0.5,                    # > 0.5 (正相关)
    ts_linear_slope(C, 20), # 用趋势斜率
    neg(ts_delta(C, 5))     # 否则用反向动量
)
```

---

## 📈 指标 → 信号 → 收益 的完整管道

### 步骤分解：

#### 1. **指标生成** (Factor Generation)
```python
factor = tree.execute(data)
# 例如: [0.5, -0.2, 0.8, -0.3, 0.1, ...]
```

#### 2. **信号转换** (Factor → Signal)
使用三线策略：
```python
# 计算三条线
centerline = ts_mean(price, 20)
upper_band = centerline + 2 * ts_std(price, 20)
lower_band = centerline - 2 * ts_std(price, 20)

# 因子归一化
factor_signal = sign(factor) if abs(factor) > threshold

# 生成信号
if price > centerline and (factor_signal > 0 or price > upper_band):
    signal = 1   # 做多
elif price < centerline and (factor_signal < 0 or price < lower_band):
    signal = -1  # 做空
```

#### 3. **回测** (Signal → Asset)
```python
# 累积持仓
position = cumsum(signal)

# 计算收益
returns = position * price_change

# 扣除交易成本
net_returns = returns - transaction_costs

# 资产曲线
asset = initial_cash + cumsum(net_returns)
```

#### 4. **评估** (Asset → Fitness)
```python
# 计算年化收益
annual_return = (asset[-1] / asset[0]) ** (250 / len(asset)) - 1

# 计算波动率
volatility = std(asset.pct_change()) * sqrt(250)

# Sharpe比率
sharpe = annual_return / volatility

# 检查约束
if max_drawdown(asset) > benchmark_dd + 0.05:
    fitness = -10  # 惩罚
elif total_return(asset) < benchmark_return:
    fitness = -5   # 惩罚
else:
    fitness = sharpe  # 正常评分
```

---

## 🧮 搜索空间有多大？

### 组合爆炸：

以深度3的树为例：

**第1层**：68个函数选择
**第2层**：每个函数的参数
  - 单参数函数：68 + 6变量 + 20常数 = 94 选择
  - 双参数函数：94 × 94 = 8,836 组合

**第3层**：进一步展开...

**总搜索空间**：远超 **10^15** 种可能！

这就是为什么需要遗传规划：
- ✅ 不可能暴力枚举所有组合
- ✅ 通过进化智能搜索
- ✅ 从简单到复杂逐步优化

---

## 📊 变量集说明

我们的训练脚本使用的变量：

```python
variable_set = ["O", "H", "L", "C", "V", "vwap", "mktcap", "turnover"]
```

每个都可以用于构建因子：

| 变量 | 含义 | 用途示例 |
|-----|------|---------|
| `O` | 开盘价 | `sub(O, C)` = 跳空 |
| `H` | 最高价 | `sub(H, L)` = 波动幅度 |
| `L` | 最低价 | `div(C, L)` = 价格位置 |
| `C` | 收盘价 | `ts_delta(C, 1)` = 涨跌 |
| `V` | 成交量 | `ts_rank(V, 20)` = 量能 |
| `vwap` | VWAP | `div(C, vwap)` = 偏离度 |
| `mktcap` | 市值 | `log(mktcap)` = 规模因子 |
| `turnover` | 换手率 | `ts_mean(turnover, 5)` = 流动性 |

---

## 🎯 总结

### GPQuant在做什么？

1. **自动创造**成千上万的候选因子公式
2. **遍历**巨大的指标组合空间
3. **评估**每个因子的实际回测表现
4. **进化**优胜劣汰，保留好的基因
5. **优化**找到最适合当前数据的指标

### 它创造了什么指标？

- ✅ 简单指标：MA, EMA, STD
- ✅ 经典技术指标：MFI, ATR, ADX, CCI, Williams %R
- ✅ 统计指标：相关性、回归系数、偏度、峰度
- ✅ 复杂组合：布林带Z-score、量价背离、趋势强度
- ✅ **全新指标**：从未见过的数学组合！

### 为什么强大？

传统方法：
```
人工设计100个指标 → 测试 → 选最好的
```

GPQuant方法：
```
自动生成10^15种可能 → 智能进化 → 发现最优组合
```

**GPQuant能发现人类难以想到的指标组合！** 🚀

---

## 💡 实战建议

### 如何让GP发现更好的指标？

1. **增加种群**：500-2000个体，搜索更广
2. **更多代数**：30-50代，进化更充分
3. **丰富变量**：提供更多原始数据
4. **调整偏好**：`build_preference` 控制复杂度
5. **parsimony**：惩罚过于复杂的公式

### 查看发现的指标：

```python
print(sr.best_estimator)
# 输出例如: div(ts_corr(C, V, 15), ts_std(C, 20))
```

这就是GP为你数据**量身定制**的最优因子！
