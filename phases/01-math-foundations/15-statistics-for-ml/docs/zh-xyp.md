# 机器学习中的统计

> 统计告诉你模型是真有提升还是纯属运气。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 06-07
**时间：** ~120 分钟

## 术语对照

- Pearson r，Pearson 相关系数
- Spearman rho，Spearman 秩相关系数
- covariance matrix，协方差矩阵
- p-value，p 值
- confidence interval，置信区间
- t-test，t 检验
- chi-squared test，卡方检验
- A/B testing，A/B 测试
- effect size，效应量
- bootstrap，自助法
- Type I error，第一类错误
- Type II error，第二类错误
- CLT，中心极限定理
- cross-validation，交叉验证
## 关键术语

| 术语 | 含义 |
|------|------|
| 均值 | 值之和除以计数。对 outlier 敏感 |
| 中位数 | 排序后的中间值。对 outlier 鲁棒 |
| 标准差 | 方差的平方根。原单位度量离散度 |
| IQR | Q3-Q1。中间 50% 的范围 |
| Pearson r | 衡量两变量间线性关联。[-1, 1] |
| Spearman rho | 用秩衡量单调关联 |
| p 值 | 假设 H0 为真时获得如此极端数据的概率 |
| 置信区间 | 给定置信水平下参数的可能取值范围 |
| t-test | 检验均值是否显著不同 |
| Effect size | 差异的大小，独立于样本量 |
| Bootstrap | 有放回重采样以估计抽样分布 |
| Type I error | 误报。H0 为真时拒绝 H0 |
| Type II error | 漏报。H0 为假时未能拒绝 H0 |
| 参数检验 | 假设特定分布（通常正态） |
| 非参数检验 | 无分布假设。基于秩或符号 |

## 学习目标

- 从零计算描述性统计、Pearson/Spearman correlation 和 covariance matrix
- 执行假设检验（t-test、chi-squared）并正确解读 p 值和置信区间
- 使用 bootstrap 重采样为任意指标构建置信区间，无需分布假设
- 通过 effect size 度量区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型 A 在测试集上 0.87，模型 B 0.89。你部署了 B。三周后，生产指标比之前更差。发生了什么事？

模型 B 并没有真正优于模型 A。0.02 的差异是噪声。你的测试集太小或方差太高，或两者兼有。你把随机性当成提升发布了出去。

这在 Kaggle 排行榜上、无法复现的论文里、在基于几百个样本就宣布胜者的 A/B 测试中持续发生。根因总是一个：有人跳过了统计。

统计给你区分信号与噪声的工具。它告诉你差异何时是真实的、你该有多自信、以及你需要多少数据才能信任一个结果。

## 概念

### 描述性统计

**中心趋势度量：** 均值（敏感于 outlier）、中位数（抗 outlier）、众数（分类数据有用）。
**离散度量：** 方差、标准差（与数据同单位，更可解释）、极差（max-min）、IQR（Q3-Q1，中间 50% 的范围）。

百分位数：P50=中位延迟，P95=差但非最差情况，P99=尾延迟（通常 10 倍于中位）。

样本方差用 (n-1) 而非 n（Bessel 修正），补偿样本均值非真实总体均值带来的系统低估。

### 相关性

**Pearson r：** 衡量线性关系。范围 [-1,1]。假设线性和近似正态。对 outlier 敏感。
**Spearman rho：** 衡量单调关系。用秩替代值。如果 y=x³，Pearson < 1 但 Spearman = 1。

黄金法则：相关性 ≠ 因果关系。

### Covariance Matrix

对于 d 个特征，covariance matrix C 是 d×d matrix，C[i][j] = Cov(feature_i, feature_j)。对角线 = 各特征方差。对称正半定。

PCA 对 covariance matrix 做特征分解。Eigenvector = 主成分方向，eigenvalue = variance 大小。这正是 Lesson 10 的内容，现在你明白了为何分解 covariance matrix：它编码了数据中所有的 pairwise 线性关系。

### 假设检验

```
H0（零假设）：默认假设，通常"无效应"
H1（备择假设）：你试图证明的东西

p 值 = P(数据如此极端 | H0 为真)
NOT P(H0 为真 | 数据)！这是最常见的误解。

p < 0.05：拒绝 H0，结果"统计显著"
p >= 0.05：未能拒绝 H0。（不代表 H0 为真）
```

置信区间：`x̄ ± z · (s/√n)`。z=1.96 为 95% 置信。解释：如果重复实验多次，95% 的计算区间会包含真实均值。NOT 有 95% 概率真实均值在这个特定区间内。

### t-test

- 单样本 t-test：总体均值是否等于某假设值
- 独立双样本 t-test（Welch 版）：两组均值是否不同。不假设等方差。总用 Welch 版除非有特殊理由。
- 配对 t-test：数据成对（同一数据划分上评估两个模型），计算差值，对差值做单样本 t-test

ML 中常用配对 t-test：两个模型在相同的 10 折 cross-validation 上对比。

### Chi-squared Test

检验观察频率是否匹配期望频率。χ² = Σ((O-E)²/E)。用于分类数据。

### A/B Testing for ML Models

```
1. 相同测试集（否则对比无意义）
2. 多个指标（accuracy/precision/recall/F1/latency）
3. 估计 variance（用 cross-validation 或 bootstrap）
4. 数据隔离（测试集不能在模型选择中使用过）

流程：k 折 cross-validation → 收集配对分数 → 配对 t-test → CI
→ effect size（Cohen's d）→ 判断实际意义
```

### 统计显著性 vs 实际显著性

巨大样本量下，微小差异也会统计显著。Model A=0.9234，B=0.9237，n=1,000,000，p<0.001。统计显著但 0.03% 的提升不值部署成本。

Cohen's d：d=0.2 小效应，0.5 中，0.8 大。总同时报告 p 值和 effect size。

### 多重比较问题

检验 20 个假设，α=0.05，期望 1 个误报。P(至少一个误报) = 1-0.95²⁰ = 0.64。

Bonferroni 修正：α_adjusted = α/m。保守但简单。

### Bootstrap

```
1. 有 n 个数据点
2. 有放回抽取 n 个样本（有些重复，有些缺失）
3. 在 bootstrap 样本上计算统计量
4. 重复 B 次（1000-10000）
5. Bootstrap 统计量分布 ≈ 抽样分布

95% CI = [2.5百分位, 97.5百分位]

对模型对比：
  1. 对测试集有放回重采样 indices
  2. 在重采样集上计算 metric_A 和 metric_B
  3. diff = metric_B - metric_A
  95% CI for diff = [2.5%ile, 97.5%ile]
  若 CI 不包含 0 → 差异显著
```

### 参数 vs 非参数检验

参数检验（t-test, ANOVA, Pearson r）假设特定分布（通常正态）。大样本下 CLT 使其近似成立。
非参数检验（Mann-Whitney U, Wilcoxon signed-rank, Spearman rho）无分布假设。小样本（n<30）、有 outlier、偏态、有序数据时用。

ML 实验中通常只有 5-10 折 CV（小样本），因此 Wilcoxon signed-rank 常比 t-test 更合适。

### CLT 在 ML 中的影响

样本均值的分布趋向正态（n≥30）。为 CI 和 t-test 提供理论基础。Mini-batch GD 有效因为 batch 上的平均 gradient 近似真实 gradient。集成方法：多模型平均预测比单模型更稳定。

### ML 论文中常见统计错误

1. 在训练集上测试
2. 无置信区间
3. 忽略多重比较
4. 混淆统计显著与实际显著
5. 不平衡数据上只用 accuracy
6. 只挑有利的指标报告
7. train/test 间信息泄露
8. 测试集太小无方差估计
9. 假设独立性其实数据有组内相关（同患者的医学图像）
10. P-hacking（不断换测试/子集/排除标准直到 p<0.05）

## 动手实现

```python
# 描述性统计
def mean(values): return sum(values) / len(values)
def median(values):
    s = sorted(values); n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
def std(values, ddof=1):  # ddof=1 for Bessel correction
    m = mean(values)
    return math.sqrt(sum((x - m)**2 for x in values) / (len(values) - ddof))

# Pearson & Spearman correlation
def pearson_r(xs, ys):
    n = len(xs); mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num / den if den else 0

def spearman_rho(xs, ys):
    # rank each, then Pearson on ranks
    ...

# Bootstrap CI
def bootstrap_ci(statistic_fn, data, B=10000, alpha=0.05):
    n = len(data)
    stats = []
    for _ in range(B):
        sample = [random.choice(data) for _ in range(n)]
        stats.append(statistic_fn(sample))
    stats.sort()
    lower = int(B * alpha / 2); upper = int(B * (1 - alpha / 2))
    return stats[lower], stats[upper]
```

完整实现见 `code/statistics.py`。
