# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1, Lesson 01-04
**时间：** ~75 分钟

## 术语对照

- PMF，概率质量函数
- PDF，概率密度函数
- Bernoulli，伯努利分布
- Categorical，类别分布
- Poisson，泊松分布
- Uniform，均匀分布
- Normal，正态分布
- Gaussian，高斯分布
- Central Limit Theorem，中心极限定理
- softmax，Softmax 函数
- log-softmax，对数 Softmax
- cross-entropy，交叉熵
- logits，对数几率
- sampling，采样
- KL divergence，KL 散度
- log probability，对数概率
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 样本空间 | "所有可能性" | 实验中每个可能结果的集合 S |
| PMF | "概率函数" | 一个给出每个离散结果精确概率的函数，总和为 1 |
| PDF | "概率曲线" | 连续变量的 density 函数。在区间上积分得到概率 |
| 条件概率 | "在...条件下的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维的基础 |
| 独立性 | "它们互不影响" | P(A 且 B) = P(A) · P(B)。知道一个事件对另一个一无所知 |
| 期望值 | "平均值" | 所有结果的概率加权和。Loss function 是一个期望值 |
| 方差 | "扩散程度" | 与均值偏差平方的期望。高方差 = noisy、不稳定的估计 |
| Normal 分布 | "钟形曲线" | f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))。因 CLT 而无处不在 |
| Central Limit Theorem | "均值的分布变成正态" | 大量独立样本的均值无论来源如何都收敛到 normal 分布 |
| Joint 分布 | "两个变量一起" | P(X, Y) 描述 X 和 Y 每种结果组合的概率 |
| Marginal 分布 | "把另一个变量求和掉" | P(X) = Σ_y P(X, Y)。从 joint 中恢复单变量分布 |
| Log probability | "概率的对数" | log P(x)。将乘积转为加法，防止长序列中的数值 underflow |
| Softmax | "把分数变成概率" | softmax(z_i) = exp(z_i) / Σ exp(z_j)。将实值 logits 映射为有效的概率分布 |
| Cross-entropy | "Loss function" | -Σ(p_true · log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | "模型的原始输出" | Softmax 之前的未归一化分数。名字来自 logistic function |
| Sampling | "抽取随机值" | 按照概率分布生成值。模型生成输出的方式 |

## 扩展阅读

- [3Blue1Brown：到底什么是 Central Limit Theorem？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 为什么均值的分布变为正态的视觉证明
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖以上及更多内容的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 数值稳定性为什么重要以及如何实现

## 学习目标

- 从零实现 Bernoulli、Categorical、Poisson、Uniform 和 Normal 分布的 PMF 和 PDF
- 计算期望值和方差，并使用 Central Limit Theorem 解释为什么 Gaussian 分布无处不在
- 使用数值稳定性技巧（减去 max logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算 cross-entropy loss 并将其与 negative log-likelihood 联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50000 个候选中挑下一个词。一个 diffusion 模型通过从学到的分布中 sampling 来生成图片。这些都是概率在运行中的体现。

模型做出的每个预测都是一个概率分布。每个 loss function 衡量预测分布与真实分布的距离。每个训练步骤调整参数，使一个分布看起来更像另一个。没有概率，你无法读懂任何一篇 ML 论文、无法调试任何一个模型、也无法理解为什么你的训练 loss 变成了 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 和 1 之间的数。

```
抛硬币：
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

单次掷骰：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了全部概率论：
1. 对任何事件 A，P(A) ≥ 0
2. P(S) = 1（总有某事发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他一切（Bayes 定理、期望、分布）都从这三条规则推导出来。

### 条件概率与独立性

P(A|B) 是给定 B 已经发生时 A 的概率。

```
P(A|B) = P(A 且 B) / P(B)

例子：一副牌
  P(国王 | 人头牌) = P(国王且人头牌) / P(人头牌)
                   = (4/52) / (12/52)
                   = 4/12 = 1/3
```

两个事件独立，当知道其中一个对另一个一无所知：

```
独立：  P(A|B) = P(A)
等价于：P(A 且 B) = P(A) · P(B)
```

抛硬币是独立的。不放回地抽牌不是。

### Probability Mass Function vs Probability Density Function

离散随机变量有 probability mass function（PMF）。每个结果有特定的、可直接读取的概率。

```
PMF：P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有 probability density function（PDF）。单点处的 density 不是一个概率。概率来自在区间上对 density 积分。

```
PDF：f(x)

P(a ≤ X ≤ b) = ∫_a^b f(x) dx

f(x) 可以大于 1（是 density，不是概率）
∫_{-∞}^{+∞} f(x) dx = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择）。VAE latent space 使用 PDF（连续）。

### 常见分布

**Bernoulli：** 一次试验，两种结果。对 binary classification 建模。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p，方差 = p(1-p)
```

**Categorical：** 一次试验，k 种结果。对 multi-class classification 建模（softmax 输出）。

```
P(X = i) = p_i，其中 Σ p_i = 1
例子：P(猫) = 0.7，P(狗) = 0.2，P(鸟) = 0.1
```

**Uniform：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，对 k ∈ {1, ..., n}
连续：f(x) = 1/(b-a)，对 x ∈ [a, b]
```

**Normal (Gaussian)：** 钟形曲线。由均值 μ 和方差 σ² 参数化。

```
f(x) = (1 / √(2πσ²)) · exp(-(x - μ)² / (2σ²))

标准正态：μ = 0, σ = 1
  68% 的数据在 ±1σ 内
  95% 在 ±2σ 内
  99.7% 在 ±3σ 内
```

**Poisson：** 固定区间内稀有事件的计算。对事件发生率建模。

```
P(X = k) = (λᵏ · e^(-λ)) / k!
均值 = λ，方差 = λ
```

### 期望值与方差

期望值是概率加权的平均结果。

```
离散：  E[X] = Σ x_i · P(X = x_i)
连续：  E[X] = ∫ x · f(x) dx
```

方差衡量围绕均值的扩散程度。

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
标准差 = √Var(X)
```

在 ML 中，期望值表现为 loss function（数据分布上的平均 loss）。方差告诉你模型稳定性。Gradient 的高方差意味着 noisy 训练。

### Joint 与 Marginal 分布

Joint 分布 P(X, Y) 同时描述两个随机变量。

Joint PMF 示例（X = 天气，Y = 伞）：

| | Y=0（无伞） | Y=1（有伞） | Marginal P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Marginal 分布将另一个变量求和掉：

```
P(X = x) = Σ_y P(X = x, Y = y)
```

上表中的行列合计就是 marginal。

### 为什么 Normal 分布无处不在

Central Limit Theorem：大量独立随机变量的和（或均值）收敛到 normal 分布，无论原始分布是什么。

```
掷 1 个骰子：uniform 分布（平坦）
2 个骰子的均值：三角分布（有峰）
30 个骰子的均值：几乎完美的钟形曲线

这对任何初始分布都成立。
```

这就是为什么：
- 测量误差近似 normal（许多小的独立来源）
- 神经网络中的 weight 初始化使用 normal 分布
- SGD 中的 gradient noise 近似 normal（许多 sample gradient 的和）
- Normal 分布是给定均值和方差下的最大熵分布

### Log Probability

原始概率引起数值问题。将许多小概率相乘会迅速 underflow 到零。

```
P(句子) = P(词1) · P(词2) · ... · P(词_n)
        = 0.01 · 0.003 · 0.02 · ...
        → 0.0（约 30 项后 underflow）
```

Log probability 解决了这个问题。乘法变成了加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
             = -4.6 + -5.8 + -3.9 + ...
             → 有限数值（无 underflow）
```

规则：
- log(a · b) = log(a) + log(b)
- Log probability 始终 ≤ 0（因为 0 < P ≤ 1）
- 越负 = 越不可能
- Cross-entropy loss 是正确类别的 negative log probability

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / Σⱼ exp(z_j)

性质：
  - 所有输出在 (0, 1) 内
  - 所有输出之和为 1
  - 保持输入的相对顺序
  - exp() 放大了 logits 之间的差异
```

Softmax 技巧：在指数运算前减去最大 logit 以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

相同的结果，无溢出。
```

Log-softmax 将 softmax 和 log 组合以实现数值稳定。PyTorch 在 cross-entropy loss 内部使用它。

### Sampling

Sampling 意味着从分布中抽取随机值。在 ML 中：
- Dropout 随机采样哪些 neuron 置零
- 数据增强采样随机 transformation
- 语言模型从预测分布中 sampling 下一个 token
- Diffusion 模型采样 noise 并逐步去噪

从任意分布 sampling 需要逆变换 sampling、拒绝 sampling 或 reparameterization trick（用于 VAE）等技术。

## 动手实现

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(国王 | 人头牌) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子：E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中 sampling

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 和 log probability

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：Central Limit Theorem 演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化见 `code/probability.py`。

## 实际使用

用 NumPy 和 SciPy，以上所有都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零实现了这些。现在你知道库调用在做什么了。

## 练习

1. 为 exponential 分布实现逆变换 sampling。通过 sampling 10000 个值并比较直方图与真实 PDF 来验证。

2. 为两个不均匀的骰子构建 joint 分布表。计算 marginal 分布并检查骰子是否独立。

3. 为一个 5 类分类器计算 cross-entropy loss，当 logits 为 `[2.0, 0.5, -1.0, 3.0, 0.1]` 且正确类别是 index 3 时。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 写一个函数，接受 log probability 列表，返回最可能的序列、总 log probability 和等价原始概率。用 50 个词的句子测试，每个词的概率为 0.01。
