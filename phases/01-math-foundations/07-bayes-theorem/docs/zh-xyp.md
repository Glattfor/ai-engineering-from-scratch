# Bayes 定理

> 概率关乎你的预期。Bayes 定理关乎你学到了什么。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 06（概率基础）
**时间：** ~75 分钟

## 术语对照

- prior，先验
- likelihood，似然
- posterior，后验
- evidence，证据
- Naive Bayes，朴素贝叶斯
- Laplace smoothing，拉普拉斯平滑
- MLE，最大似然估计
- MAP，最大后验估计
- log-probability，对数概率
- false positive，假阳性
- conjugate prior，共轭先验
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Prior | "我的初始猜测" | 观察证据前的 P(假设)。在 ML 中：regularization 项。 |
| Likelihood | "数据拟合得有多好" | P(证据\|假设)。观察到的数据在特定假设下有多可能。 |
| Posterior | "我更新后的信念" | P(假设\|证据)。Prior 乘 likelihood 然后归一化。 |
| Evidence | "归一化常数" | P(data) 在所有假设上的总和。确保 posterior 和为 1。 |
| Naive Bayes | "那个简单的文本分类器" | 假设特征在给定类别后条件独立的分类器。尽管假设错误但效果很好。 |
| Laplace smoothing | "加一 smoothing" | 给每个特征加一个小计数，防止未见过数据的零概率。 |
| MLE | "就用频率" | 选择最大化 P(data\|params) 的参数。无 prior。小数据可能 overfit。 |
| MAP | "带 prior 的 MLE" | 选择最大化 P(data\|params)·P(params) 的参数。等价于正则化 MLE。 |
| Log-probability | "在 log 空间中工作" | 用 log(P) 代替 P，避免乘许多小数时的浮点 underflow。 |
| False positive | "错误警报" | 检测说阳性，但真实状态是阴性。驱动了基础概率谬误。 |

## 扩展阅读

- [3Blue1Brown：Bayes 定理](https://www.youtube.com/watch?v=HZGCoVF3YvM) —— 带医学检测例子的视觉解释
- [Stanford CS229：生成学习算法](https://cs229.stanford.edu/notes2022fall/cs229-notes2.pdf) —— Naive Bayes 及其与判别模型的关系
- [Think Bayes](https://greenteapress.com/wp/think-bayes/) —— 免费书，带 Python 代码的贝叶斯统计
- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 生产实现及各变体使用时机

## 学习目标

- 应用 Bayes 定理从 prior、likelihood 和 evidence 计算 posterior 概率
- 从零构建带 Laplace smoothing 和 log 空间计算的 Naive Bayes 文本分类器
- 比较 MLE 与 MAP 估计，并解释 MAP 如何对应 L2 regularization
- 使用 Beta-Binomial conjugate prior 实现用于 A/B testing 的序列贝叶斯更新

## 问题

一个医学检测准确率 99%。你检测结果为阳性。你真正患病的概率是多少？

大多数人说是 99%。真正的答案取决于疾病有多罕见。如果每万人中只有 1 人患病，阳性结果只意味着你约有 1% 的概率患病。其余 99% 的阳性结果来自健康人的误报。

这不是一个陷阱题。这就是 Bayes 定理。每一个垃圾邮件过滤器、每一个医疗诊断、每一个量化不确定性的机器学习模型都使用完全相同的推理。你从一个信念出发。你看到证据。你更新信念。

如果你在不懂这个的情况下构建 ML 系统，你会误读模型输出、设置错误的阈值、并交付过分自信的预测。

## 概念

### 从 joint probability 到 Bayes

你从 Lesson 06 已经知道条件概率是：

```
P(A|B) = P(A 且 B) / P(B)
```

对称地：

```
P(B|A) = P(A 且 B) / P(A)
```

两个表达式共享相同的分子：P(A 且 B)。令它们相等并重排：

```
P(A 且 B) = P(A|B) · P(B) = P(B|A) · P(A)

因此：

P(A|B) = P(B|A) · P(A) / P(B)
```

这就是 Bayes 定理。四个量，一个方程。

### 四个部分

| 部分 | 名称 | 含义 |
|------|------|------|
| P(A\|B) | Posterior | 看到证据 B 后你对 A 的更新信念 |
| P(B\|A) | Likelihood | 如果 A 为真，证据 B 出现的概率 |
| P(A) | Prior | 看到任何证据前你对 A 的信念 |
| P(B) | Evidence | 在所有可能性下看到 B 的总概率 |

Evidence 项 P(B) 充当归一化器。你可以使用全概率公式展开它：

```
P(B) = P(B|A) · P(A) + P(B|¬A) · P(¬A)
```

### 医学检测示例

某疾病每 10000 人 1 例。检测准确率 99%（检出 99% 的病人，对健康人有 1% 的误报率）。

```
P(患病)           = 0.0001    （prior：疾病罕见）
P(阳性|患病)      = 0.99      （likelihood：检测能检出）
P(阳性|健康)      = 0.01      （false positive rate）

P(阳性) = P(阳性|患病) · P(患病) + P(阳性|健康) · P(健康)
        = 0.99 · 0.0001 + 0.01 · 0.9999
        = 0.000099 + 0.009999
        = 0.010098

P(患病|阳性) = P(阳性|患病) · P(患病) / P(阳性)
            = 0.99 · 0.0001 / 0.010098
            = 0.0098
            = 0.98%
```

不到 1%。Prior 占主导。当一种情况罕见时，即使准确的检测也主要产生误报。这就是医生要求确认检测的原因。

### 垃圾邮件过滤器示例

你收到一封包含"彩票"这个词的邮件。它是垃圾邮件吗？

```
P(垃圾)                = 0.3     （30% 的邮件是垃圾邮件）
P("彩票"|垃圾)         = 0.05    （5% 的垃圾邮件包含"彩票"）
P("彩票"|非垃圾)       = 0.001   （0.1% 的正常邮件包含"彩票"）

P("彩票") = 0.05 · 0.3 + 0.001 · 0.7 = 0.0157

P(垃圾|"彩票") = 0.05 · 0.3 / 0.0157 = 0.955 = 95.5%
```

一个词将概率从 30% 推到 95.5%。真实的垃圾邮件过滤器同时对数百个词应用 Bayes。

### Naive Bayes：独立性假设

Naive Bayes 通过假设所有特征给定类别后条件独立，将其扩展到多个特征：

```
P(类别 | 特征1, 特征2, ..., 特征n)
  ∝ P(类别) · P(特征1|类别) · P(特征2|类别) · ... · P(特征n|类别)
```

"Naive"之处在于独立性假设。在文本中，词的出现并不独立（"New"和"York"是关联的）。但该假设在实践中出人意料地有效，因为分类器只需要对类别排序，不需要产生精确校准的概率。

由于分母对所有类别相同，你可以跳过它，只比较分子：

```
score(类别) = P(类别) · Π P(特征_i | 类别)
```

选 score 最高的类别。

### 最大似然估计（MLE）

如何从训练数据中得到 P(特征|类别)？计数。

```
P("free"|垃圾) = 包含"free"的垃圾邮件数 / 垃圾邮件总数
```

这就是 MLE：选择使观察到的数据最可能的参数值。对于离散计数，它退化为相对频率。

问题：如果一个词在训练时从未在垃圾邮件中出现过，MLE 给它概率零。一个未见过的词会毁掉整个乘积。用 Laplace smoothing 修正：

```
P(词|类别) = (count(词, 类别) + 1) / (类别中总词数 + 词汇量)
```

给每个计数加 1，确保概率永不为零。

### 最大后验估计（MAP）

MLE 问：什么参数最大化 P(数据|参数)？

MAP 问：什么参数最大化 P(参数|数据)？

由 Bayes 定理：

```
P(参数|数据) ∝ P(数据|参数) · P(参数)
```

MAP 在参数本身上加了一个 prior。如果你相信参数应该较小，你就把它编码为一个惩罚大值的 prior。这与 ML 中的 L2 regularization 完全相同。Ridge regression 中的"ridge"惩罚实际上是 weight 上的 Gaussian prior。

| 估计方法 | 优化目标 | ML 等价 |
|---------|---------|--------|
| MLE | P(data\|params) | 无正则化训练 |
| MAP | P(data\|params) · P(params) | L2 / L1 regularization |

### 贝叶斯 vs 频率主义：实践差异

频率派把参数视为固定的未知数。他们问："如果我多次重复这个实验，会发生什么？"

贝叶斯派把参数视为分布。他们问："鉴于我已观察到的，我对参数相信什么？"

对构建 ML 系统，实践差异：

| 方面 | 频率派 | 贝叶斯派 |
|------|--------|---------|
| 输出 | 点估计 | 值上的分布 |
| 不确定性 | 置信区间（关于过程） | 可信区间（关于参数） |
| 小数据 | 可能过拟合 | Prior 充当 regularization |
| 计算 | 通常更快 | 常需要 sampling（MCMC） |

### 为什么贝叶斯思维对 ML 重要

**Prior 即 regularization。** Weight 上的 Gaussian prior 就是 L2 regularization。Laplace prior 是 L1。每次你加一个正则化项，你就在做一个关于参数值该是什么的贝叶斯陈述。

**Posterior 即不确定性。** 一个单一的预测概率不告诉你模型对该估计有多自信。贝叶斯方法给你一个分布："我认为 P(垃圾邮件) 在 0.8 到 0.95 之间。"

**Bayes 更新即在线学习。** 今天的 posterior 成为明天的 prior。当你的模型看到新数据，它增量式更新信念，而不是从头重新训练。

## 动手实现

### 第 1 步：Bayes 定理函数

```python
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(患病|阳性) = {result:.4f}")
```

### 第 2 步：Naive Bayes 分类器

```python
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log(
                    (count + self.smoothing) / (total + self.smoothing * vocab_size)
                )
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class
```

Log probability 防止 underflow。乘许多小概率得到浮点无法表示的数。加总 log-probability 数值稳定且数学等价。

### 第 3 步：在垃圾邮件数据上训练

```python
train_docs = [
    "win free money now", "free lottery ticket winner",
    "claim your prize today free", "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon", "project update attached",
    "can we schedule a call", "quarterly report review",
    "lunch on thursday sounds good", "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]
for msg in test_messages:
    print(f"  '{msg}' → {classifier.predict(msg)}")
```

### Conjugate Prior

当 prior 和 posterior 属于同一族分布，该 prior 称为"conjugate"。这使贝叶斯更新在代数上干净。

| Likelihood | Conjugate Prior | Posterior | 例子 |
|-----------|----------------|-----------|------|
| Bernoulli | Beta(a, b) | Beta(a + 成功数, b + 失败数) | 硬币偏差估计 |
| Normal（方差已知） | Normal | Normal（加权均值，更小方差） | 传感器校准 |
| Poisson | Gamma(a, b) | Gamma(a + 计数和, b + n) | 到达率建模 |
| Multinomial | Dirichlet(alpha) | Dirichlet(alpha + 计数) | 主题建模 |

Beta-Bernoulli 更新规则极简：

```
Prior:     Beta(a, b)
Data:      s 次成功, f 次失败
Posterior: Beta(a + s, b + f)

没有积分。没有 sampling。只有加法。
```

### 序列贝叶斯更新

贝叶斯推断天然是序列式的。今天的 posterior 变成明天的 prior。

**Day 1：** Beta(1, 1) = uniform。均值 0.5。
**Day 2：** 7H, 3T → Beta(8, 4)。均值 0.667。
**Day 3：** 5H, 5T → 以 Beta(8,4) 为 prior → Beta(13, 9)。均值 0.591。

顺序无关紧要。Batch 更新和序列更新数学等价。但序列更新让你无需存储原始数据即可在每一步做决策。

### A/B Testing 的连接

1. **Prior：** Beta(1, 1) 对两个变体。
2. **Data：** A：50 点击/1000 浏览。B：65 点击/1000 浏览。
3. **Posteriors：** A: Beta(51, 951)。B: Beta(66, 936)。
4. **决策：** Monte Carlo 计算 P(B > A)。如果 > 0.95，部署 B。

优势：直接的概率陈述（"有 97% 的概率 B 更好"），没有 p 值困扰，可随时查看结果而不会放大误报率。

## 练习

1. **多次检测。** 患者两次独立检测均为阳性（准确率 99%，患病率 1/10000）。两次测试后 P(患病) 是多少？（用第一次检测的 posterior 作第二次的 prior）

2. **Smoothing 影响。** 用 smoothing 值 0.01、0.1、1.0、10.0 运行垃圾邮件分类器。top 词概率如何变化？当 smoothing=0 且一个词只出现在 ham 中时会怎样？

3. **添加特征。** 扩展 NaiveBayes 类，同时使用消息长度（短/长）作为特征。

4. **手算 MAP。** 给定数据（10 次掷硬币，7 次正面），用 Beta(2,2) prior 计算偏差的 MAP 估计。与 MLE (7/10) 比较。
