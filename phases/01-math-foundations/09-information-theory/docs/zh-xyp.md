# 信息论

> 信息论衡量惊讶。Loss function 建立于其上。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1, Lesson 06（概率）
**时间：** ~60 分钟

## 术语对照

- entropy，熵
- cross-entropy，交叉熵
- KL divergence，KL 散度
- mutual information，互信息
- perplexity，困惑度
- bits，比特
- nats，纳特
- negative log-likelihood，负对数似然
- label smoothing，标签平滑
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 信息量 | "意外程度" | 编码一个事件所需的 bits (或 nats) 数：-log(p) |
| Entropy | "随机性" | 分布所有结果的平均意外程度。衡量不可减少的不确定性。 |
| Cross-entropy | "Loss function" | 用模型分布 Q 编码来自真实分布 P 的事件时的平均意外程度。 |
| KL divergence | "分布之间的距离" | 用 Q 而不是 P 多浪费的额外 bits。等于 cross-entropy 减 entropy。不对称。 |
| Mutual information | "X 和 Y 的关联度" | 知道 Y 后对 X 的不确定性减少了多少。零 = 独立。 |
| Perplexity | "模型有多困惑" | Cross-entropy 的指数。模型在每一步从多大的等效词表中选择。 |
| Bits | "Shannon 的单位" | 以 log 底 2 衡量的信息。1 bit 解决一次公平硬币投掷。 |
| Nats | "ML 的单位" | 以自然 log 衡量的信息。PyTorch 和 TensorFlow 默认使用。 |
| Negative log-likelihood | "NLL loss" | 对 one-hot 标签与 cross-entropy loss 相同。最小化它 = 最大化正确预测的概率。 |

## 扩展阅读

- [Shannon 1948：通信的数学理论](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 原始论文
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) —— entropy 和 KL divergence 的最佳视觉解释

## 学习目标

- 从零计算 entropy、cross-entropy 和 KL divergence 并解释其关系
- 推导为什么最小化 cross-entropy loss 等价于最大化 log-likelihood
- 计算特征与目标之间的 mutual information 以对特征重要性排序
- 将 perplexity 解释为语言模型从中选择的等效词汇量

## 问题

你在每个分类模型中都调 `CrossEntropyLoss()`。每篇语言模型论文你都看到"perplexity"。你在 VAE、蒸馏和 RLHF 中读到 KL divergence。这些不是孤立的概念。它们是同一个想法换了不同的外衣。

信息论给了你推理不确定性、压缩和预测的语言。Claude Shannon 在 1948 年发明了它来解决通信问题。事实是，训练一个神经网络也是一个通信问题：模型试图通过学到的 weight 构成的 noisy channel 来传输正确的标签。

## 概念

### 信息量（信息量 = 意外程度）

当不可能的事发生，它携带更多信息。掷硬币正面？不惊讶。中彩票？非常惊讶。概率为 p 的事件的信息量为：

```
I(x) = -log(p(x))
```

log 以 2 为底得 bits。以 e 为底得 nats。同一种思想，不同单位。

确定事件携带零信息。你早就知道会发生。

### Entropy（平均意外程度）

Entropy 是分布中所有可能结果的期望意外量。

```
H(P) = -Σ p(x) · log(p(x))
```

公平硬币有最大的二元变量 entropy：1 bit。偏倚硬币（99% 正面）entropy 低：0.08 bits。Entropy 衡量分布中不可减少的不确定性。你无法压缩到它以下。

### Cross-Entropy（你每天用的 loss function）

Cross-entropy 衡量当你用分布 Q 来编码实际来自分布 P 的事件时的平均意外程度。

```
H(P, Q) = -Σ p(x) · log(q(x))
```

P 是真实分布（标签）。Q 是模型的预测。如果 Q 完美匹配 P，cross-entropy 等于 entropy。任何不匹配都会让它更大。

在分类中，P 是 one-hot vector（正确类概率为 1，其余为 0）。这使 cross-entropy 简化为：

```
H(P, Q) = -log(q(正确类))
```

这就是分类的 cross-entropy loss 公式的全部。最大化正确类的预测概率。

### KL Divergence（分布之间的距离）

KL divergence 衡量你因用 Q 而不是 P 而多出的意外量。

```
D_KL(P || Q) = Σ p(x) · log(p(x)/q(x)) = H(P,Q) - H(P)
```

Cross-entropy = entropy + KL divergence。训练中真实分布的 entropy 是常数，所以最小化 cross-entropy 等于最小化 KL divergence。你把模型的分布推向真实分布。

KL divergence 不对称：D_KL(P||Q) ≠ D_KL(Q||P)。它不是真正的距离度量。

### Mutual Information

Mutual information 衡量知道一个变量对了解另一个变量有多大帮助。

```
I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)
```

如果 X 和 Y 独立，mutual information 为零。完美相关则等于任一变量的 entropy。

在特征选择中，特征与目标之间的高 mutual information 意味着该特征有用。低则意味着噪声。

### 条件 Entropy 与联合 Entropy

H(Y|X) 衡量观察到 X 后关于 Y 还剩多少不确定性。0 ≤ H(Y|X) ≤ H(Y)。决策树在每次分裂时选择最小化 H(Y|X) 的特征。

H(X,Y) 是联合分布的 entropy。H(X,Y) ≤ H(X) + H(Y)，在独立时取等号。被"缺失"的 entropy 正是 mutual information。

### 为什么 Cross-Entropy 是分类的 THE loss function

三重视角，同一结论：

- **信息论视角：** Cross-entropy 衡量你用模型的分布代替真实分布浪费了多少 bits。
- **最大似然视角：** 对 N 个训练样本，-Σ log(q(y_i)) 正是 cross-entropy loss。最小化 cross-entropy = 最大化训练数据的似然。
- **Gradient 视角：** Cross-entropy 关于 logits 的 gradient 就是 (predicted - true)。干净、稳定、计算快。

### Label Smoothing

用 soft target 替代 hard target。ε=0.1 时：`[0, 0, 1, 0]` → `[0.025, 0.025, 0.925, 0.025]`。增加 target 的 entropy，防止模型过度自信，改善 calibration。

### Perplexity

Perplexity = e^H(P,Q)。一个 perplexity 为 50 的语言模型平均来说如同要从 50 个等可能的 token 中随机选择。越低越好。

## 动手实现

### 第 1 步：信息量与 entropy

```python
import math

def information_content(p, base=2):
    if p <= 0: return float('inf')
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(p * information_content(p, base) for p in probs if p > 0)

print(f"公平硬币 entropy:   {entropy([0.5, 0.5]):.4f} bits")
print(f"偏倚硬币 entropy:   {entropy([0.99, 0.01]):.4f} bits")
```

### 第 2 步：Cross-entropy 与 KL divergence

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0: return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true = [0.7, 0.2, 0.1]
good = [0.6, 0.25, 0.15]
bad = [0.1, 0.1, 0.8]
print(f"CE (good): {cross_entropy(true, good):.4f} bits")
print(f"CE (bad):  {cross_entropy(true, bad):.4f} bits")
```

### 第 3 步：Classification loss

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
loss = cross_entropy_loss(0, logits)
print(f"Loss: {loss:.4f} nats, Perplexity: {math.exp(loss):.2f}")
```

### 第 4 步：Cross-entropy = Negative log-likelihood

```python
ce_loss = sum(cross_entropy_loss(label, logits)
              for label, logits in zip(true_labels, model_logits)) / n
nll = -sum(math.log(softmax(logits)[label])
           for label, logits in zip(true_labels, model_logits)) / n
# ce_loss == nll (二者相同)
```

### 第 5 步：Mutual information

```python
def mutual_information(joint_probs, base=2):
    rows, cols = len(joint_probs), len(joint_probs[0])
    margin_x = [sum(row) for row in joint_probs]
    margin_y = [sum(col) for col in zip(*joint_probs)]
    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi
```

## 练习

1. 计算均匀分布（26 字母）下英文的 entropy。然后用实际字母频率估计。哪个更高，为什么？
2. 手算 logits=[5.0, 2.0, 0.5]，true class=1 的 cross-entropy loss。什么 logits 给出零 loss？
3. 证明 KL divergence 不对称。选两个分布 P 和 Q，计算 D_KL(P||Q) 和 D_KL(Q||P)。
4. 构建 perplexity 计算函数。给定 (true_token_index, predicted_logits) 对列表，返回序列的 perplexity。
