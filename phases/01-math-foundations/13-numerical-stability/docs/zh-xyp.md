# 数值稳定性

> 浮点数是漏水的抽象。它会在训练中咬你，而你预见不到。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-04
**时间：** ~120 分钟

## 术语对照

- IEEE 754，IEEE 754 标准
- machine epsilon，机器精度
- catastrophic cancellation，灾难性抵消
- overflow，溢出
- underflow，下溢
- log-sum-exp trick，对数求和指数技巧
- stable softmax，稳定 Softmax
- mixed precision，混合精度
- loss scaling，损失缩放
- bfloat16，bfloat16 格式
- gradient clipping，梯度裁剪
## 关键术语

| 术语 | 含义 |
|------|------|
| IEEE 754 | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准 |
| Machine epsilon | 某浮点格式下使 1.0+e≠1.0 的最小 e。float32 约 1.19e-7 |
| Catastrophic cancellation | 近似相等的浮点数相减，有效数字抵消，舍入噪声主导结果 |
| Overflow | 结果超过最大可表示值变 inf。exp(89) 使 float32 overflow |
| Underflow | 结果比最小可表示正数更接近零变 0.0。exp(-104) 使 float32 underflow |
| Log-sum-exp trick | 先提出 exp(max(x))，防止 exp 的 overflow 和 log(0) 的 underflow |
| Stable softmax | 先减 max(logits) 再 exp。数值相同，无 overflow 可能 |
| Mixed precision | float16 forward/backward 求速度，float32 master weight 求精度。典型加速 2-3× |
| Loss scaling | 在 backward 前乘大常数让 gradient 保持在 float16 的可表示范围内 |
| bfloat16 | Google 的 16 位格式：8 位指数（同 float32 范围）+ 7 位尾数（较低精度）。训练首选 |
| Gradient clipping | 若 ||grad|| > max_norm，缩放使 norm = max_norm。防梯度爆炸 |

## 学习目标

- 使用 max-subtraction trick 实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点运算中的 overflow、underflow 和 catastrophic cancellation
- 使用 centered finite difference 验证解析 gradient 与数值 gradient
- 解释为什么 bfloat16 优于 float16 用于训练，以及 loss scaling 如何防止 gradient underflow

## 问题

模型跑了三小时，loss 变 NaN。加 print 发现 logits 在 9000 步还正常，9001 步变 inf，9002 步所有 gradient 变 nan。或者：你用 float16 训练，精度比论文低 2%。问题在于论文用 float32，你用 float16 且没有正确的 scaling。32 bits 的累积舍入误差静默吃掉了你的精度。

数值稳定性不是理论问题。它是一次训练成功与静默失败之间的区别。

## 概念

### IEEE 754：浮点数的存储

```
Float32: [1 符号] [8 指数] [23 尾数]
Float16: [1 符号] [5 指数] [10 尾数]    范围: ±65,504
bfloat16: [1 符号] [8 指数] [7 尾数]    范围: ±3.4e38（同 float32）
```

float32 约 7 位十进制精度。float16 约 3 位，最大 65,504（ML 中太小）。bfloat16 是 Google 的答案：与 float32 相同范围，较少精度。训练神经网络范围比精度更重要。

### 为什么 0.1 + 0.2 ≠ 0.3

0.1 在二进制中无限循环，float32 截断到 23 位尾数。0.1 和 0.2 的存储值都不是精确的，它们的和不等于 0.3。别用 `==` 比较浮点数。

### Catastrophic Cancellation

两个接近的浮点数相减时，有效数字互相抵消，舍入噪声被提升到前导位。

```
a = 1.0000001  b = 1.0000000
真实差: 0.0000001    计算差: 0.00000011920929   相对误差: 19.2%
```

ML 中出现场景：方差计算（大数据均值下 E[x²]-E[x]²）、近似相等的 log-probability 相减、过小 epsilon 的有限差分 gradient。修复：重排公式避免大数相减。方差用 Welford 算法。

### Overflow 和 Underflow

exp(89) 在 float32 中溢出为 inf。exp(-104) 下溢为 0.0。

`exp()` 是 ML 中 overflow 的主要来源（softmax、sigmoid 中）。`log()` 往另一方向出问题：log(0.0) = -inf。

### Log-Sum-Exp Trick

```
不可靠: log(sum(exp(x_i)))
可靠:   c + log(sum(exp(x_i - c)))，其中 c = max(x)
```

证明：提出 exp(max(x)) 为因子。减去 max 后最大指数为 exp(0)=1，无 overflow。和至少为 1，log(1)=0，无 underflow 到 -inf。

### Stable Softmax

```
不可靠: softmax(x_i) = exp(x_i) / sum(exp(x_j))
可靠:   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

概率相同。计算安全。这不是优化，是正确性的要求。

### NaN 和 Inf：检测与预防

NaN 来源：0/0, inf-inf, inf*0, sqrt(-1), log(-1), 任何含 NaN 的运算。Inf 来源：exp(大正数), 1.0/0.0, float32 累积溢出。

预防策略：clamp exp(的输入，-80~80)、分母加 epsilon、log 内加 epsilon、gradient clipping、调试期每步 forward pass 后检测 NaN/Inf。

### 数值 Gradient Checking

```python
数值 gradient ≈ (f(x+h) - f(x-h)) / (2h)    # O(h²) 精度

相对误差 = |grad_analytical - grad_numerical| / max(|analytical|, |numerical|, 1e-8)
```

- <1e-7：完美 ✓
- <1e-5：可接受 ✓
- >1e-3：出错了 ✗

### Mixed Precision Training

```
1. float32 保存 master weight 副本
2. Forward pass 用 float16（快）
3. Loss 用 float32 计算（防 overflow）
4. Backward pass 用 float16（快）
5. 将 gradient 缩放回 float32
6. 用 float32 更新 master weight
```

问题：gradient 常极小（1e-8 或更小）。float16 将低于 ~6e-8 的全部下溢为 0。模型停止学习。修复：loss scaling。

```
1. Loss × 1024（放大）
2. 计算 (loss×1024) 的 gradient → 所有 gradient 放大 1024 倍
3. Gradient / 1024 再更新 weight → 净效果相同，无 underflow
```

### bfloat16 vs float16

float16：更多精度（10 vs 7 尾数位），但范围有限（最大 ~65,504）。训练尖峰期间 activation/logits 常超 65,504 → overflow。loss scaling 必需。

bfloat16：较少精度，但范围等同 float32（最大 ~3.4e38）。loss scaling 通常不需要。只是 float32 截掉低 16 位尾数。TPU 和 A100/H100 有原生支持。训练首选。

### Gradient Clipping

梯度爆炸时，单个大 gradient 一步毁掉所有权重。

Clip by value：`grad = clamp(grad, -max, max)`。简单但可改变 gradient 方向。
Clip by norm：`grad = grad * (max_norm / ||grad||)`。保持方向。`torch.nn.utils.clip_grad_norm_()`。标准选择。

Transformer 典型值：max_norm=1.0。RL：0.5。简单网络：5.0。

### 常见数值 Bug 速查

| Bug | 原因 | 修复 |
|-----|------|------|
| Loss 变 NaN | Softmax overflow 或 lr 太高 | stable softmax，降 lr，gradient clipping |
| Loss 卡在 log(num_classes) | 输出接近均匀分布 | 检查数据标签，检查 loss function |
| 精度低 1-3% | float16 无 loss scaling | 动态 loss scaling 或切 bfloat16 |
| 某层 gradient norm=0 | Dead ReLU 或 float16 underflow | LeakyReLU/GELU，gradient scaling |
| 不同 GPU 不同结果 | 非确定性浮点累加顺序 | 接受 1e-6 差异，或用确定性模式 |
| exp() 在 loss 中返回 inf | 未用 max-subtraction trick | `torch.nn.functional.log_softmax()` |

## 动手实现

### Stable softmax 与 cross-entropy

```python
def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_prob = shifted[true_class] - math.log(sum(math.exp(s) for s in shifted))
    return -log_prob

logits = [100.0, 101.0, 102.0]
# softmax_stable(logits) → [0.090, 0.245, 0.665] ✓
# 不用 stable 版本 → [nan, nan, nan] ✗
```

### Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus, x_minus = x[:], x[:]
        x_plus[i] += h; x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
```

完整实现见 `code/numerical.py`。
