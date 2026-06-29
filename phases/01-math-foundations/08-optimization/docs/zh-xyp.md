# 最优化

> 训练神经网络不过是找山谷底部而已。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 04-05（Derivative、Gradient）
**时间：** ~75 分钟

## 术语对照

- SGD，随机梯度下降
- momentum，动量
- Adam，Adam 优化器
- bias correction，偏差修正
- learning rate schedule，学习率调度
- step decay，阶梯衰减
- exponential decay，指数衰减
- cosine annealing，余弦退火
- warmup，预热
- convex，凸的
- non-convex，非凸的
- saddle point，鞍点
- loss landscape，损失景观
- mini-batch，小批量
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Gradient descent | "向下走" | 将 weight 减去 gradient 乘 learning rate。最基本的 optimizer。 |
| Learning rate | "步长" | 控制每次更新移动多远的标量。太大会发散。太小浪费算力。 |
| Momentum | "保持滚动" | 累积过去 gradient 为速度向量。减弱震荡，沿一致方向加速移动。 |
| SGD | "随机采样" | 在随机子集而非全数据集上计算 gradient。实践中几乎总是 mini-batch SGD。 |
| Adam | "默认 optimizer" | 自适应矩估计。逐 weight 追踪 gradient 和 gradient² 的运行均值，给每个 weight 自己的学习率。 |
| Bias correction | "修复冷启动" | Adam 的第一和第二矩初始化为零。Bias correction 除以 (1-βᵗ) 以补偿早期步。 |
| Learning rate schedule | "随时间调 lr" | 训练中调整 learning rate 的函数。早期大步，后期小步。 |
| Convex function | "一个谷" | 函数中任何局部最小值就是全局最小值。Gradient descent 总能找到。神经网络 loss 不是 convex 的。 |
| Saddle point | "平但不是最小值" | Gradient 为零，但在某些方向是最小值，另一些方向是最大值的点。高维中常见。 |
| Loss landscape | "地形" | 在 weight 空间上绘制的 loss function。通常沿两个随机方向切片可视化。 |

## 学习目标

- 从零实现 vanilla gradient descent、带 momentum 的 SGD 和 Adam
- 在 Rosenbrock 函数上比较 optimizer 的收敛性，解释为什么 Adam 使用逐 weight 自适应学习率
- 区分 convex 和 non-convex loss 曲面，解释 saddle point 在高维空间中的角色
- 配置 learning rate schedule（step decay、cosine annealing、warmup）以保证训练稳定性

## 问题

你有一个 loss function。它告诉你模型有多错。你有 gradient。它告诉你哪个方向让 loss 更差。现在你需要一个走下坡的策略。

幼稚的方法很简单：向 gradient 的反方向移动。用某个叫 learning rate 的数缩放步长。重复。这就是 gradient descent，它确实有效。但"有效"有保留条件。Learning rate 太大，你会越过山谷，在两面墙上弹跳。太小，你在通往答案的路上多爬几千步。遇到 saddle point，你还没找到最小值就停住了。

深度学习中的每一个 optimizer 都在回答同一个问题：如何更快、更稳地到达山谷底部？

## 概念

### 什么是最优化

最优化是找到使函数最小化（或最大化）的输入值。在机器学习中，函数是 loss。输入是模型的 weight。训练就是最优化。

```
最小化 L(w)，其中：
  L = loss function
  w = 模型 weight（可能是数百万个参数）
```

### Gradient descent（vanilla）

最简单的 optimizer。计算 loss 关于每个 weight 的 gradient。向 gradient 反方向移动每个 weight。用 learning rate 缩放步长。

```
w = w - lr · gradient
```

这就是全部算法。一行。

### Learning rate：最重要的超参数

Learning rate 控制步长。它决定一切关于收敛的事。太大则发散。太小则缓慢爬行。没有公式给出正确的 learning rate。你要通过实验找到。常见起点：Adam 用 0.001，带 momentum 的 SGD 用 0.01。

### SGD vs batch vs mini-batch

- **Batch GD：** 全数据集计算 gradient，一步。慢但稳定。
- **SGD：** 单一样本计算 gradient，立即迈步。Noisy 但快。
- **Mini-batch：** 小 batch（32-256 样本）计算 gradient。这是实际通用的方式。

| 变体 | Batch 大小 | Gradient 质量 | 每步速度 | 噪声 |
|------|-----------|-------------|---------|------|
| Batch GD | 全数据集 | 精确 | 慢 | 无 |
| SGD | 1 个样本 | 很 noisy | 快 | 高 |
| Mini-batch | 32-256 | 良好估计 | 平衡 | 中等 |

SGD 和 mini-batch 中的噪声不是 bug。它有助于逃离浅的局部最小值和 saddle point。

### Momentum：滚下山坡的球

Vanilla gradient descent 只看当前 gradient。如果 gradient 走锯齿形（在狭窄峡谷中常见），进展缓慢。Momentum 通过累积过去 gradient 成速度向量来修正。

```
v = β · v + gradient
w = w - lr · v
```

类比：一个球滚下山坡。它不会在每个凸起处停住重来。它在一致的方向上积累速度，减弱震荡。

`β`（通常 0.9）控制保留多少历史。更高的 β 意味着更多动量、更平滑的路径，但方向变化响应更慢。

### Adam：自适应学习率

不同 weight 需要不同的 learning rate。Adam 为每个 weight 追踪两件事：

1. 第一矩（m）：gradient 的运行均值（类似 momentum）
2. 第二矩（v）：梯度平方的运行均值（gradient magnitude）

```
m = β₁ · m + (1-β₁) · gradient
v = β₂ · v + (1-β₂) · gradient²

m̂ = m / (1-β₁ᵗ)     bias correction
v̂ = v / (1-β₂ᵗ)

w = w - lr · m̂ / (√v̂ + ε)
```

除以 `√v̂` 是关键洞察。大 gradient 的 weight 被大数除（小有效步长）。小 gradient 的 weight 被小数除（大有效步长）。每个 weight 得到自己的自适应 learning rate。

默认超参数：`lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8`。这些默认值对大多数问题都有效。

### Learning Rate Schedule

固定 learning rate 是妥协。训练早期要大步快走，晚期要小步微调。

| Schedule | 公式 | 使用场景 |
|----------|------|---------|
| Step decay | lr = lr · factor 每 N epoch | 简单，手动控制 |
| Exponential decay | lr = lr₀ · decayᵗ | 平滑减少 |
| Cosine annealing | lr = lr_min + 0.5(lr_max-lr_min)(1+cos(πt/T)) | Transformer，现代训练 |
| Warmup + decay | 线性升温，然后衰减 | 大模型，防止早期不稳定 |

### Convex vs Non-Convex

Convex 函数只有一个最小值。Gradient descent 总能找到。如 `f(x) = x²`。

神经网络 loss function 是 non-convex 的。它们有很多局部最小值、saddle point 和平坦区域。

在实践中，高维神经网络中的局部最小值很少成为问题。大多数局部最小值与全局最小值的 loss 值接近。Saddle point（有些方向平、有些方向弯曲）才是真正的障碍。Momentum 和 mini-batch 的噪声有助于逃离它们。

Sharp minima 泛化差。Flat minima 泛化好。这是 SGD with momentum 在最终测试精度上常优于 Adam 的原因之一：其噪声阻止了在 sharp minima 中定居。

## 动手实现

### 第 1 步：测试函数

```python
def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]
```

Rosenbrock 函数的最小值在 (1, 1)，在一个狭窄弯曲的峡谷中。

### 第 2 步：Vanilla Gradient Descent

```python
class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]
```

### 第 3 步：SGD with Momentum

```python
class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]
```

### 第 4 步：Adam

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [self.beta2 * v + (1 - self.beta2) * g**2 for v, g in zip(self.v, grads)]
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]
        return [p - self.lr * mh / (vh ** 0.5 + self.epsilon)
                for p, mh, vh in zip(params, m_hat, v_hat)]
```

### 第 5 步：运行并比较

```python
def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]
gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name:6s} → x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
```

## 实际使用

```python
import torch

model = torch.nn.Linear(784, 10)
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)
```

经验法则：
- 默认用 Adam（lr=0.001）。大多数问题无需调参。
- 需要最佳最终精度且有调参余力时换 SGD with momentum（lr=0.01, momentum=0.9）。
- Transformer 用 AdamW（带解耦 weight decay 的 Adam）。
- 训练超过几个 epoch 总要用 learning rate schedule。
- 训练不稳定就降 learning rate。太慢就升 learning rate。

## 练习

1. **Learning rate 扫描。** 在 Rosenbrock 函数上用 [0.0001, 0.0005, 0.001, 0.005, 0.01] 运行 vanilla GD。找出仍能收敛的最大 learning rate。

2. **Momentum 对比。** 用 momentum 值 [0.0, 0.5, 0.9, 0.99] 运行 SGD。哪个最快收敛？哪个过冲？

3. **Saddle point 逃离。** 定义 `f(x,y) = x² - y²`（原点是 saddle point）。从 (0.01, 0.01) 出发。比较 vanilla GD、SGD with momentum 和 Adam 的行为。

4. **实现 learning rate decay。** 给 GradientDescent 类加 exponential decay：`lr = lr₀ · 0.999ˢᵗᵉᵖ`。
