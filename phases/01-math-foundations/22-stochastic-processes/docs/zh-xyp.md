# 随机过程

> 有结构的随机性。Random walk、Markov chain 和 diffusion 模型背后的数学。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1, Lesson 06-07
**时间：** ~75 分钟

## 术语对照

- random walk，随机游走
- Markov property，马尔可夫性质
- transition matrix，转移矩阵
- stationary distribution，平稳分布
- Brownian motion，布朗运动
- Langevin dynamics，朗之万动力学
- MCMC，马尔可夫链蒙特卡洛
- Metropolis-Hastings，Metropolis-Hastings 算法
- temperature，温度
- diffusion process，扩散过程
## 关键术语

| 术语 | 含义 |
|------|------|
| Random walk | 以随机增量每步改变位置的过程 |
| Markov property | 未来仅依赖当前状态，不依赖历史 |
| Transition matrix | P[i][j]=从 i 到 j 的概率 |
| Stationary distribution | πP=π。链的均衡分布 |
| Brownian motion | Random walk 的连续时间极限。B(t)~N(0,t) |
| Langevin dynamics | gradient descent + noise = 探索式采样 |
| MCMC | 构造以目标分布为 stationary 的 Markov chain |
| Metropolis-Hastings | 用接受比确保收敛到目标分布的 MCMC |
| Temperature | 探索与利用的权衡参数。低=确定，高=随机 |
| Diffusion process | 前向：逐步加噪声。反向：逐步去噪生成数据 |

## 学习目标

- 模拟 1D 和 2D random walk 并验证位移的 √n 标度律
- 构建 Markov chain 模拟器并通过特征分解计算其 stationary distribution
- 实现 Metropolis-Hastings MCMC 和 Langevin dynamics 从目标分布中采样
- 将前向 diffusion 过程与 Brownian motion 联系起来，解释反向过程如何生成数据

## 概念

### Random Walk

从位置 0 开始。每步掷硬币：正面右移 +1，反面左移 -1。n 步后期望位置为 0（无偏），但距原点的期望距离以 √n 增长。

为何 √n？每步方差=1，独立→S_n 方差=n，标准差=√n。CLT 保证 S_n/√n 收敛到标准正态。

这种 √n 标度律在 ML 中无处不在：SGD 噪声 ~ 1/√(batch_size)，embedding 维度 ~ √d。

**2D random walk** 同样 √n 标度。路径描出分形图样。

**Brownian motion：** 步长 1/√n、每秒 n 步的 random walk 的连续极限。B(t) ~ N(0,t)。Diffusion 的数学基础——扩散模型中正是这个噪声过程。

### Markov Chain

系统按固定概率在状态间转移。关键性质：下一状态仅依赖当前状态（Markov property）。

```
P(X_{t+1}=j | X_t=i) = P[i][j]
```

Transition matrix P 每行和为 1。长期运行，状态分布收敛到 stationary distribution π，满足 πP=π。这是 P 的 eigenvalue=1 的左 eigenvector。

收敛条件：不可约（每状态可达其他所有状态）、非周期。不满足则可能不收敛或循环。

**Mixing time：** 达到趋于 stationary 的步数。由 spectral gap（1 减去 P 的次大 eigenvalue）控制。gap 越大→混合越快。

**吸收状态：** P[i][i]=1。进入后再也不离开。模型终止状态（游戏结束、客户流失、EOS token）。

### 与语言模型的联系

LLM token 生成近似为 Markov 过程。给定当前上下文，模型输出下一 token 分布，从中采样，继续。Temperature/top-k/top-p 修改转移概率。

### Langevin Dynamics

```
x_{t+1} = x_t - dt·∇U(x_t) + √(2T·dt)·z_t
```

两个力：gradient force 推向低能（像 gradient descent），random force 随机推动（探索）。T=0→纯 gradient descent。高温→近乎 random walk。合适温度下，粒子探索能量 landscape，在低能区停留更多时间。

### Diffusion Models

前向过程（Markov chain）：x_t = √α_t·x_{t-1} + √(1-α_t)·noise。T 步后 x_T ~ 纯噪声。

反向过程（learned Markov chain）：学习预测每步加入的噪声，然后减去。从噪声逐渐生成数据。每一步生成是 learned Markov chain 中的一步。

### MCMC：Metropolis-Hastings

从只能求值（不能直接采样）的分布 p(x) 中采样。Bayesian posterior 经典例子：知道 likelihood×prior，但归一化常数不可算。

```
1. 在位置 x
2. 提议 x' ~ Q(x'|x)
3. 接受比 α = p(x')·Q(x|x')/(p(x)·Q(x'|x))
4. 以概率 min(1,α) 接受 x'。否则停留在 x
```

对称提议（Q 对称）下 α=p(x')/p(x)。归一化常数约掉。

保证收敛到 p(x)。Burn-in（丢弃前 N 个样本让链达到平稳分布）、thinning（每 k 步保留减少自相关）、多链验证收敛。Gaussian 提议在高维下最优接受率 ≈ 23%。

## 动手实现

```python
def random_walk_1d(n_steps):
    steps = np.random.choice([-1,1], size=n_steps)
    return np.concatenate([[0], np.cumsum(steps)])

class MarkovChain:
    def __init__(self, transition_matrix):
        self.P = np.array(transition_matrix, dtype=float)
        self.n = len(self.P)

    def simulate(self, start, n_steps):
        states = [start]
        current = start
        for _ in range(n_steps):
            current = np.random.choice(self.n, p=self.P[current])
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        return np.abs(pi) / pi.sum()

def langevin_dynamics(grad_U, x0, dt, temperature, n_steps):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = x - dt*grad_U(x) + np.sqrt(2*temperature*dt)*np.random.randn(*x.shape)
        trajectory.append(x.copy())
    return np.array(trajectory)

def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples):
    x = np.array(x0, dtype=float)
    samples, accepted = [x.copy()], 0
    for _ in range(n_samples-1):
        x_proposed = x + np.random.randn(*x.shape)*proposal_std
        if np.log(np.random.rand()) < target_log_prob(x_proposed)-target_log_prob(x):
            x, accepted = x_proposed, accepted+1
        samples.append(x.copy())
    return np.array(samples), accepted/(n_samples-1)
```
