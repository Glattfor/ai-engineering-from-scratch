# 范数与距离

> 你的距离函数定义了"相似"的含义。选错，下游一切崩塌。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-02
**时间：** ~90 分钟

## 术语对照

- norm，范数
- L1 norm，L1 范数
- L2 norm，L2 范数
- Lp norm，Lp 范数
- L-infinity，L 无穷范数
- cosine similarity，余弦相似度
- Mahalanobis distance，马氏距离
- Jaccard similarity，Jaccard 相似度
- edit distance，编辑距离
- KL divergence，KL 散度
- Wasserstein distance，Wasserstein 距离
- HNSW，分层可导航小世界
- LASSO，LASSO 回归
- Ridge，岭回归
- ANN，近似最近邻
## 关键术语

| 术语 | 含义 |
|------|------|
| Norm | 将 vector 映射为非负标量的函数，满足三角不等式。 |
| L1 norm | 绝对值之和。产生稀疏性，对 outlier 鲁棒。 |
| L2 norm | 平方和的平方根。欧氏空间直线距离。 |
| Lp norm | 绝对值 p 次幂之和的 p 次根。L1 和 L2 是特例。 |
| L-infinity | 最大绝对值分量。最坏情况偏差。 |
| Cosine similarity | 归一化 dot product。忽略 magnitude，只看方向。 |
| Mahalanobis | 考虑 covariance 的 L2 距离。用于 outlier 检测。 |
| Jaccard | 交集/并集。集合用，非 vector。 |
| 编辑距离 | 最小插入/删除/替换数。字符串用。 |
| KL divergence | 非真正的距离（不对称）。衡量信息损失。 |
| Wasserstein | Earth Mover's 距离。真正度量。即使分布不重叠也有 gradient。 |
| HNSW | 分层可导航小世界图。主流 vector database 的 ANN 算法。 |
| L1/L2 正则化 | 在 loss 中加 weight norm 惩罚。L1→稀疏，L2→收缩。 |

## 学习目标

- 从零实现 L1、L2、cosine、Mahalanobis、Jaccard 和编辑距离函数
- 为给定 ML 任务选择合适的距离度量并解释为什么其他选择会失败
- 将 L1 和 L2 norm 与 LASSO 和 Ridge regularization 及其几何约束区域联系起来
- 展示同一数据集在不同度量下产生不同的最近邻

## 问题

你有两个 vector。也许是词 embedding，也许是用户画像，也许是像素数组。你需要知道：它们有多近？答案完全取决于你选哪个距离函数。两个点在一种度量下是最近邻，在另一种度量下可能相距甚远。你的 KNN 分类器、推荐引擎、vector database、聚类算法、loss function——都依赖这个选择。

不存在普适最佳距离。L2 适合空间数据。Cosine similarity 主导 NLP。Jaccard 处理集合。编辑距离处理字符串。Mahalanobis 考虑相关性。Wasserstein 移动概率质量。每一种都编码了关于"相似"含义的不同假设。

## 概念

### Norm：衡量 vector 的 magnitude

每个距离函数都可写为差的 norm：d(a,b) = ||a-b||。

### L1 Norm（曼哈顿距离）

||x||_1 = |x₁|+|x₂|+...+|xₙ|。城市街区距离，只能沿轴移动。

何时用：高维稀疏数据、需要抗 outlier 鲁棒性（单个大差异不主导）、特征选择（L1 regularization 促进稀疏性）。

L1 regularization（LASSO）：在 loss 中加 ||w||_1 惩罚权重绝对值之和。将小 weight 推到精确零，实现自动特征选择。约束区域是菱形，角在轴上（某 weight=0）。MAE loss = 平均 L1 距离。

### L2 Norm（欧氏距离）

||x||_2 = √(x₁²+x₂²+...+xₙ²)。直线距离，n 维毕达哥拉斯。

何时用：低到中维连续数据、特征尺度可比、物理距离、像素级图像相似。

L2 regularization（Ridge）：加 ||w||_2² 惩罚大 weight。不像 L1，不会推到零，而是等比缩小所有 weight。约束区域是圆形，无角在轴上。MSE loss = 平均 L2 距离平方。Square 对大量误差惩罚更重。

### Lp Norms：通用族

||x||_p = (|x₁|^p + ... + |xₙ|^p)^(1/p)。p=1 菱形，p=2 圆，p→∞（L-infinity）方形。

L-infinity norm = max(|x₁|,...,|xₙ|)。距离由差异最大的单维度决定。场景：最坏情况偏差、棋盘、制造公差。

### Cosine Similarity

cos_sim(a,b) = (a·b) / (||a||_2·||b||_2)。范围 -1 到 +1。忽略 magnitude，只看方向。

为何统治 NLP：文档长度不应影响相似性。一篇关于猫的长文和一篇关于猫的短文应被判定相似。相同词分布但不同长度的两文档指向相同方向，cosine similarity = 1.0。

何时用：文本相似（TF-IDF、word embedding、sentence embedding）、magnitude 是噪声而方向是信号的任何领域、推荐系统（用户偏好 vector）、embedding 搜索（vector database 几乎都用 cosine 或 dot product）。

**Dot product vs cosine：** dot product = ||a||·||b||·cos(angle)。当两 vector 已归一化（magnitude=1），两者相同。dot product 包含 magnitude 信息，大 magnitude 得高分（隐含质量/重要性信号）。

### Mahalanobis 距离

d_M(x,y) = √((x-y)^T·S^(-1)·(x-y))。考虑数据的 covariance 结构。先对数据做 whitening（去相关+归一化），再算 L2。如果 S 是单位阵，退化为欧氏距离。

身高体重相关：6'2"/180lbs 不异常，5'0"/180lbs 异常。欧氏距离可能说两者离均值同样远。Mahalanobis 正确识别第二个为 outlier。

何时用：outlier 检测、特征尺度和相关性不同的分类、足够数据估计可靠 covariance matrix 时、制造质量控制。

### Jaccard Similarity（集合用）

J(A,B) = |A∩B| / |A∪B|。范围 0 到 1。Jaccard distance = 1-J。

何时用：比较标签/类别/特征集合、基于词出现（非频率）的文档相似、近似去重（MinHash）、比较 binary feature vector、评估分割模型（IoU = Jaccard）。

### 编辑距离（Levenshtein）

将一字符串变为另一字符串所需的最少单字符操作数（插入/删除/替换）。动态规划计算。

"kitten"→"sitting" = 3 步。

何时用：拼写检查、DNA 序列比对、模糊字符串匹配、脏文本数据去重。

### KL Divergence

D_KL(P||Q) = Σ p(x)·log(p(x)/q(x))。不对称，不满足三角不等式。不是真正的距离。

Forward KL (D_KL(P||Q))：mean-seeking，Q 试图覆盖 P 的所有模态。
Reverse KL (D_KL(Q||P))：mode-seeking，Q 聚焦于 P 的单一模态。

使用场景：VAE（ELBO 的 KL 项）、知识蒸馏、RLHF、策略梯度方法。

### Wasserstein 距离（Earth Mover's Distance）

将一种概率分布转化为另一种所需的最小"功"。是真正的度量（对称、满足三角不等式）。即使分布不重叠也提供 gradient（KL divergence 会到 infinity）。WGAN 的核心。

### 不同任务需不同距离

| 任务 | 最佳距离 | 原因 |
|------|---------|------|
| 文本相似 | Cosine | Magnitude 是噪声，方向是含义 |
| 图像像素对比 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 鲁棒，不放大罕见的巨大差异 |
| 集合重叠 | Jaccard | 数据天然是集合值 |
| 字符串匹配 | 编辑距离 | 操作映射人类编辑直觉 |
| Outlier 检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 分布对比 | KL divergence | 衡量用 Q 替代 P 损失的信息 |
| GAN 训练 | Wasserstein | 分布不重叠也提供 gradient |
| Embedding 搜索 | Cosine 或 dot | Embedding 训练为在方向上编码含义 |
| 推荐系统 | Dot product | Magnitude 可编码流行度或置信度 |

### 距离 ⇒ Loss Function ⇒ Regularization

均方误差（MSE）= L2²。平均绝对误差（MAE）= L1。交叉熵 = KL divergence。

L1 正则化（LASSO）= loss + λ·||w||_1 → 稀疏权重（精确零）。
L2 正则化（Ridge）= loss + λ·||w||_2² → 小权重（无稀疏）。
Elastic Net = L1 + L2，结合 L1 的稀疏性和 L2 的稳定性。

为何 L1 产生稀疏性：在 2D weight 空间中，L1 约束区域是菱形，loss 轮廓（椭圆）最可能接触角（某 weight=0）。L2 约束是圆，接触光滑点（两 weight 均非零）。

### 近似最近邻搜索

精确搜索 O(n·d)。大规模用 ANN：KD-trees（低维）、Ball trees（中维）、LSH（去重）、HNSW（FAISS/Qdrant/Weaviate 主流）、IVF（十亿级）、Product quantization（内存受限）。

HNSW：多层图，顶层稀疏长跳，底层密集短跳。

## 动手实现

完整实现见 `code/distances.py`，所有函数纯 Python 从零构建。

```python
def l1_distance(a, b): return sum(abs(ai - bi) for ai, bi in zip(a, b))
def l2_distance(a, b): return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))
def cosine_similarity(a, b):
    dot = sum(ai*bi for ai, bi in zip(a, b))
    return dot / (math.sqrt(sum(ai**2 for ai in a)) * math.sqrt(sum(bi**2 for bi in b)))

def jaccard_similarity(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set)

def levenshtein(s1, s2):
    # 动态规划矩阵。O(n·m)
```

包含演示脚本展示同一数据集不同度量产生完全不同的最近邻。

## 练习

1. 计算 (1,2,3) 和 (4,0,6) 的 L1、L2 和 L-infinity 距离。验证 L∞ ≤ L2 ≤ L1 始终成立。
2. 创建 cosine similarity 高（>0.9）但 L2 距离大（>10）的两个 vector。解释几何上发生了什么。再创建 cosine 低（<0.3）但 L2 小（<0.5）的两个 vector。
3. 实现在 L1、L2、cosine 和 Mahalanobis 下返回最近邻的函数。找一个所有四种度量都不一致的 dataset。
4. 手算 Wasserstein 距离。P=[0.5,0.5,0,0] 到 Q=[0,0,0.5,0.5]。
5. 为近似 Jaccard similarity 实现 MinHash。
