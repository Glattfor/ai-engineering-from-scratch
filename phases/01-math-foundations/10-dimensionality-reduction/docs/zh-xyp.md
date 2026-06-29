# 降维

> 高维数据有结构。你只需从正确的角度看。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-03, 06
**时间：** ~90 分钟

## 术语对照

- PCA，主成分分析
- t-SNE，t-分布随机邻域嵌入
- UMAP，统一流形近似与投影
- kernel PCA，核主成分分析
- covariance matrix，协方差矩阵
- eigenvector，特征向量
- eigenvalue，特征值
- explained variance ratio，解释方差比
- perplexity，困惑度
- elbow method，肘部法则
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 维度灾难 | "特征太多" | 随维度增长，距离、体积和数据密度均表现出反直觉行为。 |
| PCA | "降维" | 旋转坐标系使轴与最大 variance 方向对齐，然后丢弃低 variance 轴。 |
| 主成分 | "重要方向" | Covariance matrix 的 eigenvector。数据变化最大的特征空间方向。 |
| Explained variance ratio | "这个成分有多少信息" | 一个主成分捕获的 variance 占总量的比例。 |
| Covariance matrix | "特征如何关联" | 对称 matrix，项 (i,j) 衡量特征 i 和 j 如何一起变化。 |
| t-SNE | "那个聚类图" | 通过保留 pairwise 邻域概率将高维映射到 2D 的非线性方法。可视化用，不做预处理。 |
| UMAP | "更快的 t-SNE" | 基于拓扑数据分析的非线性方法。保留局部和部分全局结构。比 t-SNE 更可扩展。 |
| Perplexity | "t-SNE 的旋钮" | 控制每个点考虑的有效邻居数。低 perplexity 聚焦于极局部结构。 |

## 学习目标

- 从零实现 PCA：中心化数据，计算 covariance matrix，特征分解，投影
- 使用 explained variance ratio 和 elbow method 选择主成分数量
- 比较 PCA、t-SNE 和 UMAP 在 MNIST 数字 2D 可视化上的表现并解释其取舍
- 应用带 RBF kernel 的 kernel PCA 分离标准 PCA 无法处理的非线性数据结构

## 问题

你有一个每样本 784 个特征的数据集。但大多数特征是冗余的。真实的信息存在于一个小得多的曲面上。降维找到那个更小的曲面。它将你的 784 维数据压缩到 2、10 或 50 维，同时保持重要的结构。

## 概念

### 维度灾难

高维空间是反直觉的：
- **距离变得无意义。** 随机点之间的距离收敛到相同值。
- **体积集中在角落。** 几乎所有体积在边缘，远离中心。
- **需要指数级更多数据。** 从 2D 到 20D 需要 10¹⁸ 倍数据。

### PCA：找重要的方向

PCA 找到数据变化最大的轴，旋转坐标系使第一轴捕获最大 variance，第二轴捕获次大，以此类推。

算法：中心化数据 → 计算 covariance → 特征分解 → 按 eigenvalue 排序 → 保留前 k 个 eigenvector 投影。

为什么特征分解？Covariance matrix 的 eigenvector 是特征空间中的正交方向。Eigenvalue 告诉你每个方向捕获了多少 variance。最大 eigenvalue 的 eigenvector 指向最大 variance 的方向。

### Explained variance ratio

每个主成分捕获总 variance 的一部分。累计到 0.95 意味着前 k 个成分捕获了 95% 的信息。之后的基本是噪声。

### 选择成分数

1. **阈值法。** 保留足够成分以解释 90-95% 的 variance。
2. **Elbow method。** 画每成分 explained variance。找急剧下降处。
3. **下游性能。** 使用 PCA 作预处理。扫描 k 并测量模型准确率。

### t-SNE：保留邻域关系

将高维数据映射到 2D 同时保留哪些点彼此靠近。非线性的，可以展开 PCA 无法处理的复杂流形。不同运行产生不同布局。perplexity 参数控制考虑多少邻居（典型 5-50）。簇间距离无意义。

### UMAP：更快，更好的全局结构

类似 t-SNE 但更快（使用近似最近邻图）且更好地保留了全局结构。`n_neighbors` 控制局部结构，`min_dist` 控制输出中点有多紧密。

### 何时用哪个

| 方法 | 场景 | 保留什么 | 速度 |
|------|------|---------|------|
| PCA | 训练前预处理 | 全局 variance | 快 |
| t-SNE | 发表级 2D 图 | 局部邻域 | 慢 |
| UMAP | 大规模 2D 可视化 | 局部 + 部分全局 | 中 |
| PCA | 模型特征降维 | Variance 排序特征 | 快 |

### Kernel PCA

标准 PCA 找线性子空间。Kernel PCA 应用 kernel trick 在隐式高维空间中做 PCA。RBF kernel 最适合大多数非线性数据。经典例子：同心圆。标准 PCA 投影到同一条线，kernel PCA 将其分开。

## 动手实现

### 第 1 步：从零实现 PCA

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

### 第 2 步：在合成数据上测试

```python
t = np.random.uniform(0, 2 * np.pi, 500)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, 500)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, 500)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, 500)
X = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")
```

### 第 3 步：MNIST 与 t-SNE 对比

```python
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data[:5000].astype(float), mnist.target[:5000].astype(int)

pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
```

### 第 4 步：PCA 作为预处理

```python
for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)
    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"k={k:>3d}  accuracy={acc:.4f}  variance={sum(pca_k.explained_variance_ratio_):.4f}")
```

性能在远不到 784 维时就趋于稳定。该稳定点就是你的工作点。

## 练习

1. 给 PCA 类添加 `inverse_transform`。从 10、50、200 个成分重建 MNIST 数字，打印重建误差。
2. 用 perplexity 值 5、30、100 在 MNIST 上运行 t-SNE。描述输出如何变化。
3. 生成 50 个特征中只有 5 个有信息的 synthetic 数据集。用 PCA 检查 explained variance 曲线是否正确识别出有效维度为 5。
