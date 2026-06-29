# 图论 for ML

> Graph 是关系的数学结构。如果你的数据有连接，你需要图论。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-03
**时间：** ~90 分钟

## 术语对照

- graph，图
- adjacency matrix，邻接矩阵
- degree，度
- Laplacian，拉普拉斯矩阵
- Fiedler value，Fiedler 值
- BFS，广度优先搜索
- DFS，深度优先搜索
- message passing，消息传递
- spectral clustering，谱聚类
- connected component，连通分量
- GCN，图卷积网络
## 关键术语

| 术语 | 含义 |
|------|------|
| Graph | G=(V,E)。节点和边的数学结构 |
| Adjacency matrix | A[i][j]=1 若节点 i 和 j 相连 |
| Degree | 触及节点的边数 |
| Laplacian | L=D-A。Eigenvalue 揭示图结构 |
| Fiedler value | L 的最小非零 eigenvalue。度量图连通性 |
| BFS | 队列遍历。先邻居后深入。找最短路径 |
| DFS | 栈遍历。先深入到底再回溯 |
| Message passing | 节点聚合邻居信息。GNN 核心 |
| Spectral clustering | 用 Laplacian 的 eigenvector 分割图 |
| 连通分量 | 每节点可达其他所有节点的最大子图 |

## 学习目标

- 构建带 adjacency matrix/list 表示的图类，实现 BFS 和 DFS 遍历
- 计算图 Laplacian，用其 eigenvalue 检测连通分量和聚类节点
- 实现一轮 GNN 风格的 message passing，即归一化 adjacency matrix 乘法
- 应用 spectral clustering 用 Fiedler vector 分割图

## 概念

### 图：节点与边

图 G=(V,E)。有向（Twitter 关注）vs 无向（Facebook 好友）。加权 vs 无权。

### Adjacency Matrix

A[i][j] = 1 若有边 i→j。无向图 A 对称。加权图 A[i][j]=权重。每个 GNN 的输入。

### Degree

节点连接的边数。Degree matrix D 对角：D[i][i] = deg(i)。高 degree=枢纽节点。

### BFS 和 DFS

BFS（队列 FIFO）：先访问所有邻居再深入。找无权图最短路径。
DFS（栈 LIFO/递归）：先深入到底再回溯。用于连通分量、环检测、拓扑排序。

### Graph Laplacian

L = D - A。图论中最重要的 matrix。

性质：
1. L 正半定（所有 eigenvalue ≥ 0）
2. 零 eigenvalue 数 = 连通分量数
3. 最小非零 eigenvalue = Fiedler value。度量连通性。大→图紧密，小→有瓶颈
4. Fiedler vector（该 eigenvalue 的 eigenvector）：揭示最佳分割。正值一组，负值另一组 → spectral clustering

### Spectral Clustering

1. 计算 Laplacian L
2. 找 L 的前 k 个最小 eigenvector（跳过第一个全一）
3. 用这些 eigenvector 作为节点的新坐标
4. 在新坐标上跑 k-means

Eigenvector 编码图上"最光滑"的函数。紧密连接的节点有相似 eigenvector 值。瓶颈分离的节点有不同值。

### Message Passing

GNN 的核心操作。每个节点收集邻居消息、聚合、更新。

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ neighbors(v)}))
```

最简单形式：聚合=均值，更新=线性变换+activation：

```
H^(k+1) = σ(A_norm · H^(k) · W)
```

一轮 message passing=节点看到直接邻居。K 轮=看到 K 跳邻域。

### 图概念与 ML 应用

| 概念 | 应用 |
|------|------|
| Adjacency matrix | GCN/GAT/GraphSAGE 输入 |
| Laplacian | Spectral clustering, ChebNet |
| BFS/DFS | 知识图谱遍历，路径查找 |
| Message passing | 每个 GNN 层 |
| Spectral gap | 图连通性，随机游走混合时间 |
| PageRank | 节点重要性排序 |

GCN 层（Kipf & Welling, 2017）：H^(l+1) = σ(D̂^(-½)·Â·D̂^(-½)·H^(l)·W^(l))，Â=A+I（含自环）。自环确保节点在聚合时包含自身特征。理解 Laplacian 即理解 GCN 为何有效。

## 动手实现

```python
from collections import deque

class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed: self.adj[v][u] = weight

    def adjacency_matrix(self):
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        return np.diag([len(self.adj[i]) for i in range(self.n)])

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()

def bfs(graph, start):
    visited, order, queue = {start}, [], deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        for nb in graph.adj[node]:
            if nb not in visited:
                visited.add(nb); queue.append((nb, dist+1))
    return order

def spectral_clustering(graph, k=2):
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]
    labels = (features[:, 0] >= 0).astype(int)
    return labels

def message_passing(graph, features, W):
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    A_norm = A / row_sums
    return A_norm @ features @ W
```
