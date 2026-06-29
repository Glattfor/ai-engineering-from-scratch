# 奇异值分解

> SVD 是线性代数的瑞士军刀。每个 matrix 都有一个。每个数据科学家都需要一个。

**类型：** 动手实现
**语言：** Python, Julia
**前置要求：** Phase 1, Lesson 01-03
**时间：** ~120 分钟

## 术语对照

- SVD，奇异值分解
- singular value，奇异值
- left singular vector，左奇异向量
- right singular vector，右奇异向量
- truncated SVD，截断 SVD
- pseudoinverse，伪逆
- condition number，条件数
- latent factor，隐因子
- power iteration，幂迭代
- Eckart-Young theorem，Eckart-Young 定理
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SVD | "因式分解任何 matrix" | 将 A 分解为 U·Σ·V^T，其中 U 和 V 正交，Σ 对角且非负。适用于任何形状的任何 matrix。 |
| 奇异值 | "这个成分有多重要" | Σ 的第 i 个对角项。衡量 matrix 沿第 i 个主方向拉伸多少。始终非负，降序排列。 |
| 左奇异向量 | "输出方向" | U 的一列。第 i 个右奇异向量在缩放 σ_i 后映射到的输出空间方向。 |
| 右奇异向量 | "输入方向" | V 的一列。matrix 在缩放 σ_i 后映射到第 i 个左奇异向量的输入空间方向。 |
| Truncated SVD | "低 rank 近似" | 仅保留前 k 个奇异值及其向量。产生可证明最优的 rank-k 近似（Eckart-Young 定理）。 |
| Pseudoinverse | "广义逆" | V·Σ⁺·U^T。对非零奇异值求逆，零保持不变。解非方阵或奇异 matrix 的最小二乘问题。 |
| Condition number | "对误差的敏感度" | σ_max / σ_min。大条件数意味着小输入变化导致大输出变化。SVD 直接揭示这一点。 |
| 隐因子 | "隐藏变量" | SVD 发现的低 rank 空间中的一个维度。在推荐中可能对应类型偏好。在 NLP 中对应主题。 |
| Eckart-Young 定理 | "SVD 给出最佳压缩" | 对任何目标 rank k，truncated SVD 在所有可能的 rank-k matrix 中最小化近似误差。 |
| Power iteration | "找最大的 eigenvector" | 将随机向量反复乘 matrix 并归一化。收敛到最大 eigenvalue 的 eigenvector。许多 SVD 算法的基石。 |

## 学习目标

- 通过 power iteration 实现 SVD 并解释 U、Σ 和 V^T 的几何含义
- 应用 truncated SVD 进行图像压缩并衡量压缩比与重建误差
- 通过 SVD 计算 Moore-Penrose pseudoinverse 求解超定最小二乘系统
- 将 SVD 与 PCA、推荐系统（隐因子）和 NLP 中的潜在语义分析联系起来

## 问题

你有一个 1000×2000 的 matrix。也许是用户-电影评分，也许是文档-词频表，也许是图片像素。你需要压缩它、去噪、发现隐藏结构，或求解最小二乘系统。Eigen 分解只在方阵上工作。SVD 在任何 matrix 上工作。任何形状。任何 rank。无条件。

## 概念

### SVD 在几何上做什么

每个 matrix 依次执行三个操作：旋转、缩放、旋转。SVD 使这个分解明确呈现。

```
A = U · Σ · V^T
    m×n   m×m   m×n   n×n

- V^T 在输入空间（n 维）旋转向量
- Σ 沿每条轴缩放（拉伸或压缩）
- U 将结果旋转到输出空间（m 维）
```

想象：你给 SVD 一个 matrix。它告诉你："这个 matrix 取一个输入球体，先用 V^T 旋转它，然后用 Σ 把它拉伸成椭球体，再用 U 旋转该椭球体。"奇异值是椭球体轴的长度。

### 完整分解

- U（m×m）正交，列是左奇异向量——输出空间中的方向
- Σ（m×n）对角，对角线上的奇异值 σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0，r = rank(A)
- V（n×n）正交，列是右奇异向量——输入空间中的方向

核心关系：`A · v_i = σ_i · u_i`。Matrix A 取第 i 个右奇异向量 v_i，以 σ_i 缩放，映射到第 i 个左奇异向量 u_i。

### 外积形式

```
A = σ₁·u₁·v₁^T + σ₂·u₂·v₂^T + ... + σᵣ·uᵣ·vᵣ^T

每一项是 rank-1 matrix。截断这个和就得到给定 rank 下的最佳近似。
Rank-k 近似: A_k = 前 k 项之和（Eckart-Young 定理保证最优）
```

### Truncated SVD

Eckart-Young-Mirsky 定理：保留前 k 个奇异值及其对应向量，得到对 A 的、可证明最佳的 rank-k 近似。

```
A_k = U_k · Σ_k · V_k^T

近似误差 = σ_{k+1}（谱范数）
         = √(σ_{k+1}² + ... + σᵣ²)（Frobenius 范数）
```

如果奇异值衰减快，小的 k 就捕获了 matrix 的大部分。衰减慢则 matrix 没有低 rank 结构。

### 与特征分解的关系

右奇异向量 V 是 A^T A 的 eigenvector。奇异值平方 σ_i² 是 A^T A 的 eigenvalue。左奇异向量 U 是 A A^T 的 eigenvector。

这意味着奇异值始终非负实数。但当 A 是方阵且对称正半定时，SVD 和特征分解是一回事。

### 图像压缩

灰度图是像素强度 matrix。自然图像的奇异值衰减快。前几个奇异值捕获大体结构（形状、渐变）。后面的捕获细节和噪声。Rank 50 截断通常产生视觉上几乎相同的图像，存储量减少 85%。

### 推荐系统

用户-电影评分 matrix 有低 rank。SVD 分解出：U = 隐因子空间中的用户画像，Σ = 各因子重要性，V^T = 隐因子空间中的电影画像。用户对电影的预测评分 = 用户画像与电影画像的 dot product（以奇异值加权）。

### Latent Semantic Analysis

应用 SVD 于 term-document matrix。"cat"和"dog"在概念空间中靠近（陆地宠物）。"fish"和"ocean"靠近（水生概念）。同义词最终聚在一起，因为它们在类似文档中共现。

### 伪逆

Moore-Penrose pseudoinverse：`A⁺ = V · Σ⁺ · U^T`，其中 Σ⁺ 将非零 σ_i 替换为 1/σ_i。求解最小二乘问题：如果 Ax=b 无精确解，x = A⁺b 给出最小化 ||Ax-b|| 的解。数值上比正规方程更稳定。

### SVD 与 PCA

PCA 就是中心化数据上的 SVD。给定中心化数据 matrix X，X = U·Σ·V^T，则主成分正是右奇异向量 V。每个成分的 explained variance 是 σ_i²/(n-1)。Sklearn 的 PCA 用 SVD 实现而非特征分解——更快且数值更稳定。

## 动手实现

### 第 1 步：从零用 Power Iteration 实现 SVD

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)
    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None: k = min(m, n)
    sigmas, us, vs = [], [], []
    A_residual = A.copy().astype(float)
    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)
        if eigenvalue < 1e-10: break
        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma
        sigmas.append(sigma); us.append(u); vs.append(v)
        A_residual = A_residual - sigma * np.outer(u, v)
    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))
    return U, S, V
```

### 第 2 步：与 NumPy 对比

```python
A = np.random.randn(5, 4)
U_o, S_o, V_o = svd_from_scratch(A)
U_n, S_n, Vt_n = np.linalg.svd(A, full_matrices=False)
A_reconstructed = U_o @ np.diag(S_o) @ V_o.T
print(f"重建误差: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### 第 3 步：图像压缩

```python
def compress_image_svd(image, k):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    ratio = k * (rows + cols + 1) / (rows * cols)
    print(f"k={k:>3d}  error={error:.4f}  存储比例={ratio:.1%}")
```

### 第 4 步：伪逆

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
A_pinv = Vt.T @ np.diag(1.0 / S) @ U.T
x_svd = A_pinv @ b
print(f"SVD pseudoinverse 解: {x_svd}")
```

## 练习

1. 不使用 power iteration 实现完整 SVD：计算 A^T A 的特征分解得到 V 和奇异值，然后 U = A·V·Σ⁻¹。比较数值精度。
2. 加载真实灰度图。在 rank 1, 5, 10, 25, 50, 100 压缩。计算压缩比和相对误差。找到图像肉眼可接受的 rank。
3. 构建 10×8 用户-电影评分 matrix，用行均值填缺失项。SVD rank-3 重建预测缺失评分。
4. 用 3 个合成主题创建 100×50 文档-词 matrix。验证前 3 个奇异值远大于其余。将文档投影到 3D 隐空间，检查同主题文档是否聚类。
5. 生成干净低 rank matrix（rank 3）并加不同水平噪声。通过扫描 k 找到各噪声水平下的最优截断 rank。
