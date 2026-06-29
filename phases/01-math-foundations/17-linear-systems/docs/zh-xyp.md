# 线性系统

> 解 Ax = b 是中行数学里最古老的问题之一，至今仍在驱动你的神经网络。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-03
**时间：** ~120 分钟

## 术语对照

- Gaussian elimination，高斯消元
- partial pivoting，部分主元
- LU decomposition，LU 分解
- QR decomposition，QR 分解
- Cholesky decomposition，Cholesky 分解
- least squares，最小二乘法
- normal equation，正规方程
- pseudoinverse，伪逆
- condition number，条件数
- ridge regression，岭回归
- conjugate gradient，共轭梯度
## 关键术语

| 术语 | 含义 |
|------|------|
| Gaussian elimination | 行运算消元成上三角，back substitution 求解 |
| Partial pivoting | 选列中最大绝对值为 pivot，防除以小数 |
| LU decomposition | A=LU，平摊 O(n³) 到多次求解 |
| QR decomposition | A=QR，正交 factor，最小二乘更稳定 |
| Cholesky | A=LL^T，对称正定专用，半价 LU |
| 最小二乘 | 超定系统下最小化平方残差和 |
| 正规方程 | A^T A x = A^T b。线性回归的闭式解 |
| Pseudoinverse | 广义逆，SVD 求。最小 norm 最小二乘解 |
| Condition number | σ_max/σ_min，衡量数值敏感度 |
| Ridge regression | (X^T X + λI) x = X^T y，改善 conditioning |

## 学习目标

- 用带 partial pivoting 的 Gaussian elimination 和 back substitution 解 Ax = b
- 用 LU、QR 和 Cholesky 分解 factor matrix，解释各自适用场景
- 推导最小二乘的正规方程，将其与线性回归和岭回归联系起来
- 用 condition number 诊断病态系统并用 regularization 稳定之

## 问题

每次训练线性回归，你在解线性系统。每次计算最小二乘拟合，你在解线性系统。每次神经网络层计算 y=Wx+b，你在计算线性系统的一边。加 regularization 时你在修改系统。用 Gaussian process 时你在 factor matrix。算 Mahalanobis 距离求逆 covariance matrix 时你在解线性系统。

Ax=b 无处不在。A 是已知系数 matrix，b 是已知输出 vector，x 是要找的未知 vector。

## 概念

### 几何含义

每方程定义一个超平面。解是所有超平面的交点。三种情况：唯一解（A 可逆）、无解（不相容系统，超定）、无穷多解（A 有 null space）。

列图像：b 是 A 的列的何种线性组合？若 b 在 A 的列空间中，有解。若不在，找列空间中最近的点——这就是最小二乘解。

### Gaussian Elimination

将 Ax=b 转化为上三角系统 Ux=c，通过 back substitution 求解。每列选最大 pivot（partial pivoting），用该行消去下面行的该列。O(n³)。Without pivoting，小 pivot 放大舍入误差。

### LU Decomposition

A = LU。L 存储消元乘数，U 是消元结果。一次 O(n³) factor 后，每次解新 b 只需 O(n²)（forward + back substitution）。带 pivoting：PA = LU。

### QR Decomposition

A = QR。Q 正交（Q^T Q = I），R 上三角。数值上比 LU 更稳定，用于最小二乘。Gram-Schmidt 逐列构建 Q。

### Cholesky Decomposition

A = LL^T，仅适用于对称正定 matrix。速度是 LU 的两倍，存储减半。用于 covariance matrix、Gaussian process kernel matrix、ridge regression。

### 最小二乘

当 m>n（超定），无精确解。最小化 ||Ax-b||²。解满足正规方程：A^T A x = A^T b。这正好是线性回归的闭式解。加 λI 得 ridge regression：(X^T X + λI)w = X^T y。

### Pseudoinverse

通过 SVD：A⁺ = V Σ⁺ U^T。Σ⁺ 将非零奇异值取倒数。无解时给最小二乘解，无穷解时给最小 norm 解。

### Condition Number

κ(A) = σ_max / σ_min。衡量解对输入扰动的敏感度。κ ~ 10^k：丢失约 k 位精度。κ ~ 10¹⁶（float64 下）：解无意义，matrix 接近奇异。Regularization 改善 condition number：κ 变成 (σ_max+λ)/(σ_min+λ)。

### Conjugate Gradient

大型稀疏对称正定系统的迭代方法。无需存储/求逆，仅需 matrix-vector 积。收敛速度取决于 condition number。

### 方法速查表

| 方法 | 要求 | 成本 | 场景 |
|------|------|------|------|
| Gaussian elimination | 方阵非奇异 | O(n³) | 单次求解 |
| LU | 方阵非奇异 | factor O(n³)，解 O(n²) | 同 A 多 b |
| QR | 任意 A (m≥n) | O(mn²) | 最小二乘 |
| Cholesky | 对称正定 | O(n³/3) | Covariance、GP、Ridge |
| SVD/Pseudoinverse | 任意 A | O(mn²) | Rank-deficient |

## 动手实现

```python
def gaussian_elimination(A, b):
    # partial pivoting + row elimination + back substitution

def lu_decompose(A):
    # 返回 P, L, U

def cholesky(A):
    # 返回下三角 L，A = LL^T

def least_squares_normal(A, b):
    # 解 A^T A x = A^T b

def ridge_regression(A, b, lam):
    # 用 Cholesky 解 (A^T A + λI) x = A^T b

def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```
