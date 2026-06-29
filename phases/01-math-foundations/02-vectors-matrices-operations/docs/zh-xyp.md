# Vector、Matrix 与运算

> 每个神经网络不过是 matrix 乘法加了一些额外的步骤。

**类型：** 动手实现
**语言：** Python, Julia
**前置要求：** Phase 1, Lesson 01（线性代数直觉）
**时间：** ~60 分钟

## 术语对照

- vector，向量
- matrix，矩阵
- transpose，转置
- determinant，行列式
- inverse，逆矩阵
- broadcasting，广播
- dense layer，全连接层
- activation，激活函数
- scalar，标量
- identity matrix，单位矩阵
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vector | "一个箭头" | 一个有序的数字列表。在 AI 中：高维空间中的一个点。 |
| Matrix | "一张数字表" | 一个线性 transformation。它把 vector 从一个空间映射到另一个空间。 |
| 矩阵乘法 | "把数字乘起来就是了" | 第一个 matrix 的每一行与第二个 matrix 的每一列之间的 dot product。顺序很重要。 |
| Transpose | "翻转它" | 交换行和列。把 m×n 的 matrix 变成 n×m。在 backpropagation 中至关重要。 |
| Determinant | "从 matrix 算出来的一个数" | 衡量 matrix 放大面积（2D）或体积（3D）的倍数。零意味着 transformation 压缩了一个维度。 |
| Inverse | "撤销这个 matrix" | 逆转该 transformation 的 matrix。仅当 determinant 不为零时才存在。 |
| Identity matrix | "那个无聊的 matrix" | 相当于数学中乘以 1 的 matrix。用于 residual connection（ResNet）。 |
| Broadcasting | "魔法般的 shape 修复" | 将较小的数组沿缺失维度重复以匹配较大的数组。 |
| 逐元素 | "常规乘法" | 将对应位置的数相乘。两个数组必须同 shape（或可 broadcast）。 |

## 扩展阅读

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) — 每个运算的直观视觉解释
- [NumPy broadcasting 文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) — NumPy 遵循的精确规则
- [Stanford CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) — ML 专用线性代数的简明参考

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、transpose、determinant 和 inverse
- 区分逐元素乘法和矩阵乘法，并解释各自的适用场景
- 仅使用从零实现的 Matrix 类实现一个单层 dense 神经网络（`relu(W @ x + b)`）
- 解释 broadcasting 规则以及 bias 加法在神经网络框架中的工作方式

## 问题

你想搭建一个神经网络。读代码时看到这行：

```
output = activation(weights @ input + bias)
```

那个 `@` 是矩阵乘法。`weights` 是一个 matrix。`input` 是一个 vector。如果你不知道这些运算在做什么，这行就是魔法。如果你知道，这就是一层 forward pass 的全部——三次运算。

你模型处理的每一张图片都是像素值的 matrix。每一个词 embedding 都是一个 vector。每一个神经网络的每一层都是一次 matrix transformation。你不可能在不精通 matrix 运算的情况下构建 AI 系统，就像你不可能不理解变量就写代码一样。

这节课从零开始建立这种熟练度。

## 概念

### Vector：有序的数字列表

一个 vector 是一串有方向和 magnitude 的数字。在 AI 中，vector 代表数据点、特征或参数。

```
v = [3, 4]        -- 一个 2D vector
w = [1, 0, -2]    -- 一个 3D vector
```

2D vector `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（magnitude）是 5（3-4-5 三角形）。

### Matrix：数字网格

一个 matrix 是 2D 的网格。有行和列。一个 m×n 的 matrix 有 m 行 n 列。

```
A = | 1  2  3 |     -- 2×3 matrix（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，weight matrix 把输入 vector 变换为输出 vector。一个有 784 个输入和 128 个输出的层使用 128×784 的 weight matrix。

### 为什么 shape 重要

矩阵乘法有严格规则：`(m × n) @ (n × p) = (m × p)`。内层维度必须匹配。

```
(128 × 784) @ (784 × 1) = (128 × 1)
   weights       input        output

内层维度：784 = 784  ——合法
```

如果你在 PyTorch 中遇到 shape mismatch 错误，就是这个原因。

### 运算速查表

| 运算 | 做什么 | 神经网络中的使用 |
|------|--------|----------------|
| 加法 | 逐元素相加 | 给输出加上 bias |
| Scalar 乘法 | 缩放每个元素 | learning rate × gradient |
| 矩阵乘法 | 变换 vector | 层的 forward pass |
| Transpose | 翻转行和列 | Backpropagation |
| Determinant | 单个数值概括 | 检查可逆性 |
| Inverse | 撤销一次变换 | 求解线性系统 |
| Identity | 不做任何事的 matrix | 初始化、residual connection |

### 逐元素乘法 vs 矩阵乘法

这个区别经常绊倒初学者。

逐元素：将对应位置的数相乘。两个 matrix 必须同 shape。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行的 dot product 与列。内层维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### Broadcasting

当你把一个 bias vector 加到一个输出 matrix 上时，shape 不匹配。Broadcasting 会拉伸较小的数组以适配较大的。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting 将 vector 按行拉伸：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都自动做这件事。理解它能避免你以为 shape 不对、但代码却正常运行时的困惑。

## 动手实现

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix 是奇异矩阵，逆不存在")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：看它跑起来

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个 single dense layer：`output = relu(W @ x + b)`。每个神经网络的每个 dense 层做的就是这个。

## 实际使用

NumPy 用更少的代码和数量级更快的速度完成上述所有事情。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (逐元素) =\n", A * B)
print("A @ B (矩阵乘法) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层：{weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 用 C 和 Fortran 编写的优化 BLAS 例程来实现。同样的数学，快 100 倍。

NumPy 的 Broadcasting：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动将 1D 的 bias 按行 broadcast 到所有行。每个神经网络框架都是这样处理 bias 加法的。

## 产出

本节课产出一个通过几何直觉教授 matrix 运算的 prompt。见 `outputs/prompt-matrix-operations.md`。

此处构建的 Matrix 类是我们在 Phase 3 Lesson 10 中构建 mini 神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 确认结果为单位 matrix。用三个不同的 2×2 matrix 试试。当 determinant 为零时会发生什么？

2. **实现 3×3 逆矩阵。** 用伴随矩阵法扩展 Matrix 类以计算 3×3 matrix 的逆。与 NumPy 的 `np.linalg.inv` 对比测试。

3. **构建一个两层网络。** 仅使用你的 Matrix 类（不用 NumPy），创建一个两层神经网络：输入 (3) → 隐藏层 (4) → 输出 (2)。初始化随机 weights，运行一次 forward pass，验证所有 shape 正确。
