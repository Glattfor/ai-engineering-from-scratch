# Tensor 操作

> Tensor 是数据与深度学习之间的通用语言。每张图片、每个句子、每个 gradient 都流经它们。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-02
**时间：** ~90 分钟

## 术语对照

- tensor，张量
- rank，秩
- shape，形状
- stride，步幅
- broadcasting，广播
- contiguous，连续的
- einsum，爱因斯坦求和
- NCHW，NCHW 格式
- NHWC，NHWC 格式
- multi-head attention，多头注意力
## 关键术语

| 术语 | 含义 |
|------|------|
| Tensor | 带统一类型和定义好的 shape、stride、运算的多维数组 |
| Rank | 维度数。Matrix rank=2（不要与矩阵秩混淆）|
| Shape | 每条轴的大小。`(2,3)` = 2行3列 |
| Stride | 沿每条轴前进一位需跳过的元素数 |
| Broadcasting | 严格规则：右对齐，相等或1才兼容 |
| Contiguous | 元素按逻辑布局连续存储。transpose 后为 non-contiguous |
| Einsum | 一行表达任何 tensor contraction、外积、trace 或 transpose 的通用记法 |
| NCHW / NHWC | 图像 tensor 内存布局。NCHW 通道在前（PyTorch），NHWC 通道在后（TensorFlow） |

## 学习目标

- 从零实现带 shape、stride、reshape、transpose 和逐元素运算的 tensor 类
- 应用 broadcasting 规则在不同 shape 的 tensor 上运算而不复制数据
- 编写 einsum 表达式完成 dot product、矩阵乘法、外积和 batch 操作
- 追踪 multi-head attention 每一步的精确 tensor shape

## 问题

你在搭 transformer。Forward pass 看起来干净。你运行：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着 shape。你试 transpose。现在报 `Expected 4D input (got 3D input)`。你加 unsqueeze。别的东西又崩了。

Shape 错误是深度学习代码中最常见的 bug。Tensor 就是多维数组。rank = 维度数。shape = 每维大小。32 张 224×224 RGB 图是 4D tensor：(32, 3, 224, 224)。12 头 self-attention 也是 4D：(batch, heads, seq_len, head_dim)。

## 概念

### 内存布局与 stride

Stride 告诉你沿着每条轴移动一步要跳多少个元素。Row-major（C order）：shape (2,3)，strides (3,1)。Transpose 不移动数据，只交换 stride，使 tensor 变成 non-contiguous。

### Broadcasting

在不同 shape 的 tensor 上运算而不复制数据。规则：从右对齐 shape，维度兼容当相等或其中一个为 1。维度少的左侧补 1。

```
A: (8, 1, 6, 1)    B: (7, 1, 5)
B padded: (1, 7, 1, 5)    Result: (8, 7, 6, 5)
```

### Einsum

用字母标注每条轴。仅在输入中出现但不在输出中的轴被求和。关键模式：`i,i->`(dot), `i,j->ij`(外积), `bij,bjk->bik`(batch matmul), `bhtd,bhsd->bhts`(attention scores)。

### Attention 通过 einsum

```python
Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)
concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
```

每一步都是 tensor 操作：投影（einsum matmul）、头拆分（reshape + transpose）、attention 分数（batch matmul）、加权和（batch matmul）、头合并（transpose + reshape）。

## 动手实现

```python
class Tensor:
    def __init__(self, data, shape=None):
        # 将嵌套列表展平，计算 shape 和 strides

    @staticmethod
    def _compute_strides(shape):
        # shape (3,4) → strides (4,1)：跳 4 个元素前进一行

    def reshape(self, new_shape):
        # 改 shape 不改变元素顺序。-1 自动推断

    def squeeze(self, dim=None):
        # 移除大小为 1 的轴

    def unsqueeze(self, dim):
        # 插入大小为 1 的轴，对 broadcasting 关键

    def transpose(self, dim0, dim1):
        # 交换两条轴，交换 strides 不移动数据

    def permute(self, dims):
        # 重排所有轴。NCHW ↔ NHWC 转换
```

代码见 `code/tensors.py`。
