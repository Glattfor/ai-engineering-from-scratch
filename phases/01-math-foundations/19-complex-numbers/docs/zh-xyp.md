# AI 中的复数

> -1 的平方根不是虚构的。它是旋转、频率和半个信号处理的钥匙。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1, Lesson 01-04
**时间：** ~60 分钟

## 术语对照

- complex number，复数
- conjugate，共轭
- magnitude，模
- phase，辐角
- polar form，极坐标形式
- Euler's formula，欧拉公式
- phasor，相量
- root of unity，单位根
- RoPE，旋转位置编码
- DFT，离散傅里叶变换
## 关键术语

| 术语 | 含义 |
|------|------|
| 复数 | a+bi，i²=-1 |
| 共轭 | a-bi。镜像 |
| 模 | √(a²+b²)。到原点距离 |
| 辐角 | atan2(b,a)。与正实轴的夹角 |
| 极坐标 | r·e^(iθ)。乘法变模乘角加 |
| Euler 公式 | e^(iθ)=cosθ+i·sinθ |
| Phasor | 旋转复数，表示正弦信号 |
| 单位根 | e^(2πi·k/N)，DFT 的基 |
| RoPE | 用复数乘法编码相对位置的旋转位置 Embedding |

## 学习目标

- 在直角坐标和极坐标形式下执行复数运算（加、乘、除、共轭）
- 应用 Euler 公式在复指数与三角函数间转换
- 使用单位复根实现离散 Fourier 变换
- 解释复数旋转如何构成 transformer 中 RoPE 和正弦位置编码的基础

## 概念

### 什么是复数

z = a + bi。a 实部，b 虚部，i² = -1。将数轴扩展为平面。实轴水平，虚轴垂直。

### 复数运算

加法：(a+bi)+(c+di) = (a+c)+(b+d)i
乘法：(a+bi)(c+di) = (ac-bd)+(ad+bc)i（分配律，记住 i²=-1）
共轭：(a+bi)* = a-bi。z·z* = a²+b²（总为实数）
除法：分子分母同乘分母共轭即可化为 a²+b² 分母

### 极坐标形式

z = r·(cos θ + i·sin θ)，r = |z| = √(a²+b²)（模），θ = atan2(b,a)（辐角）。直角坐标适合加法，极坐标适合乘法：模相乘，角相加。乘 magnitude=1 的复数是纯旋转。

### Euler 公式

**e^(iθ) = cos θ + i·sin θ。** 本课最重要的公式。

当 θ=π：e^(iπ) = -1 ⇒ e^(iπ) + 1 = 0。五个基本常数（e,i,π,1,0）在单一方程中。

e^(iθ) 随 θ 变化描出单位圆。复指数就是旋转。而信号处理和 ML 中到处是旋转。

### 与 2D 旋转的联系

(x+yi) × e^(iθ) 绕原点旋转点 (x,y) θ 角。与 2×2 旋转 matrix 乘法的结果完全相同。复数乘法就是 2D 旋转。

### Phasor

e^(iωt) 是以角频率 ω 绕单位圆旋转的点。实部 = cos(ωt)（余弦波），虚部 = sin(ωt)（正弦波）。正弦信号是旋转复数的"影子"。

### 单位根

N 次单位根是单位圆上 N 个等距点：w_k = e^(2πi·k/N)，k = 0,...,N-1。DFT 的基础。DFT 将信号分解为这些 N 个等距频率的成分。

### 与 Transformer 的联系

**正弦位置编码：** PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(...)。sin/cos 对是不同频率复指数的实部和虚部。低频变慢（粗粒度位置），高频变快（细粒度位置）。

**RoPE：** 显式用复数旋转 matrix 乘 query 和 key vector。两 token 的相对位置成为旋转角度。Attention 用这些旋转后的 vector 计算。

### 为什么 i 不是"imaginary"

"Imaginary"是历史事故（Descartes 的轻蔑）。i 不比如负数更"imaginary"。更有用的理解：i 是 90° 旋转算子。乘一次 i→旋转 90°，再乘一次 i²→再旋转 90° = 指向负实方向。这就是为何 i²=-1。

## 动手实现

```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real, self.imag = real, imag

    def __add__(self, other):
        return Complex(self.real+other.real, self.imag+other.imag)

    def __mul__(self, other):
        return Complex(
            self.real*other.real - self.imag*other.imag,
            self.real*other.imag + self.imag*other.real)

    def magnitude(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

def from_polar(r, theta):
    return Complex(r*math.cos(theta), r*math.sin(theta))

# DFT
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result

# 单位根
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]
```

Python 内置：`z = 3 + 2j`。
