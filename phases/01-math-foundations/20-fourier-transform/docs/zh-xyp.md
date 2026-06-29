# Fourier 变换

> 每个信号都是正弦波的和。Fourier 变换告诉你有哪些。

**类型：** 动手实现
**语言：** Python
**前置要求：** Phase 1, Lesson 01-04, 19（复数）
**时间：** ~90 分钟

## 术语对照

- DFT，离散傅里叶变换
- FFT，快速傅里叶变换
- inverse DFT，逆离散傅里叶变换
- power spectrum，功率谱
- Nyquist frequency，奈奎斯特频率
- spectral leakage，频谱泄漏
- twiddle factor，旋转因子
- convolution theorem，卷积定理
- spectrogram，频谱图
- aliasing，混叠
- STFT，短时傅里叶变换
## 关键术语

| 术语 | 含义 |
|------|------|
| DFT | N 时域样本→N 频域系数 |
| FFT | O(N log N) DFT 算法。Cooley-Tukey 递归分偶奇 |
| Inverse DFT | 重建时域信号。指数符号翻转，除以 N |
| 功率谱 | |X[k]|²。能量跨频率分布 |
| Nyquist 频率 | fs/2。可表示的最高频率 |
| Spectral leakage | 非周期信号导致的伪频率。加窗减弱 |
| Twiddle factor | e^(-2πi·k/N)，FFT 蝶形运算中组合 sub-DFT |
| Convolution theorem | 时域卷积=频域乘法。信号处理基础 |
| Spectrogram | 跨时间窗口的 FFT。时间×频率的 2D 图 |
| Aliasing | 超 Nyquist 频率表现为较低频率 |

## 学习目标

- 从零实现 DFT 并用 O(N log N) Cooley-Tukey FFT 验证
- 解读频率系数：从信号中提取振幅、相位和功率谱
- 应用 convolution theorem 通过 FFT 乘法执行卷积
- 将 Fourier 频率分解与 transformer 位置编码和 CNN 卷积层联系起来

## 概念

### DFT 定义

X[k] = Σ_n x[n]·e^(-2πi·k·n/N)，k = 0,...,N-1。每个 X[k] 是复数。|X[k]|（magnitude）= 频率 k 的振幅。angle(X[k]) = 相位偏移。

关键洞察：e^(-2πi·k·n/N) 是频率 k 处的旋转 phasor。DFT 计算信号与每个频率的相关性。

### 各系数含义

- X[0]：DC 分量（所有样本和），常偏置
- X[1...N/2-1]：正频率。X[k] 代表 N 个样本中 k 个周期
- X[N/2]：Nyquist 频率——N 样本能表示的最高频率
- X[N/2+1...N-1]：负频率。对实信号，X[N-k]=conj(X[k])

### Inverse DFT

x[n] = (1/N)·Σ_k X[k]·e^(2πi·k·n/N)。完美重建。DFT 是基变换——在不同坐标系下重述相同信息。

### FFT（Cooley-Tukey）

DFT O(N²) → FFT O(N log N)。N=1M：10¹²→2×10⁷ 次操作。

算法：信号分偶数索引和奇数索引，递归计算两半 DFT，用 twiddle factor e^(-2πi·k/N) 组合。

每个递归层 O(N)，共 log₂(N) 层。总计 O(N log N)。

### 功率谱与相位谱

功率谱：P[k] = |X[k]|²。显示各频率有多少能量。
相位谱：atan2(imag, real)。大多数分析关注功率谱。

### 频率分辨率

Δf = fs/N。区分接近频率需要更多样本。捕获高频需要更高的采样率。零填充只插值不增加分辨率。

### Convolution Theorem

**时域卷积 = 频域逐点乘法。** x * h = IFFT(FFT(x)·FFT(h))。大 kernel 时 FFT 卷积远快于直接卷积。CNN 中小 kernel（3×3）直接卷积更快，大 receptive field 用 FFT。

### 窗函数

DFT 假设信号是周期性的。非周期信号边界不连续导致 spectral leakage（伪高频）。窗函数将信号两端渐变到零：

Hann：w[n]=0.5·(1-cos(2πn/(N-1)))。通用。
Hamming：w[n]=0.54-0.46·cos(...)。音频。
Blackman：三重余弦。副瓣抑制最强。

### 与 Transformer 的联系

正弦位置编码每维度对在不同频率振荡，频率从高到低几何级分布。类似 Fourier 系数唯一标识信号，位置编码给出每个位置跨所有频段的唯一模式。pos+k 的编码可表示为 pos 编码的线性函数→模型可学会关注相对位置。

### 与 CNN 的联系

卷积层做卷积→按 convolution theorem 等价于 FFT 输入×FFT kernel×IFFT。FNet 甚至用 FFT 完全替代 self-attention，复杂度 O(N log N) 而非 O(N²)。

### Spectrogram (STFT)

单个 FFT 给出全信号频率但不说何时出现。STFT 在重叠窗口上做 FFT，产生 spectrogram：时间×频率的 2D 表示。语音识别模型（Whisper）在 mel-spectrogram 上操作。

### Aliasing

频率 > fs/2 的信号采样后表现为低频。90Hz 信号 100Hz 采样→看起来像 10Hz。下采样特征图时需抗 aliasing 低通滤波。

## 动手实现

```python
def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0,0)
        for n in range(N):
            angle = -2*math.pi*k*n/N
            total = total + Complex(x[n]) * euler(angle)
        result.append(total)
    return result

def fft(x):
    N = len(x)
    if N <= 1: return [Complex(x[0])]
    even = fft([x[i] for i in range(0,N,2)])
    odd  = fft([x[i] for i in range(1,N,2)])
    result = [Complex(0)]*N
    for k in range(N//2):
        twiddle = euler(-2*math.pi*k/N)
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k+N//2] = even[k] - t
    return result

def convolve_fft(x, h):
    N = len(x)+len(h)-1
    pad = 1
    while pad < N: pad *= 2
    X = fft(x+[0]*(pad-len(x)))
    H = fft(h+[0]*(pad-len(h)))
    Y = [xk*hk for xk,hk in zip(X,H)]
    return [y.real for y in idft(Y)[:N]]
```
