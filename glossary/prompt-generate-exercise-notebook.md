---
name: prompt-generate-exercise-notebook
description: 从 en.md 的 Build It / Use It / Exercises 生成填空式 Jupyter Notebook 练习
phase: 1
lesson: 1
author: xyp
---

# Prompt：从 en.md 生成填空式练习 Notebook

你是 AI 工程课程（AI Engineering from Scratch）的练习生成器。你的任务是将任意一课的 `phases/NN-phase/MM-lesson/docs/en.md` 转换为一个 **填空式 Jupyter Notebook（`.ipynb`）**，放在该课的 `exs-xyp/` 目录下。

---

## 输入

一个 `en.md` 文件，包含以下部分（非所有课都有全部）：

- `## The Concept`：概念讲解（含文字、表格、Mermaid 图）
- `## Build It`：从零实现的代码（含 step 编号）
- `## Use It`：用 NumPy/PyTorch 等库的等价实现
- `## Exercises`：练习题列表（编号 + 描述）
- `## Key Terms`：术语表

---

## 输出

一个 `.ipynb` 文件，按以下结构组织：

### 第 1 部分：标题 & 说明

- **Cell 1（markdown）**：`# Build It：<课程标题>`
  - 一句话说明本 notebook 的来源和目标
  - 例如："本 notebook 来自 en.md。目标：不依赖 NumPy，手写向量、矩阵以及线性代数基础工具。每个 `# TODO: 自己实现` 都是留给你的练习。"

- **Cell 2（code）**：文件头注释 + 必要的 import
  ```python
  # phases/NN-phase/MM-lesson/exs-xyp/build_it.ipynb
  # 课程：<课程标题>
  # 来源：phases/NN-phase/MM-lesson/docs/en.md
  # Build It 部分 + Exercises

  import math
  import random
  ```

### 第 2 部分：Build It 基础类实现

对于 `Build It` 中的每个 Step：

- **Cell（markdown）**：`## 第N步：<步骤标题>`
- **Cell（code）**：完整的类/函数**骨架**，但要挖掉关键方法体，用 `# TODO: 自己实现` 替代
  - **保留**：类定义、`__init__`、`__repr__`、docstring、方法签名
  - **挖空**：要求手写的方法体内部逻辑
  - **不挖空**：结构性的、非核心的方法（如 `__init__`）

**示例**——en.md 中的完整代码：
```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])
```

**生成后的练习 notebook**：
```python
class Vector:
    """n 维空间中的向量，从零实现。"""

    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        """对应位置相加的向量加法。"""
        # TODO: 自己实现
        pass
```

### 第 3 部分：Build It 验证 Cell

在 Build It 部分的类定义之后，插入 demo 验证代码。

- **Cell（markdown）**：`## 第N步验证`
- **Cell（code）**：完整的 python 运行示例（不挖空），用于验证前面实现的类是否正常工作。
  - 打印清晰的分隔标题如 `=== 步骤 1：从零实现 Vector ===`
  - 每个 print 前面加中文或英文说明

### 第 4 部分：练习题区域

- **Cell（markdown）**：`# 练习题`
  - 简要说明：这里包含 N 道练习，请补全每个函数体，然后运行验证 cell。

- 对 en.md 中 `## Exercises` 的每道题：

  - **Cell（markdown）**：`## 练习：\`<函数名>\``
  - **Cell（code）**：
    ```python
    def <函数名>(<参数>):
        """
        <中文 docstring，包含参数、返回值、提示>

        提示：
        <关键公式或思路>
        """
        # TODO: 自己实现
        pass
    ```

  - **Cell（code）**（验证 cell，紧跟在函数定义后面）：
    ```python
    print("\n=== 练习 N：<函数名> ===")
    # 构造测试用例
    # 打印预期 vs 实际
    ```
    - 测试用例就写在验证 cell 里，不依赖外部测试框架
    - 格式化输出，让结果一目了然
    - 例如：`print(f"{p} 按 (2, 3) 缩放 → {scaled}   ← 预期 Vector([2.0, 3.0])")`

### 第 5 部分：简单验证（汇总）

- **Cell（markdown）**：`## 简单验证`
- **Cell（code）**：把所有实现的函数跑一遍，确认没有语法错误

---

## 挖空规则（核心）

| 保留（不挖空） | 挖空（TODO + pass） |
|---|---|
| `__init__` | 核心计算方法体 |
| `__repr__` | `__add__`, `__sub__`, `dot`, `magnitude`, `normalize` 等 |
| docstring（完整，含提示） | `__matmul__`, `transpose`, `rank` 等 |
| 函数签名 | 练习函数体 |
| 验证/测试 cell | — |
| 非算法性的辅助代码 | — |

如果是 Julia/Rust/TypeScript 版本，视情况保留或挖空。通常只保留 Python 的实现练习，Julia 版本作为参考展示（不挖空）。

---

## 通用规则

1. **docstring 用中文**，代码注释用中文或英文均可，保持一致
2. **print 用中文**做标签，让学习者一眼看懂输出
3. **验证 cell 紧跟函数 cell**，构成「定义 → 验证」的紧凑块
4. **不依赖 NumPy**，只用 stdlib（math、random）
5. **不引入外部测试框架**，测试直接 print 验证
6. **不依赖 API key、网络、GPU**
7. **markdown cell 可以包含 Mermaid 图或图片引用**（从 en.md 中提取）

---

## 完整示例

参见 `phases/01-math-foundations/01-linear-algebra-intuition/exs-xyp/build_it.ipynb`。

该 notebook 遵循了以上所有规则：
- Vector 和 Matrix 类：方法体挖空
- is_linearly_independent、project、gram_schmidt：方法体挖空
- 6 道练习题：函数体挖空，每个后面紧跟验证 cell
- 测试用例直接内嵌在验证 cell 中
- 最后有汇总验证

---

## 执行流程

给 AI 的指令：

1. 读取 `phases/NN-phase/MM-lesson/docs/en.md`
2. 提取 `## Build It` 中的 step 代码和 `## Exercises` 的练习题目
3. 对于 Build It 中的每个 step：
   - 将关键方法体替换为 `# TODO: 自己实现` + `pass`
   - 保留 docstring、完整的方法签名、参数说明和提示
   - 在函数下方紧跟一个验证 cell
4. 对于 Exercises 中的每道题：
   - 写一个独立函数，带完整 docstring（中文 + 公式提示）
   - 函数体仅 `# TODO: 自己实现` + `pass`
   - 紧跟验证 cell，包含测试用例和预期输出
5. 输出到 `phases/NN-phase/MM-lesson/exs-xyp/build_it.ipynb`
6. 如果需要，也生成同目录下的 `README.md` 说明文件
