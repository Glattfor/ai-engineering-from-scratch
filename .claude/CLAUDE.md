# CLAUDE.md

AI 操作手册：本项目是一个 AI 工程课程仓库。以下规则用于指导 Claude Code 生成一致的课程笔记（zh-xyp.ipynb）。

---

## 项目背景

此仓库是 [AI Engineering from Scratch](https://github.com/abertsch72/ai-engineering-from-scratch) 的 fork。`my-notes` 分支包含个人学习笔记（`zh-xyp.ipynb`），这些笔记是对官方 `docs/en.md` 的中文重写与扩展。

官方课程结构见 `AGENTS.md` 和 `LESSON_TEMPLATE.md`。本文档描述的是 **zh-xyp.ipynb 笔记本的特有约定**，作为生成新笔记的参考模板。

---

## 笔记构造管线（Source → Notebook）

每个 lesson 目录下有 3 个关键文件，构成一条构造管线：

```
phases/NN-phase/NN-lesson/
├── docs/
│   ├── en.md          ← ① 官方英文教程（内容蓝本）
│   └── zh-xyp.md      ← ② 中文翻译/改编（基于 en.md，添加中文特有内容）
└── zh-xyp.ipynb       ← ③ 最终笔记本（基于 zh-xyp.md，添加可执行练习）
```

### 管线各阶段职责

| 阶段 | 文件 | 内容来源 | 新增内容 |
|------|------|---------|---------|
| ① 蓝本 | `docs/en.md` | 官方课程 | The Problem, The Concept, Build It, Use It, Ship It, Exercises, Key Terms, Further Reading |
| ② 翻译 | `docs/zh-xyp.md` | 翻译自 en.md | **术语对照**（English→中文）、**关联**表（概念→AI 应用）、中文代码注释、print 信息中文化 |
| ③ 笔记本 | `zh-xyp.ipynb` | 拆分自 zh-xyp.md | **📝 练习题**（可执行 stub + 验证 cell + output）、**讨论笔记 💬**、cell 类型拆分（md/code） |

### 从 zh-xyp.md 到 zh-xyp.ipynb 的转换规则

1. **Markdown 文本**（标题、段落、表格、mermaid）→ markdown cell
2. **Fenced code block**（`` ```python `` / `` ```julia ``）→ code cell（去掉 fence 标记）
3. **代码块的 print 输出**→ 保留为 code cell 的 output
4. **## 练习** section → 保留为 markdown cell（文字描述），然后追加 `## 📝 练习题（来自 exs-xyp）` section
5. **新增**：每个练习题拆分为 3 个 cell（markdown 标题 + code 函数 stub + code 验证调用）
6. **新增**：`## 讨论笔记 💬` section（学习过程中产生的关键理解）
7. **新增**：`## 简单验证` cell（最终语法检查）
8. **Cell 拆分粒度**：每个 `###` 子标题开始新 cell；概念 section 按主题合并（不要像原始 03 那样 20 个碎片 cell）

### 关键注意

- **zh-xyp.md 中的代码**已经带有中文 print 信息和注释，直接拆分成 code cell 即可，不需要重新翻译
- **练习题**在 zh-xyp.md 中只是文字描述，需要在 ipynb 中补充完整的函数实现（含 `# TODO: 自己实现` 标记）和验证 cell
- **讨论笔记**是学习过程中产生的，不在 zh-xyp.md 中，需要根据实际理解添加

---

## Notebook 总体结构（按 section 顺序）

每个 `zh-xyp.ipynb` 是一个 Jupyter notebook，按以下顺序组织 cell：

```
1. 标题 + 副标题（hook）
2. 元数据（类型 / 语言 / 前置要求 / 时间）
3. 术语对照（English → 中文）
4. 关键术语（三列表格）
5. [可选] 扩展阅读
6. 学习目标（4-6 条，以动词开头）
7. 问题（动机：为什么需要学这个？）
8. 概念（核心理论，mermaid 图 + 表格 + 公式 + 示例）
9. **Build It + Use It + 练习**（合并 section，代码直接取自 en.md，行内中文注释讲解；练习为引导式填空）
10. 讨论笔记 💬（关键理解，1-2 句/条）
11. 产出（本节产出的 artifact）
12. 关联（概念 → AI 应用 对照表）
13. 核心收获（要点总结）
```

---

## 第 1 部分：标题与元数据

### 标题格式

```markdown
# <中文标题>

> <英文 hook / motto — 一句话概括本节核心思想>
```

- 中文标题简洁直观（如"线性代数直觉"、"Vector、Matrix 与运算"）
- Hook 用英文写，具有冲击力（如 "所有 AI 模型不过是穿了件花哨外衣的 matrix 数学。"）

### 元数据块

```markdown
**类型：** 学习 | 动手实现
**语言：** Python, Julia
**前置要求：** Phase 0 | Phase 1, Lesson 01（<标题>）
**时间：** ~60 分钟
```

- **类型**：理论为主的写"学习"，代码实现为主的写"动手实现"
- **语言**：列出本节使用的所有语言（通常 Python + Julia）
- **前置要求**：写明依赖的前置课程编号和标题
- **时间**：估算完成时间（分钟）

---

## 第 2 部分：术语对照

```markdown
## 术语对照

- eigenvalue，特征值
- eigenvector，特征向量
- determinant，行列式
- ...
```

- **格式**：`English, 中文翻译`（英文在前，逗号分隔）
- **目的**：建立双语概念映射，帮助阅读英文论文和文档
- **范围**：列出本节涉及的所有核心术语（10-20 个）

---

## 第 3 部分：关键术语

两种格式可选用：

### 格式 A（三列详表）—— 适合基础概念课

```markdown
## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vector | "一个箭头" | 一个代表 n 维空间中某点或某方向的数字列表 |
| Matrix | "一张数字表" | 一个将 vector 从一个空间映射到另一个空间的 transformation |
```

### 格式 B（两列简表）—— 适合计算密集课

```markdown
## 关键术语

| 术语 | 一句话 |
|------|--------|
| **旋转矩阵** | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$，沿圆弧移动点，det = 1 |
```

- 格式 A 侧重**纠正误解**（"人们怎么说 vs 实际是什么"）
- 格式 B 侧重**快速查阅**（公式 + 一句话定义）
- 选择依据：概念需要破除直觉误解时用 A，需要快速索引公式/特性时用 B

---

## 第 4 部分：扩展阅读（可选）

```markdown
## 扩展阅读

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) — 每个运算的直观视觉解释
- [NumPy broadcasting 文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) — NumPy 遵循的精确规则
```

- 放置位置：关键术语之后、学习目标之前（也可放在学习目标之后）
- 每条包含：链接 + 简短说明为什么值得看
- 引用官方文档、经典教材、知名视频教程

---

## 第 5 部分：学习目标

```markdown
## 学习目标

- 用 Python 从零实现 vector 和 matrix 运算（加法、dot product、矩阵乘法）
- 用几何直觉解释 dot product、projection 和 Gram-Schmidt 过程的含义
- 将线性代数概念与 AI 应用联系起来：embedding、attention score 和 LoRA
```

- **4-6 条**，以动词开头（实现/解释/构建/区分/验证/应用）
- **具体可测**：不写"理解线性代数"，写"用 Python 从零实现 matrix 乘法"
- 每条对应后续"动手实现"部分的一个或多个 step

---

## 第 6 部分：问题（动机）

```markdown
## 问题

[2-4 段。描述没有这个知识点会遇到什么困难。
用一个具体的代码片段或场景作为钩子。
解释为什么这个知识点是 AI 工程的必要基础。]
```

**模式**：
1. 展示一段看不懂的代码 → 如果不懂 X，这就是魔法
2. 说明实际场景 → AI 系统中这个知识点无处不在
3. 过渡到本节的学习内容

---

## 第 7 部分：概念

```markdown
## 概念

### <子概念 1>

[解释 + 具体例子 + 公式]

### <子概念 2>

[解释 + mermaid 图 + 表格对比]
```

**关键约定**：

- **Mermaid 图**用于展示流程/关系/变换（不用 ASCII art）
- **表格**用于对比（如"Rank 情况 | 对 ML 意味着什么"）
- **公式**用 LaTeX：`$\sqrt{3^2 + 2^2} = \sqrt{13}$`
- **具体数值例子**优先于抽象公式——先展示 `[3, 4]` 的实际运算，再给通用形式
- **AI 连接**：每个概念说明后紧跟"为什么这对 AI 很重要"，具体到模型/算法名称
- **代码块**可嵌入概念 cell 中展示关键实现片段（用 markdown code fence + language tag）
- **关键区分**用 callout（`>` 引用块）突出显示常见混淆点

---

## 第 8 部分：Build It + Use It + 练习（合并 section）

这三个 section 不再分开，而是放在同一个大 section 下。

### 布局

```markdown
## Build It + Use It + 练习

[简短说明：本节从零实现 + NumPy 等价 + 引导练习]

### 第 1 步：<步骤名称>（Build It）

[代码直接取自 en.md 的 ## Build It，逐行添加中文注释讲解]

### 第 2 步：<步骤名称>（Build It）

...

### Use It：NumPy 等价实现

[代码直接取自 en.md 的 ## Use It，逐行添加中文注释讲解]

---

## 练习（引导式填空）

### 练习 1：<标题>

[引导式填空 —— 关键位置挖空让学习者填写]
```

### Build It / Use It 代码规则

- **直接照搬 en.md**：代码不重新发明，直接从 `docs/en.md` 的 `## Build It` 和 `## Use It` section 复制
- **逐行注释**：在每行（或每逻辑块）上方添加中文注释，解释该行在做什么
- **注释风格**：`# 创建一个绕原点旋转 theta 弧度的 2D 旋转矩阵`（解释意图，不翻译语法）
- **print 中文化**：en.md 中的英文 print 信息改为中文
- **代码本身不改动**：变量名、函数名、逻辑结构保持与 en.md 一致
- **不添加额外的演示/测试 cell**：en.md 中的测试代码已经足够，照搬即可

### 练习规则（引导式填空）

- **来源**：练习题内容取自 en.md 的 `## Exercises` section
- **引导式填空**：在关键位置用 `______` 留空，让学习者填写
- **TODO 注释**：每个空上方标注 `# TODO: <提示>` 引导思考方向
- **难度递进**：练习 1 基础（填空多但简单），练习 2 中等（核心逻辑填空），练习 3 综合（少量空但需理解全局）
- **验证 cell**：每个练习下方紧跟验证 cell，填入正确答案后直接运行验证
- **简单验证**：最后放一个全局语法检查 cell
- **不再有"📝 练习题（来自 exs-xyp）"这个独立 section**——练习直接接在 Use It 之后

### 填空示例

```python
def rotation_2d(theta):
    """
    创建 2D 旋转矩阵。
    提示：cos 和 sin 分别放在什么位置？
    """
    c, s = ______, ______  # TODO: 计算 cos(theta) 和 sin(theta)
    return [[c, -s], [s, c]]
```

### 关键注意

- Build It + Use It 部分的代码是**完整可运行的**（供参考学习），练习部分的代码是**挖空待填的**
- 两者使用相同的函数名（如 `rotation_2d`），练习可以调用 Build It 中已定义的函数
- 填空位置选择"学习关键点"：公式核心、算法关键步骤、易错边界条件

---

## 第 10 部分：讨论笔记 💬

```markdown
## 讨论笔记 💬

学习过程中产生的关键理解：

### <具体问题标题>

[1-3 段深入讨论。回答"这个数学对象到底是什么"类的问题。]
```

- **来源**：学习过程中产生的真问题，不是预设的 FAQ
- **风格**：对话式解释，用加粗标记关键结论
- **例子**："$A - \lambda I$ 是什么？" "为什么特征方程一定是 n 次多项式？"
- **可选 section**——不是每节都有

---

## 第 11 部分：产出

```markdown
## 产出

本节课产出：
- `outputs/prompt-<slug>.md`——<一段话描述这个 prompt 做什么>
```

或者：

```markdown
## 产出

这节课为 <下游课程> 建立了**<什么基础>**。<具体算法> 的代码与生产 ML 系统中 <实际应用> 的算法是**同一回事**。

**核心收获：**
1. **<要点 1>**
2. **<要点 2>**
...
```

- **格式灵活**：可以是产出文件说明，也可以是核心收获列表
- **核心收获**用加粗标题 + 简短解释（1 句）

---

## 第 12 部分：关联

```markdown
## 关联

本文的每个概念都与现代 AI 的具体部分相连：

| 概念 | 出现的地方 |
|------|-----------|
| Dot product | Transformer 中的 attention score，RAG 中的 cosine similarity |
| 矩阵乘法 | 每一个神经网络层，每一次线性变换 |
```

- **两列表格**：概念 | 出现的地方
- **具体到算法/模型/系统名称**：不写"机器学习"，写"Transformer 中的 attention score"
- **可以展开 1-2 个特别重要的关联**（如 LoRA 的详细解释）

---

## 第 13 部分：练习（已合并到 Build It + Use It + 练习）

练习不再独立成 section，而是紧跟 Use It 之后，作为 `## Build It + Use It + 练习` 大 section 的最后部分。

详见[第 8 部分：Build It + Use It + 练习（合并 section）](#第-8-部分build-it--use-it--练习合并-section) 中的练习规则。

---

## Notebook Cell 类型规范

| 内容类型 | Cell 类型 | 说明 |
|---------|----------|------|
| 标题、概念、表格、图表 | markdown | 文字内容 |
| Python/Juliacode（实现代码） | code | 函数/类定义，含执行输出 |
| Python/Julia code（测试/验证） | code | 调用上面定义的函数，含执行输出 |
| Mermaid 图表 | markdown | 用 `mermaid` 或 `` ```mermaid `` 代码块 |

---

## 代码风格

- **类命名**：PascalCase（`Vector`, `Matrix`）
- **方法/函数命名**：snake_case（`matmul`, `cosine_similarity`, `element_wise_multiply`）
- **私有/内部**：前缀下划线或不暴露
- **类型提示**：不使用（保持 notebook 简洁）
- **Docstring**：练习函数写 docstring（含提示），主体类/方法可省略
- **错误信息**：中文为主（如 `raise ValueError("Matrix 是奇异矩阵，逆不存在")`）
- **容差**：浮点比较用 `1e-10` 或 `1e-6`
- **Math**：`import math` 用于数学函数；`import random` 用于随机数
- **Import 位置**：按需导入，notebook 最前面不一定要集中 import

---

## 与官方 en.md 的关系

zh-xyp.ipynb 不是 en.md 的翻译，而是**重新组织和扩展**：

| 方面 | en.md | zh-xyp.ipynb |
|------|-------|-------------|
| 格式 | 单一 markdown 文件 | Jupyter notebook（可执行） |
| 语言 | 英文 | 中英双语（术语对照） |
| 代码 | 独立的 code/main.* | 内嵌在 notebook 中，逐行中文注释 |
| Build It / Use It | 分开的 section | 合并为一个 section，代码直接照搬 |
| 讨论 | 无 | 讨论笔记 💬 |
| 关联 | 无 | 关联对照表 |
| 练习 | 文字描述 | 引导式填空（关键位置挖空） |
| 图表 | ASCII | Mermaid（可渲染） |

**生成新笔记时**：以官方 en.md 为内容蓝本，按本文档的结构重新组织为 zh-xyp.ipynb。

---

## 生成新笔记的检查清单

在生成一个新的 zh-xyp.ipynb 后，逐项确认：

- [ ] 标题中文 + 英文 hook
- [ ] 元数据块完整（类型/语言/前置要求/时间）
- [ ] 术语对照（英文 + 中文，10-20 个）
- [ ] 关键术语表（选 A 或 B 格式）
- [ ] 扩展阅读（选填，2-4 条带说明的链接）
- [ ] 学习目标（4-6 条，动词开头，具体可测）
- [ ] 问题/动机（代码片段钩子 → 场景 → 过渡）
- [ ] 概念（mermaid 图 + 表格 + 公式 + 具体例子 + AI 连接）
- [ ] Build It + Use It + 练习（合并 section，代码直接取自 en.md，逐行中文注释）
- [ ] 练习采用引导式填空（关键位置 `______` + `# TODO: 提示`）
- [ ] 讨论笔记（选填）
- [ ] 产出/核心收获
- [ ] 关联表（概念 → AI 应用）
- [ ] 所有 code cell 可执行（含 seed 设置）
- [ ] 所有 code fence 有 language tag
- [ ] 浮点比较使用容差
- [ ] shape 注释标注在矩阵运算处
