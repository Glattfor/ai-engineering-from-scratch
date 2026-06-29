# 减少低级错误的编程习惯

> 记录如何避免 typo、缩进错误、数字敲错等低级编程错误。

---

## 为什么会犯低级错误？

### 1. 大脑在思考逻辑，没注意拼写

写代码时，注意力集中在"算法对不对"、"公式有没有漏"，手指自动敲出 `componentsm`、`0.05` 这种错误，大脑当时不会察觉。

### 2. 没有即时验证

如果一次性写完整段代码再运行，错误会累积，排查时更容易烦躁。

### 3. 缺少工具提醒

编辑器没有启用 linting、类型检查或自动格式化，错误在运行前是"隐形"的。

### 4. 复制粘贴后没检查

从别处复制代码，改了大部分，但漏改了一个变量名或数字。

---

## 改进方法

### ✅ 1. 小步快跑：写一个方法，测一个方法

不要等整个类写完再运行。写完 `magnitude()` 就立刻验证：

```bash
python3 -c "from vectors import Vector; print(Vector([3,4]).magnitude())"
```

预期输出 `5.0`，如果不是，立刻知道这个方法有问题。

---

### ✅ 2. 写单元测试

每个 lesson 的 `code/tests/` 目录就是用来放测试的。示例：

```python
import unittest
from vectors import Vector


class TestVector(unittest.TestCase):
    def test_magnitude(self):
        self.assertAlmostEqual(Vector([3, 4]).magnitude(), 5.0)

    def test_dot(self):
        self.assertEqual(Vector([1, 2]).dot(Vector([3, 4])), 11)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(
            Vector([1, 0]).cosine_similarity(Vector([1, 0])), 1.0
        )

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(
            Vector([1, 0]).cosine_similarity(Vector([0, 1])), 0.0
        )
```

运行：

```bash
python3 -m unittest test_vectors -v
```

---

### ✅ 3. 让编辑器帮你抓错

Neovim 里已经配置了 LSP 和 linting，确保这些工具生效：

- **ruff**：Python linter，发现未定义变量、语法错误、格式问题
- **mypy / pyright**：类型检查，发现属性拼写错误

检查 ruff 是否在保存时运行：

```vim
:checkhealth nvim-lint
```

---

### ✅ 4. 加类型提示

类型提示能让拼写错误更容易被发现：

```python
from typing import List


class Vector:
    def __init__(self, components: List[float]) -> None:
        self.components: List[float] = list(components)

    def dot(self, other: "Vector") -> float:
        return sum(a * b for a, b in zip(self.components, other.components))
```

如果写成 `self.componentsm`，类型检查器会报错：

```text
error: "Vector" has no attribute "componentsm"
```

---

### ✅ 5. 用自动格式化

配置 Neovim 保存时自动格式化：

```lua
vim.api.nvim_create_autocmd("BufWritePre", {
    pattern = "*.py",
    callback = function()
        vim.lsp.buf.format()
    end,
})
```

ruff / black 会自动修正：

- `a , b` → `a, b`
- 多余空行
- 缩进不一致

---

### ✅ 6. 认真读错误信息

Python 的错误信息已经把答案告诉你了：

| 错误信息 | 含义 | 解决方向 |
|----------|------|----------|
| `TabError: inconsistent use of tabs and spaces` | 缩进混用 | 统一用空格或 Tab |
| `AttributeError: 'Vector' object has no attribute 'componentsm'` | 属性名拼错 | 检查变量名 |
| `SyntaxError: invalid syntax` | 语法错误 | 检查括号、冒号 |
| `ZeroDivisionError` | 除以零 | 检查分母是否可能为零 |

养成从上到下读完错误信息的习惯，不要只看第一行。

---

### ✅ 7. 运行前快速扫一眼

提交或运行前，花 10 秒检查：

- [ ] 所有变量名是否一致
- [ ] `return` 后面是否跟对了
- [ ] 数字有没有少打（`0.5` vs `0.05`）
- [ ] 括号是否匹配
- [ ] 缩进是否对齐

---

## 针对本项目的具体做法

1. 每节课的 `code/` 目录下，写 `main.py` 时同时写 `tests/test_main.py`
2. 运行课程代码前，先跑测试：

   ```bash
   cd phases/NN-phase/MM-lesson/code
   python3 main.py && python3 -m unittest discover tests -v
   ```

3. 在 Neovim 里保存 Python 文件时，让 ruff 自动格式化
4. 遇到报错不要慌，先完整阅读错误信息

---

## 一句话总结

> **写代码 = 写 + 错 + 改 + 再错 + 再改。工具和运动是减少低级错误的最好办法。**

*最后更新：2026-06-14*
