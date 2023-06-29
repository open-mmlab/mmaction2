# 参与贡献 MMACTION2

欢迎各种形式的贡献，包括但不限于以下内容。

- 修改拼写错误或代码错误
- 新功能和组件
- 添加文档或将文档翻译成其他语言
- 添加关于视频理解算法的新项目（推荐），具体细节请参考[这里](../projectzoo.md)

## 工作流程

1. Fork 并拉取最新的 mmaction2
2. 创建一个有意义的新分支（不要使用主分支进行 PR）
3. 提交你的更改
4. 创建一个 PR

```{note}
- 如果你计划添加一些涉及大规模更改的新功能，请首先打开一个 issue 进行讨论。
- 如果你是论文的作者，并希望将你的方法包含在 mmaction2 中，请与我们联系。我们将非常感谢您的贡献。
```

## 代码风格

### Python

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选代码风格。

我们使用以下工具进行代码检查和格式化：

- [flake8](http://flake8.pycqa.org/en/latest/)：检查器
- [yapf](https://github.com/google/yapf)：格式化器
- [isort](https://github.com/timothycrosley/isort)：排序导入
- [codespell](https://github.com/codespell-project/codespell)：一个用于修复文本文件中常见拼写错误的 Python 工具。
- [mdformat](https://github.com/executablebooks/mdformat)：Mdformat 是一个自由裁量的 Markdown 格式化工具，可用于强制执行一致的 Markdown 文件样式。
- [docformatter](https://github.com/myint/docformatter)：一个格式化工具，用于格式化文档字符串。

yapf 和 isort 的样式配置可以在 [setup.cfg](https://github.com/open-mmlab/mmaction2/blob/main/setup.cfg) 中找到。

我们使用 [pre-commit hook](https://pre-commit.com/) 来保证每次提交时自动进行代码检查和格式化，启用的功能包括 `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`, 修复 `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, 对 `requirments.txt`的排序等。
预提交钩子的配置存储在 [.pre-commit-config](https://github.com/open-mmlab/mmaction2/blob/main/.pre-commit-config.yaml) 中。

在克隆仓库后，你需要安装初始化的预提交钩子。

```shell
pip install -U pre-commit
```

从仓库文件夹中

```shell
pre-commit install
```

在此之后，每次提交，代码规范检查和格式化工具都将被强制执行。

```{note}
在创建 PR 之前，请确保你的代码通过了 lint 检查并由 yapf 进行了格式化。
```

### C++ 和 CUDA

我们遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)。
