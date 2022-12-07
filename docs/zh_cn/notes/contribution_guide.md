# 参与贡献 MMAction2

欢迎任何类型的贡献，包括但不限于

- 修改拼写错误或代码错误
- 添加文档或将文档翻译成其他语言
- 添加新功能和新组件

## 工作流程

1. fork 并 pull 最新的 OpenMMLab 仓库 (MMAction2)
2. 签出到一个新分支（不要使用 master 分支提交 PR）
3. 进行修改并提交至 fork 出的自己的远程仓库
4. 在我们的仓库中创建一个 PR

```{note}
如果你计划添加一些新的功能，并引入大量改动，请尽量首先创建一个 issue 来进行讨论。
如果你是论文作者，希望在 MMAction2 中支持你的算法，请联系我们。 我们十分感谢你的贡献。
```

## 代码风格

### Python

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为统一的代码风格。

我们使用下列工具来进行代码风格检查与格式化：

- [flake8](https://github.com/PyCQA/flake8): Python 官方发布的代码规范检查工具，是多个检查工具的封装
- [isort](https://github.com/timothycrosley/isort): 自动调整模块导入顺序的工具
- [yapf](https://github.com/google/yapf): 一个 Python 文件的格式化工具。
- [codespell](https://github.com/codespell-project/codespell): 检查单词拼写是否有误
- [mdformat](https://github.com/executablebooks/mdformat): 检查 markdown 文件的工具
- [docformatter](https://github.com/myint/docformatter): 一个 docstring 格式化工具。

yapf 和 isort 的格式设置位于 [setup.cfg](../../../setup.cfg)

我们使用 [pre-commit hook](https://pre-commit.com/) 来保证每次提交时自动进行代
码检查和格式化，启用的功能包括 `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`, 修复 `end-of-files`, `double-quoted-strings`,
`python-encoding-pragma`, `mixed-line-ending`, 对 `requirments.txt`的排序等。
pre-commit hook 的配置文件位于 [.pre-commit-config](../../../.pre-commit-config.yaml)

在你克隆仓库后，你需要按照如下步骤安装并初始化 pre-commit hook。

```shell
pip install -U pre-commit
```

在仓库文件夹中执行

```shell
pre-commit install
```

在此之后，每次提交，代码规范检查和格式化工具都将被强制执行。

```{important}
在创建 PR 之前，请确保你的代码完成了代码规范检查，并经过了 yapf 的格式化。
```

### C++ 和 CUDA

C++ 和 CUDA 的代码规范遵从 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
