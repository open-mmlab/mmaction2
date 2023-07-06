#!/usr/bin/env python
import re
from pathlib import Path

from utils import replace_link

DATASETS_ROOT = Path('dataset_zoo')  # Path to save generated paper pages.
DATASETZOO_TEMPLATE = """\
# 数据集统计

在本页面中，我们列举了我们支持的[所有数据集](#所有已支持的数据集)。你可以点击链接跳转至对应的数据集详情页面。

## 所有已支持的数据集

* 数据集数量：{num_datasets}
{dataset_msg}

"""  # noqa: E501


def generate_datasets_pages():
    dataset_list = Path('../../tools/data').glob('*/README.md')
    num_datasets = 0
    dataset_msgs = []

    for file in dataset_list:
        num_datasets += 1

        copy = DATASETS_ROOT / file.parent.with_suffix('.md').name

        title_template = r'^# Preparing (.*)'
        # use chinese doc if exist
        chinese_readme = Path(
            str(file).replace('README.md', 'README_zh-CN.md'))
        if chinese_readme.exists():
            file = chinese_readme
            title_template = r'^# 准备(.*)'
        with open(file, 'r') as f:
            content = f.read()

        title = re.match(title_template, content).group(1)
        title = title.lstrip(' ')
        content = replace_link(r'\[([^\]]+)\]\(([^)]+)\)', '[{}]({})', content,
                               file)
        content = replace_link(r'\[([^\]]+)\]: (.*)', '[{}]: {}', content,
                               file)
        dataset_msgs.append(f'\t - [{title}]({copy})')

        with open(copy, 'w') as f:
            f.write(content)

    dataset_msg = '\n'.join(dataset_msgs)

    modelzoo = DATASETZOO_TEMPLATE.format(
        num_datasets=num_datasets,
        dataset_msg=dataset_msg,
    )

    with open('datasetzoo_statistics.md', 'w') as f:
        f.write(modelzoo)


DATASETS_ROOT.mkdir(exist_ok=True)
generate_datasets_pages()
