#!/usr/bin/env python
import re
from pathlib import Path

from utils import replace_link

DATASETS_ROOT = Path('dataset_zoo')  # Path to save generated paper pages.
MODELZOO_TEMPLATE = """\
# Dataset Zoo Summary

In this page, we list [all datasets](#all-supported-datasets) we support. You can click the link to jump to the corresponding dataset pages.

## All supported datasets

* Number of datasets: {num_datasets}
{dataset_msg}

"""  # noqa: E501


def generate_datasets_pages():
    dataset_list = Path('../../tools/data').glob('*/README.md')
    num_datasets = 0
    dataset_msgs = []

    for file in dataset_list:
        num_datasets += 1

        copy = DATASETS_ROOT / file.parent.with_suffix('.md').name

        with open(file, 'r') as f:
            content = f.read()

        title = re.match(r'^# Preparing (.*)', content).group(1)
        content = replace_link(r'\[([^\]]+)\]\(([^)]+)\)', '[{}]({})', content,
                               file)
        content = replace_link(r'\[([^\]]+)\]: (.*)', '[{}]: {}', content,
                               file)
        dataset_msgs.append(f'\t - [{title}]({copy})')

        with open(copy, 'w') as f:
            f.write(content)

    dataset_msg = '\n'.join(dataset_msgs)

    modelzoo = MODELZOO_TEMPLATE.format(
        num_datasets=num_datasets,
        dataset_msg=dataset_msg,
    )

    with open('datasetzoo_statistics.md', 'w') as f:
        f.write(modelzoo)


DATASETS_ROOT.mkdir(exist_ok=True)
generate_datasets_pages()
