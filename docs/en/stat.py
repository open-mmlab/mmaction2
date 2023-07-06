#!/usr/bin/env python
import re
import shutil
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load
from modelindex.models.Result import Result
from tabulate import tabulate
from utils import replace_link

MMACT_ROOT = Path(__file__).absolute().parents[2]
PAPERS_ROOT = Path('model_zoo')  # Path to save generated paper pages.
GITHUB_PREFIX = 'https://github.com/open-mmlab/mmaction2/blob/main/'
MODELZOO_TEMPLATE = """\
# Model Zoo Summary

In this page, we list [all algorithms](#all-supported-algorithms) we support. You can click the link to jump to the corresponding model pages.

And we also list all checkpoints for different tasks we provide. You can sort or search checkpoints in the table and click the corresponding link to model pages for more details.

## All supported algorithms

* Number of papers: {num_papers}
{type_msg}

* Number of checkpoints: {num_ckpts}
{paper_msg}

"""  # noqa: E501

METRIC_ALIAS = {
    'Top 1 Accuracy': 'Top-1 (%)',
    'Top 5 Accuracy': 'Top-5 (%)',
}

TASK_MAP = dict(
    detection='Spatio Temporal Action Detection Models',
    localization='Action Localization Models',
    recognition='Action Recognition Models',
    skeleton='Skeleton-based Action Recognition Models',
    retrieval='Video Retrieval Models',
    recognition_audio='Audio-based Action Recognition Models')

model_index = load(str(MMACT_ROOT / 'model-index.yml'))


def build_collections(model_index):
    # add models for collections
    col_by_name = {}
    for col in model_index.collections:
        setattr(col, 'models', [])
        col_by_name[col.name] = col

    for model in model_index.models:
        col = col_by_name[model.in_collection]
        col.models.append(model)
        setattr(model, 'collection', col)
        if model.results is None:
            setattr(model, 'tasks', [])
        else:
            setattr(model, 'tasks', [result.task for result in model.results])


build_collections(model_index)

# save a map from model name to title in README
model2title = dict()


def count_papers(collections):
    total_num_ckpts = 0
    type_count = defaultdict(int)
    paper_msgs = []

    for collection in collections:
        with open(MMACT_ROOT / collection.readme) as f:
            readme = f.read()

        ckpts = set(x.lower().strip()
                    for x in re.findall(r'\[ckpt.*\]\((https?.*)\)', readme))
        total_num_ckpts += len(ckpts)
        title = collection.paper['Title']
        papertype = collection.data.get('type', 'Algorithm')
        type_count[papertype] += 1

        readme_title = re.search(r'^#\s+.+', readme)

        readme = Path(collection.filepath).parents[1].with_suffix('.md').name
        model = Path(collection.filepath).parent.name
        model2title[model] = readme_title.group()[2:].replace(' ', '-')
        paper_msgs.append(f'\t- [{papertype}] [{title}]({PAPERS_ROOT / readme}'
                          f'#{model2title[model]}) ({len(ckpts)} ckpts)')

    type_msg = '\n'.join(
        [f'\t- {type_}: {count}' for type_, count in type_count.items()])
    paper_msg = '\n'.join(paper_msgs)

    modelzoo = MODELZOO_TEMPLATE.format(
        num_papers=len(collections),
        num_ckpts=total_num_ckpts,
        type_msg=type_msg,
        paper_msg=paper_msg,
    )

    with open('modelzoo_statistics.md', 'w') as f:
        f.write(modelzoo)


count_papers(model_index.collections)


def generate_paper_page(collection):

    # Write a copy of README
    with open(MMACT_ROOT / collection.readme) as f:
        content = f.read()
    readme_path = Path(collection.filepath)
    copy = PAPERS_ROOT / readme_path.parents[1].with_suffix('.md').name
    if not copy.exists():
        with open(copy, 'w') as copy_file:
            task = readme_path.parents[1].name
            head_content = f'# {TASK_MAP[task]}\n'
            copy_file.write(head_content)

    def lower_heading(match):
        return '#' + match.group()

    content = replace_link(r'\[([^\]]+)\]\(([^)]+)\)', '[{}]({})', content,
                           Path(collection.readme))
    content = replace_link(r'\[([^\]]+)\]: (.*)', '[{}]: {}', content,
                           Path(collection.readme))

    content = re.sub(r'^#+\s+.+', lower_heading, content, flags=re.M)

    with open(copy, 'a') as copy_file:
        copy_file.write(content)


if PAPERS_ROOT.exists():
    shutil.rmtree(PAPERS_ROOT)
PAPERS_ROOT.mkdir(exist_ok=True)
for collection in model_index.collections:
    generate_paper_page(collection)


def scatter_results(models):
    model_result_pairs = []
    for model in models:
        if model.results is None:
            result = Result(task=None, dataset=None, metrics={})
            model_result_pairs.append((model, result))
        else:
            for result in model.results:
                model_result_pairs.append((model, result))
    return model_result_pairs


def generate_summary_table(task, model_result_pairs, title=None):
    metrics = set()
    for model, result in model_result_pairs:
        if result.task == task:
            metrics = metrics.union(result.metrics.keys())
    metrics = sorted(list(metrics))

    rows = []

    def convert2float(number):
        units = {'M': 1e6, 'G': 1e9, 'T': 1e12}
        if isinstance(number, str):
            num = float(number.rstrip('MGT'))
            number = num * units[number[-1]]
        return number

    for model, result in model_result_pairs:
        if result.task != task:
            continue
        name = model.name
        if model.metadata.parameters is not None:
            params = convert2float(model.metadata.parameters)
            params = f'{params / 1e6:.2f}'  # Params
        else:
            params = None
        if model.metadata.flops is not None:
            flops = convert2float(model.metadata.flops)
            flops = f'{flops / 1e9:.2f}'  # Flops
        else:
            flops = None

        readme = Path(
            model.collection.filepath).parents[1].with_suffix('.md').name
        model = Path(model.collection.filepath).parent.name
        page = f'[link]({PAPERS_ROOT / readme}#{model2title[model]})'
        model_metrics = []
        for metric in metrics:
            model_metrics.append(str(result.metrics.get(metric, '')))

        rows.append([name, params, flops, *model_metrics, page])

    with open('modelzoo_statistics.md', 'a') as f:
        if title is not None:
            f.write(f'\n{title}')
        f.write("""\n```{table}\n:class: model-summary\n""")
        header = [
            'Model',
            'Params (M)',
            'Flops (G)',
            *[METRIC_ALIAS.get(metric, metric) for metric in metrics],
            'Readme',
        ]
        table_cfg = dict(
            tablefmt='pipe',
            floatfmt='.2f',
            numalign='right',
            stralign='center')
        f.write(tabulate(rows, header, **table_cfg))
        f.write('\n```\n')


def generate_dataset_wise_table(task, model_result_pairs, title=None):
    dataset_rows = defaultdict(list)
    for model, result in model_result_pairs:
        if result.task == task:
            dataset_rows[result.dataset].append((model, result))

    if title is not None:
        with open('modelzoo_statistics.md', 'a') as f:
            f.write(f'\n{title}')
    for dataset, pairs in dataset_rows.items():
        generate_summary_table(task, pairs, title=f'### {dataset}')


model_result_pairs = scatter_results(model_index.models)

# Generate Action Recognition Summary
generate_dataset_wise_table(
    task='Action Recognition',
    model_result_pairs=model_result_pairs,
    title='## Action Recognition',
)

# Generate Action Detection Summary
generate_dataset_wise_table(
    task='Action Detection',
    model_result_pairs=model_result_pairs,
    title='## Action Detection',
)

# Generate Skeleton-based Action Recognition Summary
generate_dataset_wise_table(
    task='Skeleton-based Action Recognition',
    model_result_pairs=model_result_pairs,
    title='## Skeleton-based Action Recognition',
)

# Generate Video Retrieval Summary
generate_dataset_wise_table(
    task='Video Retrieval',
    model_result_pairs=model_result_pairs,
    title='## Video Retrieval',
)

# Generate Temporal Action Localization Summary
generate_dataset_wise_table(
    task='Temporal Action Localization',
    model_result_pairs=model_result_pairs,
    title='## Temporal Action Localization',
)
