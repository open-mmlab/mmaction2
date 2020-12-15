#!/usr/bin/env python
import functools as func
import glob
import os
import re
from itertools import chain


def link_anchor(name_str):
    return name_str.strip().replace(' ', '-').replace('.', '')


config_dir_names = ['localization', 'recognition']

stats = []

for config_dir_name in config_dir_names:
    with open(config_dir_name + '_models.md') as content_file:
        content = content_file.read()

    # title
    title = content.split('\n')[0].replace('#', '')

    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmaction' in x)

    # count models and papers
    model_item_list = ''
    papers = set()

    readme_files = []
    all_config_dirs = glob.glob('../configs/' + config_dir_name + '*')
    for path, dirs, files in chain.from_iterable(
            os.walk(conf_dir) for conf_dir in all_config_dirs):
        readme_files += [
            os.path.join(path, f) for f in files if f.endswith('.md')
        ]

    for f in readme_files:
        with open(f, 'r') as readme_file:
            readme_content = '\n' + readme_file.read()
        model = [x.strip() for x in re.findall(r'\n# (.*)\n', readme_content)]
        assert len(model) >= 1
        _papers = set(
            x.lower().strip()
            for x in re.findall(r'\btitle *= *"?{(.*)}"?', readme_content))
        papers = papers.union(_papers)
        model_item_list += ('    - [' + model[0] + '](#' +
                            link_anchor(model[0]) + ')\n')
        model_item_list += '\n'.join(sorted('      - ' + x for x in _papers))
        model_item_list += '\n'
    # organize statsmsg
    statsmsg = f"""
## [{title.strip() + ' Statistics'}](#{link_anchor(title)})

* Number of checkpoints: {len(ckpts)}
* Number of configs: {len(configs)}
* Number of papers: {len(papers)}
{model_item_list}

    """

    stats.append((papers, configs, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _, _ in stats])
allconfigs = func.reduce(lambda a, b: a.union(b), [c for _, c, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, _, c, _ in stats])
msglist = '\n'.join(x for _, _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}

{msglist}

"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)

for f in [f_name + '_models.md' for f_name in config_dir_names]:
    with open(f, 'r') as model_content_file:
        with open('modelzoo.md', 'a') as modelzoo_file:
            for line in model_content_file:
                modelzoo_file.write(line)
