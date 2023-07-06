#!/usr/bin/env python
from pathlib import Path

from utils import replace_link

# This script reads /projects/*/README.md and generate projectzoo.md

all_files = list(Path('../../projects/').glob('*/README.md'))
example_project = '../../projects/example_project/README.md'
all_files.remove(Path(example_project))
all_files.insert(0, Path(example_project))

project_zoo = open('../../projects/README.md').read()
for file in all_files:
    with open(file) as f:
        content = f.read()
        content = replace_link(r'\[([^\]]+)\]\(([^)]+)\)', '[{}]({})', content,
                               file)
        content = replace_link(r'\[([^\]]+)\]: (.*)', '[{}]: {}', content,
                               file)

        project_zoo += content

with open('projectzoo.md', 'w') as f:
    f.write(project_zoo)
