import re
from pathlib import Path


def replace_link(pattern, template, content, file_path):
    MMACT_ROOT = Path(__file__).absolute().parents[2]
    GITHUB_PREFIX = 'https://github.com/open-mmlab/mmaction2/blob/main/'

    def replace_core(matchobj):
        name = matchobj.group(1)
        link = matchobj.group(2)
        if link.startswith('http') or link.startswith('#'):
            return template.format(name, link)
        # For link relative to project folder, such as '/configs/*/*.py'
        elif Path(link).is_absolute():
            link = link.lstrip('/')
            folder = MMACT_ROOT
        # For link relative to current file, such as './config/*.py'
        else:
            folder = file_path.parent
        file_link = link.split('#')[0]
        assert (folder / file_link).exists(), \
            f'Link not found:\n{file_path}: {folder / link}'
        rel_link = (folder / link).resolve().relative_to(MMACT_ROOT)
        link = GITHUB_PREFIX + str(rel_link)
        return template.format(name, link)

    return re.sub(pattern, replace_core, content)
