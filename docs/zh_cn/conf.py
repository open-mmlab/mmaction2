# Copyright (c) OpenMMLab. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MMAction2'
copyright = '2020, OpenMMLab'
author = 'MMAction2 Authors'
version_file = '../../mmaction/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables', 'sphinx_copybutton', 'myst_parser'
]

# numpy and torch are required
autodoc_mock_imports = ['mmaction.version', 'PIL']

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    # 'logo_url': 'https://mmocr.readthedocs.io/en/latest/',
    'menu': [
        {
            'name':
            '教程',
            'url':
            'https://colab.research.google.com/github/'
            'open-mmlab/mmaction2/blob/master/demo/'
            'mmaction2_tutorial_zh-CN.ipynb'
        },
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmaction2'
        },
        {
            'name':
            '上游代码库',
            'children': [
                {
                    'name': 'MMCV',
                    'url': 'https://github.com/open-mmlab/mmcv',
                    'description': '计算机视觉基础库'
                },
                {
                    'name': 'MMClassification',
                    'url': 'https://github.com/open-mmlab/mmclassification',
                    'description': '图像分类代码库'
                },
                {
                    'name': 'MMDetection',
                    'url': 'https://github.com/open-mmlab/mmdetection',
                    'description': '物体检测代码库'
                },
            ]
        },
    ],
    # Specify the language of shared menu
    'menu_lang':
    'cn'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 3

language = 'zh_CN'
master_doc = 'index'


def builder_inited_handler(app):
    subprocess.run(['./merge_docs.sh'])
    subprocess.run(['./stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
