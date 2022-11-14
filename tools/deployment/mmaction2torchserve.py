# Copyright (c) OpenMMLab. All rights reserved.
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from mmengine.config import Config
from mmengine.utils import mkdir_or_exist

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    raise ImportError('`torch-model-archiver` is required.'
                      'Try: pip install torch-model-archiver')


def mmaction2torchserve(
    config_file: str,
    checkpoint_file: str,
    output_folder: str,
    model_name: str,
    label_file: str,
    model_version: str = '1.0',
    force: bool = False,
):
    """Converts MMAction2 model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file (str): In MMAction2 config format.
        checkpoint_file (str): In MMAction2 checkpoint format.
        output_folder (str): Folder where `{model_name}.mar` will be created.
            The file created will be in TorchServe archive format.
        label_file (str): A txt file which contains the action category names.
        model_name (str | None): If not None, used for naming the
            `{model_name}.mar` file that will be created under `output_folder`.
            If None, `{Path(checkpoint_file).stem}` will be used.
        model_version (str): Model's version.
        force (bool): If True, if there is an existing `{model_name}.mar` file
            under `output_folder` it will be overwritten.
    """
    mkdir_or_exist(output_folder)

    config = Config.fromfile(config_file)

    with TemporaryDirectory() as tmpdir:
        config.dump(f'{tmpdir}/config.py')
        shutil.copy(label_file, f'{tmpdir}/label_map.txt')

        args = Namespace(
            **{
                'model_file': f'{tmpdir}/config.py',
                'serialized_file': checkpoint_file,
                'handler': f'{Path(__file__).parent}/mmaction_handler.py',
                'model_name': model_name or Path(checkpoint_file).stem,
                'version': model_version,
                'export_path': output_folder,
                'force': force,
                'requirements_file': None,
                'extra_files': f'{tmpdir}/label_map.txt',
                'runtime': 'python',
                'archive_format': 'default'
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


def parse_args():
    parser = ArgumentParser(
        description='Convert MMAction2 models to TorchServe `.mar` format.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Folder where `{model_name}.mar` will be created.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='If not None, used for naming the `{model_name}.mar`'
        'file that will be created under `output_folder`.'
        'If None, `{Path(checkpoint_file).stem}` will be used.')
    parser.add_argument(
        '--label-file',
        type=str,
        default=None,
        help='A txt file which contains the action category names. ')
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0',
        help='Number used for versioning.')
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='overwrite the existing `{model_name}.mar`')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    mmaction2torchserve(args.config, args.checkpoint, args.output_folder,
                        args.model_name, args.label_file, args.model_version,
                        args.force)
