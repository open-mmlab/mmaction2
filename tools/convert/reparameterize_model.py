# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch

from mmaction.apis import init_recognizer
from mmaction.models.recognizers import BaseRecognizer


def convert_recoginzer_to_deploy(model, checkpoint, save_path):
    print('Converting...')
    assert hasattr(model, 'backbone') and \
        hasattr(model.backbone, 'switch_to_deploy'), \
        '`model.backbone` must has method of "switch_to_deploy".' \
        f' But {model.backbone.__class__} does not have.'

    model.backbone.switch_to_deploy()
    checkpoint['state_dict'] = model.state_dict()
    torch.save(checkpoint, save_path)

    print('Done! Save at path "{}"'.format(save_path))


def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the repvgg block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    parser.add_argument(
        'save_path',
        help='The path where the converted checkpoint file is stored.')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != '.pth' and save_path.suffix != '.tar':
        print('The path should contain the name of the pth format file.')
        exit()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = init_recognizer(
        args.config_path, checkpoint=args.checkpoint_path, device='cpu')
    assert isinstance(model, BaseRecognizer), \
        '`model` must be a `mmpretrain.classifiers.ImageClassifier` instance.'

    checkpoint = torch.load(args.checkpoint_path)
    convert_recoginzer_to_deploy(model, checkpoint, args.save_path)


if __name__ == '__main__':
    main()
