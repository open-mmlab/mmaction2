import argparse

import torch

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('video', help='video file')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    # build the recognizer from a config file and checkpoint file
    model = init_recognizer(args.config, args.checkpoint, device=device)
    # test a single video
    results = inference_recognizer(model, args.video, args.label)

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])


if __name__ == '__main__':
    main()
