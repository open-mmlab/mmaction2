# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmaction.localization import load_localize_proposal_file
from tools.data.parse_file_list import parse_directory


def process_norm_proposal_file(norm_proposal_file, frame_dict):
    """Process the normalized proposal file and denormalize it.

    Args:
        norm_proposal_file (str): Name of normalized proposal file.
        frame_dict (dict): Information of frame folders.
    """
    proposal_file = norm_proposal_file.replace('normalized_', '')
    norm_proposals = load_localize_proposal_file(norm_proposal_file)

    processed_proposal_list = []
    for idx, norm_proposal in enumerate(norm_proposals):
        video_id = norm_proposal[0]
        frame_info = frame_dict[video_id]
        num_frames = frame_info[1]
        frame_path = osp.basename(frame_info[0])

        gt = [[
            int(x[0]),
            int(float(x[1]) * num_frames),
            int(float(x[2]) * num_frames)
        ] for x in norm_proposal[2]]

        proposal = [[
            int(x[0]),
            float(x[1]),
            float(x[2]),
            int(float(x[3]) * num_frames),
            int(float(x[4]) * num_frames)
        ] for x in norm_proposal[3]]

        gt_dump = '\n'.join(['{} {} {}'.format(*x) for x in gt])
        gt_dump += '\n' if len(gt) else ''
        proposal_dump = '\n'.join(
            ['{} {:.04f} {:.04f} {} {}'.format(*x) for x in proposal])
        proposal_dump += '\n' if len(proposal) else ''

        processed_proposal_list.append(
            f'# {idx}\n{frame_path}\n{num_frames}\n1'
            f'\n{len(gt)}\n{gt_dump}{len(proposal)}\n{proposal_dump}')

    with open(proposal_file, 'w') as f:
        f.writelines(processed_proposal_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Denormalize proposal file')
    parser.add_argument(
        'dataset',
        type=str,
        choices=['thumos14'],
        help='dataset to be denormalize proposal file')
    parser.add_argument(
        '--norm-proposal-file',
        type=str,
        help='normalized proposal file to be denormalize')
    parser.add_argument(
        '--data-prefix',
        type=str,
        help='path to a directory where rawframes are held')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f'Converting from {args.norm_proposal_file}.')
    frame_dict = parse_directory(args.data_prefix)
    process_norm_proposal_file(args.norm_proposal_file, frame_dict)


if __name__ == '__main__':
    main()
