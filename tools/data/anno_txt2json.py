import argparse

from mmcv import dump


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert txt anno list to json')
    parser.add_argument(
        'annofile', type=str, help='the txt annofile to convert')
    parser.add_argument(
        '--format',
        type=str,
        default='rawframes',
        choices=['rawframes', 'videos'],
        help='the format of the txt annofile')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=(
            'the output file name, use annofile.replace(\'.txt\', \'.json\') '
            'if the arg value is None'))
    args = parser.parse_args()

    return args


def lines2dictlist(lines, format):
    """Convert lines in 'txt' format to dictions in 'json' format. Currently
    support single-label and multi-label.

    Args:
        lines (list): List of lines in 'txt' label format.
            'frame_dir num_frame label' (rawframes + single-label)
            'frame_dir num_frame label1 label2 ...' (rawframes + multi-label)
            'filename label' (videos + single-label)
            'filename label1 label2 ...' (videos + multi-label)
        format (str): Data format, choices are 'rawframes' and 'videos'.

    Returns:
        list[diction]: For rawframes format, each diction has keys: frame_dir,
            total_frames, label; for videos format, each diction has keys:
            filename, label.
    """
    lines = [x.split() for x in lines]
    if format == 'rawframes':
        data = [
            dict(
                frame_dir=line[0],
                total_frames=int(line[1]),
                label=[int(x) for x in line[2:]]) for line in lines
        ]
    elif format == 'videos':
        data = [
            dict(filename=line[0], label=[int(x) for x in line[1:]])
            for line in lines
        ]
    return data


if __name__ == '__main__':
    # convert txt anno list to json
    args = parse_args()
    lines = open(args.annofile).readlines()
    lines = [x.strip() for x in lines]
    result = lines2dictlist(lines, args.format)
    if args.output is None:
        args.output = args.annofile.replace('.txt', '.json')
    dump(result, args.output)
