import argparse

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert txt annotation list to json')
    parser.add_argument(
        'annofile', type=str, help='the txt annotation file to convert')
    parser.add_argument(
        '--format',
        type=str,
        default='rawframes',
        choices=['rawframes', 'videos'],
        help='the format of the txt annotation file')
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
    """Convert lines in 'txt' format to dictionaries in 'json' format.
    Currently support single-label and multi-label.

    Example of a single-label rawframes annotation txt file:

    .. code-block:: txt

        (frame_dir num_frames label)
        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2

    Example of a multi-label rawframes annotation txt file:

    .. code-block:: txt

        (frame_dir num_frames label1 label2 ...)
        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2

    Example of a single-label videos annotation txt file:

    .. code-block:: txt

        (filename label)
        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2

    Example of a multi-label videos annotation txt file:

    .. code-block:: txt

        (filename label1 label2 ...)
        some/path/000.mp4 1 3 5
        some/path/001.mp4 1 4 8
        some/path/002.mp4 2 4 9

    Args:
        lines (list): List of lines in 'txt' label format.
        format (str): Data format, choices are 'rawframes' and 'videos'.

    Returns:
        list[dict]: For rawframes format, each dict has keys: frame_dir,
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
    mmcv.dump(result, args.output)
