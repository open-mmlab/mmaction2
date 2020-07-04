import csv
import fnmatch
import glob
import json
import os


def parse_directory(path,
                    key_func=lambda x: x[-11:],
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Folder path to parse frames.
        key_func (callable): Function to do key mapping.
            default: lambda x: x[-11:].
        rgb_prefix (str): Prefix of generated rgb frames name.
            default: 'img_'.
        flow_x_prefix (str): Prefix of generated flow x name.
            default: `flow_x_`.
        flow_y_prefix (str): Prefix of generated flow y name.
            default: `flow_y_`.
        level (int): Directory level for glob searching. Options are 1 and 2.
            default: 1.
    """
    print(f'parse frames under folder {path}')
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        """Count file number with a given directory and prefix.

        Args:
            directory (str): Data directory to be search.
            prefix_list (list): List or prefix

        Returns:
            list (int): Number list of the file with the prefix.
        """
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, frame_folder in enumerate(frame_folders):
        total_num = count_files(frame_folder,
                                (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = key_func(frame_folder)

        num_x = total_num[1]
        num_y = total_num[2]
        if num_x != num_y:
            raise ValueError(f'x and y direction have different number '
                             f'of flow images in video folder: {frame_folder}')
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (frame_folder, total_num[0], num_x)

    print('frame folder analysis done')
    return frame_dict


def parse_ucf101_splits(level):
    """Parse UCF-101 dataset into "train", "val", "test" splits.

    Args:
        level: directory level of data.

    Returns:
        list: "train", "val", "test" splits of UCF-101.
    """
    class_index_file = 'data/ucf101/annotations/classInd.txt'
    train_file_template = 'data/ucf101/annotations/trainlist{:02d}.txt'
    test_file_template = 'data/ucf101/annotations/testlist{:02d}.txt'

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip().split() for x in fin]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_index}

    def line_to_map(line):
        """A function to map line string to vid and label.

        Args:
            line (str): a long directory path, which is a text path.

        Returns:
            tuple[str, str]: (vid, label), vid is the video id,
                label is the video label.
        """
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = '/'.join(vid.split('/')[-level:])
        label = class_mapping[items[0].split('/')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


def parse_sthv1_splits(level):
    """Parse Something-Something dataset V1 into "train", "val" splits.

    Args:
        level: directory level of data.

    Returns:
        list: "train", "val", "test" splits of Something-Something dataset V1.
    """
    # Read the annotations
    # yapf: disable
    class_index_file = 'data/sthv1/annotations/something-something-v1-labels.csv'  # noqa
    # yapf: enable
    train_file = 'data/sthv1/annotations/something-something-v1-train.csv'
    val_file = 'data/sthv1/annotations/something-something-v1-validation.csv'
    test_file = 'data/sthv1/annotations/something-something-v1-test.csv'

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip() for x in fin]
    class_mapping = {class_index[idx]: idx for idx in range(len(class_index))}

    def line_to_map(line, test_mode=False):
        items = line.strip().split(';')
        vid = items[0]
        vid = '/'.join(vid.split('/')[-level:])
        if test_mode:
            return vid
        else:
            label = class_mapping[items[1]]
            return vid, label

    with open(train_file, 'r') as fin:
        train_list = [line_to_map(x) for x in fin]

    with open(val_file, 'r') as fin:
        val_list = [line_to_map(x) for x in fin]

    with open(test_file, 'r') as fin:
        test_list = [line_to_map(x, test_mode=True) for x in fin]

    return ((train_list, val_list, test_list), )


def parse_sthv2_splits(level):
    """Parse Something-Something dataset V2 into "train", "val" splits.

    Args:
        level: directory level of data.

    Returns:
        list: "train", "val", "test" splits of Something-Something dataset V2.
    """
    # Read the annotations
    # yapf: disable
    class_index_file = 'data/sthv2/annotations/something-something-v2-labels.json'  # noqa
    # yapf: enable
    train_file = 'data/sthv2/annotations/something-something-v2-train.json'
    val_file = 'data/sthv2/annotations/something-something-v2-validation.json'
    test_file = 'data/sthv2/annotations/something-something-v2-test.json'

    with open(class_index_file, 'r') as fin:
        class_mapping = json.loads(fin.read())

    def line_to_map(item, test_mode=False):
        vid = item['id']
        vid = '/'.join(vid.split('/')[-level:])
        if test_mode:
            return vid
        else:
            template = item['template'].replace('[', '')
            template = template.replace(']', '')
            label = class_mapping[template]
            return vid, label

    with open(train_file, 'r') as fin:
        items = json.loads(fin.read())
        train_list = [line_to_map(item) for item in items]

    with open(val_file, 'r') as fin:
        items = json.loads(fin.read())
        val_list = [line_to_map(item) for item in items]

    with open(test_file, 'r') as fin:
        items = json.loads(fin.read())
        test_list = [line_to_map(item, test_mode=True) for item in items]

    return ((train_list, val_list, test_list), )


def parse_mit_splits(level):
    """Parse Moments in Time dataset into "train", "val" splits.

    Args:
        level: directory level of data.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations
    class_mapping = {}
    with open('data/mit/annotations/moments_categories.txt') as f_cat:
        for line in f_cat.readlines():
            cat, digit = line.rstrip().split(',')
            class_mapping[cat] = int(digit)

    def line_to_map(x, test=False):
        vid = '.'.join(x[0].split('.')[:-1])
        label = class_mapping[x[0].split('/')[0]]
        return vid, label

    csv_reader = csv.reader(open('data/mit/annotations/trainingSet.csv'))
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open('data/mit/annotations/validationSet.csv'))
    val_list = [line_to_map(x) for x in csv_reader]

    test_list = val_list  # no test for mit
    return ((train_list, val_list, test_list), )
