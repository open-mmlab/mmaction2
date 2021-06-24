import csv
import fnmatch
import glob
import json
import os
import os.path as osp


def parse_directory(path,
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Directory path to parse frames.
        rgb_prefix (str): Prefix of generated rgb frames name.
            default: 'img_'.
        flow_x_prefix (str): Prefix of generated flow x name.
            default: `flow_x_`.
        flow_y_prefix (str): Prefix of generated flow y name.
            default: `flow_y_`.
        level (int): Directory level for glob searching. Options are 1 and 2.
            default: 1.

    Returns:
        dict: frame info dict with video id as key and tuple(path(str),
            rgb_num(int), flow_x_num(int)) as value.
    """
    print(f'parse frames under directory {path}')
    if level == 1:
        # Only search for one-level directory
        def locate_directory(x):
            return osp.basename(x)

        frame_dirs = glob.glob(osp.join(path, '*'))

    elif level == 2:
        # search for two-level directory
        def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

        frame_dirs = glob.glob(osp.join(path, '*', '*'))

    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        """Count file number with a given directory and prefix.

        Args:
            directory (str): Data directory to be search.
            prefix_list (list): List or prefix.

        Returns:
            list (int): Number list of the file with the prefix.
        """
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, frame_dir in enumerate(frame_dirs):
        total_num = count_files(frame_dir,
                                (rgb_prefix, flow_x_prefix, flow_y_prefix))
        dir_name = locate_directory(frame_dir)

        num_x = total_num[1]
        num_y = total_num[2]
        if num_x != num_y:
            raise ValueError(f'x and y direction have different number '
                             f'of flow images in video directory: {frame_dir}')
        if i % 200 == 0:
            print(f'{i} videos parsed')
        frame_dict[dir_name] = (frame_dir, total_num[0], num_x)

    print('frame directory analysis done')
    return frame_dict


def parse_ucf101_splits(level):
    """Parse UCF-101 dataset into "train", "val", "test" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

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
        """A function to map line string to video and label.

        Args:
            line (str): A long directory path, which is a text path.

        Returns:
            tuple[str, str]: (video, label), video is the video id,
                label is the video label.
        """
        items = line.strip().split()
        video = osp.splitext(items[0])[0]
        if level == 1:
            video = osp.basename(video)
            label = items[0]
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
            label = class_mapping[osp.dirname(items[0])]
        return video, label

    splits = []
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


def parse_jester_splits(level):
    """Parse Jester into "train", "val" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Jester dataset.
    """
    # Read the annotations
    class_index_file = 'data/jester/annotations/jester-v1-labels.csv'
    train_file = 'data/jester/annotations/jester-v1-train.csv'
    val_file = 'data/jester/annotations/jester-v1-validation.csv'
    test_file = 'data/jester/annotations/jester-v1-test.csv'

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip() for x in fin]
    class_mapping = {class_index[idx]: idx for idx in range(len(class_index))}

    def line_to_map(line, test_mode=False):
        items = line.strip().split(';')
        video = items[0]
        if level == 1:
            video = osp.basename(video)
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
        if test_mode:
            return video

        label = class_mapping[items[1]]
        return video, label

    with open(train_file, 'r') as fin:
        train_list = [line_to_map(x) for x in fin]

    with open(val_file, 'r') as fin:
        val_list = [line_to_map(x) for x in fin]

    with open(test_file, 'r') as fin:
        test_list = [line_to_map(x, test_mode=True) for x in fin]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_sthv1_splits(level):
    """Parse Something-Something dataset V1 into "train", "val" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Something-Something V1 dataset.
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
        video = items[0]
        if level == 1:
            video = osp.basename(video)
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
        if test_mode:
            return video

        label = class_mapping[items[1]]
        return video, label

    with open(train_file, 'r') as fin:
        train_list = [line_to_map(x) for x in fin]

    with open(val_file, 'r') as fin:
        val_list = [line_to_map(x) for x in fin]

    with open(test_file, 'r') as fin:
        test_list = [line_to_map(x, test_mode=True) for x in fin]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_sthv2_splits(level):
    """Parse Something-Something dataset V2 into "train", "val" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.

    Returns:
        list: "train", "val", "test" splits of Something-Something V2 dataset.
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
        video = item['id']
        if level == 1:
            video = osp.basename(video)
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
        if test_mode:
            return video

        template = item['template'].replace('[', '')
        template = template.replace(']', '')
        label = int(class_mapping[template])
        return video, label

    with open(train_file, 'r') as fin:
        items = json.loads(fin.read())
        train_list = [line_to_map(item) for item in items]

    with open(val_file, 'r') as fin:
        items = json.loads(fin.read())
        val_list = [line_to_map(item) for item in items]

    with open(test_file, 'r') as fin:
        items = json.loads(fin.read())
        test_list = [line_to_map(item, test_mode=True) for item in items]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_mmit_splits():
    """Parse Multi-Moments in Time dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Multi-Moments in Time.
    """

    # Read the annotations
    def line_to_map(x):
        video = osp.splitext(x[0])[0]
        labels = [int(digit) for digit in x[1:]]
        return video, labels

    csv_reader = csv.reader(open('data/mmit/annotations/trainingSet.csv'))
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open('data/mmit/annotations/validationSet.csv'))
    val_list = [line_to_map(x) for x in csv_reader]

    test_list = val_list  # not test for mit

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_kinetics_splits(level, dataset):
    """Parse Kinetics dataset into "train", "val", "test" splits.

    Args:
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.
        dataset (str): Denotes the version of Kinetics that needs to be parsed,
            choices are "kinetics400", "kinetics600" and "kinetics700".

    Returns:
        list: "train", "val", "test" splits of Kinetics.
    """

    def convert_label(s, keep_whitespaces=False):
        """Convert label name to a formal string.

        Remove redundant '"' and convert whitespace to '_'.

        Args:
            s (str): String to be converted.
            keep_whitespaces(bool): Whether to keep whitespace. Default: False.

        Returns:
            str: Converted string.
        """
        if not keep_whitespaces:
            return s.replace('"', '').replace(' ', '_')

        return s.replace('"', '')

    def line_to_map(x, test=False):
        """A function to map line string to video and label.

        Args:
            x (str): A single line from Kinetics csv file.
            test (bool): Indicate whether the line comes from test
                annotation file.

        Returns:
            tuple[str, str]: (video, label), video is the video id,
                label is the video label.
        """
        if test:
            # video = f'{x[0]}_{int(x[1]):06d}_{int(x[2]):06d}'
            video = f'{x[1]}_{int(float(x[2])):06d}_{int(float(x[3])):06d}'
            label = -1  # label unknown
            return video, label

        video = f'{x[1]}_{int(float(x[2])):06d}_{int(float(x[3])):06d}'
        if level == 2:
            video = f'{convert_label(x[0])}/{video}'
        else:
            assert level == 1
        label = class_mapping[convert_label(x[0])]
        return video, label

    train_file = f'data/{dataset}/annotations/kinetics_train.csv'
    val_file = f'data/{dataset}/annotations/kinetics_val.csv'
    test_file = f'data/{dataset}/annotations/kinetics_test.csv'

    csv_reader = csv.reader(open(train_file))
    # skip the first line
    next(csv_reader)

    labels_sorted = sorted({convert_label(row[0]) for row in csv_reader})
    class_mapping = {label: i for i, label in enumerate(labels_sorted)}

    csv_reader = csv.reader(open(train_file))
    next(csv_reader)
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open(val_file))
    next(csv_reader)
    val_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open(test_file))
    next(csv_reader)
    test_list = [line_to_map(x, test=True) for x in csv_reader]

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_mit_splits():
    """Parse Moments in Time dataset into "train", "val" splits.

    Returns:
        list: "train", "val", "test" splits of Moments in Time.
    """
    # Read the annotations
    class_mapping = {}
    with open('data/mit/annotations/moments_categories.txt') as f_cat:
        for line in f_cat.readlines():
            cat, digit = line.rstrip().split(',')
            class_mapping[cat] = int(digit)

    def line_to_map(x):
        video = osp.splitext(x[0])[0]
        label = class_mapping[osp.dirname(x[0])]
        return video, label

    csv_reader = csv.reader(open('data/mit/annotations/trainingSet.csv'))
    train_list = [line_to_map(x) for x in csv_reader]

    csv_reader = csv.reader(open('data/mit/annotations/validationSet.csv'))
    val_list = [line_to_map(x) for x in csv_reader]

    test_list = val_list  # no test for mit

    splits = ((train_list, val_list, test_list), )
    return splits


def parse_hmdb51_split(level):
    train_file_template = 'data/hmdb51/annotations/trainlist{:02d}.txt'
    test_file_template = 'data/hmdb51/annotations/testlist{:02d}.txt'
    class_index_file = 'data/hmdb51/annotations/classInd.txt'

    def generate_class_index_file():
        """This function will generate a `ClassInd.txt` for HMDB51 in a format
        like UCF101, where class id starts with 1."""
        frame_path = 'data/hmdb51/rawframes'
        annotation_dir = 'data/hmdb51/annotations'

        class_list = sorted(os.listdir(frame_path))
        class_dict = dict()
        if not osp.exists(class_index_file):
            with open(class_index_file, 'w') as f:
                content = []
                for class_id, class_name in enumerate(class_list):
                    # like `ClassInd.txt` in UCF-101,
                    # the class_id begins with 1
                    class_dict[class_name] = class_id + 1
                    cur_line = ' '.join([str(class_id + 1), class_name])
                    content.append(cur_line)
                content = '\n'.join(content)
                f.write(content)
        else:
            print(f'{class_index_file} has been generated before.')
            class_dict = {
                class_name: class_id + 1
                for class_id, class_name in enumerate(class_list)
            }

        for i in range(1, 4):
            train_content = []
            test_content = []
            for class_name in class_dict:
                filename = class_name + f'_test_split{i}.txt'
                filename_path = osp.join(annotation_dir, filename)
                with open(filename_path, 'r') as fin:
                    for line in fin:
                        video_info = line.strip().split()
                        video_name = video_info[0]
                        if video_info[1] == '1':
                            target_line = ' '.join([
                                osp.join(class_name, video_name),
                                str(class_dict[class_name])
                            ])
                            train_content.append(target_line)
                        elif video_info[1] == '2':
                            target_line = ' '.join([
                                osp.join(class_name, video_name),
                                str(class_dict[class_name])
                            ])
                            test_content.append(target_line)
            train_content = '\n'.join(train_content)
            test_content = '\n'.join(test_content)
            with open(train_file_template.format(i), 'w') as fout:
                fout.write(train_content)
            with open(test_file_template.format(i), 'w') as fout:
                fout.write(test_content)

    generate_class_index_file()

    with open(class_index_file, 'r') as fin:
        class_index = [x.strip().split() for x in fin]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_index}

    def line_to_map(line):
        items = line.strip().split()
        video = osp.splitext(items[0])[0]
        if level == 1:
            video = osp.basename(video)
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
        label = class_mapping[osp.dirname(items[0])]
        return video, label

    splits = []
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


def parse_diving48_splits():

    train_file = 'data/diving48/annotations/Diving48_V2_train.json'
    test_file = 'data/diving48/annotations/Diving48_V2_test.json'

    train = json.load(open(train_file))
    test = json.load(open(test_file))

    # class_index_file = 'data/diving48/annotations/Diving48_vocab.json'
    # class_list = json.load(open(class_index_file))

    train_list = []
    test_list = []

    for item in train:
        vid_name = item['vid_name']
        label = item['label']
        train_list.append((vid_name, label))

    for item in test:
        vid_name = item['vid_name']
        label = item['label']
        test_list.append((vid_name, label))

    splits = ((train_list, test_list), )
    return splits
