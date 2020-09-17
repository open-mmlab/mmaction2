import os
import os.path as osp

import mmcv

data_file = '../../../data/ActivityNet'
video_list = f'{data_file}/video_info_new.csv'
anno_file = f'{data_file}/anet_anno_action.json'
rawframe_dir = f'{data_file}/rawframes'
action_name_list = 'action_name.csv'


def get_info():
    anet_annotation = mmcv.load(anno_file)

    videos = open(video_list).readlines()
    videos = [x.strip().split(',') for x in videos]
    attr_names = videos[0][1:]
    # the first line is 'video,numFrame,seconds,fps,rfps,subset,featureFrame'
    attr_names = [x.lower() for x in attr_names]
    attr_types = [int, float, float, float, str, int]

    vidanno = {}
    for line in videos[1:]:
        name = line[0]
        data = {}
        for attr_name, attr_type, attr_val in zip(attr_names, attr_types,
                                                  line[1:]):
            data[attr_name] = attr_type(attr_val)
        vidanno[name] = data

    # only keep downloaded videos
    vidanno = {k: v for k, v in vidanno.items() if k in anet_annotation}
    # update numframe
    for vid in vidanno.keys():
        pth = osp.join(rawframe_dir, vid)
        num_imgs = len(os.listdir(pth))
        # one more rgb img than flow
        assert (num_imgs - 1) % 3 == 0
        num_frames = (num_imgs - 1) // 3
        vidanno[vid]['numframe'] = num_frames

    anet_labels = open('action_name.csv')
    anet_labels = [x.strip() for x in anet_labels]

    train_vids = {
        k: vid
        for k, vid in vidanno.items() if vid['subset'] == 'training'
    }
    val_vids = {
        k: vid
        for k, vid in vidanno.items() if vid['subset'] == 'validation'
    }

    def simple_label(vidix):
        anno = anet_annotation[vidix]
        lb = anno['annotations'][0]['label']
        return anet_labels.index(lb)

    train_lines = [
        k + ' ' + str(train_vids[k]['numframe']) + ' ' + str(simple_label(k))
        for k in train_vids.keys()
    ]
    val_lines = [
        k + ' ' + str(val_vids[k]['numframe']) + ' ' + str(simple_label(k))
        for k in val_vids.keys()
    ]

    with open(osp.join(data_file, 'anet_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'anet_val_video.txt'), 'w') as fout:
        fout.write('\n'.join(val_lines))
