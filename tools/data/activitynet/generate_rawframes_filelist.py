import os
import os.path as osp

import mmcv

data_file = '../../../data/ActivityNet'
video_list = f'{data_file}/video_info_new.csv'
anno_file = f'{data_file}/anet_anno_action.json'
rawframe_dir = f'{data_file}/rawframes'
action_name_list = 'action_name.csv'


def generate_rawframes_filelist():
    anet_annotations = mmcv.load(anno_file)

    videos = open(video_list).readlines()
    videos = [x.strip().split(',') for x in videos]
    attr_names = videos[0][1:]
    # the first line is 'video,numFrame,seconds,fps,rfps,subset,featureFrame'
    attr_names = [x.lower() for x in attr_names]
    attr_types = [int, float, float, float, str, int]

    vidannos = {}
    for line in videos[1:]:
        name = line[0]
        data = {}
        for attr_name, attr_type, attr_val in zip(attr_names, attr_types,
                                                  line[1:]):
            data[attr_name] = attr_type(attr_val)
        vidannos[name] = data

    # only keep downloaded videos
    vidannos = {k: v for k, v in vidannos.items() if k in anet_annotations}
    # update numframe
    for vid in vidannos.keys():
        pth = osp.join(rawframe_dir, vid)
        num_imgs = len(os.listdir(pth))
        # one more rgb img than flow
        assert (num_imgs - 1) % 3 == 0
        num_frames = (num_imgs - 1) // 3
        vidannos[vid]['numframe'] = num_frames

    anet_labels = open('action_name.csv')
    anet_labels = [x.strip() for x in anet_labels]

    train_vids = {
        k: vid
        for k, vid in vidannos.items() if vid['subset'] == 'training'
    }
    val_vids = {
        k: vid
        for k, vid in vidannos.items() if vid['subset'] == 'validation'
    }

    def simple_label(vidix):
        anno = anet_annotations[vidix]
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

    def clip_list(k, anno, vidanno):
        n_second = anno['duration_second']
        n_frame = vidanno['numframe']
        fps = n_frame / n_second
        segs = anno['annotations']
        lines = []
        for seg in segs:
            segment = seg['segment']
            lb = seg['label']
            lb = anet_labels.index(lb)
            start, end = int(segment[0] * fps), int(segment[1] * fps)
            if end > n_frame - 1:
                end = n_frame - 1
            newline = f'{k} {start} {end - start + 1} {lb}'
            lines.append(newline)
        return lines

    train_clips, val_clips = [], []
    for k in train_vids.keys():
        train_clips.extend(clip_list(k, anet_annotations[k], train_vids[k]))
    for k in val_vids.keys():
        val_clips.extend(clip_list(k, anet_annotations[k], val_vids[k]))

    with open(osp.join(data_file, 'anet_train_clip.txt'), 'w') as fout:
        fout.write('\n'.join(train_clips))
    with open(osp.join(data_file, 'anet_val_clip.txt'), 'w') as fout:
        fout.write('\n'.join(val_clips))


if __name__ == '__main__':
    generate_rawframes_filelist()
