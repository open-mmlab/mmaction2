import os
import os.path as osp

annotation_root = '../../../data/gym/annotations'
data_root = '../../../data/gym/subactions'
frame_data_root = '../../../data/gym/subaction_frames'

videos = os.listdir(data_root)
videos = set(videos)

train_file_org = osp.join(annotation_root, 'gym99_train_org.txt')
val_file_org = osp.join(annotation_root, 'gym99_val_org.txt')
train_file = osp.join(annotation_root, 'gym99_train.txt')
val_file = osp.join(annotation_root, 'gym99_val.txt')
train_frame_file = osp.join(annotation_root, 'gym99_train_frame.txt')
val_frame_file = osp.join(annotation_root, 'gym99_val_frame.txt')

train_org = open(train_file_org).readlines()
train_org = [x.strip().split() for x in train_org]
train = [x for x in train_org if x[0] + '.mp4' in videos]
if osp.exists(frame_data_root):
    train_frames = []
    for line in train:
        length = len(os.listdir(osp.join(frame_data_root, line[0])))
        train_frames.append([line[0], str(length // 3), line[1]])
    train_frames = [' '.join(x) for x in train_frames]
    with open(train_frame_file, 'w') as fout:
        fout.write('\n'.join(train_frames))

train = [x[0] + '.mp4 ' + x[1] for x in train]
with open(train_file, 'w') as fout:
    fout.write('\n'.join(train))

val_org = open(val_file_org).readlines()
val_org = [x.strip().split() for x in val_org]
val = [x for x in val_org if x[0] + '.mp4' in videos]
if osp.exists(frame_data_root):
    val_frames = []
    for line in val:
        length = len(os.listdir(osp.join(frame_data_root, line[0])))
        val_frames.append([line[0], str(length // 3), line[1]])
    val_frames = [' '.join(x) for x in val_frames]
    with open(val_frame_file, 'w') as fout:
        fout.write('\n'.join(val_frames))

val = [x[0] + '.mp4 ' + x[1] for x in val]
with open(val_file, 'w') as fout:
    fout.write('\n'.join(val))
