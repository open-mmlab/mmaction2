import os
import os.path as osp

annotation_root = '../../../data/gym/annotations'
data_root = '../../../data/gym/subactions'
videos = os.listdir(data_root)
videos = set(videos)

train_file_org = osp.join(annotation_root, 'gym99_train_org.txt')
val_file_org = osp.join(annotation_root, 'gym99_val_org.txt')
train_file = osp.join(annotation_root, 'gym99_train.txt')
val_file = osp.join(annotation_root, 'gym99_val.txt')

train_org = open(train_file_org).readlines()
train_org = [x.strip().split() for x in train_org]
train = [x for x in train_org if x[0] + '.mp4' in videos]
train = [x[0] + '.mp4 ' + x[1] for x in train]
with open(train_file, 'w') as fout:
    fout.write('\n'.join(train_org))

val_org = open(val_file_org).readlines()
val_org = [x.strip().split() for x in val_org]
val = [x for x in val_org if x[0] + '.mp4' in videos]
val = [x[0] + '.mp4 ' + x[1] for x in val]
with open(val_file, 'w') as fout:
    fout.write('\n'.join(val_org))
