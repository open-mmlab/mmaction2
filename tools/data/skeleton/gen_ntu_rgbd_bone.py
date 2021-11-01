

from tqdm import tqdm 
import mmcv 
import os
import os.path as osp
import numpy as np

datasets = {
    'xview', 'xsub'
}
paris = {
    'xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),

    'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
}

sets = {
    'train', 'val'
}

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

for dataset in datasets:  # benchmark
    for set in sets: # part
        print('ing--', dataset, set)

        results = []

        path = os.path.join('/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_nmtvc', dataset)
        path = '{}/{}.pkl'.format(path, set)

        data = mmcv.load(path)
        prog_bar = mmcv.ProgressBar(len(data))
        for i, item in enumerate(data):
            keypoint = item['keypoint'] # CTVM -> MTVC
            M, T, V, C = keypoint.shape
            bone = np.zeros((M, T, V, C), dtype=np.float32)

            for v1, v2 in paris[dataset]:
                v1 -= 1
                v2 -= 1 
                bone [:, :, v1, :] = keypoint[:,:,v1,:] - keypoint[:,:,v2,:]
            item['keypoint'] =  bone 
            results.append(item)
            prog_bar.update()

        out_path = os.path.join('/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_bone', dataset)
        if not osp.exists(out_path):
            os.makedirs(out_path)
        output_path = '{}/{}.pkl'.format(out_path, set)
    
        mmcv.dump(results, output_path)
        print(f'{dataset}--{set} finish!!!!~')




