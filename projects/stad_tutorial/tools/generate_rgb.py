# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import cv2

src_dir = 'data/multisports/trainval'
target_dir = 'data/multisports/rawframes'

sport_list = ['aerobic_gymnastics']
for sport in sport_list:
    video_root = osp.join(src_dir, sport)
    if not osp.exists(video_root):
        print('No {} video dir to generate rgb images.'.format(video_root))
        continue
    print('Will generate {} rgb dir for {}.'.format(
        len(os.listdir(video_root)), osp.basename(sport)))
    for clip_name in os.listdir(video_root):
        mp4_path = osp.join(video_root, clip_name)
        save_dir = osp.join(target_dir, sport, clip_name[:-4])
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'I420')
        ii = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            aa = str(ii)
            s = aa.zfill(5)
            image_name = osp.join(save_dir + '/' + s + '.jpg')
            if ret is True:
                cv2.imwrite(image_name, frame)
            else:
                break
            ii = ii + 1
        cap.release()
        print('Generate {} rgb dir successfully.'.format(clip_name[:-4]))
