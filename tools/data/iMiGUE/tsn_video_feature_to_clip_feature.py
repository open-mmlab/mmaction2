'''
Created on Dec 9, 2022

@author: micro-gesture
'''

import json
import numpy as np
import os
import pandas as pd


tsn_clip_feature_directory = '/media/micro-gesture/work/hshi/project/BSN-boundary-sensitive-network.pytorch/data/imigue/clip_feature_tsn_depth8_clip_length100_overlap0.5'
tsn_video_feature_directory = '/media/micro-gesture/work/hshi/project/BSN-boundary-sensitive-network.pytorch/data/imigue/feature_tsn_8'
video_annotation_file_path = '/media/micro-gesture/work/hshi/project/PycharmProjects/mmaction2/data/iMiGUE/label/imigue_annotation_corrected.json'
    
with open(video_annotation_file_path) as json_file:
    video_annotations = json.load(json_file)
    

feature_scale = 8
clip_temporal_dim = 100
step = clip_temporal_dim//2


if os.path.isdir(tsn_clip_feature_directory) is False:
    os.makedirs(tsn_clip_feature_directory)


video_annotations = video_annotations['database']
videoNames = os.listdir(tsn_video_feature_directory)

video_dict = {}

stats = dict()

#for i in range

for videoName in videoNames:
    df = pd.read_csv(os.path.join(tsn_video_feature_directory, videoName))
    
    feature = df.values
    N, C = feature.shape
    #aa = N * 8
    begs_feature = list(range(0, N-step, step))
    video_annotation = video_annotations[videoName.split('.')[0]]
    
    for i,  beg_feature in enumerate(begs_feature):
        assert N - beg_feature >= step 
        
        end_feature = min(N, beg_feature + clip_temporal_dim)
        print (videoName, beg_feature, end_feature)
        #newdf = pd.DataFrame(feature[beg_feature:end_feature, :])
        #newdf.to_csv(os.path.join(tsn_clip_feature_directory, videoName.split('.')[0] + '-' + str(i) + '.csv'), index=False)
    
        beg_frame = beg_feature * feature_scale
        end_frame = end_feature * feature_scale

        beg_second = beg_frame/video_annotation['fps']
        end_second = end_frame/video_annotation['fps']
        
        
        clip_annotation = []
        
        for annotation in video_annotation['annotations']:
                        
            if (annotation['segment'][0] >= beg_second and
                annotation['segment'][1] <= end_second):
                        
                annotation['segment'][0] = annotation['segment'][0] - beg_second
                annotation['segment'][1] = annotation['segment'][1] - beg_second
                clip_annotation.append(annotation)

                if annotation['label'] not in stats.keys():
                    stats[annotation['label']] = 1
                else:
                    stats[annotation['label']] = stats[annotation['label']] + 1

        


        video_dict[videoName.split('.')[0] + '-{:d}'.format(i)] = {
                        'fps': video_annotation['fps'],
                        'beg_feature': beg_feature,
                        'end_feature': end_feature,
                        'feature_frame': clip_temporal_dim,
                        'duration_frame': clip_temporal_dim * feature_scale,
                        'duration_second': clip_temporal_dim * feature_scale / video_annotation['fps'],
                        'beg_frame': beg_frame,
                        'end_frame': end_frame,
                        'beg_second': beg_second,
                        'end_second': end_second,
                        'subset': video_annotation['subset'],
                        'annotations': clip_annotation}
    

    
print (len(stats))
json_object = json.dumps(video_dict)
with open("../../../data/iMiGUE/label/imigue_clip_annotation_100_8_corrected.json", "w") as outfile:
    outfile.write(json_object)
    
    
    
    
    
    
    
