'''
Created on Dec 2, 2022

@author: micro-gesture
'''
import os



import pandas as pd
label_file = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/labels_train_protocol2.csv'
class_index_map_file = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/Class_Index_Map.csv'
dest_dir = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/annotations_train'

def imigue_to_ssn_labels(label_file, dest_dir, class_index_map_file='/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/Class_Index_Map.csv'):
    
    
    dataframe = pd.read_csv(label_file)
    class_index_map = pd.read_csv(class_index_map_file)
    
    labelTypes = set(dataframe['class'])
    
    
    for label in labelTypes:
        currentDataFrame = dataframe[dataframe['class']==label].reset_index(drop=True)
        
        labelName = class_index_map[class_index_map['index']==label].reset_index(drop=True)['type'][0]
        
        
        num_annotations = currentDataFrame.shape[0]
        
        outPath = os.path.join(dest_dir, labelName + '.txt')
        
        with open(outPath, 'w') as f:
        
            for i in range(num_annotations):
                
            
                videoName = '{:04d}'.format(int(currentDataFrame['video_id'][i]))
                start_time = currentDataFrame['start_time'][i]
                end_time = currentDataFrame['end_time'][i]
                currentline = videoName + '  ' + '{:.2f}'.format(start_time) + ' ' + '{:.2f}'.format(end_time)
                
                f.writelines(currentline)
        
    
    
imigue_to_ssn_labels(label_file, dest_dir)

label_file = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/labels_test_protocol2.csv'
class_index_map_file = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/Class_Index_Map.csv'
dest_dir = '/media/micro-gesture/work/hshi/project/mmaction2/data/iMiGUE/annotations_test'



imigue_to_ssn_labels(label_file, dest_dir)