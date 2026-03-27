import pickle
import numpy as np 

def analyze_data(data_sample):    
    print(data_sample.keys())
    print('Keys in the pickle file:', data_sample.keys())
    print('Keypoints shape:', data_sample['keypoint'])
    print('Keypoint score shape:', data_sample['keypoint_score'].shape)
    print('Total frames:', data_sample['total_frames'])
    print('Sample keypoint data:', data_sample['keypoint'][0,0,:5])
    print('Frame dir data:', data_sample['frame_dir'])
    print('Label:', data_sample['label'])

if __name__ == "__main__":
    data_ntu60 = pickle.load(open('/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/data/skeleton/ntu60_2d/ntu60_2d_train.pkl', 'rb'))
    data_ntu_sample = data_ntu60['annotations'][42]
    analyze_data(data_ntu_sample)
    print("="*50)
    data = pickle.load(open('/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/video_1.pkl', 'rb'))
    analyze_data(data)