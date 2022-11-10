from sklearn.manifold import TSNE
import json
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open('/data/junbeom/repo/mmaction2/lab/tsm_ucf101/tsm_k400_to_ucf101_feature.json', 'r') as f :
    feature_data = json.load(f)
with open('/data/junbeom/repo/mmaction2/lab/data/ann_for_plt_ucf101_test_01.txt', 'r') as f :
    ucf101_split_list = f.readlines()

dim = 2
model = TSNE(dim)
tsne_result = model.fit_transform(feature_data)

# print(tsne_result)
print(len(tsne_result)) 
print(len(tsne_result[0])) 
print(len(feature_data)) 
print(len(feature_data[0])) 
# 3783 2 3783 2048 in i3d

class_num = 101

sorted_by_classes_videos = []
for split_idx in range(len(ucf101_split_list) - 1) :
    sorted_by_classes_videos.append(tsne_result[int(ucf101_split_list[split_idx]) : int(ucf101_split_list[split_idx+1]) - 1])

# print(len(sorted_by_classes_videos))
# print(len(sorted_by_classes_videos[0]))
# print(len(sorted_by_classes_videos[0][0]))

np.random.seed(721)
color_idx = np.random.random((class_num))
cmap_idx = plt.cm.magma(color_idx)

with open('/data/junbeom/repo/mmaction2/lab/data/ucfTrainTestlist/classInd.txt', 'r') as f :
    class_bind_list = f.readlines()
    
for class_idx, same_class_videos in enumerate(sorted_by_classes_videos) :
    plt.scatter(same_class_videos[:, 0], same_class_videos[:, 1], c=cmap_idx[class_idx].reshape(1, -1), s=5)
    if class_idx in range(59, 68) :
        plt.text(same_class_videos[0][0], same_class_videos[0][1], class_bind_list[class_idx].split(' ')[1], fontsize=6)



plt.savefig('/data/junbeom/repo/mmaction2/lab/tsm_ucf101/tsm_k400_to_ucf101.png')