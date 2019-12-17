from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 shorter_edge=256,
                 input_size=224,
                 num_segments=1,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_prefix = data_prefix

    def load_annotations(self, ann_file):
        video_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                video_infos.append({''})

    def __getitem__(self, idx):
        pass
