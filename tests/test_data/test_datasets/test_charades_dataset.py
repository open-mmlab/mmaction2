from mmaction.datasets import CharadesDataset
from .base import BaseTestDataset


class TestCharadesDaataset(BaseTestDataset):

    def test_charades_dataset(self):
        charades_dataset = CharadesDataset()
        return charades_dataset
