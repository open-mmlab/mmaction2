import copy

import torch
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import LoadHVULabel


class TestLabel(TestLoading):

    def test_load_hvu_label(self):
        hvu_label_example1 = copy.deepcopy(self.hvu_label_example1)
        hvu_label_example2 = copy.deepcopy(self.hvu_label_example2)
        categories = hvu_label_example1['categories']
        category_nums = hvu_label_example1['category_nums']
        num_tags = sum(category_nums)
        num_categories = len(categories)

        loader = LoadHVULabel()
        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={False})')

        result1 = loader(hvu_label_example1)
        label1 = torch.zeros(num_tags)
        mask1 = torch.zeros(num_tags)
        category_mask1 = torch.zeros(num_categories)

        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={True})')

        label1[[0, 4, 5, 7, 8]] = 1.
        mask1[:10] = 1.
        category_mask1[:3] = 1.

        assert torch.all(torch.eq(label1, result1['label']))
        assert torch.all(torch.eq(mask1, result1['mask']))
        assert torch.all(torch.eq(category_mask1, result1['category_mask']))

        result2 = loader(hvu_label_example2)
        label2 = torch.zeros(num_tags)
        mask2 = torch.zeros(num_tags)
        category_mask2 = torch.zeros(num_categories)

        label2[[1, 8, 9, 11]] = 1.
        mask2[:2] = 1.
        mask2[7:] = 1.
        category_mask2[[0, 2, 3]] = 1.

        assert torch.all(torch.eq(label2, result2['label']))
        assert torch.all(torch.eq(mask2, result2['mask']))
        assert torch.all(torch.eq(category_mask2, result2['category_mask']))
