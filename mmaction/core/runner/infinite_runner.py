# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from torch.utils.data import DataLoader


@RUNNERS.register_module()
class InfiniteEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner supports dataloader with InfiniteSampler.

    The workers of dataloader will re-initialize, when the iterator of
    dataloader is created. InfiniteSampler is designed to avoid these time
    consuming operations, since the iterator with InfiniteSampler will never
    reach the end.
    """

    def train(self, data_loader: DataLoader, **kwargs) -> None:
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # To reuse the iterator, we only create iterator once and bind it
        # with runner. In the next epoch, the iterator will be used against
        if not hasattr(self, 'data_loader_iter'):
            self.data_loader_iter = iter(self.data_loader)

        # The InfiniteSampler will never reach the end, but we set the
        # length of InfiniteSampler to the actual length of dataset.
        # The length of dataloader is determined by the length of sampler,
        # when the sampler is not None. Therefore, we can simply forward the
        # whole dataset in a epoch by length of dataloader.

        for i in range(len(self.data_loader)):
            data_batch = next(self.data_loader_iter)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
