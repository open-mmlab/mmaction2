# Copyright (c) OpenMMLab. All rights reserved.
import gc
from typing import Dict, List, Union

from mmengine.runner import EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmaction.registry import LOOPS


class EpochMultiLoader:
    """Multi loaders based on epoch."""

    def __init__(self, dataloaders: List[DataLoader]):
        self._dataloaders = dataloaders
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __iter__(self):
        """Return self when executing __iter__."""
        return self

    def __next__(self):
        """Get the next iter's data of multiple loaders."""
        data = tuple([next(loader) for loader in self.iter_loaders])
        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])


@LOOPS.register_module()
class MultiLoaderEpochBasedTrainLoop(EpochBasedTrainLoop):
    """EpochBasedTrainLoop with multiple dataloaders.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or Dict): A dataloader object or a dict to
            build a dataloader for training the model.
        other_loaders (List of Dataloader or Dict): A list of other loaders.
            Each item in the list is a dataloader object or a dict to build
            a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 other_loaders: List[Union[Dict, DataLoader]],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval)
        multi_loaders = [self.dataloader]
        for loader in other_loaders:
            if isinstance(loader, dict):
                loader = runner.build_dataloader(loader, seed=runner.seed)
            multi_loaders.append(loader)

        self.multi_loaders = multi_loaders

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        gc.collect()
        for loader in self.multi_loaders:
            if hasattr(loader, 'sampler') and hasattr(loader.sampler,
                                                      'set_epoch'):
                loader.sampler.set_epoch(self._epoch)

        for idx, data_batch in enumerate(EpochMultiLoader(self.multi_loaders)):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
