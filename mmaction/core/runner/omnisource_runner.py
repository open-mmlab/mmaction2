# Copyright (c) Open-MMLab. All rights reserved.
import time
import warnings

import mmcv
from mmcv.runner import EpochBasedRunner, Hook
from mmcv.runner.utils import get_host_info


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class OmniSourceDistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        for data_loader in runner.data_loaders:
            if hasattr(data_loader.sampler, 'set_epoch'):
                # in case the data loader uses `SequentialSampler` in Pytorch
                data_loader.sampler.set_epoch(runner.epoch)
            elif hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                # batch sampler in pytorch wraps the sampler as its attributes.
                data_loader.batch_sampler.sampler.set_epoch(runner.epoch)


class OmniSourceRunner(EpochBasedRunner):
    """OmniSource Epoch-based Runner.

    This runner train models epoch by epoch, the epoch length is defined by the
    dataloader[0], which is the main dataloader.
    """

    def run_iter(self, data_batch, train_mode, source, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        # Since we have multiple sources, we add a suffix to log_var names,
        # so that we can differentiate them.
        if 'log_vars' in outputs:
            log_vars = outputs['log_vars']
            log_vars = {k + source: v for k, v in log_vars.items()}
            self.log_buffer.update(log_vars, outputs['num_samples'])

        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loaders = data_loaders
        self.main_loader = self.data_loaders[0]
        # Add aliasing
        self.data_loader = self.main_loader
        self.aux_loaders = self.data_loaders[1:]
        self.aux_iters = [cycle(loader) for loader in self.aux_loaders]

        auxiliary_iter_times = [1] * len(self.aux_loaders)
        use_aux_per_niter = 1
        if 'train_ratio' in kwargs:
            train_ratio = kwargs.pop('train_ratio')
            use_aux_per_niter = train_ratio[0]
            auxiliary_iter_times = train_ratio[1:]

        self._max_iters = self._max_epochs * len(self.main_loader)

        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.main_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, source='')
            self.call_hook('after_train_iter')

            if self._iter % use_aux_per_niter != 0:
                self._iter += 1
                continue

            for idx, n_times in enumerate(auxiliary_iter_times):
                for _ in range(n_times):
                    data_batch = next(self.aux_iters[idx])
                    self.call_hook('before_train_iter')
                    self.run_iter(
                        data_batch, train_mode=True, source=f'/aux{idx}')
                    self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    # Now that we use validate hook, not implement this func to save efforts.
    def val(self, data_loader, **kwargs):
        raise NotImplementedError

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training.
                `data_loaders[0]` is the main data_loader, which contains
                target datasets and determines the epoch length.
                `data_loaders[1:]` are auxiliary data loaders, which contain
                auxiliary web datasets.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2)] means running 2
                epochs for training iteratively. Note that val epoch is not
                supported for this runner for simplicity.
            max_epochs (int | None): The max epochs that training lasts,
                deprecated now. Default: None.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(workflow) == 1 and workflow[0][0] == 'train'
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        mode, epochs = workflow[0]
        self._max_iters = self._max_epochs * len(data_loaders[0])

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            if isinstance(mode, str):  # self.train()
                if not hasattr(self, mode):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an '
                        'epoch')
                epoch_runner = getattr(self, mode)
            else:
                raise TypeError(
                    f'mode in workflow must be a str, but got {mode}')

            for _ in range(epochs):
                if mode == 'train' and self.epoch >= self._max_epochs:
                    break
                epoch_runner(data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
