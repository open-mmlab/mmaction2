import torch

from ..builder import build_sampler
from ..registry import RECOGNIZERS
from .recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class Recognizer3DSampler(Recognizer3D):
    """3D recognizer model framework with sampler."""

    def __init__(self,
                 backbone,
                 cls_head,
                 sampler,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            backbone,
            cls_head,
            neck=neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        # import pdb
        # pdb.set_trace()
        self.sampler = build_sampler(sampler, test_cfg=test_cfg)

    def forward_test(self, imgs, mvs=None, i_frames=None, audios=None):
        """Defines the computation performed at every call when evaluation and
        testing."""
        # import pdb
        # pdb.set_trace()
        test_crop = self.test_cfg.get('test_crop', 3)
        test_clip = self.test_cfg.get('test_clip', 30)
        if test_crop and test_clip:
            assert test_clip * test_crop == imgs.shape[1]
        if imgs.shape[0] != 1:
            raise NotImplementedError(
                'Only supports single batch testing with samplers.')

        top_k_clip_inds = self.sampler(mvs, i_frames, audios)
        # batch x (clips x crops) x C x T x H x W
        imgs = torch.reshape(imgs, (1, ) + (
            test_clip,
            test_crop,
        ) + imgs.shape[2:])
        # batch x  clips x crops x C x T x H x W
        # import pdb
        # pdb.set_trace()
        num_segs = self.sampler.top_k * test_crop
        # pdb.set_trace()
        imgs = imgs[:, top_k_clip_inds, :, :, :, :, :]
        # pdb.set_trace()
        imgs = imgs.reshape((-1, ) + imgs.shape[3:])
        # pdb.set_trace()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_segs)

        return cls_score.cpu().numpy()

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
