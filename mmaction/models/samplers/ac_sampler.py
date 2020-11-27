import numpy as np
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..builder import build_recognizer
from ..registry import SAMPLERS

# ACSampler is very like a recognizer


@SAMPLERS.register_module()
class ACSampler(nn.Module):
    """class for ACSampler.

    It is a direvative of recognizer, with at most three sub-recognizers
    insides.
    """

    def __init__(self,
                 top_k,
                 combination=dict(type='av_union_list', k_prime=5),
                 audio_recognizer_config=None,
                 pretrained_audio=None,
                 mv_recognizer_config=None,
                 pretrained_mv=None,
                 if_recognizer_config=None,
                 pretrained_if=None,
                 **kwargs):
        super().__init__()
        assert (audio_recognizer_config and pretrained_audio) or (
            mv_recognizer_config
            and pretrained_mv) or (if_recognizer_config, pretrained_if)
        if audio_recognizer_config and pretrained_audio:
            self.audio_recognizer = build_recognizer(
                audio_recognizer_config, test_cfg=dict(average_clips=None))
            # do not average clips
            load_checkpoint(self.audio_recognizer, pretrained_audio)
        if mv_recognizer_config and pretrained_mv:
            self.mv_recognizer = build_recognizer(mv_recognizer_config)
            load_checkpoint(self.mv_recognizer, pretrained_mv)
        if mv_recognizer_config and pretrained_mv:
            self.mv_recognizer = build_recognizer(mv_recognizer_config)
            load_checkpoint(self.mv_recognizer, pretrained_mv)

        self.top_k = top_k
        self.combination = combination
        if combination.get('type') in ['convex', 'av_union_list']:
            if combination.get('type') == 'convex':
                self.alpha = combination.get('alpha', -1)
                assert 0 <= self.alpha < 1
            if combination.get('type') == 'av_union_list':
                self.k_prime = combination.get('k_prime', self.top_k // 2)
                assert (0 <= self.k_prime < self.top_k) and isinstance(
                    self.k_prime, int)
        else:
            raise NotImplementedError

    def forward(self, mvs=None, i_frames=None, audios=None, **kwargs):
        """Define the computation performed at every call."""
        # import pdb
        # pdb.set_trace()
        a_ac = None
        mv_ac = None
        if_ac = None
        if hasattr(self, 'audio_recognizer') and audios is not None:
            a_ac = self.audio_recognizer(audios, return_loss=False)
        if hasattr(self, 'mv_recognizer') and mvs is not None:
            mv_ac = self.mv_recognizer(mvs.squeeze(0), return_loss=False)
        if hasattr(self, 'if_recognizer') and i_frames is not None:
            if_ac = self.if_recognizer(mvs.squeeze(0), return_loss=False)
        v_ac = np.mean(np.array([i for i in [mv_ac, if_ac] if i is not None]))
        if a_ac is None:
            return np.argsort(np.max(v_ac, axis=1)[-self.top_k:])
        # import pdb
        # pdb.set_trace()
        if isinstance(v_ac, type(np.nan)):
            print(np.argsort(np.max(a_ac, axis=1)[-self.top_k:]))
            return np.argsort(np.max(a_ac, axis=1)[-self.top_k:])

        if self.combination['type'] == 'convex':
            combined_ac = self.alpha * a_ac + (1 - self.alpha) * v_ac
            top_k_inds = np.argsort(combined_ac, axis=1)[:, -self.top_k:]
        elif self.combination['type'] == 'av_union_list':
            top_k_inds = np.argsort(v_ac, axis=1)[:, :self.k_prime]
            top_k_inds = np.concatenate((top_k_inds, np.argsort(a_ac)),
                                        axis=1)[:, -self.top_k:]

        # top_k_inds = np.argsort(np.max(a_ac, axis=1))[-10:]
        print(top_k_inds)
        return top_k_inds
