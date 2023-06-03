# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CLIPTokenize(BaseTransform):
    """Tokenize text and convert to tensor."""

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`CLIPTokenize`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """

        try:
            import clip
        except ImportError:
            raise ImportError('Please run `pip install '
                              'git+https://github.com/openai/CLIP.git` '
                              'to install clip first. ')

        text = results['text']
        text_tokenized = clip.tokenize(text)[0]
        results['text'] = text_tokenized
        return results
