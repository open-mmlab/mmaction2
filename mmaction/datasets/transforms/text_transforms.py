import clip
from typing import Dict
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CLIPTokenize(BaseTransform):

    def transform(self, results: Dict) -> Dict:
        text = results['text']
        text_tokenized = clip.tokenize(text)[0]
        results['text'] = text_tokenized
        return results
