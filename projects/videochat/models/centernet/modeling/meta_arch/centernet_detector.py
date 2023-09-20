import torch
from detectron2.modeling import (build_backbone, build_proposal_generator,
                                 detector_postprocess)
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from torch import nn


@META_ARCH_REGISTRY.register()
class CenterNetDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        self.register_buffer('pixel_mean',
                             torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer('pixel_std',
                             torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape())

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]

        _, proposal_losses = self.proposal_generator(images, features,
                                                     gt_instances)
        return proposal_losses

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def inference(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        inp = images.tensor
        features = self.backbone(inp)
        proposals, _ = self.proposal_generator(images, features, None)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes):
            if do_postprocess:
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            else:
                r = results_per_image
                processed_results.append(r)
        return processed_results

    def preprocess_image(self, batched_inputs):
        """Normalize, pad and batch the input images."""
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images
