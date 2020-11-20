from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class AVRecognizer(BaseRecognizer):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:
        Methods:`forward_train`, supporting to forward when training.
        Methods:`forward_test`, supporting to forward when testing.

    Attributes:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def extract_feat(self, imgs, audios):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs, audios)
        return x

    def forward_train(self, imgs, audios, labels):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        audios = audios.reshape((-1, ) + audios.shape[2:])

        x = self.extract_feat(imgs, audios)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs, audios):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        audios = audios.reshape((-1, ) + audios.shape[2:])

        x = self.extract_feat(imgs, audios)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)

        return cls_score.cpu().numpy()

    def forward(self, imgs, audios, label, return_loss=True):
        if return_loss:
            return self.forward_train(imgs, audios, label)
        else:
            return self.forward_test(imgs, audios)

    def forward_gradcam(self, audios):
        raise NotImplementedError

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch['imgs']
        audios = data_batch['audios']
        label = data_batch['label']

        losses = self(imgs, audios, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        audios = data_batch['audios']
        label = data_batch['label']

        losses = self(imgs, audios, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
