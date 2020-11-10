import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ..models.recognizers import Recognizer2D, Recognizer3D


class GradCAM:

    def __init__(self, model, target_layer_name, colormap='viridis'):
        """
        Args:
            model (model): the model to be used.
            target_layer_name (str): name of convolutional layer to
                be used to get gradients and feature maps from for creating
                localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create
                heatmap. For more information, please visit:
                https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """
        self.model = model
        self.model.eval()
        if isinstance(model, Recognizer2D):
            self.is_recognizer2d = True
        elif isinstance(model, Recognizer3D):
            self.is_recognizer2d = False
        else:
            raise ValueError(
                'GradCAM utils only support Recognizer2D & Recognizer3D.')

        self.target_gradients = None
        self.target_activations = None
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = torch.tensor(model.cfg.img_norm_cfg['mean'])
        self.data_std = torch.tensor(model.cfg.img_norm_cfg['std'])
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        """Register forward and backward hook to a layer, given layer_name, to
        obtain gradients and activations.

        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.target_gradients = grad_output[0].detach()

        def get_activations(module, input, output):
            self.target_activations = output.clone().detach()

        layer_ls = layer_name.split('/')
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _calculate_localization_map(self, inputs, use_labels):
        """Calculate localization map for all inputs with Grad-CAM.

        Args:
            inputs (dict): model inputs, at least including two keys,
                `imgs` and `label`.
            use_labels (bool): Whether to use given labels to generate
                localization map. Labels are in `inputs['label']`.
        Returns:
            localization_map (ndarray(s)): the localization map for input imgs.
            preds (tensor): Model predictions for `inputs`.
                shape (n_instances, n_class).
        """
        inputs['imgs'] = inputs['imgs'].clone()

        # model forward & backward
        preds = self.model(gradcam=True, **inputs)
        if use_labels:
            labels = inputs['label']
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)
        else:
            score = torch.max(preds, dim=-1)[0]
        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()

        if self.is_recognizer2d:
            # [batch_size, num_segments, 3, H, W]
            B, T, _, H, W = inputs['imgs'].size()
        else:
            # [batch_size, num_crops*num_clips, 3, clip_len, H, W]
            b1, b2, _, T, H, W = inputs['imgs'].size()
            B = b1 * b2

        gradients = self.target_gradients
        activations = self.target_activations
        if self.is_recognizer2d:
            # [B*Tg, C', H', W']
            BxTg, C, _, _ = gradients.size()
            Tg = BxTg // B
        else:
            # source shape: [B, C', Tg, H', W']
            _, C, Tg, _, _ = gradients.size()
            # target shape: [B, Tg, C', H', W']
            gradients = gradients.permute(0, 2, 1, 3, 4)
            activations = activations.permute(0, 2, 1, 3, 4)

        # calculate & resize to [B, 1, T, H, W]
        weights = torch.mean(gradients.view(B, Tg, C, -1), dim=3)
        weights = weights.view(B, Tg, C, 1, 1)
        activations = activations.view([B, Tg, C] +
                                       list(activations.size()[-2:]))
        localization_map = torch.sum(
            weights * activations, dim=2, keepdim=True)
        localization_map = F.relu(localization_map)
        localization_map = localization_map.permute(0, 2, 1, 3, 4)
        localization_map = F.interpolate(
            localization_map,
            size=(T, H, W),
            mode='trilinear',
            align_corners=False,
        )

        # Normalize the localization map.
        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[0],
            torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[0],
        )
        localization_map_min = torch.reshape(
            localization_map_min, shape=(B, 1, 1, 1, 1))
        localization_map_max = torch.reshape(
            localization_map_max, shape=(B, 1, 1, 1, 1))
        # delta must be small enough.
        # make sure `localization_map_max - localization_map_min >> delta`
        delta = 1e-20
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + delta)
        localization_map = localization_map.data

        return localization_map.squeeze(dim=1), preds

    def _alpha_blending(self, localization_map, input_imgs, alpha):
        """Blend heatmap and model input imgs and get visulization results.

        Args:
            localization_map (tensor): localization map for all inputs,
                generated with Grad-CAM
            input_imgs (tensor): model inputs, normed images.
            alpha (float): transparency level of the heatmap,
                in the range [0, 1].
        Returns:
            blended_imgs (tensor): blending results for localization map and
                input images. shape [B, T, H, W, 3], pixel values in
                range [0, 1], in RGB order.
        """
        # localization_map shape [B, T, H, W]
        localization_map = localization_map.cpu()

        # heatmap shape [B, T, H, W, 3] in RGB order
        heatmap = self.colormap(localization_map.detach().numpy())
        heatmap = heatmap[:, :, :, :, :3]
        heatmap = torch.from_numpy(heatmap)

        # Permute input imgs to [B, T, H, W, 3], like heatmap
        if self.is_recognizer2d:
            # Recognizer2D input (B, T, C, H, W)
            curr_inp = input_imgs.permute(0, 1, 3, 4, 2)
        else:
            # Recognizer3D input (B', num_clips*num_crops, C, T, H, W)
            # B = B' * num_clips * num_crops
            curr_inp = input_imgs.view([-1] + list(input_imgs.size()[2:]))
            curr_inp = curr_inp.permute(0, 2, 3, 4, 1)

        # renormalize input imgs to [0, 1]
        curr_inp = curr_inp.cpu()
        curr_inp *= self.data_std
        curr_inp += self.data_mean
        curr_inp /= 255.

        # alpha blending
        blended_imgs = alpha * heatmap + (1 - alpha) * curr_inp

        return blended_imgs

    def __call__(self, inputs, use_labels=False, alpha=0.5):
        """Visualize the localization maps on their corresponding inputs as
        heatmap, using Grad-CAM.

        Generate visualization results for **ALL CROPS**.
        For example, for I3D model, if `clip_len=32, num_clips=10` and
        use `ThreeCrop` in test pipeline, then for every model inputs,
        there are 960(32*10*3) images generated.

        Args:
            inputs (dict): model inputs, generated by test pipeline.
            use_labels (bool): Whether to use given labels to generate
                localization map. Labels are in `inputs['label']`.
            alpha (float): transparency level of the heatmap,
                in the range [0, 1].
        Returns:
            blended_imgs (tensor): Visualization results, blended by
                localization maps and model inputs.
            preds (tensor): Model predictions for inputs.
        """
        imgs = inputs['imgs']

        # localization_map shape [B, T, H, W]
        # preds shape [batch_size, num_classes]
        localization_map, preds = self._calculate_localization_map(
            inputs, use_labels=use_labels)

        # blended_imgs shape [B, T, H, W, 3]
        blended_imgs = self._alpha_blending(localization_map, imgs, alpha)

        # blended_imgs shape [B, T, H, W, 3]
        # preds shape [batch_size, num_classes]
        # 2D: B = batch_size, T = num_segments
        # 3D: B = batch_size * num_crops * num_clips, T = clip_len
        return blended_imgs, preds
