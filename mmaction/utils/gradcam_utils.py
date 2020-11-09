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
            raise ValueError('GradCAM utils only support Recognizer models.')

        self.target_gradients = None
        self.target_activations = None
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = torch.tensor(model.cfg.img_norm_cfg['mean'])
        self.data_std = torch.tensor(model.cfg.img_norm_cfg['std'])
        self._register_single_hook(target_layer_name)

    def _register_single_hook(self, layer_name):
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
            BB, C, _, _ = gradients.size()
            Tg = BB // B
        else:
            # source shape: [B, Tg, C', H', W']
            _, C, Tg, _, _ = gradients.size()
            # target shape: [B, C', Tg, H', W']
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
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + 1e-12)
        localization_map = localization_map.data

        return localization_map.squeeze(dim=1), preds

    def __call__(self, inputs, use_labels=False, alpha=0.5):
        """Visualize the localization maps on their corresponding inputs as
        heatmap, using Grad-CAM.

        Args:
            inputs (dict): model inputs, generated by test pipeline.
            use_labels (bool): Whether to use given labels to generate
                localization map. Labels are in `inputs['label']`.
            alpha (float): transparency level of the heatmap,
                in the range [0, 1].
        Returns:
        """
        imgs = inputs['imgs']

        # localization_map shape [B, T, H, W]
        # 2D: B = batch_size, T = num_segments
        # 3D: B = batch_size * num_crops * num_clips, T = clip_len
        localization_map, preds = self._calculate_localization_map(
            inputs, use_labels=use_labels)
        if localization_map.device != torch.device('cpu'):
            localization_map = localization_map.cpu()

        # heatmap shape [B, T, H, W, 3] in RGB order
        heatmap = self.colormap(localization_map.detach().numpy())
        heatmap = heatmap[:, :, :, :, :3]

        # Permute input imgs to [B, T, H, W, 3], like heatmap
        if self.is_recognizer2d:
            # Recognizer2D input (B, T, C, H, W)
            curr_inp = imgs.permute(0, 1, 3, 4, 2)
        else:
            # Recognizer3D input (B', num_clips*num_crops, C, T, H, W)
            curr_inp = imgs.view([-1] + list(imgs.size()[2:]))
            curr_inp = curr_inp.permute(0, 2, 3, 4, 1)

        if curr_inp.device != torch.device('cpu'):
            curr_inp = curr_inp.cpu()

        # renormalize input imgs to [0, 1]
        curr_inp *= self.data_std
        curr_inp += self.data_mean
        curr_inp /= 255.

        heatmap = torch.from_numpy(heatmap)
        curr_inp = alpha * heatmap + (1 - alpha) * curr_inp

        return curr_inp, preds
