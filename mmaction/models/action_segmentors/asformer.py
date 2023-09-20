# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmaction.registry import MODELS


@MODELS.register_module()
class ASFormer(BaseModel):
    """Boundary Matching Network for temporal action proposal generation."""

    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim,
                 num_classes, channel_masking_rate, sample_rate):
        super().__init__()
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps,
                                   input_dim, num_classes,
                                   channel_masking_rate)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.num_classes = num_classes
        self.mse = MODELS.build(dict(type='MeanSquareErrorLoss'))
        self.ce = MODELS.build(dict(type='CrossEntropyLoss'))

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        pass

    def forward(self, inputs, data_samples, mode, **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[:obj:`ActionDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        input = torch.stack(inputs)
        if mode == 'tensor':
            return self._forward(inputs, **kwargs)
        if mode == 'predict':
            return self.predict(input, data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(input, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """
        device = batch_inputs.device
        batch_target_tensor = torch.ones(
            len(batch_inputs),
            max(tensor.size(1) for tensor in batch_inputs),
            dtype=torch.long) * (-100)
        mask = torch.zeros(
            len(batch_inputs),
            self.num_classes,
            max(tensor.size(1) for tensor in batch_inputs),
            dtype=torch.float)
        for i in range(len(batch_inputs)):
            batch_target_tensor[i,
                                :np.shape(batch_data_samples[i].classes)[0]] \
                = torch.from_numpy(batch_data_samples[i].classes)

            mask[i, i, :np.shape(batch_data_samples[i].classes)[0]] = \
                torch.ones(self.num_classes,
                           np.shape(batch_data_samples[i].classes)[0])

        batch_target_tensor = batch_target_tensor.to(device)
        batch_target_tensor = batch_target_tensor.to(device)
        mask = mask.to(device)
        batch_inputs = batch_inputs.to(device)
        ps = self.model(batch_inputs, mask)
        loss = 0
        for p in ps:
            loss += self.ce(
                p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                batch_target_tensor.view(-1),
                ignore_index=-100)
            loss += 0.15 * torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                    min=0,
                    max=16) * mask[:, :, 1:])

        loss_dict = dict(loss=loss)
        return loss_dict

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        """Define the computation performed at every call when testing."""
        device = batch_inputs.device
        actions_dict = batch_data_samples[0].actions_dict
        batch_target_tensor = torch.ones(
            len(batch_inputs),
            max(tensor.size(1) for tensor in batch_inputs),
            dtype=torch.long) * (-100)
        batch_target = [
            data_sample.classes for data_sample in batch_data_samples
        ]
        mask = torch.zeros(
            len(batch_inputs),
            self.num_classes,
            max(tensor.size(1) for tensor in batch_inputs),
            dtype=torch.float)
        for i in range(len(batch_inputs)):
            batch_target_tensor[i, :np.shape(batch_data_samples[i].classes
                                             )[0]] = torch.from_numpy(
                                                 batch_data_samples[i].classes)
            mask[i, :, :np.
                 shape(batch_data_samples[i].classes)[0]] = torch.ones(
                     self.num_classes,
                     np.shape(batch_data_samples[i].classes)[0])
        batch_target_tensor = batch_target_tensor.to(device)
        mask = mask.to(device)
        batch_inputs = batch_inputs.to(device)
        predictions = self.model(batch_inputs, mask)
        for i in range(len(predictions)):
            confidence, predicted = torch.max(
                F.softmax(predictions[i], dim=1).data, 1)
            confidence, predicted = confidence.squeeze(), predicted.squeeze()
            confidence, predicted = confidence.squeeze(), predicted.squeeze()
        recognition = []
        ground = [
            batch_data_samples[0].index2label[idx] for idx in batch_target[0]
        ]
        for i in range(len(predicted)):
            recognition = np.concatenate((recognition, [
                list(actions_dict.keys())[list(actions_dict.values()).index(
                    predicted[i].item())]
            ]))
        output = [dict(ground=ground, recognition=recognition)]
        return output

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        print(x.shape)

        return x.shape


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):

    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """scalar dot attention.

        :param proj_query: shape of (B, C, L) =>
            (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        """
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage,
                 att_type):  # r1 = r2 (2)
        super(AttLayer, self).__init__()
        self.query_conv = nn.Conv1d(
            in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(
            in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(
            in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        """construct window mask of shape (1, l, l + l//2 + l//2), used for
        sliding window self attention."""
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, i, i:i + self.bl] = 1
        return window_mask

    def forward(self, x1, x2, mask):
        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        device = q.device
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones(
            (m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(
            q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        device = q.device
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([
                q,
                torch.zeros(
                    (m_batchsize, c1, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            k = torch.cat([
                k,
                torch.zeros(
                    (m_batchsize, c2, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            v = torch.cat([
                v,
                torch.zeros(
                    (m_batchsize, c3, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            nb += 1

        padding_mask = torch.cat([
            torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
            torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)
        ],
                                 dim=-1)

        q = q.reshape(m_batchsize, c1, nb,
                      self.bl).permute(0, 2, 1,
                                       3).reshape(m_batchsize * nb, c1,
                                                  self.bl)
        padding_mask = padding_mask.reshape(
            m_batchsize, 1, nb,
            self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, 1, self.bl)
        k = k.reshape(m_batchsize, c2, nb,
                      self.bl).permute(0, 2, 1,
                                       3).reshape(m_batchsize * nb, c2,
                                                  self.bl)
        v = v.reshape(m_batchsize, c3, nb,
                      self.bl).permute(0, 2, 1,
                                       3).reshape(m_batchsize * nb, c3,
                                                  self.bl)

        output, attentions = self.att_helper.scalar_dot_att(
            q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(
            0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        device = q.device
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([
                q,
                torch.zeros(
                    (m_batchsize, c1, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            k = torch.cat([
                k,
                torch.zeros(
                    (m_batchsize, c2, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            v = torch.cat([
                v,
                torch.zeros(
                    (m_batchsize, c3, self.bl - L % self.bl)).to(device)
            ],
                          dim=-1)
            nb += 1
        padding_mask = torch.cat([
            torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
            torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)
        ],
                                 dim=-1)
        q = q.reshape(m_batchsize, c1, nb,
                      self.bl).permute(0, 2, 1,
                                       3).reshape(m_batchsize * nb, c1,
                                                  self.bl)
        k = torch.cat([
            torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k,
            torch.zeros(m_batchsize, c2, self.bl // 2).to(device)
        ],
                      dim=-1)
        v = torch.cat([
            torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v,
            torch.zeros(m_batchsize, c3, self.bl // 2).to(device)
        ],
                      dim=-1)
        padding_mask = torch.cat([
            torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask,
            torch.zeros(m_batchsize, 1, self.bl // 2).to(device)
        ],
                                 dim=-1)
        k = torch.cat([
            k[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2]
            for i in range(nb)
        ],
                      dim=0)  # special case when self.bl = 1
        v = torch.cat([
            v[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2]
            for i in range(nb)
        ],
                      dim=0)
        padding_mask = torch.cat([
            padding_mask[:, :, i * self.bl:(i + 1) * self.bl +
                         (self.bl // 2) * 2] for i in range(nb)
        ],
                                 dim=0)  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.to(device).repeat(
            m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(
            0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type,
                 num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList([
            copy.deepcopy(
                AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type))
            for i in range(num_head)
        ])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):

    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1), nn.ReLU(), nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1))

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):

    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type,
                 stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels,
                                            out_channels)
        self.instance_norm = nn.InstanceNorm1d(
            in_channels, track_running_stats=False)
        self.att_layer = AttLayer(
            in_channels,
            in_channels,
            out_channels,
            r1,
            r1,
            r2,
            dilation,
            att_type=att_type,
            stage=stage)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f,
                                          mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    #         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]


class Encoder(nn.Module):

    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes,
                 channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type,
                      'encoder', alpha) for i in  # 2**i
            range(num_layers)
        ])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):

    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes,
                 att_type, alpha):
        super(Decoder, self).__init__(
        )  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type,
                      'decoder', alpha) for i in  # 2 ** i
            range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class MyTransformer(nn.Module):

    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim,
                 num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type='sliding_att',
            alpha=1)
        self.decoders = nn.ModuleList([
            copy.deepcopy(
                Decoder(
                    num_layers,
                    r1,
                    r2,
                    num_f_maps,
                    num_classes,
                    num_classes,
                    att_type='sliding_att',
                    alpha=exponential_descrease(s)))
            for s in range(num_decoders)
        ])  # num_decoders

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(
                F.softmax(out, dim=1) * mask[:, 0:1, :],
                feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs
