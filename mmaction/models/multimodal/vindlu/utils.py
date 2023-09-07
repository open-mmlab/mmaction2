# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.dist as dist
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from scipy import interpolate


def all_gather_concat(data: torch.Tensor) -> torch.Tensor:
    """Gather tensors with different first-dimension size and concat to one
    tenosr.

    Note:
        Only the first dimension should be different.

    Args:
        data (Tensor): Tensor to be gathered.

    Returns:
        torch.Tensor: The concatenated tenosr.
    """
    if dist.get_world_size() == 1:
        return data

    data_size = torch.tensor(data.size(0), device=data.device)
    sizes_list = dist.all_gather(data_size)

    total_length = sum(sizes_list)
    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = dist.all_gather(data)

    # gather all data according to the default DDP sampler. For instance,
    # 8 samples on 2 GPUs, GPU0: [0,2,4,6], GPU1: [1,3,5,7], will be gathered
    # as [0,1,2,3,4,5,6,7]
    all_data = []
    for gather_batch in zip(*gather_list):
        all_data.extend(gather_batch)

    return torch.stack(all_data)[:total_length]


def interpolate_pos_embed_beit(state_dict, new_model):
    """interpolate the positional embeddings. The spatial pe is relative and
    temporal pe is absolute. additional temporal pe is padded with 0.

    Args:
        state_dict (dict): The state_dict.
        new_model (nn.Module): The created model.

    Returns: dict. The state_dict with updated positional embeddings.
    """
    state_dict = interpolate_pos_relative_bias_beit(
        state_dict_old=state_dict,
        state_dict_new=new_model.state_dict(),
        patch_shape_new=new_model.vision_encoder.embeddings.patch_embeddings.
        patch_shape,
    )
    # absolute temporal pos bias
    temporal_pe_key = 'vision_encoder.embeddings.temporal_position_embeddings'
    if temporal_pe_key in state_dict:
        logger = MMLogger.get_current_instance()
        logger.info(
            f'interpolate temporal positional embeddings: {temporal_pe_key}')
        state_dict[temporal_pe_key] = load_temp_embed_with_mismatch(
            temp_embed_old=state_dict[temporal_pe_key],
            temp_embed_new=new_model.state_dict()[temporal_pe_key],
        )
    return state_dict


def load_temp_embed_with_mismatch(temp_embed_old,
                                  temp_embed_new,
                                  add_zero=True):
    """Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger = MMLogger.get_current_instance()
    logger.info(
        f'Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}')
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[:, :num_frms_old] \
                = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(
                temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new


def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1)  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new,
        mode='linear')  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2)  # (1, num_frames_new, 1, d)
    return temp_embed_new


def interpolate_pos_relative_bias_beit(state_dict_old, state_dict_new,
                                       patch_shape_new):
    """
    Args:
        state_dict_old: loaded state dict
        state_dict_new: state dict for model with new image size
        patch_shape_new: new model patch_shape
    ref: https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py  # noqa: E501
    """
    all_keys = list(state_dict_old.keys())
    for key in all_keys:
        if 'relative_position_index' in key:
            state_dict_old.pop(key)

        if 'relative_position_bias_table' in key:
            rel_pos_bias = state_dict_old[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = state_dict_new[key].size()
            dst_patch_shape = patch_shape_new
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens)**0.5)
            dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
            if src_size != dst_size:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q**(i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size,
                                                src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(
                            rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens),
                                             dim=0)
                state_dict_old[key] = new_rel_pos_bias
    return state_dict_old
