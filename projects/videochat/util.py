import numpy as np
from decord import VideoReader, cpu


def loadvideo_decord(sample,
                     sample_rate_scale=1,
                     new_width=384,
                     new_height=384,
                     clip_len=8,
                     frame_sample_rate=2,
                     num_segment=1):
    fname = sample
    vr = VideoReader(
        fname, width=new_width, height=new_height, num_threads=1, ctx=cpu(0))
    # handle temporal segments
    # converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) // num_segment
    duration = max(len(vr) // vr.get_avg_fps(), 8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i * seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


def loadvideo_decord_origin(sample,
                            sample_rate_scale=1,
                            new_width=384,
                            new_height=384,
                            clip_len=8,
                            frame_sample_rate=2,
                            num_segment=1):
    fname = sample
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    # handle temporal segments
    # converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) // num_segment
    duration = max(len(vr) // vr.get_avg_fps(), 8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i * seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


def loadvideo_decord_time_segment(sample,
                                  segment_length=10,
                                  sample_rate_scale=1,
                                  new_width=384,
                                  new_height=384,
                                  frame_sample_rate=2):
    fname = sample
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    total_duration = len(vr) // vr.get_avg_fps()  # 计算视频的总时长（秒）
    num_segment = int(total_duration // segment_length)  # 计算可以分割的段数
    remaining_time = total_duration % segment_length  # 计算剩余的时间（秒）

    all_index = []
    buffer = []
    time_index = []
    for i in range(num_segment):
        start_time = i * segment_length
        end_time = start_time + segment_length
        index = np.arange(
            start_time * vr.get_avg_fps(),
            end_time * vr.get_avg_fps(),
            step=int(sample_rate_scale))
        all_index.append(list(index))
        vr.seek(0)
        buffer.append(vr.get_batch(list(index)).asnumpy())
        time_index.append(i * segment_length)
    time_index.append(i * segment_length)

    if remaining_time > 0:  # 如果有剩余的时间，则将其作为一个单独的片段处理
        start_time = num_segment * segment_length
        index = np.arange(
            start_time * vr.get_avg_fps(),
            len(vr),
            step=int(sample_rate_scale))
        all_index.append(list(index))
        vr.seek(0)
        buffer.append(vr.get_batch(list(index)).asnumpy())

    return buffer, time_index
