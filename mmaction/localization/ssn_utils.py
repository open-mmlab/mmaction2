import fnmatch
import glob
import os
import os.path as osp
from itertools import groupby


def load_localize_proposal_file(filename):
    lines = list(open(filename))

    # Split the proposal file into many parts which contain one video's
    # information separately.
    groups = groupby(lines, lambda x: x.startswith('#'))

    video_infos = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(video_info):
        """Template information of a video in a standard file:

            # index
            video_id
            num_frames
            fps
            num_gts
            label, start_frame, end_frame
            label, start_frame, end_frame
            ...
            num_proposals
            label, best_iou, overlap_self, start_frame, end_frame
            label, best_iou, overlap_self, start_frame, end_frame
            ...
        Example of a standard annotation file:
        .. code-block:: txt
            # 0
            video_validation_0000202
            5666
            1
            3
            8 130 185
            8 832 1136
            8 1303 1381
            5
            8 0.0620 0.0620 790 5671
            8 0.1656 0.1656 790 2619
            8 0.0833 0.0833 3945 5671
            8 0.0960 0.0960 4173 5671
            8 0.0614 0.0614 3327 5671
        """
        offset = 0
        video_id = video_info[offset]
        offset += 1

        num_frames = int(float(video_info[1]) * float(video_info[2]))
        num_gts = int(video_info[3])
        offset = 4

        gt_boxes = [x.split() for x in video_info[offset:offset + num_gts]]
        offset += num_gts
        num_proposals = int(video_info[offset])
        offset += 1
        proposal_boxes = [
            x.split() for x in video_info[offset:offset + num_proposals]
        ]

        return video_id, num_frames, gt_boxes, proposal_boxes

    return [parse_group(video_info) for video_info in video_infos]


def process_localize_proposal_list(norm_proposal_list, out_list_name,
                                   frame_dict):
    norm_proposals = load_localize_proposal_file(norm_proposal_list)

    processed_proposal_list = []
    for idx, proposal in enumerate(norm_proposals):
        video_id = proposal[0]
        frame_info = frame_dict[video_id]
        num_frames = frame_info[1]
        frame_path = osp.basename(frame_info[0])

        gts = [[
            int(x[0]),
            int(float(x[1]) * num_frames),
            int(float(x[2]) * num_frames)
        ] for x in proposal[2]]

        proposals = [[
            int(x[0]),
            float(x[1]),
            float(x[2]),
            int(float(x[3]) * num_frames),
            int(float(x[4]) * num_frames)
        ] for x in proposal[3]]

        gts_dump = '\n'.join(['{} {} {}'.format(*x) for x in gts])
        gts_dump += '\n' if len(gts) else ''
        proposals_dump = '\n'.join(
            ['{} {:.04f} {:.04f} {} {}'.format(*x) for x in proposal])
        proposals_dump += '\n' if len(proposal) else ''

        processed_proposal_list.append(
            f'# {idx}\n{frame_path}\n{num_frames}\n1'
            f'\n{len(gts)}\n{gts_dump}{len(proposals)}\n{proposals_dump}')

    open(out_list_name, 'w').writelines(processed_proposal_list)


def parse_frame_folder(path,
                       key_func=lambda x: x[-11:],
                       rgb_prefix='img_',
                       flow_x_prefix='flow_x_',
                       flow_y_prefix='flow_y_',
                       level=1):
    """Parse directories holding extracted frames from standard benchmarks."""
    print('parse frames under folder {}'.format(path))
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        frames = os.listdir(directory)
        num_list = [len(fnmatch.filter(frames, x + '*')) for x in prefix_list]
        return num_list

    # check RGB
    frame_dict = {}
    for i, frame_folder in enumerate(frame_folders):
        num_all = count_files(frame_folder,
                              (rgb_prefix, flow_x_prefix, flow_y_prefix))
        key = key_func(frame_folder)

        num_flow_x = num_all[1]
        num_flow_y = num_all[2]
        if num_flow_x != num_flow_y:
            raise ValueError('x and y direction have different number '
                             'of flow images. video: ' + frame_folder)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[key] = (frame_folder, num_all[0], num_flow_x)

    print('Frame folder analysis done')
    return frame_dict
