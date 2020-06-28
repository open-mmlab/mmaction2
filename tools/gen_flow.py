import os
import os.path as osp

import cv2
import numpy as np


def FlowToImg(raw_flow, bound=20.):
    """Convert flow to gray image

    Args:
        raw_flow (np.array float): estimated flow, the shape is [w, h]
        bound (float): bound for the flow-to-image normalization

    return:
        flow (np.array uint8): normalized flow
    """
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255 / float(2 * bound))
    flow = flow.astype(np.uint8)
    return flow


def gen_flow(frames, algo='tvl1'):
    """Estimate flow given frames

    Args:
        frames (list): list of rgb frames
        algo (str): which algorithm to use, in ['tvl1', 'farneback']

    return:
        flow (list): list of flow
    """
    assert algo in ['tvl1', 'farneback']
    gray_frames = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in frames]

    if algo == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        def op(x, y):
            return tvl1.calc(x, y, None)
    elif algo == 'farneback':

        def op(x, y):
            return cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5,
                                                1.2, 0)

    gray_st = gray_frames[:-1]
    gray_ed = gray_frames[1:]

    flow = [op(x, y) for x, y in zip(gray_st, gray_ed)]
    return flow


def extract_dense_flow(path,
                       dest,
                       bound=20.,
                       write_image=False,
                       start_idx=0,
                       algo='tvl1'):
    """Extract dense flow given video or frames, save them as gray-scale images

    Args:
        path (str): location of the video or frames. If use a video as input,
                    pass the location of the video. If use frames as input,
                    pass the template of frames, like '/path/{:05d}.jpg'.
        dest (str): the directory to store the extracted flow images
        bound (float): bound for the flow-to-image normalization
        write_image (bool): whether to save the extracted images to dest
        start_idx (int): denotes the starting index if use frames as input, the
                         the first image is path.format(start_idx)
        algo (str): which algorithm to use, in ['tvl1', 'farneback']
    """

    if osp.exists(path):
        frames = []
        vid = cv2.VideoCapture(path)
        flag, f = vid.read()
        while flag:
            frames.append(f)
            flag, f = vid.read()
    else:
        idx = start_idx
        im_name = path.format(idx)
        while osp.exists(im_name):
            frames.append(cv2.imread(im_name))
            idx += 1
            im_name = path.format(im_name)

    flow = gen_flow(frames, algo=algo)

    flow_x = [FlowToImg(x[:, :, 0], bound) for x in flow]
    flow_y = [FlowToImg(x[:, :, 1], bound) for x in flow]

    if not osp.exists(dest):
        os.system('mkdir -p ' + dest)
    flow_x_names = [
        osp.join(dest, 'x_{:05d}.jpg'.format(ind))
        for ind in range(len(flow_x))
    ]
    flow_y_names = [
        osp.join(dest, 'y_{:05d}.jpg'.format(ind))
        for ind in range(len(flow_y))
    ]

    for imx, namex in zip(flow_x, flow_x_names):
        cv2.imwrite(namex, imx)
    for imy, namey in zip(flow_y, flow_y_names):
        cv2.imwrite(namey, imy)
    if write_image:
        im_names = [
            osp.join(dest, 'img_{:05d}.jpg'.format(ind))
            for ind in range(len(frames))
        ]
        for im, name in zip(frames, im_names):
            cv2.imwrite(name, im)


if __name__ == '__main__':
    extract_dense_flow('demo/demo.mp4', 'flow_data', 20, 1, None, 'tvl1')
