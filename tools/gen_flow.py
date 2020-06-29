import os
import os.path as osp

import cv2
import numpy as np


def flow_to_img(raw_flow, bound=20.):
    """Convert flow to gray image.

    Args:
        raw_flow (np.ndarray[float]): Estimated flow with the shape (w, h).
        bound (float): Bound for the flow-to-image normalization. Default: 20.

    Returns:
        np.ndarray[uint8]: The result list of np.ndarray[uint8], with shape
                        (w, h).
    """
    flow = np.clip(raw_flow, -bound, bound)
    flow += bound
    flow *= (255 / float(2 * bound))
    flow = flow.astype(np.uint8)
    return flow


def generate_flow(frames, method='tvl1'):
    """Estimate flow with given frames.

    Args:
        frames (list[np.ndarray[uint8]]): List of rgb frames, with shape
                                        (w, h, 3).
        method (str): Use which method to generate flow. Options are 'tvl1'
                    and 'farneback'. Default: 'tvl1'.

    Returns:
        list[np.ndarray[float]]: The result list of np.ndarray[float], with
                                shape (w, h, 2).
    """
    assert method in ['tvl1', 'farneback']
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    if method == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        def op(x, y):
            return tvl1.calc(x, y, None)
    elif method == 'farneback':

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
                       method='tvl1'):
    """Extract dense flow given video or frames, save them as gray-scale
    images.

    Args:
        path (str): Location of the video or frames. If use a video as input,
                    pass the location of the video. If use frames as input,
                    pass the template of frames, like '/path/{:05d}.jpg'.
        dest (str): The directory to store the extracted flow images.
        bound (float): Bound for the flow-to-image normalization. Default: 20.
        write_image (bool): Whether to save the extracted images to dest.
                            Default: False.
        start_idx (int): The starting frame index if use frames as input, the
                        first image is path.format(start_idx). Default: 0.
        method (str): Use which method to generate flow. Options are 'tvl1'
                        and 'farneback'. Default: 'tvl1'.
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
        img_name = path.format(idx)
        while osp.exists(img_name):
            frames.append(cv2.imread(img_name))
            idx += 1
            img_name = path.format(idx)

    flow = generate_flow(frames, method=method)

    flow_x = [flow_to_img(x[:, :, 0], bound) for x in flow]
    flow_y = [flow_to_img(x[:, :, 1], bound) for x in flow]

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

    num_frames = len(flow)
    for i in range(num_frames):
        cv2.imwrite(flow_x[i], flow_x_names[i])
        cv2.imwrite(flow_y[i], flow_y_names[i])

    if write_image:
        img_names = [
            osp.join(dest, 'img_{:05d}.jpg'.format(ind))
            for ind in range(len(frames))
        ]
        for frame, name in zip(frames, img_names):
            cv2.imwrite(name, frame)


if __name__ == '__main__':
    extract_dense_flow('demo/demo.mp4', 'flow_data', 20, 1, None, 'tvl1')
