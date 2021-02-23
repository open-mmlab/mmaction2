import argparse
import os
import os.path as osp

import numpy as np


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2tensorrt(onnx_file,
                  trt_file,
                  input_shape,
                  verify=False,
                  workspace_size=1):
    """Create tensorrt engine from onnx model.

    Args:
        onnx_file (str): Filename of the input ONNX model file.
        trt_file (str): Filename of the output TensorRT engine file.
        input_shape (list[int]): Input shape of the model.
            eg [1, 3, 224, 224].
        verify (bool, optional): Whether to verify the converted model.
            Defaults to False.
        workspace_size (int, optional): Maximum workspace of GPU.
            Defaults to 1.
    """
    import onnx
    from mmcv.tensorrt import TRTWraper, onnx2trt, save_trt_engine

    onnx_model = onnx.load(onnx_file)
    # create trt engine and wraper
    opt_shape_dict = {'input': [input_shape, input_shape, input_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        fp16_mode=False,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        import torch
        import onnxruntime as ort

        input_tensor = torch.randn(*input_shape)
        input_tensor_cpu = input_tensor.detach().cpu().numpy()
        input_tensor_cuda = input_tensor.cuda()

        # Get results from ONNXRuntime
        session_options = ort.SessionOptions()
        sess = ort.InferenceSession(onnx_file, session_options)

        # get input and output names
        input_names = [_.name for _ in sess.get_inputs()]
        output_names = [_.name for _ in sess.get_outputs()]

        onnx_outputs = sess.run(None, {
            input_names[0]: input_tensor_cpu,
        })

        # Get results from TensorRT
        trt_model = TRTWraper(trt_file, input_names, output_names)
        with torch.no_grad():
            trt_outputs = trt_model({input_names[0]: input_tensor_cuda})
        trt_outputs = [
            trt_outputs[_].detach().cpu().numpy() for _ in output_names
        ]

        # Compare results
        np.testing.assert_allclose(
            onnx_outputs[0], trt_outputs[0], rtol=1e-05, atol=1e-05)
        print('The numerical values are the same ' +
              'between ONNXRuntime and TensorRT')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models from ONNX to TensorRT')
    parser.add_argument('model', help='Filename of the input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        default='tmp.trt',
        help='Filename of the output TensorRT engine')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 8, 224, 224],
        help='input video size')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size of GPU in GiB')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # Create TensorRT engine
    onnx2tensorrt(
        args.model,
        args.trt_file,
        args.input_shape,
        verify=args.verify,
        workspace_size=args.workspace_size)
