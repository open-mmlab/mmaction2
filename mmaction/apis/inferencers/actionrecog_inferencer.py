# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.fileio import list_from_file
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmaction.registry import INFERENCERS
from mmaction.structures import ActionDataSample
from mmaction.utils import ConfigType, get_str_type

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='action-recognition')
@INFERENCERS.register_module()
class ActionRecogInferencer(BaseInferencer):
    """The inferencer for action recognition.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb" or
            "configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py".
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        label_file (str, optional): label file for dataset.
        input_format (str): Input video format, Choices are 'video',
            'rawframes', 'array'. 'video' means input data is a video file,
            'rawframes' means input data is a video frame folder, and 'array'
            means input data is a np.ndarray. Defaults to 'video'.
        pack_cfg (dict, optional): Config for `InferencerPackInput` to load
            input. Defaults to empty dict.
        scope (str, optional): The scope of the model. Defaults to "mmaction".
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'vid_out_dir', 'draw_pred', 'fps',
        'out_type', 'target_resolution'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample'
    }

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 label_file: Optional[str] = None,
                 input_format: str = 'video',
                 pack_cfg: dict = {},
                 scope: Optional[str] = 'mmaction') -> None:
        # A global counter tracking the number of videos processed, for
        # naming of the output videos
        self.num_visualized_vids = 0
        self.input_format = input_format
        self.pack_cfg = pack_cfg.copy()
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        if label_file is not None:
            self.visualizer.dataset_meta = dict(
                classes=list_from_file(label_file))

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 vid_out_dir: str = '',
                 out_type: str = 'video',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 target_resolution: Optional[Tuple[int]] = None,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            vid_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
            out_type (str): Output type of visualization results.
                Defaults to 'video'.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        return super().__call__(
            inputs,
            return_datasamples,
            batch_size,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            vid_out_dir=vid_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            out_type=out_type,
            target_resolution=target_resolution,
            **kwargs)

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list. The main difference from mmengine
        version is that we don't list a directory cause input could be a frame
        folder.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        test_pipeline = cfg.test_dataloader.dataset.pipeline
        # Alter data pipelines for decode
        if self.input_format == 'array':
            for i in range(len(test_pipeline)):
                if 'Decode' in get_str_type(test_pipeline[i]['type']):
                    test_pipeline[i] = dict(type='ArrayDecode')
            test_pipeline = [
                x for x in test_pipeline if 'Init' not in x['type']
            ]
        elif self.input_format == 'video':
            if 'Init' not in get_str_type(test_pipeline[0]['type']):
                test_pipeline = [dict(type='DecordInit')] + test_pipeline
            else:
                test_pipeline[0] = dict(type='DecordInit')
            for i in range(len(test_pipeline)):
                if 'Decode' in get_str_type(test_pipeline[i]['type']):
                    test_pipeline[i] = dict(type='DecordDecode')
        elif self.input_format == 'rawframes':
            if 'Init' in get_str_type(test_pipeline[0]['type']):
                test_pipeline = test_pipeline[1:]
            for i in range(len(test_pipeline)):
                if 'Decode' in get_str_type(test_pipeline[i]['type']):
                    test_pipeline[i] = dict(type='RawFrameDecode')
        # Alter data pipelines to close TTA, avoid OOM
        # Use center crop instead of multiple crop
        for i in range(len(test_pipeline)):
            if get_str_type(
                    test_pipeline[i]['type']) in ['ThreeCrop', 'TenCrop']:
                test_pipeline[i]['type'] = 'CenterCrop'
        # Use single clip for `Recognizer3D`
        if cfg.model.type == 'Recognizer3D':
            for i in range(len(test_pipeline)):
                if get_str_type(test_pipeline[i]['type']) == 'SampleFrames':
                    test_pipeline[i]['num_clips'] = 1
        # Pack multiple types of input format
        test_pipeline.insert(
            0,
            dict(
                type='InferencerPackInput',
                input_format=self.input_format,
                **self.pack_cfg))

        return Compose(test_pipeline)

    def visualize(
        self,
        inputs: InputsType,
        preds: PredType,
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        draw_pred: bool = True,
        fps: int = 30,
        out_type: str = 'video',
        target_resolution: Optional[Tuple[int]] = None,
        vid_out_dir: str = '',
    ) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            fps (int): Frames per second for saving video. Defaults to 4.
            out_type (str): Output format type, choose from 'img', 'gif',
                'video'. Defaults to ``'img'``.
            target_resolution (Tuple[int], optional): Set to
                (desired_width desired_height) to have resized frames. If
                either dimension is None, the frames are resized by keeping
                the existing aspect ratio. Defaults to None.
            vid_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if self.visualizer is None or (not show and vid_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                frames = single_input
                video_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                frames = single_input.copy()
                video_num = str(self.num_visualized_vids).zfill(8)
                video_name = f'{video_num}.mp4'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_path = osp.join(vid_out_dir, video_name) if vid_out_dir != '' \
                else None

            visualization = self.visualizer.add_datasample(
                video_name,
                frames,
                pred,
                show_frames=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                fps=fps,
                out_type=out_type,
                out_path=out_path,
                target_resolution=target_resolution,
            )
            results.append(visualization)
            self.num_visualized_vids += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        pred_out_file: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasample=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred)
                results.append(result)
        # Add video to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

    def pred2dict(self, data_sample: ActionDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (ActionDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['pred_labels'] = data_sample.pred_label.tolist()
        result['pred_scores'] = data_sample.pred_score.tolist()
        return result
