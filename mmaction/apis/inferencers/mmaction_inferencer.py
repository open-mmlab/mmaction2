# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.structures import InstanceData

from .actionrecog_inferencer import ActionRecogInferencer
from .base_mmaction_inferencer import BaseMMAction2Inferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class MMAction2Inferencer(BaseMMAction2Inferencer):
    """MMAction2 Inferencer. It's a wrapper around base task inferenecers:
    ActionRecog, and it can be used to perform end-to-end action recognition
    inference.

    Args:
        rec (str, optional): Pretrained action recognition
            algorithm. It's the path to the config file or the model name
            defined in metafile. Defaults to None.
        rec_weights (str, optional): Path to the custom checkpoint file of
            the selected rec model. If it is not specified and "rec" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        label_file (str, optional): label file for dataset.
        input_format (str): Input video format, Choices are 'video',
            'rawframes', 'array'. 'video' means input data is a video file,
            'rawframes' means input data is a video frame folder, and 'array'
            means input data is a np.ndarray. Defaults to 'video'.
    """

    def __init__(self,
                 rec: Optional[str] = None,
                 rec_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 label_file: Optional[str] = None,
                 input_format: str = 'video') -> None:

        if rec is None:
            raise ValueError('rec algorithm should provided.')

        self.visualizer = None
        self.num_visualized_imgs = 0

        if rec is not None:
            self.actionrecog_inferencer = ActionRecogInferencer(
                rec, rec_weights, device, label_file, input_format)
            self.mode = 'rec'

    def forward(self, inputs: InputType, batch_size: int,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
            batch_size (int): Batch size. Defaults to 1.

        Returns:
            Dict: The prediction results. Possibly with keys "rec".
        """
        result = {}
        if self.mode == 'rec':
            predictions = self.actionrecog_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            result['rec'] = [[p] for p in predictions]

        return result

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
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
        """

        if 'rec' in self.mode:
            return self.actionrecog_inferencer.visualize(
                inputs, preds['rec'][0], **kwargs)

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer. It can be a path
                to image / image directory, or an array, or a list of these.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        preds = self.forward(ori_inputs, batch_size, **forward_kwargs)

        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(preds, visualization, **postprocess_kwargs)
        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    pred_out_file: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (Dict): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            print_result (bool): Whether to print the result.
                Defaults to False.
            pred_out_file (str): Output file name to store predictions
                without images. Supported file formats are “json”, “yaml/yml”
                and “pickle/pkl”. Defaults to ''.

        Returns:
            Dict or List[Dict]: Each dict contains the inference result of
            each image. Possible keys are "rec_labels", "rec_scores"
        """

        result_dict = {}
        pred_results = [{} for _ in range(len(next(iter(preds.values()))))]
        if 'rec' in self.mode:
            for i, rec_pred in enumerate(preds['rec']):
                result = dict(rec_labels=[], rec_scores=[])
                for rec_pred_instance in rec_pred:
                    rec_dict_res = self.actionrecog_inferencer.pred2dict(
                        rec_pred_instance)
                    result['rec_labels'].append(rec_dict_res['pred_labels'])
                    result['rec_scores'].append(rec_dict_res['pred_scores'])
                pred_results[i].update(result)

        result_dict['predictions'] = pred_results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict
