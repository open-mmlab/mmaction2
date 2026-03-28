import itertools
import json
import os

from detectron2.evaluation.coco_evaluation import (
    COCOEvaluator, _evaluate_predictions_on_coco)
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


class GRiTCOCOEvaluator(COCOEvaluator):

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {'image_id': input['image_id']}

            if 'instances' in output:
                instances = output['instances'].to(self._cpu_device)
                prediction['instances'] = instances_to_coco_json(
                    instances, input['image_id'])

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        self._logger.info('Preparing results for COCO format ...')
        coco_results = list(
            itertools.chain(*[x['instances'] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if self._output_dir:
            file_path = os.path.join(self._output_dir,
                                     'coco_instances_results.json')
            self._logger.info('Saving results to {}'.format(file_path))
            with PathManager.open(file_path, 'w') as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info('Annotations are not available for evaluation.')
            return

        self._logger.info('Evaluating predictions with {} COCO API...'.format(
            'unofficial' if self._use_fast_impl else 'official'))

        coco_results = self.convert_classname_to_id(coco_results)

        for task in sorted(tasks):
            assert task in {'bbox', 'segm',
                            'keypoints'}, f'Got unknown task: {task}!'
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                ) if len(coco_results) > 0 else
                None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval,
                task,
                class_names=self._metadata.get('thing_classes'))
            self._results[task] = res

    def convert_classname_to_id(self, results):
        outputs = []
        class_name_to_id = {}
        categories = sorted(
            self._coco_api.dataset['categories'], key=lambda x: x['id'])

        for cat in categories:
            class_name_to_id[cat['name']] = cat['id']

        for pred in results:
            if pred['object_descriptions'] in class_name_to_id:
                pred['category_id'] = class_name_to_id[
                    pred['object_descriptions']]
                del pred['object_descriptions']
                outputs.append(pred)

        return outputs


class GRiTVGEvaluator(COCOEvaluator):

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            assert input['image_id'] == int(
                input['file_name'].split('/')[-1].split('.')[0])
            prediction = {'image_id': input['image_id']}

            if 'instances' in output:
                instances = output['instances'].to(self._cpu_device)
                prediction['instances'] = instances_to_coco_json(
                    instances, input['image_id'], output_logits=True)
                h = input['height']
                w = input['width']
                scale = 720.0 / max(h, w)
                scaled_inst = []
                for inst in prediction['instances']:
                    inst['bbox'][0] = inst['bbox'][0] * scale
                    inst['bbox'][1] = inst['bbox'][1] * scale
                    inst['bbox'][2] = inst['bbox'][2] * scale
                    inst['bbox'][3] = inst['bbox'][3] * scale
                    scaled_inst.append(inst)
                if len(scaled_inst) > 0:
                    prediction['instances'] = scaled_inst
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        """This is only for saving the results to json file."""
        self._logger.info('Preparing results for COCO format ...')
        coco_results = list(
            itertools.chain(*[x['instances'] for x in predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir,
                                     'vg_instances_results.json')
            self._logger.info('Saving results to {}'.format(file_path))
            with PathManager.open(file_path, 'w') as f:
                f.write(json.dumps(coco_results))
                f.flush()


def instances_to_coco_json(instances, img_id, output_logits=False):
    """Add object_descriptions and logit (if applicable) to detectron2's
    instances_to_coco_json."""
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    object_descriptions = instances.pred_object_descriptions.data
    if output_logits:
        logits = instances.logits.tolist()

    results = []
    for k in range(num_instance):
        result = {
            'image_id': img_id,
            'category_id': classes[k],
            'bbox': boxes[k],
            'score': scores[k],
            'object_descriptions': object_descriptions[k],
        }
        if output_logits:
            result['logit'] = logits[k]

        results.append(result)
    return results
