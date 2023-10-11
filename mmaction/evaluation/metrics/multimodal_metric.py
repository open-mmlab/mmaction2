# Copyright (c) OpenMMLab. All rights reserved.
# Copied from mmpretrain
# Partly adopted from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.utils import is_seq_of

from mmaction.registry import METRICS
from mmaction.structures.action_data_sample import format_label
from .acc_metric import to_tensor


def _process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


@METRICS.register_module()
class VQAAcc(BaseMetric):
    '''VQA Acc metric.
    Args:

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    '''
    default_prefix = 'VQA'

    def __init__(self,
                 full_score_weight: float = 0.3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.full_score_weight = full_score_weight

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            gt_answer = sample.get('gt_answer')
            gt_answer_weight = sample.get('gt_answer_weight')
            if isinstance(gt_answer, str):
                gt_answer = [gt_answer]
            if gt_answer_weight is None:
                gt_answer_weight = [1. / (len(gt_answer))] * len(gt_answer)

            result = {
                'pred_answer': sample.get('pred_answer'),
                'gt_answer': gt_answer,
                'gt_answer_weight': gt_answer_weight,
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        acc = []
        for result in results:
            pred_answer = self._process_answer(result['pred_answer'])
            gt_answer = [
                self._process_answer(answer) for answer in result['gt_answer']
            ]
            answer_weight = result['gt_answer_weight']

            weight_sum = 0
            for i, gt in enumerate(gt_answer):
                if gt == pred_answer:
                    weight_sum += answer_weight[i]
            vqa_acc = min(1.0, weight_sum / self.full_score_weight)
            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100

        metrics = {'acc': accuracy}
        return metrics

    def _process_answer(self, answer):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = _process_punctuation(answer)
        answer = _process_digit_article(answer)
        return answer


@METRICS.register_module()
class ReportVQA(BaseMetric):
    """Dump VQA result to the standard json format for VQA evaluation.

    Args:
        file_path (str): The file path to save the result file.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """
    default_prefix = 'VQA'

    def __init__(self,
                 file_path: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not file_path.endswith('.json'):
            raise ValueError('The output file must be a json file.')
        self.file_path = file_path

    def process(self, data_batch, data_samples) -> None:
        """transfer tensors in predictions to CPU."""
        for sample in data_samples:
            question_id = sample['question_id']
            pred_answer = sample['pred_answer']

            result = {
                'question_id': int(question_id),
                'answer': pred_answer,
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Dump the result to json file."""
        mmengine.dump(results, self.file_path)
        logger = MMLogger.get_current_instance()
        logger.info(f'Results has been saved to {self.file_path}.')
        return {}


@METRICS.register_module()
class VQAMCACC(BaseMetric):
    '''VQA multiple choice Acc metric.
    Args:

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    '''
    default_prefix = 'VQAMC'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            # gt_labels in datasample is a LabelData
            label = sample['gt_label'].item()
            result = {
                'pred_label': sample.get('pred_label'),
                'gt_label': label,
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        preds = np.array([x['pred_label'] for x in results])
        labels = np.array([x['gt_label'] for x in results])

        accuracy = np.sum(preds == labels) / len(preds) * 100

        metrics = {'acc': accuracy}
        return metrics


@METRICS.register_module()
class RetrievalRecall(BaseMetric):
    r"""Recall evaluation metric for image retrieval.

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k recall will
            be calculated and outputted together. Defaults to 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    """
    default_prefix: Optional[str] = 'retrieval'

    def __init__(self,
                 topk: Union[int, Sequence[int]],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        topk = (topk, ) if isinstance(topk, int) else topk

        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        self.topk = topk
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_score = data_sample['pred_score'].cpu()
            gt_label = format_label(data_sample['gt_label'])

            if 'gt_score' in data_sample:
                target = data_sample.get('gt_score').clone()
            else:
                num_classes = pred_score.size()[-1]
                target = F.one_hot(gt_label, num_classes)

            # Because the retrieval output logit vector will be much larger
            # compared to the normal classification, to save resources, the
            # evaluation results are computed each batch here and then reduce
            #  all results at the end.
            result = RetrievalRecall.calculate(
                pred_score.unsqueeze(0), target.unsqueeze(0), topk=self.topk)
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        result_metrics = dict()
        for i, k in enumerate(self.topk):
            recall_at_k = sum([r[i].item() for r in results]) / len(results)
            result_metrics[f'Recall@{k}'] = recall_at_k

        return result_metrics

    @staticmethod
    def calculate(pred: Union[np.ndarray, torch.Tensor],
                  target: Union[np.ndarray, torch.Tensor],
                  topk: Union[int, Sequence[int]],
                  pred_indices: (bool) = False,
                  target_indices: (bool) = False) -> float:
        """Calculate the average recall.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            topk (int, Sequence[int]): Predictions with the k-th highest
                scores are considered as positive.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. Defaults to False.

        Returns:
            List[float]: the average recalls.
        """
        topk = (topk, ) if isinstance(topk, int) else topk
        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        max_keep = max(topk)
        pred = _format_pred(pred, max_keep, pred_indices)
        target = _format_target(target, target_indices)

        assert len(pred) == len(target), (
            f'Length of `pred`({len(pred)}) and `target` ({len(target)}) '
            f'must be the same.')

        num_samples = len(pred)
        results = []
        for k in topk:
            recalls = torch.zeros(num_samples)
            for i, (sample_pred,
                    sample_target) in enumerate(zip(pred, target)):
                sample_pred = np.array(to_tensor(sample_pred).cpu())
                sample_target = np.array(to_tensor(sample_target).cpu())
                recalls[i] = int(np.in1d(sample_pred[:k], sample_target).max())
            results.append(recalls.mean() * 100)
        return results


def _format_pred(label, topk=None, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`pred` must be Sequence of indices when' \
                f' `pred_indices` set to True, but get {type(label)}'
        for i, sample_pred in enumerate(label):
            assert is_seq_of(sample_pred, int) or isinstance(
                sample_pred, (np.ndarray, torch.Tensor)), \
                '`pred` should be Sequence of indices when `pred_indices`' \
                f'set to True. but pred[{i}] is {sample_pred}'
            if topk:
                label[i] = sample_pred[:min(topk, len(sample_pred))]
        return label
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')
    topk = topk if topk else label.size()[-1]
    _, indices = label.topk(topk)
    return indices


def _format_target(label, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`target` must be Sequence of indices when' \
                f' `target_indices` set to True, but get {type(label)}'
        for i, sample_gt in enumerate(label):
            assert is_seq_of(sample_gt, int) or isinstance(
                sample_gt, (np.ndarray, torch.Tensor)), \
                '`target` should be Sequence of indices when ' \
                f'`target_indices` set to True. but target[{i}] is {sample_gt}'
        return label

    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif isinstance(label, Sequence) and not mmengine.is_str(label):
        label = torch.tensor(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')

    indices = [sample_gt.nonzero().squeeze(-1) for sample_gt in label]
    return indices
