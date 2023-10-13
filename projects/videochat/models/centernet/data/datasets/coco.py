import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.register_coco import register_coco_instances


def register_distill_coco_instances(name, metadata, json_file, image_root):
    """add extra_annotation_keys."""
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(
            json_file, image_root, name, extra_annotation_keys=['score']))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type='coco',
        **metadata)


_PREDEFINED_SPLITS_COCO = {
    'coco_2017_unlabeled':
    ('coco/unlabeled2017', 'coco/annotations/image_info_unlabeled2017.json'),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
    register_coco_instances(
        key,
        _get_builtin_metadata('coco'),
        os.path.join('datasets', json_file)
        if '://' not in json_file else json_file,
        os.path.join('datasets', image_root),
    )

_PREDEFINED_SPLITS_DISTILL_COCO = {
    'coco_un_yolov4_55_0.5':
    ('coco/unlabeled2017',
     'coco/annotations/yolov4_cocounlabeled_55_ann0.5.json'),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_DISTILL_COCO.items():
    register_distill_coco_instances(
        key,
        _get_builtin_metadata('coco'),
        os.path.join('datasets', json_file)
        if '://' not in json_file else json_file,
        os.path.join('datasets', image_root),
    )
