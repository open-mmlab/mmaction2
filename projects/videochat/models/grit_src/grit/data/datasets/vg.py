import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from lvis import LVIS

logger = logging.getLogger(__name__)

__all__ = ['load_vg_json', 'register_vg_instances']


def register_vg_instances(name, metadata, json_file, image_root):
    """"""
    DatasetCatalog.register(name,
                            lambda: load_vg_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type='vg',
        **metadata)


def get_vg_meta():
    categories = [{'supercategory': 'object', 'id': 1, 'name': 'object'}]
    vg_categories = sorted(categories, key=lambda x: x['id'])
    thing_classes = [k['name'] for k in vg_categories]
    meta = {'thing_classes': thing_classes}
    return meta


def load_vg_json(json_file, image_root, dataset_name=None):

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info('Loading {} takes {:.2f} seconds.'.format(
            json_file, timer.seconds()))

    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann['id'] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), \
        "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info('Loaded {} images in the LVIS v1 format from {}'.format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if 'file_name' in img_dict:
            file_name = img_dict['file_name']
            record['file_name'] = os.path.join(image_root, file_name)

        record['height'] = int(img_dict['height'])
        record['width'] = int(img_dict['width'])
        image_id = record['image_id'] = img_dict['id']

        objs = []
        for anno in anno_dict_list:
            assert anno['image_id'] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {'bbox': anno['bbox'], 'bbox_mode': BoxMode.XYWH_ABS}
            obj['category_id'] = 0
            obj['object_description'] = anno['caption']

            objs.append(obj)
        record['annotations'] = objs
        if len(record['annotations']) == 0:
            continue
        record['task'] = 'DenseCap'
        dataset_dicts.append(record)

    return dataset_dicts


_CUSTOM_SPLITS_LVIS = {
    'vg_train': ('vg/images', 'vg/annotations/train.json'),
    'vg_test': ('vg/images', 'vg/annotations/test.json'),
}

for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS.items():
    register_vg_instances(
        key,
        get_vg_meta(),
        os.path.join('datasets', json_file)
        if '://' not in json_file else json_file,
        os.path.join('datasets', image_root),
    )
