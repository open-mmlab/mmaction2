import os

from detectron2.data.datasets.register_coco import register_coco_instances

categories = [
    {
        'id': 0,
        'name': 'car'
    },
    {
        'id': 1,
        'name': 'truck'
    },
    {
        'id': 2,
        'name': 'trailer'
    },
    {
        'id': 3,
        'name': 'bus'
    },
    {
        'id': 4,
        'name': 'construction_vehicle'
    },
    {
        'id': 5,
        'name': 'bicycle'
    },
    {
        'id': 6,
        'name': 'motorcycle'
    },
    {
        'id': 7,
        'name': 'pedestrian'
    },
    {
        'id': 8,
        'name': 'traffic_cone'
    },
    {
        'id': 9,
        'name': 'barrier'
    },
]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        'thing_dataset_id_to_contiguous_id': thing_dataset_id_to_contiguous_id,
        'thing_classes': thing_classes
    }


_PREDEFINED_SPLITS = {
    'nuimages_train':
    ('nuimages', 'nuimages/annotations/nuimages_v1.0-train.json'),
    'nuimages_val':
    ('nuimages', 'nuimages/annotations/nuimages_v1.0-val.json'),
    'nuimages_mini':
    ('nuimages', 'nuimages/annotations/nuimages_v1.0-mini.json'),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join('datasets', json_file)
        if '://' not in json_file else json_file,
        os.path.join('datasets', image_root),
    )
