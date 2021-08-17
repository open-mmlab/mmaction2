# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

tag_list = '../../../data/hvu/annotations/hvu_categories.csv'

lines = open(tag_list).readlines()
lines = [x.strip().split(',') for x in lines[1:]]
tag_categories = {}
for line in lines:
    tag, category = line
    tag_categories.setdefault(category, []).append(tag)

for k in tag_categories:
    tag_categories[k].sort()

mmcv.dump(tag_categories, 'hvu_tags.json')
