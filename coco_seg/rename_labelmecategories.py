# -*- coding:utf-8 -*-
"""
coco tools
Written by Sanjun
rename the original json categories id to new label_name
"""

from __future__ import print_function
import os, sys
import shutil
import numpy as np
import json
from random import shuffle
from collections import OrderedDict

#########################################################
# coco detailed label information
#{
#    "info": info, # dict
#    "licenses": [license], # list, inner is dict
#    "images": [image], # list, inner is dict
#    "annotations": [annotation], # list, inner is dict
#    "categories": # list, inner is dict
#}
#########################################################

# set the label map 'Old':'New'
label_name_map = {'OldLabel':'NewLabel'}
# save result label name
result_label_name = OrderedDict()

# get the category_name by category_id
def get_category_name(in_json_file, in_id):
    data = json.load(open(in_json_file, 'r'))
    for cat in data['shapes']:
        cat_name = str(cat['label'])
    if not cat_name.isspace():
        return cat_name
    else:
        raise RuntimeError('{} has no name'.format(cat_id))

def rename_labelme_category_json(in_json_dir, json_file, save_dir, save_ann_name):

    # annotations
    annotation=[]

    json_file_path = os.path.join(in_json_dir, json_file)
    data = json.load(open(json_file_path, 'r'))
    #print("image num:", len(data['images']))
    for ann in data['shapes']:
        org_name = str(ann['label'])
        if org_name in label_name_map.keys():
            new_name = label_name_map[org_name]
            print(org_name, new_name)
        else:
            print('label keep ', org_name)
            new_name = org_name
        ## record the results label name
        if new_name in result_label_name.keys():
            result_label_name[new_name] += 1
        else:
            result_label_name[new_name] = 1
        ann['label'] = new_name
        annotation.append(ann)
        #print('=== rename label {}:{} ---> {}:{} ==='.format(org_name, catID,
        #    new_name, new_catID))

    data['shapes']=annotation

    # save
    save_ann_file = os.path.join(save_dir, save_ann_name)
    json.dump(data, open(save_ann_file, 'w'), indent=4) # indent=4 for more elegant to show
    print('save json done:', save_ann_file)

if __name__ == '__main__':

    in_path = '/datasets/orgCOCODir'
    out_path = '/datasets/newCOCOCatIDDir'
    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    print("in_path:", in_path)
    print("out_path:", out_path)
    in_annotation_path = os.path.join(in_path, 'annotations', 'OldLabelDir')
    out_annotation_path = os.path.join(out_path, 'annotations')
    if not os.path.exists(out_annotation_path):
        os.makedirs(out_annotation_path)

    for root_dir, sub_dir, files in os.walk(in_annotation_path):
        for json_file in files:
            save_file = json_file
            print(os.path.join(root_dir, json_file))
            rename_labelme_category_json(root_dir, json_file, out_annotation_path, save_file)


