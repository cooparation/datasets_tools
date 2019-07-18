# -*- coding:utf-8 -*-
"""
coco tools
Written by Sanjun
convert to coco dir structure
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

#'./annotations/instances_val2017.json' # Object Instance label info
# person_keypoints_val2017.json  # Object Keypoint label info
# captions_val2017.json  # Image Caption label info

new_annId = int(0)
new_imgId = int(0)

# get the train and val list by read the json file in json dir
# and save the result to train_lists.txt val_lists.txt
def get_json_train_val_lists(in_json_dir, out_dir):
    ratio = 0.8
    lines = []
    for root_dir, sub_dirs, json_files in os.walk(in_json_dir):
        for json_file in json_files:
            lines.append(json_file + '\n')
    f = open(os.path.join(out_dir, 'file_list.txt'), 'w')
    f.writelines(lines)
    f.close()
    shuffle(lines)
    L = int(len(lines) * ratio)
    f = open(os.path.join(out_dir, 'train_lists.txt'), 'w')
    f.writelines(lines[:L])
    f.close()
    f = open(os.path.join(out_dir, 'val_lists.txt'), 'w')
    f.writelines(lines[L:])
    f.close()

# get the category_name by category_id
def get_name_byID(in_json_file, in_id):
    data = json.load(open(in_json_file, 'r'))
    for cat in data['categories']:
        cat_id = cat['id']
        if in_id == int(cat_id):
            cat_name =  str(cat['name'])
            break
    if not cat_name.isspace():
        return cat_name
    else:
        raise RuntimeError('{} has no name'.format(cat_id))

# get the dict[category_id name bbox_num]
def get_categories_id_map(in_json_file, out_dir):
    result_record = OrderedDict()
    data = json.load(open(in_json_file, 'r'))
    for ann in data['annotations']:
        cat_id = int(ann['category_id'])
        if cat_id in result_record.keys():
            result_record[cat_id][1] += 1
        else:
            result_record[cat_id] = [get_name_byID(in_json_file, cat_id), 1]

    result_record = OrderedDict([(k, result_record[k]) for k in sorted(result_record.keys())])
    file_name = os.path.basename(in_json_file).split('.')[0] + '_cat_map.txt'
    f = open(os.path.join(out_dir, file_name), 'w')
    for key in result_record.keys():
        f.write('{:5} {:30} {:10}\n'.format(key, result_record[key][0], result_record[key][1]))
    f.close()
    print('save key map', result_record)

# write the train or val instance json
def write_instance_json(in_json_dir, in_json_file_list, save_dir, save_ann_name):
    # images
    image = []
    # annotations
    annotation = []

    # from json lines to get json files
    # and read the json info,
    # and copy the image to corresponding dir
    f = open(in_json_file_list, 'r')
    json_lines = f.readlines()
    image_num = 0
    global new_annId
    global new_imgId
    for json_file in json_lines:
        json_file = json_file.strip()
        json_file_path = os.path.join(in_json_dir, json_file)
        if not os.path.exists(json_file_path):
            print('Warning: annotation {} not exists'.format(json_file_path))
            continue
        if not json_file.endswith('json'):
            continue
        data = json.load(open(json_file_path, 'r'))
        #print("image num:", len(data['images']))
        #print('-----', json_file_path)
        #for i in range(0, len(data['images'])):
        for imageI in data['images']:
            # get all related object through image id
            imgId = imageI['id']
            file_name = imageI['file_name']
            image_path = os.path.join(in_images_path, file_name)
            have_image = False
            for ann in data['annotations']: # one ann just has one seg-box
                if ann['image_id'] == imgId and os.path.exists(image_path):
                    ann['image_id'] = int(new_imgId)
                    ann['id'] = int(new_annId) # update ann_id
                    annotation.append(ann)
                    new_annId += 1
                    have_image = True

            if have_image:
                new_fileName = str(new_imgId) + '.jpg'
                imageI['file_name'] = new_fileName
                imageI['id'] = int(new_imgId)
                image.append(imageI)
                new_imgId += 1
                shutil.copy(image_path, os.path.join(save_dir, new_fileName))
                #shutil.copy(image_path, save_dir)
                print('copy image {}/{}'.format(image_num, len(json_lines)))
                image_num += 1
            else:
                print('Warning: image {} has no annotations'.format(image_path))
    data_2={}
    data_2['info']=data['info']
    data_2['licenses']=data['licenses']
    data_2['images']=image
    data_2['annotations']=annotation
    data_2['categories']=data['categories']

    # save to new json file to check
    json.dump(data_2, open(save_ann_name, 'w'),indent=4) # indent=4 for more elegant to show
    print('save json done:', save_ann_name)

if __name__ == '__main__':

    in_path = '/datasets/food_coco_org11'
    out_path = '/datasets/food_coco_test11'
    in_path = '/datasets/imageAndAnnotationsDir'
    out_path = '/datasets/cocoStyleDir'
    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    print('input path:', in_path)
    print('save path:', in_path)
    in_annotatios_path = os.path.join(in_path, 'annotations')
    in_images_path = os.path.join(in_path, 'images')

    out_train_path = os.path.join(out_path, 'train2018')
    out_val_path = os.path.join(out_path, 'val2018')
    out_annotation_path = os.path.join(out_path, 'annotations')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_train_path):
        os.makedirs(out_train_path)
    if not os.path.exists(out_val_path):
        os.makedirs(out_val_path)
    if not os.path.exists(out_annotation_path):
        os.makedirs(out_annotation_path)

    # get train_lists.txt and val_lists.txt and save to out_path
    get_json_train_val_lists(in_annotatios_path, out_path)

    # get train instance
    in_json_file_list = os.path.join(out_path, 'train_lists.txt')
    save_dir = out_train_path
    save_ann_name = os.path.join(out_annotation_path, 'instances_train2018.json')
    write_instance_json(in_annotatios_path, in_json_file_list, save_dir, save_ann_name)
    # get categories map txt
    get_categories_id_map(save_ann_name, out_path)

    # get val instance
    in_json_file_list = os.path.join(out_path, 'val_lists.txt')
    save_dir = out_val_path
    save_ann_name = os.path.join(out_annotation_path, 'instances_val2018.json')
    write_instance_json(in_annotatios_path, in_json_file_list, save_dir, save_ann_name)
    # get categories map txt
    get_categories_id_map(save_ann_name, out_path)
