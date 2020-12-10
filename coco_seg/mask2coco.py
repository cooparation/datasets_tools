#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Convert one rgb-mask into multiple binary-masks,
 then convert them into coco format json file.
'''
import os
import sys
import re
import glob
import json
import fnmatch
import datetime
import cv2
import numpy as np
from PIL import Image
#sys.path.append('/datasets/datasets_tools')
from coco_seg import pycococreatortools


def rgb2binary(label_name, save_dir):
    # convert one rgb-mask to multiple binary masks
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    lbl = cv2.imread(label_name, 1)
    h, w = lbl.shape[:2]
    leaf_dict = {}
    idx = 0
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(w):
            if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
                continue
            leaf_dict[tuple(lbl[i][j])] = idx
            mask = (lbl == lbl[i][j]).all(-1)
            # leaf = lbl * mask[..., None]      # rgb-mask with black background
            # np.repeat(mask[...,None],3,axis=2)    # 3D mask
            leaf = np.where(mask[..., None], white_mask, 0)
            mask_name = os.path.join(save_dir, lbl_id + '_leaf_' + str(idx) + '.png')
            cv2.imwrite(mask_name, leaf)
            idx += 1


def filter_for_image(root, files):
    #file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = ['*.bmp', '*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


def mask2coco(save_dir='cocoType_dir/annotations'):
    #coco_output = {
    #    "info": INFO,
    #    "licenses": LICENSES,
    #    "categories": CATEGORIES,
    #    "images": [],
    #    "annotations": []
    #}

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
        image_files = filter_for_image(root, files)
        if len(image_files) >= 1:
            sub_dir = image_files[0].split('/')[-2] ## Get The Sub Images Dir
        else:
            continue

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

        save_cats_dir = os.path.join(save_dir, sub_dir)
        if not os.path.exists(save_cats_dir):
                os.makedirs(save_cats_dir)
        with open('{}/instances_{}_train2018.json'.format(save_cats_dir, sub_dir), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
            print('save json: ', output_json_file)

    #with open('{}/instances_train2018.json'.format(save_dir), 'w') as output_json_file:
    #    json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    label_dir = './labels/masks'
    SAVE_DIR = 'binary_dir'
    label_list = glob.glob(os.path.join(label_dir, '*.png'))
    count = 0
    for label_name in label_list:
        count += 1
        rgb2binary(label_name, SAVE_DIR)
        print('to binary mask {}/{}'.format(count, len(label_list)), end = '', flush=True)

    ROOT_DIR = './shapes/train'
    IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2017")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

    SAVE_DIR = './cocoType_dir/annotations'

    INFO = {
        "description": "Defect Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2018,
        "contributor": "contributor",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    # 根据自己的需要添加种类
    CATEGORIES = [
        {
            'id': 1,
            'name': 'MT_Break',
            'supercategory': 'MT',
        }
    ]

    mask2coco(save_dir=SAVE_DIR)

