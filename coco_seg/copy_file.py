# -*- coding:utf-8 -*-
"""
coco tools
Written by Sanjun
copy jsons and images to dir:[annotations, images]
"""

from __future__ import print_function
#from pycocotools.coco import COCO
#import zipfile
import os, sys
import shutil
import numpy as np
import json
from random import shuffle
from collections import OrderedDict

#########################################################
# coco detailed label information
#{
#    {
#    root_dir: [root_dir] # the root_dir
#    ---- images: [image], # images dir
#    ---- annotations: [annotation], # annotations dir
#    }
#}
#########################################################

def copy_files(in_root_dir, out_image_dir, out_json_dir):
    file_num = 0
    for root_dir, sub_dirs, files in os.walk(in_root_dir):
        for each_file in files:
            if each_file.split('.')[-1] in ['json', 'jpg']:
                file_path = os.path.join(root_dir, each_file)
                if each_file.split('.')[-1] == 'json':
                    out_dir = out_json_dir
                elif each_file.split('.')[-1] == 'jpg':
                    out_dir = out_image_dir
                    file_num += 1
                shutil.copy(file_path, out_dir)
                print('copy {} to {}'.format(file_path, out_dir))

if __name__ == '__main__':

    in_path = '/datasets/imageAndjsonDir'
    out_path = '/datasets/coco_org'
    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    print('in_path', in_path)
    print('out_path', out_path)
    out_images_path = os.path.join(out_path, 'images')
    out_annotations_path = os.path.join(out_path, 'annotations')
    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)
    if not os.path.exists(out_annotations_path):
        os.makedirs(out_annotations_path)

    copy_files(in_path, out_images_path, out_annotations_path)
