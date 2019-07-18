# -*- coding:utf-8 -*-

from __future__ import print_function
#from pycocotools.coco import COCO
#import zipfile
import os, sys
import shutil
import numpy as np
import json

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

json_file='/workspace/D2/sanjun/rawfood_coco_all/annotations_0-19_new2/405_instances_train2018.json' # Object Instance label info
#json_file='/workspace/D2/sanjun/coco/annotations/instances_train2017.json'
new_json_file = 'new_instance.json'

data=json.load(open(json_file,'r'))

data_2={}
data_2['info']=data['info']
data_2['licenses']=data['licenses']
data_2['images']=[data['images'][0]] # just get the first image
print("image num:", len(data['images']))
data_2['categories']=data['categories']
annotation=[]

# get all related object through image id
imgID=data_2['images'][0]['id']
for ann in data['annotations']:
    if ann['image_id']==imgID:
        annotation.append(ann)

data_2['annotations']=annotation

# save to new json file to check
json.dump(data_2,open(new_json_file,'w'),indent=4) # indent=4 for more elegant to show
print('save new json', new_json_file)
