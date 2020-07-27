# -*- coding: utf-8 -*-
'''
Convert coco json to labelme json
'''
import json
import cv2
import numpy as np
import os

label_num_dict = {}

# use the json format of labelme as a reference, because most info can be reused
def reference_labelme_json():
    ref_json_path = 'reference_labelme.json'
    data = json.load(open(ref_json_path))
    return data

def ExtractEachCocoAnnos(json_file, save_dir):
    images_annos_list = []
    data = json.load(open(json_file, 'r'))
    print("image num:", len(data['images']))
    for i in range(len(data['images'])):
        data_2={}
        data_2['info']=data['info']
        data_2['licenses']=data['licenses']
        data_2['images'] = [data['images'][i]]
        data_2['categories'] = data['categories']

        annotation = []
        # get all related object through image id
        imgID = data_2['images'][0]['id']
        for ann in data['annotations']:
            if ann['image_id'] == imgID:
                annotation.append(ann)

        data_2['annotations']=annotation
        images_annos_list.append(data_2)
        image_file_name = data_2['images'][0]['file_name']
        json_file_name = image_file_name.split('.')[-2] + '.json'
        new_json_file = os.path.join(save_dir, json_file_name)
        print('save ', new_json_file)
        json.dump(data_2, open(new_json_file,'w'),indent=4)
        #image_file_path =
    return images_annos_list

# get single image shapes from single image coco 'annotations'
def labelme_shapes(data, data_ref):
    shapes = []
    # get each segmentation info
    for ann in data['annotations']:
        # every category just has one segmentation list
        class_name = [i['name'] for i in data['categories'] if i['id'] == ann['category_id']]
        assert(len(class_name) == 1), 'error {}'.format(class_name)

        # ~ print(ann['segmentation'])
        if not type(ann['segmentation']) == list:
            continue
        else:
            if class_name[0] not in label_num_dict.keys():
                label_num_dict[class_name[0]] = 1
            else:
                label_num_dict[class_name[0]] += 1

            shape = {}
            #shape['label'] = class_name[0] + '_' + str(label_num_dict[class_name[0]])
            shape['label'] = class_name[0]
            #shape['line_color'] = data_ref['shapes'][0]['line_color']
            #shape['fill_color'] = data_ref['shapes'][0]['fill_color']
            # every segmentation list may have more than one segmentation-points
            for i in range(len(ann['segmentation'])):
                shape['points'] = []
                x = ann['segmentation'][i][::2]  # the odd one is x
                y = ann['segmentation'][i][1::2] # the even one is y
                for j in range(len(x)):
                    shape['points'].append([x[j], y[j]])

                shape['shape_type'] =  data_ref['shapes'][0]['shape_type']
                shape['flags'] = data_ref['shapes'][0]['flags']
                shapes.append(shape)
    return shapes


def Coco2labelme(json_path, save_dir, data_ref):
    with open(json_path,'r') as fp:
        data = json.load(fp)  # load coco json
        data_labelme={}
        data_labelme['version'] = data_ref['version']
        data_labelme['flags'] = data_ref['flags']

        data_labelme['shapes'] = labelme_shapes(data, data_ref)

        #data_labelme['lineColor'] = data_ref['lineColor']
        #data_labelme['fillColor'] = data_ref['fillColor']
        data_labelme['imagePath'] = data['images'][0]['file_name']

        data_labelme['imageData'] = None # TODO

        data_labelme['imageHeight'] = data['images'][0]['height']
        data_labelme['imageWidth'] = data['images'][0]['width']

        # save json file
        file_name = data_labelme['imagePath']
        new_json_file = os.path.join(save_dir, file_name.split('.')[0]+'.json')
        json.dump(data_labelme, open(new_json_file, 'w'),indent=4)

if __name__ == '__main__':

    coco_jsons_dir = './cocoType_dir/annotations'

    ### 1. extract every single json from coco_json ###
    save_extracted_dir = 'coco_extracted/annotations'
    if not os.path.exists(save_extracted_dir):
        os.makedirs(save_extracted_dir)
    for root_dir, sub_dir, file_names in os.walk(coco_jsons_dir):
        for json_file in file_names:
            if json_file.split('.')[-1] == 'json':
                json_path = os.path.join(root_dir, json_file)
                print('load：', json_path)
                ExtractEachCocoAnnos(json_path, save_extracted_dir)

    data_ref = reference_labelme_json()
    ### 2. convert coco json to labelme json ###
    save_labelme_dir = './labelmeType_dir/annotations'
    if not os.path.exists(save_labelme_dir):
        os.makedirs(save_labelme_dir)
    for root_dir, sub_dir, file_names in os.walk(save_extracted_dir):
        for json_file in file_names:
            if json_file.split('.')[-1] == 'json':
                json_path = os.path.join(root_dir, json_file)
                print('load：', json_path)
                Coco2labelme(json_path, save_labelme_dir, data_ref)
    print(label_num_dict)
