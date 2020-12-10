#!/usr/bin/python
# -*- coding: UTF-8 -*-
# !H:\Anaconda3\envs\new_labelme\python.exe
import argparse
import json
import os
import os.path as osp
import base64
import warnings

import PIL.Image
import yaml

from labelme import utils

import cv2
import numpy as np
from skimage import img_as_ubyte

import labelmeutils


# from sys import argv

def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    json_file = "results/annotations/"
    images_path = "results/images/"
    save_dir = "results"

    # freedom
    #list_path = os.listdir(json_file)
    print('freedom =', json_file)
    #for i in range(0, len(list_path)):
    for root_dir, _, files in os.walk(json_file):
        #path = os.path.join(json_file, list_path[i])
        for file in files:
            path = os.path.join(root_dir, file)
            sub_dir = path.split('/')[-2]
            if os.path.isfile(path):
                print('---', path)
                try:
                    data = json.load(open(path), encoding='utf-8')
                    if data['imageData']:
                         imageData = data['imageData']
                    else:
                        #imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                        imagePath = os.path.join(images_path, sub_dir, data['imagePath'])
                        with open(imagePath, 'rb') as f:
                            imageData = f.read()
                            imageData = base64.b64encode(imageData).decode('utf-8')

                    img = utils.img_b64_to_arr(imageData)

                    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

                    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]

                    # lbl_viz = utils.draw_label(lbl, img, captions)
                    lbl_viz = labelmeutils.draw_label(lbl, img, captions)
                    out_dir = osp.basename(path).split('.json')[0]
                    save_file_name = out_dir

                    if not osp.exists(osp.join(save_dir, 'mask')):
                        os.mkdir(osp.join(save_dir,'mask'))
                    maskdir = osp.join(save_dir,'mask')

                    if not osp.exists(osp.join(save_dir, 'mask_viz')):
                        os.mkdir(osp.join(save_dir, 'mask_viz'))
                    maskvizdir = osp.join(save_dir, 'mask_viz')

                    out_dir1 = maskdir

                    PIL.Image.fromarray(lbl).save(osp.join(out_dir1, save_file_name + '.png'))

                    PIL.Image.fromarray(lbl_viz).save(osp.join(maskvizdir, save_file_name + '_label_viz.png'))

                    with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
                        for lbl_name in lbl_names:
                            f.write(lbl_name + '\n')

                    warnings.warn('info.yaml is being replaced by label_names.txt')
                    info = dict(label_names=lbl_names)
                    with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
                        yaml.safe_dump(info, f, default_flow_style=False)

                    print('Saved to: %s' % out_dir1)
                except Exception as exc:
                    print('generated an exception: %s' % (exc))
                    continue


if __name__ == '__main__':
    # base64path = argv[1]
    main()

