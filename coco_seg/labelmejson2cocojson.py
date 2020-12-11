#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

from coco_seg import labelmeutils 

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'images'))
    os.makedirs(osp.join(args.output_dir, 'annotations'))
    print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    # coco data format
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    supercategory_name = "None" # TODO
    ## get label dict
    cat_id_map = labelmeutils.get_labelme_categories_id_map(args.input_dir, args.output_dir)
    for class_name in cat_id_map.keys():
        class_id = cat_id_map[class_name][0]
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=supercategory_name,
            id=class_id,
            name=class_name,
        ))
    print('class_name_to_id:', class_name_to_id)

    # write json and image
    label_files = glob.glob(osp.join(args.input_dir, '**/*.json'), recursive=True)
    for image_id, label_file in enumerate(label_files):
        #print('Generating dataset from:', label_file)
        print('Generating dataset --- {} / {}'.format(image_id+1, len(label_files)))
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            args.output_dir, 'images', base + '.jpg'
        )
        out_ann_file = osp.join(args.output_dir, 'annotations',base+'.json')

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file))
        PIL.Image.fromarray(img).save(out_img_file)
        #data['images'].append(dict(
        #    license=0,
        #    url=None,
        #    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
        #    height=img.shape[0],
        #    width=img.shape[1],
        #    date_captured=None,
        #    id=image_id,
        #))
        data['images'] = [dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        )]

        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            mask = labelmeutils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if label in masks:
                masks[label] = masks[label] | mask
            else:
                masks[label] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[label].append(points)

        for label, mask in masks.items():
            cls_name = label.split('-')[0]
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            #data['annotations'].append(dict(
            #    id=len(data['annotations']),
            #    image_id=image_id,
            #    category_id=cls_id,
            #    segmentation=segmentations[label],
            #    area=area,
            #    bbox=bbox,
            #    iscrowd=0,
            #))
            data['annotations'] = [dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[label],
                area=area,
                bbox=bbox,
                iscrowd=0,
            )]

        with open(out_ann_file, 'w') as f:
            json.dump(data, f)
    print('save json and image done:', args.output_dir)


if __name__ == '__main__':
    main()
# @input data_annotated directory structure:
#    --- [image.jpg, image.json, ...]
# @output data_dataset_coco directory structure:
#    --- images:[image1.jpg, image2,jpg, ...]
#    --- annotations:[image1.json, image2.json, ...]
# ./labelmejson2cocojson.py data_annotated data_dataset_coco
