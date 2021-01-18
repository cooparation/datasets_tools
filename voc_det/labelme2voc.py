# -*- coding: utf-8 -*-
#!/usr/bin/env python
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

import labelme
sys.path.append('./coco_seg/')
import labelmeutils
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file, note start with background', required=True)
    args = parser.parse_args()


    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
#        sys.exit(1)
    else:
        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
    #    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'))
    saved_path = args.output_dir
    if not os.path.exists(os.path.join(saved_path , 'ImageSets','Segmentation')):
        os.makedirs(os.path.join(saved_path , 'ImageSets','Segmentation'))
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        print(i, line)
        #### Notes
        class_id = i # starts with 0, background
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    print('class_name_to_id:',class_name_to_id)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelmeutils.label_colormap(255)

    #for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
    for label_file in glob.glob(osp.join(args.input_dir, '**/*.json')):
        print('Generating dataset from:', label_file)
        try:
            with open(label_file) as f:
                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    args.output_dir, 'JPEGImages', base + '.jpg')
    #            out_lbl_file = osp.join(
    #                args.output_dir, 'SegmentationClass', base + '.npy')
    #                args.output_dir, 'SegmentationClass', base + '.npy')
                out_png_file = osp.join(
                    args.output_dir, 'SegmentationClass', base + '.png')
                out_viz_file = osp.join(
                    args.output_dir,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

                data = json.load(f)

                #img_file = osp.join(args.input_imgs_dir, label_file.split('.json')[0]+'.jpg')
                img_file = label_file.replace('annotations', 'images')
                img_file = img_file.replace('json', 'bmp')
                img = np.asarray(PIL.Image.open(img_file))
                PIL.Image.fromarray(img).save(out_img_file)
        except:
            with open('wrongdata.txt','w') as f:
                f.write(label_file+'\n')
            print('==== Warning: this picture is wrong ====')
            continue


        lbl = labelmeutils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)

    #    np.save(out_lbl_file, lbl)

        viz = labelmeutils.draw_label(
            lbl, img, class_names, colormap=colormap)
        PIL.Image.fromarray(viz).save(out_viz_file)

    #6.split files for txt
    txtsavepath = os.path.join(saved_path , 'ImageSets','Segmentation')
    ftrainval = open(os.path.join(txtsavepath,'trainval.txt'), 'w')
    ftest = open(os.path.join(txtsavepath,'test.txt'), 'w')
    ftrain = open(os.path.join(txtsavepath,'train.txt'), 'w')
    fval = open(os.path.join(txtsavepath,'val.txt'), 'w')
    total_files = os.listdir(osp.join(args.output_dir, 'SegmentationClass'))
    total_files = [i.split("/")[-1].split(".png")[0] for i in total_files]
    #test_filepath = ""
    for file in total_files:
        ftrainval.write(file + "\n")
    #test
    #for file in os.listdir(test_filepath):
    #    ftest.write(file.split(".jpg")[0] + "\n")
    #split
    #train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
    ##train
    #for file in train_files:
    #    ftrain.write(file + "\n")
    ##val
    #for file in val_files:
    #    fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()



if __name__ == '__main__':
    main()
