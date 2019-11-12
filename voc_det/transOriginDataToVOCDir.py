# coding=utf-8
#!/usr/bin/env python

# Function: this python is try to convert the origin data dir labeled with
#           labelImg to VOC dir formart with "JPEGImages Annotations"
import os
import sys
import cv2
from random import shuffle
from collections import OrderedDict
import shutil

import xml.etree.cElementTree as ET # parse xml

file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']
#file_type_list = ['jpg']

##############################
# Brief: copy a lists of origin classes images and xmls to dst_data_dir
# note: the origin images and xmls are in the same dir
#       JPEGImages and Annotations are contained in dst_data_dir
def copyOriginDataToVOCFormatDirAndRename(src_data_dir, dst_data_dir):
    # paths_src = [src_data_dir + '/JPEGImages', src_data_dir + '/Annotations']
    paths_dst = [dst_data_dir + '/JPEGImages', dst_data_dir + '/Annotations']
    if not os.path.exists(paths_dst[0]):
        os.makedirs(paths_dst[0])
    if not os.path.exists(paths_dst[1]):
        os.makedirs(paths_dst[1])

    types = set()
    write_lines = []
    num_images = 0
    for dir_info in os.walk(src_data_dir):
        root_dir, sub_dirs, file_names = dir_info
        for each in file_names:
            image_type = each.split('.')[1]
            image_name = each.split('.')[0]
            # image and xml are the same dir
            image_file_path = os.path.join(root_dir, each)
            xml_file_path = os.path.join(root_dir, image_name + '.xml')
            new_name_image = image_file_path.split(src_data_dir)[1].replace('/', '_')
            new_name_image = new_name_image.replace('(', '_')
            new_name_image = new_name_image.replace(')', '_')
            new_name_image = new_name_image.replace(' ', '_')
            new_name_xml = xml_file_path.split(src_data_dir)[1].replace('/', '_')
            new_name_xml = new_name_xml.replace('(', '_')
            new_name_xml = new_name_xml.replace(')', '_')
            new_name_xml = new_name_xml.replace(' ', '_')
            if image_type not in file_type_list:
                print 'ignore ', image_type
                continue
            num_images += 1

            # copy images and xml
            shutil.copy(image_file_path, paths_dst[0] + '/')
            shutil.copy(xml_file_path, paths_dst[1] + '/')
            print 'copy', num_images, image_file_path, 'to', paths_dst[0]
            print 'copy', num_images, xml_file_path, 'to', paths_dst[1]
            #newname = '{:0>6}'.format(total_num) +'.jpg'
            # rename images and xml
            os.rename(os.path.join(paths_dst[0], each),
                    os.path.join(paths_dst[0], new_name_image))
            os.rename(os.path.join(paths_dst[1], image_name + '.xml'),
                    os.path.join(paths_dst[1], new_name_xml))
            print 'rename', num_images, os.path.join(paths_dst[0], each),\
                    'to', os.path.join(paths_dst[0], new_name_image)
            print 'rename', num_images, os.path.join(paths_dst[1], image_name + '.xml'), \
                    'to', os.path.join(paths_dst[1], new_name_xml)

            write_lines.append(paths_dst[0]+'/'+new_name_image + ' ' + \
                           paths_dst[1]+'/'+new_name_xml + '\n')
    print 'have num_images:', num_images

    ###  write full path of image_list_path and conresponding xml_file_path
    ### to dst_data_dir/ImageSets/Main/
    shuffle(write_lines)

    L  = int(len(write_lines)*0.1)

    #f = open('./data/test.txt','w')
    #f.writelines(write_lines[:L])
    #f.close()

    #f = open( './data/trainval.txt','w')
    #f.writelines(write_lines[L:])
    #f.close()

    image_lists = []
    for line in range(0, len(write_lines)):
        image_path = write_lines[line].split()[0]
        image_name = image_path.split('/')[-1]
        if image_name.split('.')[-1] in file_type_list:
            image_name = image_name.split('.')[0] + '\n'
            image_lists.append(image_name)
        else:
            print 'image error', image_path
            sys.exit(2)

    image_name_files_dir = os.path.join(dst_data_dir, 'ImageSets/Main')
    if not os.path.exists(image_name_files_dir):
        os.makedirs(image_name_files_dir)
    f = open(dst_data_dir + '/ImageSets/Main/test.txt','w')
    f.writelines(image_lists[:L])
    f.close()
    f = open(dst_data_dir + '/ImageSets/Main/trainval.txt','w')
    f.writelines(image_lists[L:])
    f.close()

if __name__ == '__main__':

    # input data_dir and data list, copy to JPEGImages and Annotations of dst_data_dir
    src_data_dir = '/workspace/D2/sanjun/fridgeFood/fridgeImageLabels'
    dst_data_dir = '/workspace/D2/sanjun/fridgeFood_VOC'

    copyOriginDataToVOCFormatDirAndRename(src_data_dir, dst_data_dir)
