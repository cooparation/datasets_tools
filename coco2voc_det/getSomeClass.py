# coding=utf-8
#!/usr/bin/env python
import os
import sys
import cv2
import global_dir
from random import shuffle
from collections import OrderedDict
import shutil

import xml.etree.cElementTree as ET # parse xml

file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']
#file_type_list = ['jpg']

##############################
### generate the labelname_voc
# example:
# edit label
# label_info = [
#     dict(name='none', label=0, display_name='background'),  # backgroud
#     dict(name="cat",label=1, display_name='cat'),  # class1
#     dict(name="dog",label=2, display_name='dog'),  # class2
# ]
# labelmap('./labelmap_voc.prototxt', label_info)
#############################
def labelmap(labelmap_file, label_info):
    labelmap = caffe_pb2.LabelMap()
    for i in range(len(label_info)):
        labelmapitem = caffe_pb2.LabelMapItem()
        labelmapitem.name = label_info[i]['name']
        labelmapitem.label = label_info[i]['label']
        labelmapitem.display_name = label_info[i]['display_name']
        labelmap.item.add().MergeFrom(labelmapitem)
    with open(labelmap_file, 'w') as f:
        f.write(str(labelmap))

#########################
### rename image
### rename Img,output image name is the format
### 000000011.jpg,000003456.jpg,000000000.jpg, 最高9位，前补0
########################
def rename_img(Img_dir):
    listfile=os.listdir(Img_dir) # get the image lists
    total_num = 0
    for line in listfile:
        if line[-4:] == '.jpg':
            newname = '{:0>9}'.format(total_num) +'.jpg'
            os.rename(os.path.join(Img_dir, line), os.path.join(Img_dir, newname))
            total_num+=1

# note: out_data_dir contains the Annotations
# Brief: if the xml file has the classes
#        then remove the not needed classes and save to out_data_dir
#        else do nothing
def deleteChildNodesFromeXML(xml_list_path, CLASSES, out_data_dir):
#path_root = ['./VOC2007/Annotations',
#             './VOC2012/Annotations']
#CLASSES = ["dog",  "person"]
    xml_name = os.path.basename(xml_list_path)
    tree = ET.parse(xml_list_path)
    root = tree.getroot() # get root node
    for child in root.findall('object'):
        name = child.find('name').text
        if not name in CLASSES:
            root.remove(child)

    tree.write(os.path.join(out_data_dir, 'Annotations', xml_name))

##############################
# Brief: extract a lists of classes images and xmls to dst_data_dir
# note: JPEGImages and Annotations are contained in data_dir
def extractVOCSomeClass(src_data_dir, dst_data_dir, classes, list_files):
    paths_src = [src_data_dir + '/JPEGImages', src_data_dir + '/Annotations']
    paths_dst = [dst_data_dir + '/JPEGImages', dst_data_dir + '/Annotations']
    lines = open(list_files, 'r').readlines()
    if not os.path.exists(paths_dst[0]):
        os.makedirs(paths_dst[0])
    if not os.path.exists(paths_dst[1]):
        os.makedirs(paths_dst[1])

    types = set()
    write_lines = []
    num_images = 0
    for line in lines:
        image_name = line.split()[0]
        label = line.split()[1]
        print label
        if label != '1':
            continue
        image_name_jpg = image_name + '.jpg'
        image_file_path = os.path.join(src_data_dir, 'JPEGImages', image_name_jpg)
        #write_lines.append((file_path.split('/')[-1]).split('.')[0] + '\n')
        xml_file_path = image_file_path.replace('JPEGImages', 'Annotations')
        xml_file_path = xml_file_path.replace('jpg', 'xml')

        hasTheClass = deleteChildNodesFromeXML(xml_file_path, classes, dst_data_dir)
        if not hasTheClass:
            continue
        write_lines.append(paths_dst[0]+'/'+image_name+'.jpg'+ ' ' + \
                           paths_dst[1]+'/'+image_name+'.xml' + '\n')
        shutil.copy(image_file_path, paths_dst[0] + '/')
        #shutil.copy(xml_file_path, paths_dst[1] + '/')
        num_images += 1
        print 'copy', num_images, image_file_path, 'to', paths_dst[0]
        print 'copy', num_images, xml_file_path, 'to', paths_dst[1]
    print 'have num_images:', num_images

    ###  write full path of image_list_path and conresponding xml_file_path
    ### to dst_data_dir/ImageSets/Main/
    shuffle(write_lines)

    L  = int(len(write_lines)*0.1)
    f = open('./data/test.txt','w')
    f.writelines(write_lines[:L])
    f.close()

    f = open( './data/trainval.txt','w')
    f.writelines(write_lines[L:])
    f.close()

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
    src_data_dir = '/workspace/D2/sanjun/VOCdevkit/VOC2012'
    dst_data_dir = '/workspace/D2/sanjun/ExtractedVOCClasses/person'

    list_files = '/workspace/D2/sanjun/VOCdevkit/VOC2012/ImageSets/Main/person_trainval.txt'

    CLASSES = ['person']

    extractVOCSomeClass(src_data_dir, dst_data_dir, CLASSES, list_files)
