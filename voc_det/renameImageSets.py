import os, shutil
import cv2
from random import shuffle
from collections import OrderedDict

#paths = ['/home/liusj/datasets/rawImages']
#label_paths = ['/home/liusj/datasets/labelFiles']

# require Annotations and JPEGImages are in the same dir
paths = ['/apps/liusj/zhangxiaoDatasets/footDetection/Annotations']

dst_dir = '/apps/liusj/FoodDetDatasets'
sub_annotations_dst_dir = 'Annotations'
sub_jpegs_dst_dir = 'JPEGImages'

file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']

write_lines = []
types = set()
num_images = 0
for path in paths:
    for root, _, files in os.walk(path):
        for fname in files:
            types.add(fname.split('.')[-1])
            if fname.split('.')[-1] == 'xml':
                num_images += 1
                file_path = os.path.join(root, fname)
                newPrefname = fname.split('.')[0] + '_'\
                            + str(num_images)
                new_file_path = dst_dir + '/'\
                            + sub_annotations_dst_dir + '/'\
                            + newPrefname + '.xml'
                shutil.copyfile(file_path, new_file_path)

                file_path = file_path.replace('Annotations', 'JPEGImages')
                file_path = file_path.replace('xml', 'jpg')
                print 'images', file_path
                new_jpg_path = dst_dir + '/'\
                            + sub_jpegs_dst_dir + '/'\
                            + newPrefname + '.jpg'
                shutil.copyfile(file_path, new_jpg_path)

print 'copy image num:', num_images
