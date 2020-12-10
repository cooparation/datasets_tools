# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

def annsToSeg(anns, coco_instance):
    '''
    converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
    :param anns: COCO annotations as returned by 'coco.loadAnns'
    :param coco_instance: an instance of the COCO class from pycocotools
    :return: three 2D numpy arrays where the value of each pixel is the class id, instance number, and instance id.
    '''
    image_details = coco_instance.loadImgs(anns[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']

    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, anns = annsToMask(anns, h, w)

    for i, mask in enumerate(masks):
        class_seg = np.where(class_seg>0, class_seg, mask*anns[i]['category_id'])
        instance_seg = np.where(instance_seg>0, instance_seg, mask*(i+1))
        id_seg = np.where(id_seg > 0, id_seg, mask * anns[i]['id'])

    return class_seg, instance_seg, id_seg.astype(np.int64)


def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annsToMask(anns, h, w):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    anns = sorted(anns, key=lambda x: x['area'])  # Smaller items first, so they are not covered by overlapping segs
    for ann in anns:
        rle = annToRLE(ann, h, w)
        m = maskUtils.decode(rle)
        masks.append(m)
    return masks, anns

def coco2voc(anns_file, target_folder, n=None, compress=True):
    '''
    This function converts COCO style annotations to PASCAL VOC style instance and class
        segmentations. Additionaly, it creates a segmentation mask(1d ndarray) with every pixel contatining the id of
        the instance that the pixel belongs to.
    :param anns_file: COCO annotations file, as given in the COCO data set
    :param Target_folder: path to the folder where the results will be saved
    :param n: Number of image annotations to convert. Default is None in which case all of the annotations are converted
    :param compress: if True, id segmentation masks are saved as '.npz' compressed files. if False they are saved as '.npy'
    :return: All segmentations are saved to the target folder, along with a list of ids of the images that were converted
    '''

    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    if n is None:
        n = len(coco_imgs)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    instance_target_path = os.path.join(target_folder, 'instance_labels')
    class_target_path = os.path.join(target_folder, 'class_labels')
    id_target_path = os.path.join(target_folder, 'id_labels')

    os.makedirs(instance_target_path, exist_ok=True)
    os.makedirs(class_target_path, exist_ok=True)
    os.makedirs(id_target_path, exist_ok=True)

    image_id_list = open(os.path.join(target_folder, 'images_ids.txt'), 'a+')
    start = time.time()

    for i, img in enumerate(coco_imgs):

        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            continue

        class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

        Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img) + '.png')
        Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')

        if compress:
            np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
        else:
            np.save(os.path.join(id_target_path, str(img)+'.npy'), id_seg)

        image_id_list.write(str(img)+'\n')

        if i%100==0 and i>0:
            print(str(i)+" annotations processed" +
                  " in "+str(int(time.time()-start)) + " seconds")
        if i>=n:
            break

    image_id_list.close()
    return
if __name__ == '__main__':
    # !!Change paths to your local machine!!
    annotations_file = './COCO2017/annotations/instances_train2017.json'
    labels_output_folder = './vocType_dir/output'
    data_folder = './MSCOCO2017/train2017'
    if len(sys.argv) == 3:
        annotations_file = sys.argv[1]
        labels_output_folder = sys.argv[2]

    # Convert n=25 annotations
    coco2voc(annotations_file, labels_output_folder, n=None, compress=True)
