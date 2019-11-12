# --------------------------------------------------------
#
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import numpy as np

from voc_dataset import VOCDataset

def group_voc_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        #gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            #all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
            all_gt_boxes[class_index][image_id] = np.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = all_gt_boxes[class_index][image_id]
    return true_case_stat, all_gt_boxes, all_difficult_cases

def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()

def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

# get acc from prediction_file = 'det_test_classname.txt'
def compute_ap_from_predfile_per_class(num_true_cases, gt_boxes, difficult_cases,
                prediction_file, confi_thresh, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = [float(v)-1.0 for v in t[2:]]
            #box -= [1.0]  # convert to python format where indexes start from 0
            #box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        sorted_indexes1 = np.where(scores[sorted_indexes]>=confi_thresh)
        sorted_indexes = sorted_indexes[sorted_indexes1]
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]

            ixmin = np.maximum(gt_box[:, 0], box[0])
            iymin = np.maximum(gt_box[:, 1], box[1])
            ixmax = np.minimum(gt_box[:, 2], box[2])
            iymax = np.minimum(gt_box[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_box[:, 2] - gt_box[:, 0] + 1.) *
                   (gt_box[:, 3] - gt_box[:, 1] + 1.) - inters)
            ious = inters / uni

            max_iou = np.max(ious)
            max_arg = np.argmax(ious)

            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
                #if (image_id, max_arg) not in matched:
                #    true_positive[i] = 1
                #    matched.add((image_id, max_arg))
                #else:
                #    false_positive[i] = 1
            else:
                false_positive[i] = 1

    numtp = np.sum(true_positive)
    numfp = np.sum(false_positive)
    num_dets = len(image_ids)
    print('numtp: {:}, numfp: {:} num_true_cases: {:}'.format(numtp, numfp, num_true_cases))
    #assert (numfp + numtp == num_dets)
    rec = numtp/ np.maximum(float(num_true_cases), np.finfo(np.float64).eps)
    prec = numtp / np.maximum(float(numtp + numfp), np.finfo(np.float64).eps)

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        average_precision = compute_voc2007_average_precision(precision, recall)
    else:
        average_precision = compute_average_precision(precision, recall)
    return prec, average_precision, rec

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['name'] = obj.find('name').text # get the class name
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

