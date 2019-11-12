#!/usr/bin/env python

# --------------------------------------------------------
# get the eval statistics
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

import os, sys, argparse
import numpy as np

from voc_dataset import VOCDataset
from voc_eval import group_voc_annotation_by_class
from voc_eval import compute_ap_from_predfile_per_class

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('--eval_results_dir', nargs=1,
            help='results directory (det_test_classname.txt)', type=str)
    parser.add_argument('--voc_root', dest='voc_root',
            default='./VOC2007', type=str)
    parser.add_argument('--imageset_txt', dest='imageset_txt',
            default='./test.txt', type=str)
    parser.add_argument('--class_file', dest='class_file',
            default='./voc-model-labels.txt', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_voc_results_file_template(out_dir = 'results'):
    filename = 'det_test' + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path

if __name__ == '__main__':
    args = parse_args()

    class_file = args.class_file
    voc_root = args.voc_root
    imageset_txt = args.imageset_txt
    eval_results_dir = args.eval_results_dir[0]
    class_file = args.class_file

    #voc_root = "/datasets/VOCdevkit/VOC2007"
    #imageset_txt = "test.txt"
    #eval_results_dir = "./eval_results"
    #class_file = "./models/voc-model-labels.txt"
    save_file = 'static.txt'

    with open(class_file, "r") as f:
        lines = f.readlines()
    class_names = [t.strip("\n") for t in lines]

    dataset = VOCDataset(voc_root, is_test=True)
    true_case_stat, all_gb_boxes, all_difficult_cases = group_voc_annotation_by_class(dataset)

    static_result = open(save_file, "a")
    precs = []
    aps = []
    recs = []
    print("\n\nAverage Precision Per-class:")
    for confi_thresh in np.arange(0.3, 0.85, 0.05):
        static_result.write('conf_thresh: ' +str(confi_thresh) + "\n")
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue
            prediction_path = get_voc_results_file_template(
                    out_dir=eval_results_dir).format(class_name)
            prec, ap, rec = compute_ap_from_predfile_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                confi_thresh,
                0.30, #args.iou_threshold,
                True, #args.use_2007_metric
            )
            precs.append(prec)
            aps.append(ap)
            recs.append(rec)
            print("{:20s} PREC = {:.4f} AP = {:.4f}  REC = {:.4f}\n".format(class_name, prec, ap, rec))

            static_result.write("{:20s} PREC = {:.4f} AP = {:.4f}  REC = {:.4f}\n".format(class_name, prec, ap, rec))

        static_result.write('Mean PREC = {:.4f} Mean AP = {:.4f} Mean REC = {:.4f}\n'.format(np.mean(precs), np.mean(aps), np.mean(recs)))
        print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    static_result.close()
