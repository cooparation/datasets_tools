voc_root="./VOCdevkit/VOC2007"
imageset_txt="test.txt"
eval_results_dir="./eval_results"
class_file="voc-model-labels.txt"
python ./voc_det/get_prec_ap_rec.py \
    --voc_root $voc_root \
    --imageset_txt $imageset_txt \
    --class_file $class_file \
    --eval_results_dir $eval_results_dir
