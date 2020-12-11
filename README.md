# tools for VOC and COCO style datasets

## Requirements
* cocodataset/cocoapi

## Instructions
## coco tools
* ```extract_coco_sample.py```  extract the sample coco data to check label infomation
* ```copy_file.py``` copy labeled files to dir[images, annotations]
* ```convert2coco.py``` convert to COCO style dir structure
* ```rename_categories.py``` rename the categories
* ```labelme2coco.py``` convert the labelme json to coco json
* ```labelme2mask.py``` convert the labelme json to png mask
* ```mask2coco.py``` convert the png mask to coco json
## voc tools
* ```getSomeClass.py``` extract some class from the VOC
* ```get_prec_ap_rec.sh``` get the eval results precision, mean average precision
and recall
