# -*- coding: utf-8 -*-
'''
Do some polygons based augmentation with imgaug tool
'''
import os
import sys
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import polys
import json
from json import encoder
import glob

import cv2

####### Example: Augment Images and Polygons #######
# images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
# images[:, 64, 64, :] = 255
# polygons = [
#     [ia.Polygon([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
#     [ia.Polygon([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0)])]
# ]
# images_aug, polygons_aug = seq(images=images, polygons=polygons)
# psoi = ia.PolygonsOnImage([
#         ia.Polygon([(10.5, 20.5), (50.5, 30.5), (10.5, 50.5)])
#         ], shape=images_aug[0].shape)
####################################################

def Save_LabelMe_ImageAndAnno(json_dict, image_aug, polygon_aug, image_path_new, anno_file_new):
    #annotations = {}
    from collections import OrderedDict
    annotations = OrderedDict()
    annotations["imageData"] = "" ## TODO
    annotations["imageWidth"] = json_dict["imageWidth"]
    annotations["imageHeight"] = json_dict["imageHeight"]
    annotations["imagePath"] = os.path.basename(image_path_new)
    annotations["shapes"] = []
    annotations["flags"] = json_dict["flags"]
    annotations["version"] = json_dict["version"]
    shapes_dict = {}
    shapes_dict["label"] = json_dict["shapes"]
    for shape_dict in json_dict["shapes"]:
        #print('++++++ polygon_aug', polygon_aug)
        ### add polygon ###
        for i in range(len(polygon_aug)):
            shapes_dict["label"] = shape_dict["label"]
            shapes_dict["points"] = []
            shapes_dict["group_id"] = shape_dict["group_id"]
            shapes_dict["shape_type"] = shape_dict["shape_type"]
            shapes_dict["flags"] = shape_dict["flags"]
            ### add points ###
            for point in polygon_aug[i]:
                #print("================= ", point)
                shapes_dict["points"].append([float(point[0]), float(point[1])])
        annotations["shapes"].append(shapes_dict)
    #print(annotations)

    ### save json and image ###
    result_json = json.dump(annotations,
            open(anno_file_new, "w", encoding='utf-8'), sort_keys=True, indent=3)
    cv2.imwrite(image_path_new, image_aug)
    print('save json: ', anno_file_new)
    print('save image: ', image_path_new)


class Aug_Polygons():
    def __init__(self, anno_type='labelme'):
        self.anno_type = anno_type

    ### get images and annotations ###
    def read_images_annos(self, images_path, annos_path):
        self.images = []
        self.images_name = []
        self.images_format = []
        self.polygons = []
        self.json_dicts = []
        if self.anno_type == 'labelme':
            assert(len(images_path) == len(annos_path)), '{} not the same {}'.format(len(images_path), len(annos_path))
            ## read polygons points from labelme json
            for i in range(len(images_path)):
                image_path = images_path[i]
                image_name = os.path.basename(image_path).split('.')[0]
                image_format = os.path.basename(image_path).split('.')[1]
                try:
                    image = cv2.imread(image_path)
                except:
                    print('read {} failed'.format(image_path))
                    continue

                self.images.append(image)
                self.images_name.append(image_name)
                self.images_format.append(image_format)

                anno_path = annos_path[i]
                with open(anno_path, "r") as f:
                    json_dict = json.load(f)
                self.json_dicts.append(json_dict)
                polygon = [] ## polygon with one picture
                for j in range(len(json_dict["shapes"])):
                    keypoints = []
                    for kp_list in json_dict["shapes"][j]["points"]:
                        keypoints.append((kp_list[0], kp_list[1]))
                    polygon.append(ia.Polygon(keypoints))
                    #print('keypoints', keypoints)
                self.polygons.append(polygon)
        else:
            print('{} is not supported'.format(self.anno_type))
            sys.exit(-1)

    ## extract the objects of the pointed mask areas
    def extract_objects(self, augmenter_seq, isCropMask=True, save_dir='crop_results'):
        num_image_patchs = 0
        image_patchs = []
        for i in range(len(self.images)):
            (image_h, image_w, image_c) = self.images[i].shape

            ### one image shapes ###
            for polygon in self.polygons[i]:
                poly_points = []
                #print('====== polygon', polygon)
                for point in polygon.exterior:
                    poly_points.append([point[0], point[1]])
                poly_points = np.array([poly_points], 'int32')
                rect = cv2.boundingRect(poly_points) # returns (x,y,w,h)
                if isCropMask:
                    image_crop_mask = np.zeros((image_h, image_w), np.uint8)
                    cv2.fillPoly(image_crop_mask, poly_points, (255))
                    res = cv2.bitwise_and(self.images[i],self.images[i], mask = image_crop_mask)
                    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                else:
                    cropped = self.images[i][rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                crop_image_name = 'crop_' + self.images_name[i] + \
                               str(num_image_patchs) + '.' + self.images_format[i]
                save_image_name = os.path.join(save_dir, crop_image_name)
                cv2.imwrite(save_image_name, cropped)
                image_patchs.append(cropped)

        #images_pos_aug = augmenter_seq(images=images_pos_tmp)
        for i in range(num_image_patchs):
            self.images[i] = image_patchs[i]
            #cv2.imwrite("img_color_aug.jpg", self.images[i])

    ## change the color for the pointed mask areas
    def color_change(self, augmenter_seq, isSeg=True):
        images_pos_tmp = []
        images_neg_tmp = []
        for i in range(len(self.images)):
            #(image_h, image_w, image_c) = self.images[i].shape
            image_pos_mask = np.zeros(self.images[i].shape, np.uint8)

            pts = []
            ### one image shapes ###
            for polygon in self.polygons[i]:
                poly_points = []
                #print('====== polygon', polygon)
                for point in polygon.exterior:
                    poly_points.append([point[0], point[1]])
                pts.append(poly_points)
            cv2.fillPoly(image_pos_mask, np.array(pts, 'int32'), (1, 1, 1))
            #cv2.fillPoly(image_pos_mask, np.array(pts, 'int32'), (0, 255, 255))
            #cv2.imwrite('fillImage.jpg', image_pos_mask)
            image_neg_mask = 1 - image_pos_mask
            image_pos_tmp = image_pos_mask * self.images[i]
            image_neg_tmp = image_neg_mask * self.images[i]

            images_pos_tmp.append(image_pos_tmp)
            images_neg_tmp.append(image_neg_tmp)
        images_pos_aug = augmenter_seq(images=images_pos_tmp)
        for i in range(len(self.images)):
            self.images[i] = images_pos_aug[i] + images_neg_tmp[i]
            #cv2.imwrite("img_color_aug.jpg", self.images[i])

    def do_augmentation(self, augmenter_seq):
        ### do augmentations ###
        self.images_aug, self.polygons_aug = augmenter_seq(images=self.images, polygons=self.polygons)

        ### if the shapes change,then update it
        for i in range(len(self.images_aug)):
            (image_h, image_w, image_c) = self.images_aug[i].shape
            if self.json_dicts[i]["imageHeight"] != image_h:
                self.json_dicts[i]["imageHeight"] = image_h
            if self.json_dicts[i]["imageWidth"] != image_w:
                self.json_dicts[i]["imageWidth"] = image_w
        print('polygons_aug', self.polygons_aug)

    ### save images and write labelme annotations ###
    def save_images_annos(self, save_dir = 'results', aug_times=0):
        for i in range(len(self.images)):
            new_image_name = self.images_name[i] + '_aug_' + str(aug_times)
            save_images_dir = os.path.join(save_dir, 'images')
            save_annos_dir = os.path.join(save_dir, 'annotations')
            if not os.path.exists(save_images_dir):
               os.makedirs(save_images_dir)
            if not os.path.exists(save_annos_dir):
               os.makedirs(save_annos_dir)
            image_path_new = os.path.join(save_images_dir, new_image_name + '.'+self.images_format[i])
            anno_file_new = os.path.join(save_annos_dir, new_image_name + '.json')
            if self.anno_type == 'labelme':
                Save_LabelMe_ImageAndAnno(self.json_dicts[i], self.images_aug[i], self.polygons_aug[i], image_path_new, anno_file_new)

    ### show results ###
    def vis_results(self, vis_dir = 'vis_dir'):
        for i in range(len(self.images_aug)):
            image_with_polygon = self.images_aug[i]
            for j in range(len(self.polygons_aug[i])):
                psoi = ia.PolygonsOnImage([
                         self.polygons_aug[i][j]
                        ], shape=self.images_aug[i].shape)
                image_with_polygon = psoi.draw_on_image(
                        image_with_polygon, alpha_points=0, alpha_face=0.0, color_lines=(255, 0, 0))
                        #image_with_polygon, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0))
            if not os.path.exists(vis_dir):
               os.makedirs(vis_dir)
            save_path = os.path.join(vis_dir,
                    self.images_name[i]+'_polygon'+'.'+self.images_format[i])
            cv2.imwrite(save_path, image_with_polygon)


'''
changes the color temperature of images to a random value between 1100 and 10000 Kelvin
'''
aug_colorTemperature = iaa.ChangeColorTemperature((1100, 10000))

'''
Convert each image to a colorspace with a brightness-related channel, extract
that channel, multiply it by a factor between 0.5 and 1.5, add a value between
-30 and 30 and convert back to the original colorspace
'''
aug_brightness = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))

'''
Multiply the hue and saturation of images by random values;
Sample random values from the discrete uniform range [-50..50],and add them
'''
aug_hueSaturation = [iaa.MultiplyHue((0.5, 1.5)), iaa.MultiplySaturation((0.5, 1.5)),
                     iaa.AddToHue((-50, 50)), iaa.AddToSaturation((-50, 50))
                    ]

'''
Increase each pixel’s R-value (redness) by 10 to 100
'''
aug_redChannels = iaa.WithChannels(0, iaa.Add((10, 100)))

### add the augmenters ###
seq = iaa.Sequential([
    ## 0.5 is the probability, horizontally flip 50% of the images
    iaa.Fliplr(0.5),
    #iaa.Flipud(0.5),
    ## crop images from each side by 0 to 16px(randomly chosen)
    #iaa.Crop(percent=(0, 0.1)),
    #iaa.LinearContrast((0.75, 1.5)),
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #iaa.AdditiveGaussianNoise(scale=0.05*255),
    ## blur images with a sigma of 0 to 3.0
    #iaa.GaussianBlur(sigma=(0, 3.0)),
    # iaa.Affine(translate_px={"x": (1, 5)}),
    #iaa.Affine(
    #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #    rotate=(-25, 25),
    #    shear=(-8, 8)
    #)
])

# Pick two of four given augmenters and apply them in random order
color_seq = iaa.SomeOf(1, [
    aug_brightness,
    aug_colorTemperature,
    #aug_hueSaturation[0],
    #aug_hueSaturation[1],
    #aug_hueSaturation[2],
    #aug_hueSaturation[3]
], random_order=True)

seq = iaa.Sequential([
      #iaa.Affine(rotate=(-90, 90)),
      #iaa.Rot90(1, keep_size=False),
      iaa.SomeOf(1, [
        ## 0.5 is the probability, horizontally flip 50% of the images
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        ## crop images from each side by 0 to 16px(randomly chosen)
        #iaa.Crop(percent=(0, 0.1)),
        #iaa.LinearContrast((0.75, 1.5)),
        #iaa.Multiply((0.8, 1.2), per_channel=0.2),
        #iaa.AdditiveGaussianNoise(scale=0.05*255),
        ## blur images with a sigma of 0 to 3.0
        #iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Affine(translate_px={"x": (1, 5)}),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-7, 7),
            shear=(-8, 8)
        )
      ]),
], random_order=True)

if __name__ == '__main__':

    #images_dir = './labelmeType_dir/images'
    #annos_dir = './labelmeType_dir/annotations/'
    #images_path = glob.glob(os.path.join(images_dir, '**/*.jpg'), recursive=True)
    #annos_file = glob.glob(os.path.join(annos_dir, '**/*.json'), recursive=True)

    ## input images path list and anno-files path list
    images_path = ['images/2020_04_02_13_59_46_640.jpg',
                   'images/2020_04_02_13_59_46_640_test.jpg'
                  ]
    annos_path = ['images/2020_04_02_13_59_46_640.json',
                   'images/2020_04_02_13_59_46_640_test.json'
                 ]
    img_format = 'bmp'
    #img_format = 'jpg'
    if len(sys.argv) == 2:
        root_path = sys.argv[1]
        #in_images = glob.glob(os.path.join(root_path, 'images', '**/*.jpg'), recursive=True)
        in_images = glob.glob(os.path.join(root_path, 'images', '**/*.'+img_format), recursive=True)
        in_annos = glob.glob(os.path.join(root_path, 'annotations', '**/*.json'), recursive=True)
    else:
        print("Usage: {} [input_root_path]".format(sys.argv[0]))

    total_num,file_num = len(in_images), 0
    total_aug_times = 5
    for i in range(total_num):
        anno_file = in_images[i].replace('images', 'annotations')
        anno_file = anno_file.replace(img_format, 'json')
        if not os.path.exists(anno_file):
            print('======= Warning:{} not exists'.format(anno_file))
            continue
        file_num += 1
        print('---- processing {} / {}'.format(file_num, total_num), end = "",flush=True)
        for aug_times in range(total_aug_times):
            images_path = [in_images[i]]
            annos_path = [anno_file]

            aug_handle = Aug_Polygons()
            aug_handle.read_images_annos(images_path, annos_path)
            aug_handle.color_change(color_seq, isSeg=True)
            aug_handle.do_augmentation(seq)
            aug_handle.save_images_annos(save_dir='results', aug_times=aug_times)
            aug_handle.vis_results(vis_dir='vis_dir')

            #aug_handle.extract_objects(color_seq, isCropMask=False)
