import os
import cv2

#img_path_a = './datasets/single_image_monet_trumpbiden/trainA'
#img_path_aa = './datasets/single_image_monet_trumpbiden/trainA'
#img_path_b = './datasets/single_image_monet_trumpbiden/trainB'
#img_path_bb = './datasets/single_image_monet_trumpbiden/trainB'
#img_path_a = './datasets/single_image_monet_trumpbiden_bg/trainA'
#img_path_aa = './datasets/single_image_monet_trumpbiden_bg/trainA'
#img_path_b = './datasets/single_image_monet_trumpbiden_bg/trainB'
#img_path_bb = './datasets/single_image_monet_trumpbiden_bg/trainB'
#img_path_a = './datasets/JiZhongChiZhuPaiGu/trainA'
#img_path_aa = './datasets/JiZhongChiZhuPaiGu/trainA'
#img_path_b = './datasets/JiZhongChiZhuPaiGu/trainB'
#img_path_bb = './datasets/JiZhongChiZhuPaiGu/trainB'
img_path_a = '/datasets/GANDatasets/shoes_classes'
img_path_aa = './datasets/shoesTypeA2TypeB/trainA'
img_path_b = './datasets/shoesTypeA2TypeB/trainB'
img_path_bb = './datasets/shoesTypeA2TypeB/trainB'

def resize_save(input_path, output_path, shape_w=156, shape_h=318, cover=False):
    os.makedirs(output_path, exist_ok=True)
    for root_dir, sub_dir, img_files in os.walk(input_path):
        for img_file in img_files:
            img_file = os.path.join(root_dir, img_file)
            if img_file.split('.')[-1] not in ['jpg', 'png', 'bmp']:
                continue
            img_data = cv2.imread(img_file)
            img = cv2.resize(img_data, (shape_w, shape_h), interpolation=cv2.INTER_CUBIC)
            if cover:
                output_img_name = img_file
            else:
                img_name = os.path.basename(img_file)
                output_img_name = os.path.join(output_path, img_name)
            cv2.imwrite(output_img_name, img)
            print('image save to :', output_img_name)
if __name__=='__main__':
    #resize_save(img_path_a, img_path_aa, shape_w=160, shape_h=192)
    #resize_save(img_path_a, img_path_aa, shape_w=288, shape_h=359)
    #resize_save(img_path_b, img_path_bb, shape_w=288, shape_h=359)
    #resize_save(img_path_a, img_path_aa, shape_w=420, shape_h=420)
    #resize_save(img_path_b, img_path_bb, shape_w=420, shape_h=420)
    resize_save(img_path_a, img_path_a, shape_w=256, shape_h=256, cover=True)
    #resize_save(img_path_b, img_path_bb, shape_w=200, shape_h=100)

