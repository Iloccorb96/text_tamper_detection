#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import tqdm
import argparse

def process_dataset(image_dir, label_dir, output_dir, patch_size=512, stride=512):
    '''
    将原图和mask图切为512*512大小的patch，并去掉无用的patch
    '''
    # 确保输出目录存在
    output_image_dir = os.path.join(output_dir, 'image')
    output_label_dir = os.path.join(output_dir, 'label')
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    print(output_image_dir, output_label_dir)
    # 获取image目录下所有图片的路径
    image_files = os.listdir(image_dir)

    for image_file in tqdm.tqdm(image_files):
        # 构建图片和mask标签的路径
        if '.jpg' in image_file or '.png' in image_file:
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace('.jpg', '.png').replace('image', 'label'))

            # 读取原图和mask标签
            image = cv2.imread(image_path)
            label = cv2.imread(label_path)

            # 获取图片尺寸
            height, width, channel = image.shape
            assert channel == 3

            # 计算填充后的尺寸
            pad_height = patch_size - height % patch_size if height % patch_size != 0 else 0
            pad_width = patch_size - width % patch_size if width % patch_size != 0 else 0

            # 填充原图和标签
            image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = cv2.copyMakeBorder(label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # 获取填充后的图片尺寸
            padded_height, padded_width = image.shape[:2]

            # 遍历原图，切片并保存
            patch_index = 0
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    # 获取patch
                    image_patch = image[y:y + patch_size, x:x + patch_size]
                    label_patch = label[y:y + patch_size, x:x + patch_size]

                    # 检查mask标签patch是否存在mask值
                    if np.any(label_patch == 255):
                        # 保存patch
                        patch_image_name = f'{image_file[:-4]}_patch{patch_index}.jpg'.replace('image_', '')
                        patch_label_name = f'{image_file[:-4]}_patch{patch_index}.png'.replace('image_', '')
                        image_patch_path = os.path.join(output_image_dir, patch_image_name)
                        label_patch_path = os.path.join(output_label_dir, patch_label_name)
                        cv2.imwrite(image_patch_path, image_patch)
                        cv2.imwrite(label_patch_path, label_patch)
                        patch_index += 1
        else:
            pass
if __name__ == '__main__':
    #从命令行获取parent_dir的目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir', type=str, default='/mnt/text_tamper/data', help='parent directory of dataset')
    args = parser.parse_args()
    parent_dir = args.parent_dir
for dataset in os.listdir(parent_dir):
    not_train_dataset =[]
    for ntd in not_train_dataset:
        if ntd in dataset:
            continue
        else:
            dataset_path = os.path.join(parent_dir, dataset)
            image_dir = os.path.join(dataset_path, 'img')
            label_dir = os.path.join(dataset_path, 'mask')
            output_dir = os.path.join(dataset_path, 'patch')
            process_dataset(image_dir=image_dir, label_dir=label_dir, output_dir=output_dir)




