#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn as nn
import random
from dataset.docudataset import *

torch.backends.cudnn.enabled = False
from models.Tifdm import Tifdm
from utils.trainer import *



def choose_dataset(root_dirs):
    '''

    Args:
        root_dirs: 根据输入的root_dirs，选择数据集
    Returns:

    '''
    ##############数据集准备,先用大数据集训练一个base
    train_index = []
    valid_index = []
    # 构建索引
    for i, root_dir in enumerate(root_dirs):
        index = []
        image_dir = os.path.join(root_dir, 'image')
        label_dir = os.path.join(root_dir, 'label')
        image_files = os.listdir(image_dir)

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file).replace('jpg', 'png')
            index.append((image_path, label_path, i))  # 添加图片路径和标签路径，以及数据集索引
        # 随机打乱索引
        random.seed(42)
        random.shuffle(index)
        # 分割索引为训练集和验证集
        train_size = int(0.8 * len(index))
        train_index = train_index + index[:train_size]
        valid_index = valid_index + index[train_size:]
    #确认文件都存在
    for aaa in [train_index,valid_index]:
        for aa in aaa:
            image_path,label_path,bb = aa
            if os.path.isfile(image_path):
                pass
            else:
                print(f"文件 {image_path} 不存在")
                exit()
            if os.path.isfile(label_path):
                pass
            else:
                print(f"文件 {label_path} 不存在")
                exit()
    # 构建训练、验证集list
    train_img_dir = []
    train_mask_dir = []
    for aa in train_index:
        train_img_dir.append(aa[0])
        train_mask_dir.append(aa[1])
    valid_img_dir = []
    valid_mask_dir = []
    for aa in valid_index:
        valid_img_dir.append(aa[0])
        valid_mask_dir.append(aa[1])
    return sorted(train_img_dir),sorted(train_mask_dir), sorted(train_img_dir),sorted(train_mask_dir)

if __name__ == '__main__':
#==========================全局参数======================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/text_tamper/', help='parent directory of dataset')
    parser.add_argument('--load_model_path', type=str, default='/mnt/text_tamper_detection/cache/weight_Ours_0.0218_0.0141_0.0077_0.5005_1.pth', help='load_model_path')
    parser.add_argument('--save_model_path', type=str, default='/mnt/text_tamper_detection/ckps/model_ckps', help='save_model_path')
    parser.add_argument('--ckps_dir', type=str, default='/mnt/text_tamper_detection/ckps', help='ckps_dir')
    parser.add_argument('--process_test', type=str, default='0', help='parent directory of dataset')
    args = parser.parse_args()
    data_dir = args.data_dir
#==========================参数设置======================================
    params = {
        "model_name": 'Tifdm',
        "mode": "train",  # train,predict
        "lr": 0.0001,
        "batch_size": 4,
        "test_batch_size": 4,
        "num_workers": 4,
        "epochs": 100,
        "non_blocking_": True,
        "dataset_name": '02-03-04-05-06FCDSCD-07',
    }
    # 获得当前父目录的上一层目录
    params["load_model_path"] = args.load_model_path  # best_acc+'_'+f1+'_'+iou+'_'+auc+'_'+epoch
    params["save_model_path"] = os.path.join(args.save_model_path, 'weight_{}_version.pth'.format(params["model_name"]))
    params["save_dir"] = args.ckps_dir  #保存中间过程图片的根目录

#=============================Dataset===================================
    root_dirs = [
        # os.path.join(data_dir,'02_tianchi_2022_RIFLC/patch/'),
        os.path.join(data_dir, '03_tianchi_2023TTI/train/tampered/patch'),
        # os.path.join(data_dir,'04_tianchi_securityAI_s2/patch'),
        # os.path.join(data_dir,'05_tianchi_securityAI_long/data/patch/'),
        # os.path.join(data_dir,'06_FCD/patch'),
        # os.path.join(data_dir,'06_SCD/pic_FCD/patch'),
        os.path.join(data_dir, '06_test/pic_FCD/patch'),
        os.path.join(data_dir, '06_train/pic_FCD/patch'),
        os.path.join(data_dir, '07_Text_tamper/patch')]
    train_img_dir,train_mask_dir, valid_img_dir, valid_mask_dir = choose_dataset(root_dirs)

    tmp_train_img_dir = train_img_dir[:16]
    tmp_train_mask_dir = train_mask_dir[:16]
    tmp_valid_img_dir = valid_img_dir[:8]
    tmp_valid_mask_dir = valid_mask_dir[:8]
    if args.process_test == '0':
        train_dataset = UNetDataset(tmp_train_img_dir, train_mask_dir, mode='train')
        val_dataset = UNetDataset(tmp_valid_img_dir, tmp_valid_mask_dir, mode='predict')
    else:
        train_dataset = UNetDataset(train_img_dir, train_mask_dir, mode='train')
        val_dataset = UNetDataset(valid_img_dir, valid_mask_dir, mode='predict')
    predict_results_save_dir = os.path.join(args.ckps_dir, 'predict_results')

#================================加载模型================================
    model = Tifdm()
    model = model.cuda()
    model = nn.DataParallel(model).cuda()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    start_epoch,best_acc = load_model(model,params['load_model_path'],optimizer)

    train_and_validate(model,optimizer, train_dataset,val_dataset, params,start_epoch,best_acc)