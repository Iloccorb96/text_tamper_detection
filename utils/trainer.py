#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import shutil
import time
import torch
import tqdm
import cv2
from Metric import *
import numpy as np
from MetricMonitor import MetricMonitor
from torch.utils.data import DataLoader
import torch.nn as nn
from ..lossModel.MultLoss import WeightedDiceBCE
from sklearn.metrics import roc_auc_score
from torchvision.utils import save_image
import torch.nn.functional as F
def load_model(model,load_model_path,optimizer):
    """
    :param model:
    :param params:path of model checkpoint
    :param optimizer:
    :return:
    """
    if os.path.exists(load_model_path):
        checkpoint = torch.load(load_model_path)
        #model.module.load_state_dict(checkpoint['model'], strict=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model_name = checkpoint['model_name']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('The { '+model_name+' } model load weight successful!')
        return start_epoch,best_acc
    else:
        print('The model path is not exists. We will train the model from scratch.')
        return 1,0



def save_model(model, optimizer, save_dir, params, epoch=0, best_acc=0,cur_acc=0, model_name='model_name', f1=0, iou=0, auc=0,mode='best'):
    '''

    Args:
        model: 模型
        optimizer: 优化器
        save_dir:
        params:
        epoch:
        best_acc:
        cur_acc:
        model_name:
        f1:
        iou:
        auc:
        mode:

    Returns:

    '''
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch,
        'model': model.state_dict(),
        'model_name': model_name,
        'optimizer': optimizer.state_dict(),
    }
    best_acc = str(round(best_acc, 4))
    cur_acc = str(round(cur_acc, 4))
    f1 = str(round(f1, 4))
    iou = str(round(iou, 4))
    auc = str(round(auc, 4))
    epoch = str(epoch)
    if mode == 'best':
        params = params.replace('version', mode+'_' +best_acc + '_' + f1 + '_' + iou + '_' + auc + '_' + epoch)
    elif mode == 'lastest':
        params = params.replace('version', mode+'_' +cur_acc + '_' + f1 + '_' + iou + '_' + auc + '_' + epoch)
    elif mode == 'per5':
        params = params.replace('version', mode+'_' +cur_acc + '_' + f1 + '_' + iou + '_' + auc + '_' + epoch)
    torch.save(checkpoint, params)
    print('Time: {}, save weight successful! Best score is:{}'.format(time.strftime('%H:%M:%S', time.localtime()),
                                                                      best_acc))


def cal_score(tmp_valid_label_dir, save_dir):
    dir_1 = save_dir  # 预测值
    file_list_1 = os.listdir(dir_1)  # 预测值
    file_list_1.sort()  # 预测值
    file_list_2 = tmp_valid_label_dir
    file_list_2.sort()  # label
    f1_list = []
    iou_list = []
    auc_list = []

    stream = tqdm(file_list_1, desc='processing', colour='CYAN')
    for i, _ in enumerate(stream, start=0):
        file_path_1 = os.path.join(dir_1, file_list_1[i])
        file_path_2 = file_list_2[i]

        img_1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(file_path_2, cv2.IMREAD_GRAYSCALE)
        h, w = img_1.shape
        h_, w_ = img_2.shape
        if h != h_ or w != w_:
            img_2 = cv2.resize(img_2, (w, h))

        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_1[img_1 >= 0.5] = 1
        img_1[img_1 < 0.5] = 0

        img_2_temp = img_2
        img_2_temp[img_2_temp >= 0.5] = 1
        img_2_temp[img_2_temp < 0.5] = 0
        # 计算f1 score 和 iou score
        f1, iou = metric_numpy(img_1, img_2_temp)

        # 计算auc
        img_1_ = img_1.flatten()
        img_2_ = img_2.flatten()
        try:
            # 计算auc score
            auc = roc_auc_score(img_2_, img_1_)
            auc_list.append(auc)
        except ValueError:
            pass

        f1_list.append(f1)
        iou_list.append(iou)

    f1_avg = np.mean(f1_list)
    iou_avg = np.mean(iou_list)
    auc_avg = np.mean(auc_list)
    # auc_avg = 0
    score = f1_avg + iou_avg
    return score, f1_avg, iou_avg, auc_avg


def train(train_loader, model, criterion1, optimizer, epoch, params, global_step):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader, desc='processing', colour='CYAN')

    for i, (images, masks, _) in enumerate(stream, start=1):
        images = images.cuda(non_blocking=params['non_blocking_'])  # 非阻塞的，数据转移将不会等待GPU完成之前的操作
        masks = masks.cuda(non_blocking=params['non_blocking_'])

        reg_outs = model(images)
        reg_outs = torch.sigmoid(reg_outs)

        loss_region = criterion1(reg_outs, masks)

        optimizer.zero_grad()
        loss_region.backward()

        optimizer.step()

        global_step += 1

        metric_monitor.update("Loss", loss_region.item())
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor} Time: {time}".format(epoch=epoch, metric_monitor=metric_monitor,
                                                                          time=time.strftime('%H:%M:%S',
                                                                                             time.localtime()))
        )

def val(val_loader, model, criterion1, epoch, params, global_step):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            masks = batch_y_val
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])

def predict(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        avg_dice_list = []
        avg_iou_list = []
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            masks = batch_y_val
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])
            masks = masks.cuda(non_blocking=params['non_blocking_'])

            output_val = model(batch_x_val)

            batch_x_val_h_flip = batch_x_val.clone().detach()
            batch_x_val_h_flip = torch.flip(batch_x_val_h_flip, [3])

            batch_x_val_v_flip = batch_x_val.clone().detach()
            batch_x_val_v_flip = torch.flip(batch_x_val_v_flip, [2])

            image_h_flip = model(batch_x_val_h_flip)
            image_v_flip = model(batch_x_val_v_flip)
            image_h_flip = torch.flip(image_h_flip, [3])
            image_v_flip = torch.flip(image_v_flip, [2])

            result_output_1 = (output_val + image_h_flip + image_v_flip) / 3.

            result_output = result_output_1
            result_output = torch.sigmoid(result_output)
            for i in range(len(result_output)):
                orig_w = w_s[i]
                orig_h = h_s[i]
                result_output_ = F.interpolate(result_output[i:i + 1], size=[orig_h, orig_w], mode="bicubic",
                                               align_corners=False)
                result_output[result_output >= threshold] = 1
                result_output[result_output < threshold] = 0
                str_ = name[i].split('/')[-1]
                name_str = str_.replace('.jpg', '.png')
                save_img_name = os.path.join(save_dir, name_str)
                save_image(result_output_, save_img_name)


def predict_simple(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        avg_dice_list = []
        avg_iou_list = []
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            masks = batch_y_val
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])
            ############## 1 ##############
            output_val = model(batch_x_val)
            result_output = torch.sigmoid(output_val)
            for i in range(len(result_output)):
                orig_w = w_s[i]
                orig_h = h_s[i]
                result_output_ = F.interpolate(result_output[i:i + 1], size=[orig_h, orig_w], mode="bicubic",
                                               align_corners=False)
                result_output[result_output >= threshold] = 1
                result_output[result_output < threshold] = 0
                str_ = name[i].split('/')[-1]
                name_str = str_.replace('.jpg', '.png')
                save_img_name = os.path.join(save_dir, name_str)
                save_image(result_output_, save_img_name)


def train_and_validate(model, optimizer, train_dataset, val_dataset, valid_label_dir,params, epoch_start=1, best_acc=0):
    save_dir = os.path.join(params['save_dir'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["test_batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    # Define Loss
    criterion_1 = WeightedDiceBCE(dice_weight=0.3, BCE_weight=0.7).cuda()
    # criterion_2 = nn.BCELoss().cuda()

    log_path = '/data/jinrong/wcw/PS/text-image-forgery-detection/ckps/logs'
    global_step = 0

    if params["mode"] == 'train':
        for epoch in range(epoch_start, params["epochs"] + 1):
            train(train_loader, model, criterion_1, optimizer, epoch, params, global_step)
            f1 = 0
            iou = 0
            predict_simple(val_loader, model, params, threshold=0.35)
            cur_acc, f1, iou, auc = cal_score(valid_label_dir, save_dir)
            print('current model is:{} ,current epoch is:{} ,current score is:{} ,best score is:{}'.format(
                params["model_name"], epoch, cur_acc, best_acc))
            if epoch//5 == 0:
                save_model(model, optimizer, params["save_dir"], params["save_model_path"], epoch, best_acc,
                           cur_acc, params["model_name"], f1, iou, auc,mode='per5')
            if cur_acc > best_acc:
                best_acc = cur_acc
                save_model(model, optimizer, params["save_dir"], params["save_model_path"], epoch, best_acc,
                           cur_acc,params["model_name"], f1, iou, auc)
            val(val_loader, model, criterion_1, epoch, params, global_step)


    elif params["mode"] == 'val':
        predict(val_loader, model, params, threshold=0.35)
        cur_acc, f1, iou, auc = cal_score(valid_label_dir, save_dir)
        print('current score is:{:3f} f1 score is:{:3f} iou score is:{:3f} auc score is:{:3f}'.format(cur_acc, f1, iou,
                                                                                                      auc))

    elif params["mode"] == 'predict':
        # threshold=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.75]
        threshold = [0.35]
        log_path = "./logs/{}_{}.txt".format(params["model_name"], params["dataset_name"])
        for thre in threshold:
            predict(val_loader, model, params, threshold=thre)
            cur_acc, f1, iou, auc = cal_score(valid_label_dir, save_dir)
            cur_acc = str(round(cur_acc, 3))
            f1 = str(round(f1, 3))
            iou = str(round(iou, 3))
            auc = str(round(auc, 3))
            print_str = 'threshold:{} current score is:{} f1 score is:{} iou score is:{} auc score is:{} \n'.format(
                thre, cur_acc, f1, iou, auc)
            print(print_str)
            with open(log_path, 'a') as f:
                f.write(print_str)
            f.close()

