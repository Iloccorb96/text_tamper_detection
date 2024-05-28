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


def cal_f1_iou_auc(batch_pred_mask, batch_gt_mask):
    """
    :param pred_mask:
    :param label_mask:
    :return:
    """
    """
        计算模型预测结果的评分（F1、IoU、AUC）
        :param batch_pred_mask: 批量预测的掩码图像，形状为 [batch_size, H, W]，值范围为 [0, 1]
        :param batch_gt_mask: 批量真实标签的掩码图像，形状为 [batch_size, H, W]，值范围为 [0, 1]
        :return: 总评分（score）、F1平均值（f1_avg）、IoU平均值（iou_avg）、AUC平均值（auc_avg）
        """
    f1_list, iou_list, auc_list = [], [], []
    # 循环处理每个样本
    for pred_mask, gt_mask in zip(batch_pred_mask, batch_gt_mask):
        # 将预测结果和真实标签转换为二进制图像
        pred_mask[pred_mask >= 0.5] = 1
        pred_mask[pred_mask < 0.5] = 0
        gt_mask[gt_mask >= 0.5] = 1
        gt_mask[gt_mask < 0.5] = 0

        # 计算 F1 分数和 IoU
        f1, iou = metric_numpy(pred_mask, gt_mask)

        # 计算 AUC
        try:
            auc = roc_auc_score(gt_mask.flatten(), pred_mask.flatten())
            auc_list.append(auc)
        except ValueError:
            pass
        f1_list.append(f1)
        iou_list.append(iou)
    return f1_list, iou_list, auc_list


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

def predict(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        f1_list, iou_list, auc_list = [], [], []
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])
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
            result_output = result_output.cpu().numpy()
            f1_list_tmp, iou_list_tmp, auc_list_tmp = cal_f1_iou_auc(result_output, batch_y_val)
            f1_list += f1_list_tmp
            iou_list += iou_list_tmp
            auc_list += auc_list_tmp
        f1_avg, iou_avg, auc_avg = np.mean(f1_list), np.mean(iou_list), np.mean(auc_list) if auc_list else 0
        score = f1_avg + iou_avg
        return score, f1_avg, iou_avg, auc_avg

def predict_simple(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with (torch.no_grad()):
        f1_list, iou_list, auc_list = [],[],[]
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])
            batch_pred = model(batch_x_val)
            batch_pred = torch.sigmoid(batch_pred)
            batch_pred = batch_pred.cpu().numpy()
            f1_list_tmp, iou_list_tmp, auc_list_tmp = cal_f1_iou_auc(batch_pred, batch_y_val)
            f1_list += f1_list_tmp
            iou_list += iou_list_tmp
            auc_list += auc_list_tmp
        f1_avg,iou_avg,auc_avg = np.mean(f1_list),np.mean(iou_list),  np.mean(auc_list) if auc_list else 0
        score = f1_avg + iou_avg
    return score,f1_avg, iou_avg, auc_avg

def train_and_validate(model, optimizer, train_dataset, val_dataset, valid_label_dir,params, epoch_start=1, best_score=0):
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

    # log_path = '/data/jinrong/wcw/PS/text-image-forgery-detection/ckps/logs'
    global_step = 0

    if params["mode"] == 'train':
        for epoch in range(epoch_start, params["epochs"] + 1):
            train(train_loader, model, criterion_1, optimizer, epoch, params, global_step)
            cur_score,f1_avg, iou_avg, auc_avg = predict_simple(val_loader, model, params, threshold=0.35)

            print('current model is:{} ,current epoch is:{} ,current score is:{} ,best score is:{}'.format(
                params["model_name"], epoch, cur_score, best_score))
            if epoch//5 == 0:
                save_model(model, optimizer, params["save_dir"], params["save_model_path"], epoch, best_score,
                           cur_score, params["model_name"], f1_avg, iou_avg, auc_avg,mode='per5')
            if cur_score > best_score:
                best_score = cur_score
                save_model(model, optimizer, params["save_dir"], params["save_model_path"], epoch, best_score,
                           cur_score,params["model_name"], f1_avg, iou_avg, auc_avg)



    elif params["mode"] == 'val':
        predict(val_loader, model, params, threshold=0.35)
        cur_score,f1_avg, iou_avg, auc_avg = predict_simple(val_loader, model, params, threshold=0.35)
        print('current score is:{:3f} f1 score is:{:3f} iou score is:{:3f} auc score is:{:3f}'.format(cur_score, f1_avg, iou_avg, auc_avg))

    elif params["mode"] == 'predict':
        # threshold=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.75]
        threshold = [0.35]
        log_path = "./logs/{}_{}.txt".format(params["model_name"], params["dataset_name"])
        for thre in threshold:
            predict(val_loader, model, params, threshold=thre)
            f1, iou, auc = cal_f1_iou_auc(valid_label_dir, save_dir)
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

