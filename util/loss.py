# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/12/24 9:25
# @Author   : Yang Jiaxiong
# @File     : loss.py
import torch.nn as nn

import pytorch_iou
import pytorch_ssim

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_iou_loss(pred, target):  # hybrid loss
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss


# mix up loss to do the deep supervision with d0 as the final Salient Map
def multi_mix_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):  # d7,d0
    loss0 = bce_ssim_iou_loss(d0, labels_v)
    loss1 = bce_ssim_iou_loss(d1, labels_v)
    loss2 = bce_ssim_iou_loss(d2, labels_v)
    loss3 = bce_ssim_iou_loss(d3, labels_v)
    loss4 = bce_ssim_iou_loss(d4, labels_v)
    loss5 = bce_ssim_iou_loss(d5, labels_v)
    loss6 = bce_ssim_iou_loss(d6, labels_v)
    # loss7 = bce_ssim_loss(d7, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6  # + 5.0*lossa
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (  deprecate: 版本兼容性的问题，从[0]改为.item(0)
    #     loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


# bce to do the deep supervision with d0 as the final Salient Map
def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % ( deprecate: same as above
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #     loss5.data.item(),
    #     loss6.data.item()))

    return loss0, loss

# todo more loss functions may work
# https://github.com/siyueyu/SCWSSOD
# Structure-ConsistentWeakly Supervised Salient Object Detection with Local Saliency Coherence
# 这是一种新的方法进行损失，因此需要将模型进行改变，最后SM先不上采样，而是先进行结构loss求解，之后再上采样，得到相关loss
