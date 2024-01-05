# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/2/28 22:15
# @Author   : Yang Jiaxiong
# @File     : AGCAInside.py
import torch
from torch import nn

from attention_block import DANet_ChannelAttentionModule, Attention_Gate
from model.u2net import _upsample_like, REBNCONV


class AGCAU_Net7(nn.Module):  # stage1  七层encoder
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AGCAU_Net7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilate=2)  # 最深的一层，两侧对称

        self.cam = DANet_ChannelAttentionModule()

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention3 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)  # 最后重整

    def forward(self, x):
        hx = x  # hx不断在变化（/2)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)  # 由于这里hx6没有过pool，所以hx6和hx7的维度相同！

        ac6 = self.cam(hx7)  # a6的维度减半
        hx6d = self.rebnconv6d(torch.cat((ac6, hx6), dim=1))  # 平行传递层  9

        ac5 = self.cam(hx6d)
        hx6dup = _upsample_like(ac5, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx5, hx6dup), 1))  # 18

        ac4 = self.cam(hx5d)
        hx5dup = _upsample_like(ac4, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx4, hx5dup), 1))  # 36

        ag3 = self.attention3(g=hx4d, x=hx3)  # 72
        ac3 = self.cam(hx4d)
        hx4dup = _upsample_like(ac3, hx3)
        hx3d = self.rebnconv3d(torch.cat((ag3, hx4dup), 1))

        ag2 = self.attention2(g=hx3d, x=hx2)  # 144
        ac2 = self.cam(hx3d)
        hx3dup = _upsample_like(ac2, hx2)
        hx2d = self.rebnconv2d(torch.cat((ag2, hx3dup), 1))

        ag1 = self.attention1(g=hx2d, x=hx1)  # 288
        ac1 = self.cam(hx2d)
        hx2dup = _upsample_like(ac1, hx1)
        hx1d = self.rebnconv1d(torch.cat((ag1, hx2dup), 1))

        return hx1d + hxin


class AGCAU_Net6(nn.Module):  # 128,144,144
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AGCAU_Net6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.cam = DANet_ChannelAttentionModule()

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        ac5 = self.cam(hx6)  # 9
        hx5d = self.rebnconv5d(torch.cat((ac5, hx5), 1))  # 平行传递层

        ac4 = self.cam(hx5d)  # 18
        hx5dup = _upsample_like(ac4, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx4, hx5dup), 1))

        ac3 = self.cam(hx4d)  # 36
        hx4dup = _upsample_like(ac3, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx3, hx4dup), 1))

        ag2 = self.attention2(g=hx3d, x=hx2)  # 72
        ac2 = self.cam(hx3d)
        hx3dup = _upsample_like(ac2, hx2)
        hx2d = self.rebnconv2d(torch.cat((ag2, hx3dup), 1))

        ag1 = self.attention1(g=hx2d, x=hx1)  # 144
        ac1 = self.cam(hx2d)
        hx2dup = _upsample_like(ac1, hx1)
        hx1d = self.rebnconv1d(torch.cat((ag1, hx2dup), 1))

        return hx1d + hxin


class AGCAU_Net5(nn.Module):  # 256,72,72
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AGCAU_Net5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.cam = DANet_ChannelAttentionModule()

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        ac4 = self.cam(hx5)  # 9
        hx4d = self.rebnconv4d(torch.cat((ac4, hx4), 1))  # 平行传递

        ac3 = self.cam(hx4d)  # 18
        hx4dup = _upsample_like(ac3, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx3, hx4dup), 1))

        ac2 = self.cam(hx3d)  # 36
        hx3dup = _upsample_like(ac2, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx2, hx3dup), 1))

        ag1 = self.attention1(g=hx2d, x=hx1)
        ac1 = self.cam(hx2d)
        hx2dup = _upsample_like(ac1, hx1)
        hx1d = self.rebnconv1d(torch.cat((ag1, hx2dup), 1))

        return hx1d + hxin


class AGCAU_Net4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AGCAU_Net4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.cam = DANet_ChannelAttentionModule()

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # 平行传递层

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx2, hx3dup), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx1, hx2dup), 1))

        return hx1d + hxin


class AttU_Net4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class AGCAInside(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AGCAInside, self).__init__()

        self.stage1 = AGCAU_Net7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = AGCAU_Net6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = AGCAU_Net5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = AGCAU_Net4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = AttU_Net4F(512, 256, 512)  # 没有升降采样
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = AttU_Net4F(512, 256, 512)  # 最下面一层

        # decoder
        self.stage5d = AttU_Net4F(1024, 256, 512)
        self.stage4d = AGCAU_Net4(1024, 128, 256)
        self.stage3d = AGCAU_Net5(512, 64, 128)
        self.stage2d = AGCAU_Net6(256, 32, 64)
        self.stage1d = AGCAU_Net7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1  这个过程就是不断增加通道数，减少图片大小的过程
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
