# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/24 11:09
# @Author   : Yang Jiaxiong
# @File     : AGCABothRef.py
import torch
from torch import nn

from attention_block import DANet_ChannelAttentionModule, Attention_Gate
from model.AGCAInsidePro import AGCAU_Net7, AGCAU_Net6, AGCAU_Net5, AGCAU_Net4, AttU_Net4F
from model.u2net import _upsample_like


class AGCABothPro(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AGCABothPro, self).__init__()

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

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.attention5 = Attention_Gate(512, 512)
        self.stage5d = AttU_Net4F(1024, 256, 512)
        self.attention4 = Attention_Gate(512, 512)
        self.stage4d = AGCAU_Net4(1024, 128, 256)
        self.attention3 = Attention_Gate(256, 256)
        self.stage3d = AGCAU_Net5(512, 64, 128)
        self.attention2 = Attention_Gate(128, 128)
        self.stage2d = AGCAU_Net6(256, 32, 64)
        self.attention1 = Attention_Gate(64, 64)
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

        # -------------------- decoder --------------------
        ag5 = self.attention5(g=hx6, x=hx5)  # 512, 18, 18
        ca5 = self.cam5(hx6)  # 512, 9, 9
        hx6up = _upsample_like(ca5, hx5)  # 512, 18, 18
        hx5d = self.stage5d(torch.cat((ag5, hx6up), 1))  # 512, 18, 18

        ag4 = self.attention4(g=hx5d, x=hx4)  # 512, 36, 36
        ca4 = self.cam4(hx5d)  # 512, 18, 18
        hx5dup = _upsample_like(ca4, hx4)  # 512, 36, 36
        hx4d = self.stage4d(torch.cat((ag4, hx5dup), 1))  # 256, 36, 36

        ag3 = self.attention3(g=hx4d, x=hx3)  # 256, 72, 72
        ca3 = self.cam3(hx4d)  # 256, 36, 36
        hx4dup = _upsample_like(ca3, hx3)  # 256, 72, 72
        hx3d = self.stage3d(torch.cat((ag3, hx4dup), 1))  # 128, 72, 72

        ag2 = self.attention2(g=hx3d, x=hx2)  # 128, 144, 144
        ca2 = self.cam2(hx3d)  # 128, 72, 72
        hx3dup = _upsample_like(ca2, hx2)  # 128, 144, 144
        hx2d = self.stage2d(torch.cat((ag2, hx3dup), 1))  # 64, 144, 144

        ag1 = self.attention1(g=hx2d, x=hx1)  # 64, 288, 288
        ca1 = self.cam1(hx2d)  # 64, 144, 144
        hx2dup = _upsample_like(ca1, hx1)  # 64, 288, 288
        hx1d = self.stage1d(torch.cat((ag1, hx2dup), 1))  # 64, 288, 288

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

