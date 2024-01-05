# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/2/28 22:10
# @Author   : Yang Jiaxiong
# @File     : AGInsideCAOutside.py
import torch
from torch import nn

from attention_block import DANet_ChannelAttentionModule
from model.attentionU2Net import AttU_Net7, AttU_Net6, AttU_Net5, AttU_Net4, AttU_Net4F

from model.u2net import _upsample_like


class AGInsideCAOutside(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AGInsideCAOutside, self).__init__()

        self.stage1 = AttU_Net7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = AttU_Net6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = AttU_Net5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = AttU_Net4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = AttU_Net4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = AttU_Net4F(512, 256, 512)

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = AttU_Net4F(1024, 256, 512)  # 因为先进行了上采样所以这里1024了
        self.stage4d = AttU_Net4(1024, 128, 256)  # 512
        self.stage3d = AttU_Net5(512, 64, 128)  # 256
        self.stage2d = AttU_Net6(256, 32, 64)  # 128
        self.stage1d = AttU_Net7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)  # 64s
        hx = self.pool12(hx1)  # 64

        # stage 2
        hx2 = self.stage2(hx)  # 128
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)  # 256
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)  # 512
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)  # 512
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)  # 512

        # -------------------- decoder --------------------
        hx6up = _upsample_like(hx6, hx5)  # 1024
        a5 = self.cam5(hx6up)  # 这里的结构问题  先注意力还是先上采样
        hx5d = self.stage5d(torch.cat((a5, hx5), 1))  # 512

        hx5dup = _upsample_like(hx5d, hx4)  # 1024
        a4 = self.cam4(hx5dup)
        hx4d = self.stage4d(torch.cat((a4, hx4), 1))  # 256

        hx4dup = _upsample_like(hx4d, hx3)  # 512
        a3 = self.cam3(hx4dup)
        hx3d = self.stage3d(torch.cat((a3, hx3), 1))  # 128

        hx3dup = _upsample_like(hx3d, hx2)  # 256
        a2 = self.cam2(hx3dup)
        hx2d = self.stage2d(torch.cat((a2, hx2), 1))  # 64

        hx2dup = _upsample_like(hx2d, hx1)  # 128
        a1 = self.cam1(hx2dup)
        hx1d = self.stage1d(torch.cat((a1, hx1), 1))  # 64

        # side output 生成不同stage的SMAP
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

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))  # 最后合成一张最终的SMap

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
