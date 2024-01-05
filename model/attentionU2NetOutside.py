# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/12/24 16:48
# @Author   : Yang Jiaxiong
# @File     : attentionU2NetOutside.py  在外面的U型结构中增加attention gate 而stage里面就不增加了

from attention_block import Attention_Gate
from model.u2net import *
from model.u2net import _upsample_like

'''
考虑到内层没有使用attention，所以外层每一层都用了attention
'''


class AttU2NetOutside(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AttU2NetOutside, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.attention5 = Attention_Gate(512, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.attention4 = Attention_Gate(512, 512)
        self.stage4d = RSU4(1024, 128, 256)  # 512
        self.attention3 = Attention_Gate(256, 256)
        self.stage3d = RSU5(512, 64, 128)  # 256
        self.attention2 = Attention_Gate(128, 128)
        self.stage2d = RSU6(256, 32, 64)  # 128
        self.attention1 = Attention_Gate(64, 64)
        self.stage1d = RSU7(128, 16, 64)

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
        hx1 = self.stage1(hx)  # 64
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
        a5 = self.attention5(g=hx6, x=hx5)
        hx6up = _upsample_like(hx6, hx5)  # 1024
        hx5d = self.stage5d(torch.cat((hx6up, a5), 1))  # 512

        a4 = self.attention4(g=hx5d, x=hx4)  # 先上采样，之后再进行的注意力
        hx5dup = _upsample_like(hx5d, hx4)  # 1024
        hx4d = self.stage4d(torch.cat((a4, hx5dup), 1))  # 256

        a3 = self.attention3(g=hx4d, x=hx3)
        hx4dup = _upsample_like(hx4d, hx3)  # 512
        hx3d = self.stage3d(torch.cat((a3, hx4dup), 1))  # 128

        a2 = self.attention2(g=hx3d, x=hx2)
        hx3dup = _upsample_like(hx3d, hx2)  # 256
        hx2d = self.stage2d(torch.cat((a2, hx3dup), 1))  # 64

        a1 = self.attention1(g=hx2d, x=hx1)
        hx2dup = _upsample_like(hx2d, hx1)  # 128
        hx1d = self.stage1d(torch.cat((a1, hx2dup), 1))  # 64

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
