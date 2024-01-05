# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/28 10:28
# @Author   : Yang Jiaxiong
# @File     : NewNet.py


# A Bad Try


import torch
from torch import nn

from attention_block import DANet_ChannelAttentionModule, Attention_block
from model.u2net import _upsample_like


class REBNCONV(nn.Module):  # relu and batchNorm conv 的组合  输入和输出的维度相同！只进行卷积但是不降低分辨率solution
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.l_relu_s1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.l_relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class AGCAU_Net7(nn.Module):  # stage1  维持通道数  七层encoder
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
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

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilate=2)  # 这里使用了dilate扩展感受野，维持分辨率

        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()
        self.cam6 = DANet_ChannelAttentionModule()

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention3 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention2 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)  # 最后重整 从64 到64 没有增加维度

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

        ac6 = self.cam6(hx7)  # a6的维度减半
        hx6d = self.rebnconv6d(torch.cat((ac6, hx6), dim=1))  # 平行传递层  9

        ac5 = self.cam5(hx6d)
        hx6dup = _upsample_like(ac5, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx5, hx6dup), 1))  # 18

        ac4 = self.cam4(hx5d)
        hx5dup = _upsample_like(ac4, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx4, hx5dup), 1))  # 36 36 36

        hx4dup = _upsample_like(hx4d, hx3)  # 32 72 72
        ag3 = self.attention3(g=hx4dup, x=hx3)  # 72
        hx3d = self.rebnconv3d(torch.cat((ag3, hx4dup), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        ag2 = self.attention2(g=hx3dup, x=hx2)  # 144
        hx2d = self.rebnconv2d(torch.cat((ag2, hx3dup), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        ag1 = self.attention1(g=hx2dup, x=hx1)  # 288
        hx1d = self.rebnconv1d(torch.cat((ag1, hx2dup), 1))

        return hx1d + hxin


class AGCAU_Net6(nn.Module):  # 128,144,144
    def __init__(self, in_ch=64, mid_ch=32, out_ch=128):
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

        self.cam5 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention2 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)  # 这里是直接从64变为128了

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

        ac5 = self.cam5(hx6)  # 9
        hx5d = self.rebnconv5d(torch.cat((ac5, hx5), 1))  # 平行传递层

        ac4 = self.cam4(hx5d)  # 18
        hx5dup = _upsample_like(ac4, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx4, hx5dup), 1))

        ac3 = self.cam3(hx4d)  # 36
        hx4dup = _upsample_like(ac3, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx3, hx4dup), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        ag2 = self.attention2(g=hx3dup, x=hx2)  # 32*72*72
        hx2d = self.rebnconv2d(torch.cat((ag2, hx3dup), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        ag1 = self.attention1(g=hx2dup, x=hx1)  # 32*32*144
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

        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.attention1 = Attention_block(mid_ch, mid_ch, mid_ch // 2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)  # 这里从128变为256

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

        ac4 = self.cam4(hx5)  # 9
        hx4d = self.rebnconv4d(torch.cat((ac4, hx4), 1))  # 平行传递

        ac3 = self.cam3(hx4d)  # 18
        hx4dup = _upsample_like(ac3, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx3, hx4dup), 1))

        ac2 = self.cam2(hx3d)  # 36
        hx3dup = _upsample_like(ac2, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx2, hx3dup), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        ag1 = self.attention1(g=hx2dup, x=hx1)
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

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()

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

        ac3 = self.cam3(hx4)
        hx3d = self.rebnconv3d(torch.cat((ac3, hx3), 1))  # 平行传递层

        ac2 = self.cam2(hx3d)
        hx3dup = _upsample_like(ac2, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx2, hx3dup), 1))

        ac1 = self.cam1(hx2d)
        hx2dup = _upsample_like(ac1, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx1, hx2dup), 1))

        return hx1d + hxin


class AttU_Net4F(nn.Module):  # 解决空洞卷积 griding 问题
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=3)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=5)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=3)
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


class NewNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(NewNet, self).__init__()

        self.stage1 = AGCAU_Net7(in_ch, 32, 64)  # 3-》64
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 288*288-》144*144

        self.stage2 = AGCAU_Net6(64, 32, 128)  # 64-》128
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 144*144-》72*72

        self.stage3 = AGCAU_Net5(128, 64, 256)  # 128-》256
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 72*72-》36*36

        self.stage4 = AGCAU_Net4(256, 128, 512)  # 256-》512
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 36*36-》18*18

        self.stage5 = AttU_Net4F(512, 256, 512)  # 512-》512  块内没有升降采样，通道数不变
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 18*18-》9*9  必要性 这里将18变成了9
        # self.drop5 = nn.Dropout2d(0.5)  # todo

        self.stage6 = AttU_Net4F(512, 256, 512)  # 512-》512 *9*9  块内没有升降采样，通道数不变
        # self.drop6 = nn.Dropout2d(0.5)  # todo

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = AttU_Net4F(1024, 256, 512)  # 512*18*18  因为是从上层concat来的，所以input channel是*2
        self.stage4d = AGCAU_Net4(1024, 128, 256)  # 256*36*36
        self.attention3 = Attention_block(256, 256, 128)
        self.stage3d = AGCAU_Net5(512, 64, 128)  # 128*72*72
        self.attention2 = Attention_block(128, 128, 64)
        self.stage2d = AGCAU_Net6(256, 32, 64)  # 64*144*144
        self.attention1 = Attention_block(64, 64, 32)
        self.stage1d = AGCAU_Net7(128, 16, 64)  # 64*288*288

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)  # 六张图合在一起，所以通道数是6*out

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
        # hx = self.drop5(hx)  # todo

        # stage 6
        hx6 = self.stage6(hx)
        # hx6 = self.drop5(hx6)  # todo

        # -------------------- decoder --------------------
        # ag5 = self.attention5(g=hx6, x=hx5)  # 512, 18, 18
        hx6up = _upsample_like(hx6, hx5)  # 512, 18, 18
        ca5 = self.cam5(hx6up)
        hx5d = self.stage5d(torch.cat((ca5, hx5), 1))  # 512, 18, 18

        # ag4 = self.attention4(g=hx5d, x=hx4)  # 512, 36, 36
        hx5dup = _upsample_like(hx5d, hx4)  # 512, 36, 36
        ca4 = self.cam4(hx5dup)
        hx4d = self.stage4d(torch.cat((ca4, hx4), 1))  # 256, 36, 36

        hx4dup = _upsample_like(hx4d, hx3)  # 256, 72, 72
        ag3 = self.attention3(g=hx4dup, x=hx3)  # 256, 72, 72
        ca3 = self.cam3(hx4dup)
        hx3d = self.stage3d(torch.cat((ca3, ag3), 1))  # 128, 72, 72

        hx3dup = _upsample_like(hx3d, hx2)  # 128, 144, 144
        ag2 = self.attention2(g=hx3dup, x=hx2)  # 128, 144, 144
        ca2 = self.cam2(hx3dup)
        hx2d = self.stage2d(torch.cat((ca2, ag2), 1))  # 64, 144, 144

        hx2dup = _upsample_like(hx2d, hx1)  # 64, 288, 288
        ag1 = self.attention1(g=hx2dup, x=hx1)  # 64, 288, 288
        ca1 = self.cam1(hx2dup)
        hx1d = self.stage1d(torch.cat((ca1, ag1), 1))  # 64, 288, 288

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


class NewNetP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NewNetP, self).__init__()

        self.stage1 = AGCAU_Net7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = AGCAU_Net6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = AGCAU_Net5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = AGCAU_Net4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = AttU_Net4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.drop5 = nn.Dropout2d(0.5)

        self.stage6 = AttU_Net4F(64, 16, 64)
        self.drop6 = nn.Dropout2d(0.5)

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = AttU_Net4F(128, 16, 64)
        self.stage4d = AGCAU_Net4(128, 16, 64)
        self.attention3 = Attention_block(64, 64, 32)
        self.stage3d = AGCAU_Net5(128, 16, 64)
        self.attention2 = Attention_block(64, 64, 32)
        self.stage2d = AGCAU_Net6(128, 16, 64)
        self.attention1 = Attention_block(64, 64, 32)
        self.stage1d = AGCAU_Net7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

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
        hx = self.drop5(hx)

        # stage 6
        hx6 = self.stage6(hx)
        hx6 = self.drop5(hx6)

        # -------------------- decoder --------------------
        # ag5 = self.attention5(g=hx6, x=hx5)  # 512, 18, 18
        hx6up = _upsample_like(hx6, hx5)  # 512, 18, 18
        ca5 = self.cam5(hx6up)
        hx5d = self.stage5d(torch.cat((ca5, hx5), 1))  # 512, 18, 18

        # ag4 = self.attention4(g=hx5d, x=hx4)  # 512, 36, 36
        hx5dup = _upsample_like(hx5d, hx4)  # 512, 36, 36
        ca4 = self.cam4(hx5dup)
        hx4d = self.stage4d(torch.cat((ca4, hx4), 1))  # 256, 36, 36

        hx4dup = _upsample_like(hx4d, hx3)  # 256, 72, 72
        ag3 = self.attention3(g=hx4dup, x=hx3)  # 256, 72, 72
        ca3 = self.cam3(hx4dup)
        hx3d = self.stage3d(torch.cat((ca3, ag3), 1))  # 128, 72, 72

        hx3dup = _upsample_like(hx3d, hx2)  # 128, 144, 144
        ag2 = self.attention2(g=hx3dup, x=hx2)  # 128, 144, 144
        ca2 = self.cam2(hx3dup)
        hx2d = self.stage2d(torch.cat((ca2, ag2), 1))  # 64, 144, 144

        hx2dup = _upsample_like(hx2d, hx1)  # 64, 288, 288
        ag1 = self.attention1(g=hx2dup, x=hx1)  # 64, 288, 288
        ca1 = self.cam1(hx2dup)
        hx1d = self.stage1d(torch.cat((ca1, ag1), 1))  # 64, 288, 288

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
