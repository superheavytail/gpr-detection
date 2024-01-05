# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/4/28 15:03
# @Author   : Yang Jiaxiong
# @File     : cal_params_flops.py
# @Desc     : You can use this .py to get Params, Flops and FPS. 使用该文件可以计算模型的参数量Params、计算量Flops和处理速度FPS
# @Acknowledge : code from website

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from model import *


def net_profile(model: nn.Module):  # 计算参数量和计算量
    input = torch.randn(1, 3, 288, 288)
    input = torch.unsqueeze(input, dim=0)
    flops, params = profile(model, inputs=(input))
    print(str(type(model)) + ' flops:' + str(flops))
    print(str(type(model)) + ' params:' + str(params))


def cal_fps(model: nn.Module):  # 计算处理速度
    device = torch.device('cuda')
    model.to(device)
    dummy_input = torch.randn(1, 3, 288, 288, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)


# change this block with different model
net = AttU2Net(3, 1)
net_profile(net)
cal_fps(net)
