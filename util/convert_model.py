# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/23 22:14
# @Author   : Yang Jiaxiong
# @File     : convert_model.py
# @Desc     : Convert the Over-All Model CheckPoint to State_dict-Only Model CheckPoint
import os

import torch

from model import *


def convert_to_checkpoint(model_name):
    """
    将包含epoch、iteration、optimizer等信息的模型.pth转化为只有网络state_dict的模型
    :param model_name: 网络名称，see package 'model'
    :return:None
    """
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name)  # directory of the models
    model_path = os.path.join(model_dir, model_name + '.pth')  # model path to fetch
    if model_name == 'attu2net':
        net = AttU2Net(3, 1)
    elif model_name == 'attu2netboth':
        net = AttU2NetBoth(3, 1)
    elif model_name == 'attu2netoutside':
        net = AttU2NetOutside(3, 1)
    elif model_name == 'AGCAInside':
        net = AGCAInside(3, 1)
    elif model_name == 'AGCAOutside':
        net = AGCAOutside(3, 1)
    elif model_name == 'AGCABoth':
        net = AGCABoth(3, 1)
    elif model_name == 'AGInsideCAOutside':
        net = AGInsideCAOutside(3, 1)
    elif model_name == 'PACAOutside':
        net = PACAOutside(3, 1)
    elif model_name == 'AGCAInsidePro':
        net = AGCAInsidePro(3, 1)
    elif model_name == 'AGCAInsideRef':
        net = AGCAInsideRef(3, 1)
    elif model_name == 'AGCABothRef':
        net = AGCABothRef(3, 1)
    elif model_name == 'NewNet':
        net = NewNet(3, 1)
    elif model_name == 'AGCA5Stage':
        net = AGCA5Stage(3, 1)
    elif model_name == 'u2net':  # model_name == 'u2net'
        net = U2NET(3, 1)
    elif model_name == 'u3net':
        net = U3NET(3,1,4,True,True)

    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.cuda()
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['net_state_dict'])

    torch.save(net.state_dict(), model_dir + model_name + '_state_dict.pth')
