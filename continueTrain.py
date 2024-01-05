# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/20 17:55
# @Author   : Yang Jiaxiong     2625398619@qq.com  yangjiaxiong@whut.edu.cn
# @File     : train.py
# @Desc     : 断点续训  Resume
# reference :
#       有关Adam学习率是否有必要衰减的问题 https://blog.csdn.net/weixin_30838921/article/details/99482591
#       有关权重衰减的https://blog.csdn.net/program_developer/article/details/80867468

import glob
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from model import *
from util import RandomCrop, SalObjDataset, RescaleT, ToTensorLab

if __name__ == '__main__':

    # ------- 1. define loss function --------
    from util.loss import multi_mix_loss_fusion as loss_fnc

    # from util.loss import multi_bce_loss_fusion as loss_fnc

    # ------- 2. set the directory of training dataset --------
    model_name = 'AGCAInsideRef'  # 'u2netp' 'attu2net' 'attu2netp' 'u2net' 'attu2netoutside' 'attu2netboth'
    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('DUTS-TR', 'imgs' + os.sep)
    tra_label_dir = os.path.join('DUTS-TR', 'gt' + os.sep)

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    # hyper-param   todo 这里的参数是可以调节的，在debug的时候可能比较小，在正式实验的时候请正确
    epoch_num = 100000
    batch_size_train = 10  # 注意这里引入注意力机制之后就不能用12了，否则GPU会溢出
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([  # 组合式变形
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
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
    else:  # model_name == 'u2net'
        net = U2NET(3, 1)

    # ------- 4.loading model --------
    if torch.cuda.is_available():
        checkpoint = torch.load(model_dir + model_name + '.pth')
        net.load_state_dict(checkpoint['net_state_dict'])
        net.cuda()
        torch.nn.DataParallel(net)
    else:
        checkpoint = torch.load(model_dir + model_name + '.pth', map_location='cpu')
        net.load_state_dict(checkpoint['net_state_dict'])

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    # 定义学习率策略 learning rate schedule
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=64,
    #                            verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,eps=1e-10)

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # adjust as you like
    ite_num = checkpoint['ite_num']
    start_epoch = checkpoint['epoch']

    # ------- 6. training process --------
    print("---start training...")
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 1000  # save the model every 1000 iterations

    for epoch in range(start_epoch + 1, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
            # inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = loss_fnc(d0, d1, d2, d3, d4, d5, d6, labels_v)
            # 其中loss2保存的是最后一个SMap的loss，而loss保存的是所有SMap的loss

            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            # 为了打印所以引入的变量
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print(model_name + "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                # In fact, we change the save_frq as the training go further and we save the checkpoint more often.
                torch.save(
                    {'epoch': epoch,
                     'ite_num': ite_num,
                     'net_state_dict': net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()
                     },
                    model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                        ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
