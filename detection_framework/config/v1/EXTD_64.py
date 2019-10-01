# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers.modules.l2norm import L2Norm
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from torch.autograd import Variable

from layers import *
from config import cfg
import numpy as np

import mobileFacenet_64_PReLU as mobileFacenet_11

from torchsummary import summary

#用于包装F.inpterpolate上采样后的结果
def upsample(in_channels, out_channels): # should use F.inpterpolate
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                  stride=1, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class EXTD(nn.Module):

    def __init__(self, phase, base, head, num_classes):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # 主干网络和上采样层
        self.base = nn.ModuleList(base)
        self.upfeat = []

        #上采样层：先append，再转换为ModuleList
        for it in range(5):
            self.upfeat.append(upsample(in_channels=64, out_channels=64))
        self.upfeat = nn.ModuleList(self.upfeat)
        
        #head[0]为loc层，head[1]为conf层
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """
        Input：
        输入图片张量，大小为[batch,3,w,h].

        Return:
        取决于不同阶段
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # 使用mobilefacenet的前6个模块作为backbone,循环迭代2号-5号IR模块
        size = x.size()[2:] #[w, h],[640, 640]
        sources = list()
        loc = list()
        conf = list()

        for k in range(6):
            x = self.base[k](x)
            if k == 1:
                before = F.interpolate(x, scale_factor=0.5)
        s1 = x

        for k in range(2, 6):
            if k == 2:
                x += before
                before = F.interpolate(x, scale_factor=0.5)
            x = self.base[k](x)
        s2 = x

        for k in range(2, 6):
            if k == 2:
                x += before
                before = F.interpolate(x, scale_factor=0.5)
            x = self.base[k](x)
        s3 = x

        for k in range(2, 6):
            if k == 2:
                x += before
                before = F.interpolate(x, scale_factor=0.5)
            x = self.base[k](x)
        s4 = x

        for k in range(2, 6):
            if k == 2:
                x += before
                before = F.interpolate(x, scale_factor=0.5)
            x = self.base[k](x)
        s5 = x

        for k in range(2, 6):
            if k == 2:
                x += before
                before = F.interpolate(x, scale_factor=0.5)
            x = self.base[k](x)
        s6 = x

        # sources保存金字塔求出来的特征图
        sources.append(s6)

        u1 = self.upfeat[0](F.interpolate(s6, size=(s5.size()[2], s5.size()[3]), mode='bilinear')) + s5 # 10x10
        sources.append(u1)

        u2 = self.upfeat[1](F.interpolate(u1, size=(s4.size()[2], s4.size()[3]), mode='bilinear')) + s4 # 20x20
        sources.append(u2)

        u3 = self.upfeat[2](F.interpolate(u2, size=(s3.size()[2], s3.size()[3]), mode='bilinear')) + s3  # 40x40
        sources.append(u3)

        u4 = self.upfeat[3](F.interpolate(u3, size=(s2.size()[2], s2.size()[3]), mode='bilinear')) + s2  # 80x80
        sources.append(u4)

        u5 = self.upfeat[4](F.interpolate(u4, size=(s1.size()[2], s1.size()[3]), mode='bilinear')) + s1  # 160x160
        sources.append(u5)

        sources = sources[::-1]  # 特征金字塔

        # apply extra layers and cache source layer outputs
        #  for k, v in enumerate(self.extras):
        #    x = v(x)
        #    if k in self.extras_idx:
        #        sources.append(x)


        # 求loc_x和conf_x
        # 处理loc[0]和conf[0]
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        # 处理loc[1]-loc[5]和conf[1]-conf[5]
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        # 记录特征金字塔输出尺寸：[[160, 160], [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        #print(features_maps)

        # 先验框
        self.priorbox = PriorBox(size, features_maps, cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # 测试阶段的输出
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        # 训练阶段的输出
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        nn.init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()



def mobileFacenet():
    net = mobileFacenet_11.Net()
    #net.features记录mobilenet v2的网络结构
    return nn.ModuleList(net.features)


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def add_extras_mobileFace(in_channel=32):   #=64,返回一个模块
    layers = []
    channels = [in_channel]
    # 输出channel等于输出channel，同时特征图的尺寸减半
    for v in channels:
        layers += [
            nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(v),
            nn.ReLU(inplace=True)]
    return layers


def add_extras_dwc(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, in_channels,
                                     kernel_size=3, stride=2, padding=1, bias=False, groups=in_channels),
                           nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


# 
def multibox(base, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    net_source = [4, 4, 4, 4]
    feature_dim = []

    #feature_dim设置，最终为[64, 64, 64, 64, 64]
    feature_dim += [base[4].conv[-2].out_channels]
    for idx in net_source:
        feature_dim += [base[idx].conv[-2].out_channels]

    # loc/conf layer设置，最终共有6层
    #第1层
    loc_layers += [nn.Conv2d(feature_dim[0], 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(feature_dim[0], 3 + (num_classes - 1), kernel_size=3, padding=1)]
    #第2-5层
    for k, v in enumerate(net_source, 1):
        loc_layers += [nn.Conv2d(feature_dim[k], 4, kernel_size=3, padding=1)]
        # num_classes 而不是 3 + (num_classes - 1)
        conf_layers += [nn.Conv2d(feature_dim[k], num_classes, kernel_size=3, padding=1)]
    # for k, v in enumerate(extra_layers[3::6], 2):
    #第6层
    for v in [0]:
        # extra_layers[v].out_channels = 64
        loc_layers += [nn.Conv2d(extra_layers[v].out_channels, 4, kernel_size=3, padding=1)]
        # num_classes 而不是 3 + (num_classes - 1)
        conf_layers += [nn.Conv2d(extra_layers[v].out_channels, num_classes, kernel_size=3, padding=1)]

    # 返回网络主干+额外添加的层(好像用不上)+loc/conf共6层
    return base[:6], extra_layers, (loc_layers, conf_layers)


def build_extd(phase, num_classes=2):
    #extras_用于构建loc_layer和conf_layer
    base_, extras_, head_ = multibox(
        mobileFacenet(), add_extras_mobileFace(in_channel=64), num_classes)
    # base_为主干网络，head_为主干网络后面接的loc/conf层（共6层）
    # print(base_)
    return EXTD(phase, base_, head_, num_classes)


if __name__ == '__main__':
    net = build_extd('train', num_classes=2).cuda()
    inputs = torch.randn(4, 3, 640, 640).cuda()
    output = net(inputs)
    #summary(net,(3,640,640))

