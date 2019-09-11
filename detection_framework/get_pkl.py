# coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import xml.etree.ElementTree as ET
import _pickle as cPickle

import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import time
import pickle
from PIL import Image

from augmentations import to_chw_bgr
from EXTD_64 import build_extd
from config import cfg
from vocdataset import VOCDetection, VOCAnnotationTransform

labelmap = ['uav']

class Timer(object):
    """简单的计时器"""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(save_folder, net, dataset, thresh=0.5):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]
    
    #计时器设定
    _t = {'im_detect': Timer(), 'misc': Timer()}

    #设置保存目录
    output_dir = os.path.join(save_folder, 'EXTD')
    if not os.path.exists(output_dir):
        print("创建目录"+output_dir)
        os.makedirs(output_dir)
    print("检测结果保存到"+output_dir)
    for i in range(num_images):
        img = dataset.pull_image(i)
        img_name = dataset.pull_anno(i)[0]
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        h, w, _ = img.shape
	#缩放可能有点问题
        max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
        x = to_chw_bgr(image)
        x = x.astype('float32')
        x -= cfg.img_mean
        x = x[[2, 1, 0], :, :]
        x = Variable(torch.from_numpy(x).unsqueeze(0))
        if use_cuda:
            x = x.cuda()
        
        # 检测时间
        _t['im_detect'].tic()
        with torch.no_grad():
            detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(thresh).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
            fin_mask = np.where(scores > 0.6)[0]
            bboxes = boxes.cpu().numpy()[fin_mask]
            scores = scores[fin_mask]
            for k in range(len(scores)):
                leftup = (int(bboxes[k][0]), int(bboxes[k][1]))
                right_bottom = (int(bboxes[k][2]), int(bboxes[k][3]))
                cv2.rectangle(img, leftup, right_bottom, (0, 255, 0), 2)

        save_file = os.path.join(output_dir, '{}.jpg'.format(img_name))
        cv2.imwrite(save_file,img)	#保存检测图片
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
    #保存检测结果为pkl文件
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    net = build_extd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load("weights/extd_voc.pth"))
    net.eval()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('模型加载完成')

    dataset = VOCDetection("/media/wl/000675B10007A33A/DatasetRepo/uavsummer",
                           target_transform=VOCAnnotationTransform(),
                           mode='test')
    print("数据加载完成")
    #开始检测
    test_net('result/', net, dataset, 0.5)