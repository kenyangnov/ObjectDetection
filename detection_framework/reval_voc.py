#!/usr/bin/env python

# Adapt from ->
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# <- Written by Yaping Sun

"""Reval = re-eval. Re-evaluate saved detections."""

import os, sys, argparse
import numpy as np
import pickle as cPickle

from voc_eval import voc_eval

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('--output_dir', default='result/' ,help='results directory', 
                        type=str)
    parser.add_argument('--data_dir', dest='data_dir', default='/media/wl/000675B10007A33A/DatasetRepo/uavsummer/', type=str)
 
    parser.add_argument('--image_set', dest='image_set', default='uavindex', type=str)

    args = parser.parse_args()
    return args


def do_python_eval(data_path, image_set, classes, output_dir = 'result/'):
    annopath = os.path.join(data_path, 'Annotations','{:s}.xml')
    imagesetfile = os.path.join(data_path, image_set + '.txt')
    cachedir = os.path.join('cache')
    detpath = 'result/uav_result.txt'
    aps = []
    # VOC评估指标在2010年发生更改
    year = 2010  #设定遵循哪一年的规则
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(classes):   #其实只有一类：uav
        if cls == '__background__':
            continue
        rec, prec, ap = voc_eval(
            detpath, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        #将单类的评估结果写入对应的pkl文件，格式为[rec, prec, ap]
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            print('将该类的评估结果写入文件：{}_pr.pkl'.format(cls))
    #
    print('所有类别的Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')



if __name__ == '__main__':
    args = parse_args()

    #类名列表
    classes = ['uav']

    print("开始检测")
    do_python_eval(args.data_dir, args.image_set, classes)
