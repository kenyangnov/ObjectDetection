#!/usr/bin/env python3

# Adapt from ->
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# <- Written by kyn0v

"""Reval = re-eval. Re-evaluate saved detections."""

import os, sys, argparse
import numpy as np
import pickle as cPickle

from voc_eval import voc_eval

def parse_args():
    """
    解析输入参数
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('net_version',type = str)
    args = parser.parse_args()
    return args


def do_python_eval(data_path, det_file_path, classes, output_dir, ovthresh):
    anno_path = os.path.join( data_path, 'Annotations', '{:s}.xml')  #anno文件
    test_file = os.path.join( data_path, 'test.txt')  #待检测的文件列表
    cache_dir = os.path.join('result') #anno解析后的缓存地址
    aps = []  #用于记录不同类别的ap
    # 设置评估规则
    year = 2007
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # 开始评估不同类别
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        rec, prec, ap = voc_eval(
            det_file_path, anno_path, test_file, cls, cache_dir, ovthresh,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # 保存pr文件
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
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
    
    net_version = args.net_version
    # 检测结果与评估结果保存的目录
    result_root = '/home/wl/Desktop/EXTD/result/'+ net_version
    # 检测结果文件路径
    det_file_path = os.path.join(result_root, 'result.txt')
    # 数据集路径
    dataset_path = '/media/wl/000675B10007A33A/DatasetRepo/uavsummer/'
    # 待评估的类别
    classes = ['uav']

    print("开始检测")
    do_python_eval(dataset_path, det_file_path, classes, result_root,ovthresh=0.5)
