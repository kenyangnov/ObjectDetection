#!/usr/bin/env python3
# coding=UTF-8

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import os
from skimage.io import imsave
import numpy as np
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

imgPath = 'trainset/'
annFile = 'trainset.json'

if not os.path.exists('anno_image_coco/'):
    os.makedirs('anno_image_coco/')


def draw_rectangle(coordinates, image, imageName):
    for coordinate in coordinates:
        left = np.rint(coordinate[0])
        right = np.rint(coordinate[1])
        top = np.rint(coordinate[2])
        bottom = np.rint(coordinate[3])
        # 左上角坐标, 右下角坐标
        cv2.rectangle(image,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      2)
    imsave('anno_image_coco/'+imageName, image)


# 初始化标注数据的 COCO api
coco = COCO(annFile)

# 显示 COCO categories 和 supercategories
# cats = coco.loadCats(coco.getCatIds())
# print(cats)
# nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

imgPath = 'trainset/'
imgList = os.listdir(imgPath)
# for i in range(len(imgList)):
for i in range(7):
    imgId = i+1
    img = coco.loadImgs(imgId)[0]
    imageName = img['file_name']
    # print(img)
    # print(imageName)

    # 加载并显示图片
    # I = io.imread('%s/%s' % (imgPath, img['file_name']))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()
    
    # 'catIds=[]'表示展示所有类别的bbox，也可以指定类别,本次任务中每张图只有一个类别
    annId = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annId)
    #print(anns)
    
    coordinates = []
    imgRaw = cv2.imread(os.path.join(imgPath, imageName))
    for j in range(len(anns)):
        coordinate = []
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1]+anns[j]['bbox'][3])
        coordinate.append(anns[j]['bbox'][0]+anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][1])
        # print(coordinate)
        coordinates.append(coordinate)
    # print(coordinates)
    draw_rectangle(coordinates, imgRaw, imageName)
    print("正在处理{0}".format(i))
