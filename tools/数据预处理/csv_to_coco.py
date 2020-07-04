#!/usr/bin/env python
# coding=UTF-8

import sys
import os
import json
import cv2
import pandas as pd

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}


def convert(csvPath, imgPath, jsonFile):
    """
    csvPath : csv文件的路径
    imgPath : 存放图片的文件夹
    jsonFile : 保存生成的json文件路径
    """
    # 初始化
    jsonDict = {"images": [], "type": "instances", "annotations": [],
                "categories": []}
    bboxId = START_BOUNDING_BOX_ID
    categories = PRE_DEFINE_CATEGORIES
    
    # 读取csv文件
    csvFile = pd.read_csv(csvPath)
    
    # 统计图片
    imgNameList = os.listdir(imgPath)
    imgNum = len(imgNameList)
    print("图片总数为{0}".format(imgNum))
    
    for i in range(imgNum):
        print("正在处理{0}".format(i))
        imageId = i+1
        imgName = imgNameList[i]
        # 跳过没有目标的图片
        if imgName == '60f3ea2534804c9b806e7d5ae1e229cf.jpg' or imgName == '6b292bacb2024d9b9f2d0620f489b1e4.jpg':
            continue
        
        # 读取图片及其参数
        img = cv2.imread(os.path.join(imgPath, imgName))
        height, width, _ = img.shape    # 技巧：用_去除不需要的数据
        # height, width, channel = img.shape

        # 添加json中images字段元素
        image = {'file_name': imgName, 'height': height, 'width': width,
                 'id': imageId}
        # print(image)
        jsonDict['images'].append(image)

        # 获取图片在csv文件中的相应行数据（广播机制）
        lines = csvFile[csvFile.filename == imgName]

        # 添加json中annotations字段元素
        for j in range(len(lines)): # 因为图片和图片名没有重复项,故这里的len(lines)恒为1

            category = str(lines.iloc[j]['type'])
            
            # 添加categories元素并分配ID
            if category not in categories:
                newId = len(categories)
                categories[category] = newId
            
            categoryId = categories[category]
            xmin = int(lines.iloc[j]['X1'])
            ymin = int(lines.iloc[j]['Y1'])
            xmax = int(lines.iloc[j]['X3'])
            ymax = int(lines.iloc[j]['Y3'])
            # print(xmin, ymin, xmax, ymax)
            assert(xmax > xmin)
            assert(ymax > ymin)
            objectWidth = abs(xmax - xmin)
            objectHeight = abs(ymax - ymin)
            ann = {'area': objectWidth * objectHeight, 'iscrowd': 0, 'image_id':
                   imageId, 'bbox': [xmin, ymin, objectWidth, objectHeight],
                   'category_id': categoryId, 'id': bboxId, 'ignore': 0,
                   'segmentation': []}
            jsonDict['annotations'].append(ann)
            bboxId = bboxId + 1

    # 添加json中categories字段元素
    for c, cid in categories.items():
        temp = {'supercategory': 'none', 'id': cid, 'name': c}
        jsonDict['categories'].append(temp)

    json_fp = open(jsonFile, 'w')
    json_str = json.dumps(jsonDict, indent=4)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == '__main__':
    # 转换训练数据格式
    csvPath = 'train_label_fix.csv'
    imgPath = 'trainset/'
    jsonFile = 'trainset.json'
    convert(csvPath, imgPath, jsonFile)
    print("trainset converting finished")
    
    # 转换验证数据格式
    csvPath = 'train_label_fix.csv'
    imgPath = 'valset/'
    jsonFile = 'valset.json'
    convert(csvPath, imgPath, jsonFile)
    print("valset converting finished")
