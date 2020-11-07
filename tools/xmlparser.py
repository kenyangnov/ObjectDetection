# xml annotation parser and processing
# created by kynov

import os
import sys
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import shutil
import cv2
import imghdr
from shutil import copyfile


# 批量修改xml文件中的filename
def changeFileNameInAnnotation(xmlFiles):
    for filePath in tqdm(xmlFiles):
        imgName = filePath.split("\\")[-1][:-4] + ".jpg"
        tree = ET.parse(filePath)
        root = tree.getroot()
        fileName = root.find('filename')
        fileName.text = imgName
        tree.write(filePath)


# 可视化bbox
def showBBox(xmlFilePath):
    tree = ET.parse(xmlFilePath)
    root = tree.getroot()
    fileName = root.find('filename').text
    imagePath = xmlFilePath.replace("Annotations",
                                    "JPEGImages").replace(".xml", ".jpg")
    image = cv2.imread(imagePath)
    for objectInfo in root.findall('object'):
        bbox = objectInfo.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("bbox", image)
    cv2.waitKey(100000)


# 分析目标尺寸分布
def analyzeObjectSize(xmlFiles):
    sizeCount = []
    for filePath in tqdm(xmlFiles):
        tree = ET.parse(filePath)
        root = tree.getroot()
        for objectInfo in root.findall('object'):
            bbox = objectInfo.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objectSize = (xmax - xmin) * (ymax - ymin)
            sizeCount.append(objectSize)
    bins = range(4000, 20000, 1000)
    counts, binEdge = np.histogram(sizeCount, bins)
    print(counts)
    plt.hist(sizeCount, bins)
    plt.show()


# 选取指定尺寸范围内的数据
def selectObjectBySize(sizeRange, xmlFiles):
    for filePath in tqdm(xmlFiles):
        imageFileName = filePath.split("\\")[-1][:-4] + ".jpg"
        if not imghdr.what(os.path.join("./JPEGImages", imageFileName)):
            continue
        tree = ET.parse(filePath)
        root = tree.getroot()
        for objectInfo in root.findall('object'):
            bbox = objectInfo.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objectSize = (xmax - xmin) * (ymax - ymin)
            if objectSize >= sizeRange[0] and objectSize <= sizeRange[1]:
                dstDir = str(sizeRange[0]) + "to" + str(sizeRange[1])
                if not os.path.exists(dstDir):
                    os.makedirs(os.path.join(dstDir, "Annotations"))
                    os.makedirs(os.path.join(dstDir, "JPEGImages"))
                xmlFileName = filePath.split("\\")[-1][:-4] + ".xml"
                shutil.copyfile(
                    filePath, os.path.join(dstDir, "Annotations", xmlFileName))
                shutil.copyfile(
                    os.path.join("./JPEGImages", imageFileName),
                    os.path.join(dstDir, "JPEGImages", imageFileName))
                break


# xml to coco
def xml2coco(xmlFiles, jsonFile):
    categories = {"uav": 1, "bird": 2}
    bndId = 1
    jsonDict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    for filePath in tqdm(xmlFiles):
        tree = ET.parse(filePath)
        root = tree.getroot()
        fileName = root.find('filename').text
        imageId = int(os.path.splitext(fileName)[0])
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        image = {
            'file_name': fileName,
            'height': height,
            'width': width,
            'id': imageId
        }
        jsonDict['images'].append(image)
        for objectInfo in root.findall('object'):
            category = objectInfo.find('name').text
            if category not in categories:
                print("%s does not exist in categories!")
                os._exit()
            categoryId = categories[category]
            bbox = objectInfo.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            objectWidth = abs(xmax - xmin)
            objectHeight = abs(ymax - ymin)
            annotation = {
                'area': objectWidth * objectHeight,
                'iscrowd': 0,
                'image_id': imageId,
                'bbox': [xmin, ymin, objectWidth, objectHeight],
                'category_id': categoryId,
                'id': bndId,
                'ignore': 0,
                'segmentation': []
            }
            jsonDict['annotations'].append(annotation)
            bndId = bndId + 1
    for category, categoryId in categories.items():
        cat = {'supercategory': 'none', 'id': category, 'name': categoryId}
        jsonDict['categories'].append(cat)

    json_fp = open(jsonFile, 'w')
    json_str = json.dumps(jsonDict)
    json_fp.write(json_str)
    json_fp.close()


# 批量修改图片尺寸
def convertImageSize(imagePath, width=640, height=640):
    saveDir = str(width) + "x" + str(height)
    if not os.path.exists(saveDir):
        os.makedirs(os.path.join(saveDir))
    for imageFileName in tqdm(imagePath):
        imageName = os.path.basename(imageFileName)
        img = cv2.imread(imageFileName)
        try:
            img_temp = cv2.resize(img, (width, height))
            cv2.imwrite(os.path.join(saveDir, imageName), img_temp)
        except Exception as e:
            print(e)


# 批量重命名
def renameAllFiles(startIdx,
                   filenamePrefix,
                   srcAnno="./Annotations/",
                   srcJpeg="./JPEGImages/",
                   dstAnno="./DstAnnotations/",
                   dstJpeg="./DstJPEGImages/",
                   mode='copy'):
    annoList = os.listdir(srcAnno)
    picList = os.listdir(srcJpeg)
    if not os.path.exists(dstAnno):
        os.makedirs(os.path.join(dstAnno))
    if not os.path.exists(dstJpeg):
        os.makedirs(os.path.join(dstJpeg))
    cnt = startIdx  # 设置文件名计数起点
    for anno in tqdm(annoList):
        pic = anno[:-4] + '.jpg'
        if pic in picList:
            if mode == 'copy':
                copyfile(srcAnno + anno, (dstAnno + filenamePrefix + '_' +
                                          str(cnt).zfill(6) + '.xml'))
                copyfile(srcJpeg + pic, (dstJpeg + filenamePrefix + '_' +
                                         str(cnt).zfill(6) + '.jpg'))
            elif mode == 'cut':
                os.rename(srcAnno + anno, (dstAnno + filenamePrefix + '_' +
                                           str(cnt).zfill(6) + '.xml'))
                os.rename(srcJpeg + pic, (dstJpeg + filenamePrefix + '_' +
                                          str(cnt).zfill(6) + '.jpg'))
            else:
                print("Unknown mode!")
                break
            cnt += 1


# 批量重命名label
def changeLabel(xmlFiles, oriLabel, newLabel):
    for filePath in tqdm(xmlFiles):
        imageFileName = filePath.split("\\")[-1][:-4] + ".jpg"
        if not imghdr.what(os.path.join("./JPEGImages", imageFileName)):
            continue
        tree = ET.parse(filePath)
        root = tree.getroot()
        for objectInfo in root.findall('object'):
            className = objectInfo.find('name').text
            if className == oriLabel:
                objectInfo.find('name').text = newLabel
        tree.write(filePath)


if __name__ == '__main__':
    os.chdir("C:/Users/M/Desktop/fixedwing")

    xmlFiles = glob.glob('./Annotations/*.xml')
    renameAllFiles(1, "fixedwing")

    # changeLabel(xmlFiles, 'fixwing', 'fixedwing')

    # imgFiles = glob.glob('./JPEGImages/*.jpg')
    # convertImageSize(imgFiles)

    # xmlFiles = glob.glob('./Annotations/*.xml')
    # sizeRange = [1, 100000]
    # selectObjectBySize(sizeRange, xmlFiles)

    # jsonFile = sys.argv[1]
    # xml2coco(xmlFiles)

    # xmlFilePath = "./4000to20000/Annotations/000103.xml"
    # showBBox(xmlFilePath)
