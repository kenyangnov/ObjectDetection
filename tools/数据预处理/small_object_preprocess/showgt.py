import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import glob
import os,sys
import cv2
import numpy as np

#设置图片显示尺寸
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

def voc_parse(xml_list):
    for num, xml_file in enumerate(xml_list):
        print(xml_file)

        with open(xml_file, 'r') as fp:   
            for p in fp:
                if '<object>' in p:
                    d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                    # 边界框
                    x1 = int(d[-4])
                    y1 = int(d[-3])
                    x2 = int(d[-2])
                    y2 = int(d[-1])
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # 对应格式[x,y,w,h]
                    image_path = '/home/wl/Desktop/uavsummer/JPEGImages/'+xml_file[-10:-4]+'.jpg'

                    image = cv2.imread(image_path)
                    # 图像处理
                    uav_area = image[y1:y2, x1:x2] #[ymin:ymax, xmin:xmax]
                    """
                    uav_gray = cv2.cvtColor(uav_area, cv2.COLOR_BGR2GRAY)
                    # 提取图像梯度
                    gradX = cv2.Sobel(uav_gray, ddepth=cv2.CV_32F, dx=1, dy=0)
                    gradY = cv2.Sobel(uav_gray, ddepth=cv2.CV_32F, dx=0, dy=1)
                    gradient = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
                    # 高斯模糊
                    uav_blurred = cv2.GaussianBlur(gradient,(3,3),0)
                    # 归一化
                    uav_normal = np.zeros_like(uav_blurred)
                    cv2.normalize(uav_blurred, uav_normal,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
                    # 二值化
                    (ret, uav_thresh) = cv2.threshold(uav_normal, 127, 255, cv2.THRESH_BINARY)
                    # 扩张
                    #uav_closed = cv2.erode(uav_thresh, None, iterations=1)
                    uav_closed = cv2.dilate(uav_thresh, None, iterations=2)
                    # 找轮廓
                    contours, hierarchy = cv2.findContours(uav_closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # 找最大的轮廓
                    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                    x, y, w, h = cv2.boundingRect(c)
                    x1_new = x1 + x
                    y1_new = y1 + y
                    x2_new = x1 + (x+w)
                    y2_new = y1 + (y+h)

                    
                    # 显示截取uav
                    #cv2.rectangle(uav_area, (x,y), (x+w, y+h), (0, 255, 0), 2)
                    #uav_crop = uav_area[y:y+h, x:x+w]
                    #cv2.imshow("crop", uav_crop)
                    #cv2.imshow('origin', uav_area)
                    #cv2.waitKey(5000)
                    #cv2.destroyAllWindows()
                    """
                    break
            cv2.imwrite("./crop_test/"+xml_file[-10:-4]+".jpg", uav_area)
            #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) #画矩形框
            """
            cv2.imshow('origin', uav_area)
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            """



def main():
    xml_file = glob.glob('./new_test/*.xml')
    voc_parse(xml_file)
    
        

if __name__ == '__main__':
    main()

