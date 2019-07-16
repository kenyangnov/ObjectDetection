#encoding=utf-8
import os, sys
import glob
from PIL import Image

txt = "wider_face_train_bbx_gt.txt"
xml = "xml/"

with open(txt) as f:
    line = f.readline() # 读取第一行
    while line:
        print(line) # 图片名
        line = line.rstrip('\n')    # 移除换行符
        img = Image.open(('WIDER_train/images/'  + line))
        imgName = (line)
        width, height = img.size
        
        curLine=line.strip().split("_")
        annoName=curLine[-1] #设置annotation文件名(index)

        #写xml文件
        xml_file = open((xml + annoName[:-4] + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(imgName)  + '</filename>\n')
        xml_file.write('    <source>\n')
        xml_file.write('        <database>My Database</database>\n')
        xml_file.write('        <annotation>VOC2007</annotation>\n')
        xml_file.write('        <image>flickr</image>\n')
        xml_file.write('        <flickrid>NULL</flickrid>\n')
        xml_file.write('    </source>\n')
        xml_file.write('    <owner>\n')
        xml_file.write('        <flickrid>NULL</flickrid>\n')
        xml_file.write('        <name>J</name>\n')
        xml_file.write('    </owner>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        xml_file.write('    <segmented>0</segmented>\n')

        line = f.readline() # 图片包含目标的数量
        cnt = int(line)
        if cnt == 0:    # 若为0则跳过一行
            line = f.readline()
        for _ in range(cnt):    #写入多个object
            line = f.readline()
            curLine=line.strip().split(" ")
            xmin = curLine[0]
            ymin = curLine[1]
            xmax = int(curLine[2]) + int(curLine[0])	#根据需要转换成int/float或其他类型
            ymax = int(curLine[3]) + int(curLine[1])
            xml_file.write('    <object>\n')
            xml_file.write('        <name>scratch</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')
        xml_file.close()
        line = f.readline()

