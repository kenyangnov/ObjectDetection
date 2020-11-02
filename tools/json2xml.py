"""
"https://anti-uav.github.io"中提供的数据集处理脚本
written by kynov.
2020.08.01
"""

from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np

# The mask to remove watermark and text
rect_mask_1 = [0, 0, 230, 40]
rect_mask_2 = [420, 0, 640, 40]
rect_mask_3 = [10, 60, 240, 90]

IR_mask = [rect_mask_1, rect_mask_2, rect_mask_3]

rect_mask_4 = [7, 13, 248, 64]
rect_mask_5 = [9, 106, 232, 150]
rect_mask_6 = [268, 103, 477, 149]
rect_mask_7 = [1499, 15, 1900, 61]
rect_mask_8 = [956, 214, 965, 326]
rect_mask_9 = [1150, 322, 1350, 327]
rect_mask_10 = [1340, 319, 1348, 432]
rect_mask_11 = [1342, 537, 1539, 544]
rect_mask_12 = [1341, 646, 1348, 757]
rect_mask_13 = [1149, 754, 1348, 760]
rect_mask_14 = [958, 754, 963, 868]
rect_mask_15 = [573, 645, 580, 761]
rect_mask_16 = [570, 751, 771, 762]
rect_mask_17 = [383, 537, 579, 544]
rect_mask_18 = [572, 323, 582, 435]
rect_mask_20 = [580, 321, 771, 329]
rect_mask_21 = [0, 960, 190, 1080]
RGB_mask = [
    rect_mask_4, rect_mask_5, rect_mask_6, rect_mask_7, rect_mask_8,
    rect_mask_9, rect_mask_10, rect_mask_11, rect_mask_12, rect_mask_13,
    rect_mask_14, rect_mask_15, rect_mask_16, rect_mask_17, rect_mask_18,
    rect_mask_20, rect_mask_21
]


# Callback when clicking mouse
def onClickMouse(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:  # 当按下左键拖拽鼠标时
        tmp = param.copy()
        cv2.rectangle(tmp, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Coordinate of Rectangle", tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(param, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Coordinate of Rectangle", param)
        coordinate = "x1, y1, x2, y2: [ %s, %s, %s, %s ]" % (ix, iy, x, y)
        print(coordinate)
    elif event == cv2.EVENT_MOUSEMOVE and flags != cv2.EVENT_FLAG_LBUTTON:  # 左键没有按下的情况下,鼠标移动 标出坐标
        temp_coordinate = str(x) + ', ' + str(y)
        tmp = param.copy()
        cv2.putText(tmp, temp_coordinate, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.imshow("Coordinate of Rectangle", tmp)


# Get the coordinate of the area framed by the mouse
def getCoordinate(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("Coordinate of Rectangle")
    cv2.imshow("Coordinate of Rectangle", img)
    cv2.setMouseCallback("Coordinate of Rectangle", onClickMouse, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Overlapping two pictures
def overlap(img_bg_path, img_fg_path):
    bottom = cv2.imread(img_bg_path)
    top = cv2.resize(cv2.imread(img_fg_path),
                     (bottom.shape[1], bottom.shape[0]))
    overlapping = cv2.addWeighted(bottom, 0.8, top, 0.2, 0)
    cv2.namedWindow("Coordinate of Rectangle")
    cv2.imshow("Coordinate of Rectangle", overlapping)
    cv2.setMouseCallback("Coordinate of Rectangle", onClickMouse, overlapping)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Remove watermark/text
def inpaint(img, rects):
    height, width = img.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    for rect in rects:
        cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]),
                      (255, 255, 255), -1)
    img = cv2.inpaint(img, mask, 1.5, cv2.INPAINT_TELEA)
    return img


# Generate XML annotations
def generateXML(anno_path, image_id, frame, bboxes):
    height, width = frame.shape[0:2]
    xml_file = open(os.path.join(anno_path, "%06d.xml" % image_id), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + ("%06d.jpg" % image_id) +
                   '</filename>\n')
    xml_file.write('    <source>\n')
    xml_file.write(
        '        <database>https://anti-uav.github.io/</database>\n')
    xml_file.write('        <annotation>VOC2007</annotation>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>0</segmented>\n')
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = int(bbox[2]) + int(bbox[0])  # 根据需要转换成int/float或其他类型
        ymax = int(bbox[3]) + int(bbox[1])
        xml_file.write('    <object>\n')
        xml_file.write('        <name>uav</name>\n')
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


# Convert dataset to VOC format
def generateVOC(mode='IR'):
    # Only Support IR or RGB to evalute
    assert mode in ['IR', 'RGB']
    # setup experiments
    video_paths = glob.glob(os.path.join('test-dev(corrected)', '*'))
    image_id = 1
    # Handle each video
    for video_id, video_path in enumerate(video_paths, start=1):
        # video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4' % mode)
        gt_file = os.path.join(video_path, '%s_label.json' % mode)
        with open(gt_file, 'r') as f:
            label_gt = json.load(f)
        capture = cv2.VideoCapture(video_file)
        anno_path = os.path.join(video_path, mode, "Annotations")
        img_path = os.path.join(video_path, mode, "JPEGImages")
        if not os.path.exists(anno_path):
            os.makedirs(os.path.join(video_path, mode, "Annotations"))
        if not os.path.exists(img_path):
            os.makedirs(os.path.join(video_path, mode, "JPEGImages"))
        frame_id = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            _gt = label_gt['gt_rect'][frame_id]
            _exist = label_gt['exist'][frame_id]
            if _exist:
                frame = inpaint(frame, IR_mask)
                cv2.imwrite(os.path.join(img_path, "%06d.jpg" % (image_id)),
                            frame)
                generateXML(anno_path, image_id, frame, [_gt])
                image_id += 1
            frame_id += 1
        print("%s had been processed." % str(video_id))


# Show the ground truth bbox in video
def videoShowing(mode='IR'):
    # Only Support IR or RGB to evalute
    assert mode in ['IR', 'RGB']
    # setup experiments
    video_paths = glob.glob(os.path.join('test-dev(corrected)', '*'))
    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4' % mode)
        gt_file = os.path.join(video_path, '%s_label.json' % mode)
        with open(gt_file, 'r') as f:
            label_gt = json.load(f)
        capture = cv2.VideoCapture(video_file)
        frame_id = 0
        while True:
            ret, frame = capture.read()
            # cv2.imwrite("./%s.jpg"%mode, frame)
            # return
            if mode == 'RGB':
                frame = inpaint(frame, RGB_mask)
            else:
                frame = inpaint(frame, IR_mask)
            if not ret:
                capture.release()
                break
            _gt = label_gt['gt_rect'][frame_id]
            _exist = label_gt['exist'][frame_id]
            if _exist:
                cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])),
                              (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                              (0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2,
                            (0, 255, 0) if _exist else (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            cv2.waitKey(100)
            frame_id += 1
        cv2.destroyAllWindows()


# Show the ground truth bbox in overlapping video
def videoOverlapShowing():
    # setup experiments
    video_path = os.path.join('videos', '3')
    video_name = 'test'
    video_file_IR = os.path.join(video_path, 'IR.mp4')
    gt_file_IR = os.path.join(video_path, 'IR_label.json')
    video_file_RGB = os.path.join(video_path, 'RGB.mp4')
    gt_file_RGB = os.path.join(video_path, 'RGB_label.json')

    with open(gt_file_IR, 'r') as f:
        label_gt_IR = json.load(f)
    with open(gt_file_RGB, 'r') as f:
        label_gt_RGB = json.load(f)
    capture_IR = cv2.VideoCapture(video_file_IR)
    capture_RGB = cv2.VideoCapture(video_file_RGB)

    frame_id_RGB = 0
    frame_id_IR = 0
    # infrared image fast forward
    for i in range(3):
        ret_IR, frame_IR = capture_IR.read()
        frame_id_IR = frame_id_IR + 1

    while True:
        ret_IR, frame_IR = capture_IR.read()
        frame_IR = inpaint(frame_IR, IR_mask)
        ret_RGB, frame_RGB = capture_RGB.read()
        frame_RGB = inpaint(frame_RGB, RGB_mask)
        if not (ret_IR or ret_RGB):
            capture_IR.release()
            capture_RGB.release()
            break
        _gt_IR = label_gt_IR['gt_rect'][frame_id_IR]
        _exist_IR = label_gt_IR['exist'][frame_id_IR]
        _gt_RGB = label_gt_RGB['gt_rect'][frame_id_RGB]
        _exist_RGB = label_gt_RGB['exist'][frame_id_RGB]
        if (_exist_IR):
            cv2.rectangle(
                frame_IR, (int(_gt_IR[0]), int(_gt_IR[1])),
                (int(_gt_IR[0] + _gt_IR[2]), int(_gt_IR[1] + _gt_IR[3])),
                (0, 255, 0))
        if (_exist_RGB):
            cv2.rectangle(
                frame_RGB, (int(_gt_RGB[0]), int(_gt_RGB[1])),
                (int(_gt_RGB[0] + _gt_RGB[2]), int(_gt_RGB[1] + _gt_RGB[3])),
                (0, 255, 0))

        cv2.putText(frame_IR, 'exist' if _exist_IR else 'not exist',
                    (frame_IR.shape[1] // 2 - 20, 30), 1, 2,
                    (0, 255, 0) if _exist_IR else (0, 0, 255), 2)
        cv2.putText(frame_RGB, 'exist' if _exist_RGB else 'not exist',
                    (frame_RGB.shape[1] // 2 - 20, 30), 1, 2,
                    (0, 255, 0) if _exist_RGB else (0, 0, 255), 2)
        overlapping = cv2.addWeighted(
            frame_RGB, 0.5,
            cv2.resize(frame_IR, (frame_RGB.shape[1], frame_RGB.shape[0])),
            0.5, 0)
        cv2.imshow(video_name, overlapping)
        cv2.waitKey(100)
        frame_id_IR += 1
        frame_id_RGB += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # videoShowing(mode='IR')
    # getCoordinate("./RGB.jpg")
    # getCoordinate("./IR.jpg")
    # generateVOC('RGB')  # or 'IR'
    videoOverlapShowing()
