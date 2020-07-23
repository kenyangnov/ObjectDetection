from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np

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
RGB_mask = [
    rect_mask_4, rect_mask_5, rect_mask_6, rect_mask_7, rect_mask_8,
    rect_mask_9, rect_mask_10, rect_mask_11, rect_mask_12, rect_mask_13,
    rect_mask_14, rect_mask_15, rect_mask_16, rect_mask_17, rect_mask_18,
    rect_mask_20
]


# Callback when clicking mouse
def onClickMouse(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        tmp = param.copy()
        cv2.rectangle(tmp, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Coordinate of Rectangle", tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(param, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Coordinate of Rectangle", param)
        coordinate = "x1, y1, x2, y2: [ %s, %s, %s, %s ]" % (ix, iy, x, y)
        print(coordinate)
    elif event == cv2.EVENT_MOUSEMOVE and flags != cv2.EVENT_FLAG_LBUTTON:
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
    xml_file.write('    <filename>' + ("%06d.jpg" % image_id) + '</filename>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>https://anti-uav.github.io/</database>\n')
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


if __name__ == '__main__':
    # videoShowing(mode='IR')
    # getCoordinate("./IR.jpg"
    generateVOC('IR')  # or 'RGB'
