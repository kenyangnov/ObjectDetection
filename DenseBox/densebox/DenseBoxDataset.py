import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

#test
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#注：此处bbox有三类坐标空间:原图空间(ORI),resize后规定尺寸后的坐标空间(480*480),输出特征图的坐标空间(120*120)

class DenseBoxDataset(Dataset):
    """
    自定义数据集
    """
    CLASSES = ('scratch', )

    def __init__(self, root, ann_file = 'train.json', size=(480, 480), test_mode=False):
        """
        初始化
        将坐标空间从原图转换到size空间
        """
        self.root = root
        self.size = size
        self.img_infos = self.load_annotations(root+ann_file)
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #            std=[0.229, 0.224, 0.225])
        ])
        self.test_mode = test_mode
        self.label_maps = []
        self.mask_maps = []

        for idx in range(len(self.img_infos)):
            img_info = self.img_infos[idx]
            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            # bbox坐标转换到480*480的坐标空间(输出特征图)
            ori_shape = (img_info['width'], img_info['height'], 3)
            dw = 1.0/ori_shape[0]
            dh = 1.0/ori_shape[1]
            for i in range(len(gt_bboxes)):
                gt_bboxes[i][0] = gt_bboxes[i][0]*dw*self.size[0]
                gt_bboxes[i][1] = gt_bboxes[i][1]*dh*self.size[1]
                gt_bboxes[i][2] = gt_bboxes[i][2]*dw*self.size[0]
                gt_bboxes[i][3] = gt_bboxes[i][3]*dh*self.size[1]

            # 初始化label_map(score_map+dist_map)和mask_map
            label_map = torch.zeros([5, 120, 120], dtype=torch.float32)
            mask_map = torch.zeros([1, 120, 120], dtype=torch.float32)

            # 填充label_map(score_map+dist_map)
            score_map = label_map[0]
            self.init_score_map(score_map=score_map, bboxes = gt_bboxes, ratio=1.0)
            #print(torch.nonzero(label_map[0]))
            dxt_map, dyt_map = label_map[1], label_map[2]
            dxb_map, dyb_map = label_map[3], label_map[4]
            self.init_dist_map(dxt_map=dxt_map, dyt_map=dyt_map,
                                dxb_map=dxb_map, dyb_map=dyb_map,
                                bboxes = gt_bboxes)
            self.label_maps.append(label_map)
            # 初始化loss mask map
            self.init_mask_map(mask_map=mask_map, bboxes = gt_bboxes, ratio=1.0)
            self.mask_maps.append(mask_map)

        """
        self.transform = T.Compose([
            T.Resize(self.size),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
        """


    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        """
        :return:
        """
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        """
        先获取图片id,再得到对应的ann_id,最后得到该图片的annotation信息
        """
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)
    

    def _parse_ann_info(self, ann_info):
        """
        返回某张图片中annotation的详细信息
        """
        gt_bboxes = []
        gt_labels = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            # 将bbox转换成[xmin, ymin, xmax, ymax]的形式
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels)
        return ann

    # 初始化score_map
    def init_score_map(self,
                       score_map,
                       bboxes,
                       ratio=0.3):
        """
        初始化score_map
        ratio为保留的中心区域的比率
        """
        # assert score_map.size == torch.Size([120, 120])
        for bbox in bboxes:
            #首先转换到120*120坐标空间
            leftup_x = bbox[0] / 4.0
            leftup_y = bbox[1] / 4.0
            rightdown_x = bbox[2] / 4.0
            rightdown_y = bbox[3] / 4.0

            bbox_center_x = float(leftup_x + rightdown_x) * 0.5
            bbox_center_y = float(leftup_y + rightdown_y) * 0.5
            bbox_w = rightdown_x - leftup_x
            bbox_h = rightdown_y - leftup_y

            org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
            org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
            end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
            end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
            score_map[org_y: end_y + 1, org_x: end_x + 1] = 1.0
        # verify...
        #print(torch.nonzero(score_map))
    
    #初始化dist_map
    def init_dist_map(self,
                      dxt_map,
                      dyt_map,
                      dxb_map,
                      dyb_map,
                      bboxes):

        # assert dxt_map.size == torch.Size([120, 120])
        for bbox in bboxes:
            #首先转换到120*120坐标空间
            leftup_x = bbox[0] / 4.0
            leftup_y = bbox[1] / 4.0
            rightdown_x = bbox[2] / 4.0
            rightdown_y = bbox[3] / 4.0

            bbox_w = rightdown_x - leftup_x
            bbox_h = rightdown_y - leftup_y

            for y in range(dxt_map.size(0)):  # dim H
                for x in range(dxt_map.size(1)):  # dim W
                    dist_xt = (float(x) - leftup_x) / bbox_w
                    dist_yt = (float(y) - leftup_y) / bbox_h
                    dist_xb = (float(x) - rightdown_x) / bbox_w
                    dist_yb = (float(y) - rightdown_y) / bbox_h
                    # 行和列分别是y和x
                    dxt_map[y, x] = dist_xt
                    dyt_map[y, x] = dist_yt
                    dxb_map[y, x] = dist_xb
                    dyb_map[y, x] = dist_yb

    #初始化mask_map
    def init_mask_map(self,
                      mask_map,
                      bboxes,
                      ratio=0.3):

        # assert mask_map.size == torch.Size([1, 120, 120])
        for bbox in bboxes:
            #首先转换到输出特征图的120*120坐标空间
            leftup_x = bbox[0] / 4.0
            leftup_y = bbox[1] / 4.0
            rightdown_x = bbox[2] / 4.0
            rightdown_y = bbox[3] / 4.0

            bbox_center_x = float(leftup_x + rightdown_x) * 0.5
            bbox_center_y = float(leftup_y + rightdown_y) * 0.5
            bbox_w = rightdown_x - leftup_x
            bbox_h = rightdown_y - leftup_y

            org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
            org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
            end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
            end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
            mask_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # 加载图片
        img = Image.open(self.root+ 'JPEG/' + img_info['filename'])
        #print(img_info['filename'])
        # 转换为RGB格式以及规定尺寸
        if img.mode == 'L' or img.mode == 'I':  # 8bit or 32bit gray-scale
            img = img.convert('RGB')
        img = self.transform(img)
        #返回的数据
        return img, self.label_maps[idx], self.mask_maps[idx]


class DenseBoxDatasetOnline(Dataset):
    """
    自定义数据集
    """
    CLASSES = ('scratch', )

    def __init__(self, root, ann_file = 'train.json', size=(480, 480), test_mode=False):
        """
        初始化
        """
        self.root = root
        self.size = size
        self.img_infos = self.load_annotations(root+ann_file)
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #            std=[0.229, 0.224, 0.225])
        ])
        self.test_mode = test_mode
        self.bboxes = []

        for idx in range(len(self.img_infos)):
            img_info = self.img_infos[idx]
            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            # bbox坐标从原图空间转换到480*480(输出特征图)的坐标空间
            ori_shape = (img_info['width'], img_info['height'], 3)
            dw = 1.0/ori_shape[0]
            dh = 1.0/ori_shape[1]
            bbox = []
            for i in range(len(gt_bboxes)):
                gt_bboxes[i][0] = gt_bboxes[i][0]*dw*self.size[0]
                gt_bboxes[i][1] = gt_bboxes[i][1]*dh*self.size[1]
                gt_bboxes[i][2] = gt_bboxes[i][2]*dw*self.size[0]
                gt_bboxes[i][3] = gt_bboxes[i][3]*dh*self.size[1]
                bbox.append([gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]])
            # 初始化label_map(score_map+dist_map)和mask_map
            self.bboxes.append(bbox)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        """
        :return:
        """
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        """
        先获取图片id,再得到对应的ann_id,最后得到该图片的annotation信息
        """
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)
    

    def _parse_ann_info(self, ann_info):
        """
        返回某张图片中annotation的详细信息
        """
        gt_bboxes = []
        gt_labels = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            # 将bbox转换成[xmin, ymin, xmax, ymax]的形式
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels)
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # 加载图片
        img = Image.open(self.root+ 'JPEG/' + img_info['filename'])
        #print(img_info['filename'])
        # 转换为RGB格式以及规定尺寸
        if img.mode == 'L' or img.mode == 'I':  # 8bit or 32bit gray-scale
            img = img.convert('RGB')
        img = self.transform(img)
        #show(img, self.bboxes[idx])

        return img, self.bboxes[idx]

# 显示图片
def show(img, bboxes):
    """
    img为PIL的Image格式
    bbox为list
    """
    img = np.array(img)
    img = np.transpose(img,(1,2,0))
    plt.imshow(img)
    #画矩形框
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        top = ([xmin, xmax], [ymin, ymin])
        right = ([xmax, xmax], [ymin, ymax])
        botton = ([xmax, xmin], [ymax, ymax])
        left = ([xmin, xmin], [ymax, ymin])
        lines = [top, right, botton, left]
        for line in lines:
            plt.plot(*line, color = 'r')
            plt.scatter(*line, color = 'b')
    #调整原点到左上角
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    plt.show()


