# python3
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random 
import xml.etree.ElementTree as ET
from PIL import ImageDraw, Image

from config import cfg
from augmentations import preprocess

class VOCAnnotationTransform(object):
    """当图片进行缩放处理之后，anno也要做相应变换

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = [1]
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            #label_idx = 1
           # bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 target_transform=VOCAnnotationTransform(),
                 mode='train'):

        self.root = root
        self.mode = mode
        self.target_transform = target_transform
        self._annopath = osp.join(root, 'Annotations', '%s.xml')
        self._imgpath = osp.join(root, 'JPEGImages', '%s.jpg')
        self.ids = []
        for line in open(osp.join(root, 'uavindex.txt')):
            self.ids.append(line.strip())

    def __getitem__(self, index):
        im, gt = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        while True:
            img_id = self.ids[index]
            img_path = self._imgpath % img_id
            target = ET.parse(self._annopath % img_id).getroot()
            img = Image.open(img_path)

            if img.mode == 'L':
                img = img.convert('RGB')

            width, height = img.size

            if self.target_transform is not None:
                target = self.target_transform(target, width, height)
            bbox_labels = target
            target = np.array(target)
            if target.ndim!=2:
                index = random.randrange(0, len(self.ids))
                continue
            # train和test：mode不同，预处理不同
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, img_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, len(self.ids))

        return torch.from_numpy(img), target

    def pull_image(self, index):
        '''返回指定index的PIL格式的原始图片

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img_path = self._imgpath % img_id
        img = Image.open(img_path)
        if img.mode=='L':
            img.convert('RGB')
        img = np.array(img)
        return img

    def pull_anno(self, index):
        '''返回指定index的原始anno

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''返回指定index的图片的原始tensor

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == '__main__':
    dataset = VOCDetection(cfg.VOC.HOME)
    dataset.pull_item(0)
