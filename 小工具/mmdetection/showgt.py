import argparse
import mmcv
import torch
from mmdet.datasets import build_dataloader, get_dataset
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
#设置图片显示尺寸
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


'''
使用说明：
显示coco类型数据的GT，参数是config文件
1.在40行修改图片目录
'''

def main():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    coco = dataset.coco
    assert isinstance(coco, COCO)
    cocoGt   = coco
    
    #画图
    imgIds = cocoGt.getImgIds(list(x+1 for x in range(165)))
    imageFile = "/media/wl/000675B10007A33A/DatasetRepo/wider_face_split/WIDER_train/images/"
        
    plt.figure()
    for i in range(len(imgIds)):
        imgId = imgIds[i]
        Img_gt = cocoGt.loadImgs(imgId)[0]
        imageUrl = imageFile+Img_gt['file_name']
        #显示GT标签
        annId_gt = cocoGt.getAnnIds(Img_gt['id'])
        imgAnn_gt = cocoGt.loadAnns(ids=annId_gt)
        print(imgAnn_gt)
        I = io.imread(imageUrl)
        plt.imshow(I)
        cocoGt.showAnns(imgAnn_gt)
        plt.title('GT')
        plt.show()
        #plt.clf()
        

if __name__ == '__main__':
    main()

