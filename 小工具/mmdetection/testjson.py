import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import copy
import datetime
import time

import skimage.io as io
import matplotlib.pyplot as plt
import pylab
#设置图片显示尺寸
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


'''
使用须知：
1.生成了结果文件(pkl,json)之后，评估前注释260之后几行
2.图片目录在443行修改
3.生成结果图片在478行
'''

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        self.setDetParams()
        self.iouType = iouType

def computeIoU(params, _gts, _dts, imgId, catId):
        p = params

        gt = _gts[imgId,catId]
        dt = _dts[imgId,catId]
        
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

def evaluateImg(params,_gts, _dts, ious, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = params
        if p.useCats:
            gt = _gts[imgId,catId]
            dt = _dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in _gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in _dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = ious[imgId, catId][:, gtind] if len(ious[imgId, catId]) > 0 else ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        #匹配dt和gt
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

def main():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    args = parser.parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    #单GPU检测
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)
    
    if args.out:
        #输出pkl文件
        mmcv.dump(outputs, args.out)

        print('Starting evaluate {}'.format(' and '.join("bbox")))
        result_file = args.out + '.json'
        
        #输出json文件
        results2json(dataset, outputs, result_file)

        coco = dataset.coco
        assert isinstance(coco, COCO)
        cocoGt   = coco
        cocoDt = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox'
        params   = {}
        params = Params(iouType=iou_type)
        params.imgIds = sorted(cocoGt.getImgIds())
        params.catIds = sorted(cocoGt.getCatIds())
        evalImgs = defaultdict(list)
        eval     = {}
        _gts = defaultdict(list)
        _dts = defaultdict(list)
        _paramsEval = {}
        stats = []
        ious = {}

        #prepare
        params.imgIds = list(np.unique(params.imgIds))
        params.catIds = list(np.unique(params.catIds))
        params.maxDets = sorted(params.maxDets)
        gts=cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=params.imgIds, catIds=params.catIds))
        dts=cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=params.imgIds, catIds=params.catIds))
        _gts = defaultdict(list)
        _dts = defaultdict(list)
        catIds = params.catIds
        for gt in gts:
            _gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            _dts[dt['image_id'], dt['category_id']].append(dt)
        

        print("图片总数为：")
        print(len(params.imgIds))
        pic_num=len(params.imgIds)
        
        #未找到目标的图片id
        dtnum, gtnum=[], []
        for i in _gts:
            gtnum.append(i[0])
        s1 = set(gtnum)
        for j in _dts:
            dtnum.append(j[0])
        s2 = set(dtnum)
        miss = s1 - s2
        print("未找到目标的图片id：")
        print(miss)
        #未找到目标id对应的图片名
        missname = []
        for i in miss:
            missname.append(cocoGt.loadImgs(i)[0]['file_name'])
        print(missname)

        #含有多检测框图片id
        multi = []
        for k,v in _dts.items():
            if len(v)>1:
                multi.append(k)
        print("含有多个检测框的图片总数：")
        print(len(multi))
        

        
        #evaluate
        ious = {(imgId, catId): computeIoU(params, _gts, _dts, imgId, catId) \
                        for imgId in params.imgIds
                        for catId in catIds}
        maxDet = params.maxDets[-1]    #100
        evalImgs = defaultdict(list)   # per-image per-category evaluation results
        evalImgs = [evaluateImg(params,_gts, _dts, ious, imgId, catId, areaRng, maxDet) \
                 for catId in catIds
                 for areaRng in params.areaRng
                 for imgId in params.imgIds
                 ]
        _paramsEval = copy.deepcopy(params)
        
        #IOU不符合的图片
        cnt_50 = []
        cnt_55 = []
        cnt_60 = []
        cnt_65 = []
        cnt_70 = []
        cnt_75 = []
        cnt_80 = []
        cnt_85 = []
        cnt_90 = []
        cnt_95 = []
        cnt_100 = []
        for i in range(pic_num):
            e = evalImgs[i]['dtMatches']
            zero_50 = np.count_nonzero(e[0])
            if zero_50==0:
                cnt_50.append(i+1)
                continue
            zero_55 = np.count_nonzero(e[1])
            if zero_55==0:
                cnt_55.append(i+1)
                continue
            zero_60 = np.count_nonzero(e[2])
            if zero_60==0:
                cnt_60.append(i+1)
                continue
            zero_65 = np.count_nonzero(e[3])
            if zero_65==0:
                cnt_65.append(i+1)
                continue
            zero_70 = np.count_nonzero(e[4])
            if zero_70==0:
                cnt_70.append(i+1)
                continue
            zero_75 = np.count_nonzero(e[5])
            if zero_75==0:
                cnt_75.append(i+1)
                continue
            zero_80 = np.count_nonzero(e[6])
            if zero_80==0:
                cnt_80.append(i+1)
                continue
            zero_85 = np.count_nonzero(e[7])
            if zero_85==0:
                cnt_85.append(i+1)
                continue
            zero_90 = np.count_nonzero(e[8])
            if zero_90==0:
                cnt_90.append(i+1)
                continue
            zero_95 = np.count_nonzero(e[9])
            if zero_95==0:
                cnt_95.append(i+1)
                continue
            cnt_100.append(i+1)
        print("IoU不足相应百分比的图片数量：")
        print("#50#")
        print(len(cnt_50))
        print("#55#")
        print(len(cnt_55))
        print("#60#")
        print(len(cnt_60))
        print("#65#")
        print(len(cnt_65))
        print("#70#")
        print(len(cnt_70))
        print("#75#")
        print(len(cnt_75))
        print("#80#")
        print(len(cnt_80))
        print("#85#")
        print(len(cnt_85))
        print("#90#")
        print(len(cnt_90))
        print("#95#")
        print(len(cnt_95))
        print("#100#")
        print(len(cnt_100))
        print(len(cnt_50)+len(cnt_55)+len(cnt_60)+len(cnt_65)+
              len(cnt_70)+len(cnt_75)+len(cnt_80)+len(cnt_85)+
              len(cnt_90)+len(cnt_95)+len(cnt_100))
        for i in range(pic_num):
            e = evalImgs[i]['dtMatches']
            #print(e[0])

        #画图
        
        #显示不足指定iou的图片
        imgIds_f = cocoDt.getImgIds(list(range(1,len(img_ids)+1)))    #未识别成功的图片(false)
        imgIds_t = cocoDt.getImgIds(list(set([x+1 for x in range(pic_num)]) - set(cnt_75)))
        
        imageFile = "/media/wl/000675B10007A33A/DatasetRepo/haier/JPEG/"
        
        plt.figure()

        for i in range(len(imgIds_f)):
            imgId = imgIds_f[i]
            Img_dt = cocoDt.loadImgs(imgId)[0]
            Img_gt = cocoGt.loadImgs(imgId)[0]
            imageUrl = imageFile+Img_dt['file_name']


            #显示GT标签
            annId_gt = cocoGt.getAnnIds(Img_gt['id'])
            imgAnn_gt = cocoGt.loadAnns(ids=annId_gt)
            #gt缺少segmentation字段，补上
            for ann in imgAnn_gt:
                ann['segmentation'] = [[ann['bbox'][0], ann['bbox'][1],
                                        ann['bbox'][0], ann['bbox'][1]+ann['bbox'][3],
                                        ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3],
                                        ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]]]
            I = io.imread(imageUrl)
            plt.subplot(1,2,1)
            plt.imshow(I)
            cocoGt.showAnns(imgAnn_gt)
            plt.title('GT')

            #显示DT标签
            annId_dt = cocoDt.getAnnIds(Img_dt['id'])
            imgAnn_dt = cocoDt.loadAnns(ids=annId_dt)
            plt.subplot(1,2,2)
            plt.imshow(I)
            cocoDt.showAnns(imgAnn_dt)
            plt.title('DT')
            #plt.show()

            #保存图片到指定文件夹
            plt.rcParams['savefig.dpi']=300 #dpi为300,图片尺寸为1800*1200
            plt.savefig('/home/wl/mmdetection/pic_result/haiermask/{picname}.svg'.format(picname=Img_gt['file_name']))
            plt.clf()
        #显示未识别到的图片
        '''
        print("未识别到目标的图片")
        for failpic in  miss:
            p = cocoGt.loadImgs(failpic)[0]
            imageUrl = imageFile+p['file_name']
            I=io.imread(imageUrl)
            plt.subplot(1,2,1)
            plt.imshow(I)
            Id_gt = cocoGt.getAnnIds(p['id'])
            Ann_gt = cocoGt.loadAnns(ids=Id_gt)
            for ann in Ann_gt:
                ann['segmentation'] = [[ann['bbox'][0], ann['bbox'][1],
                                        ann['bbox'][0], ann['bbox'][1]+ann['bbox'][3],
                                        ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3],
                                        ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]]]
            
            cocoGt.showAnns(Ann_gt)
            #plt.rcParams['savefig.dpi']=300 #dpi为300,图片尺寸为1800*1200
            #plt.savefig('/home/wl/mmdetection/pic_result/miss/{picname}'.format(picname=p['file_name']))
            plt.subplot(1,2,2)
            plt.imshow(I)
            plt.show()
        '''
            

        evalsmall = [evaluateImg(params,_gts, _dts, ious, imgId, catId, [0 ** 2, 32 ** 2], maxDet) \
                 for catId in catIds
                 for imgId in params.imgIds
                 ]
        #print("evalImags:")
        #print(evalsmall[0])

        #accumulate
        p = params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        # create dictionary for future indexing
        _pe = _paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }

        #summarize
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=params.maxDets[2])
            stats[6] = _summarize(0, maxDets=params.maxDets[0])
            stats[7] = _summarize(0, maxDets=params.maxDets[1])
            stats[8] = _summarize(0, maxDets=params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        iouType = params.iouType
        summarize = _summarizeDets
        stats = summarize()

if __name__ == '__main__':
    main()


